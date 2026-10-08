use crate::fetch::is_private;
use crate::{Asset, Document, Error, Result, config, output_profiles};
use base64::Engine;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;
use std::time::Duration;
use url::{Host, Url};

const MAX_IMAGE: usize = 64 * 1024 * 1024;
const MAX_ADDED: usize = 100 * 1024 * 1024;
const MAX_REFERENCES: usize = 1024;

fn failure(message: &str) -> Error {
    Error::Conversion(format!("Image resource: {message}"))
}

pub(super) fn asset_name(target: &str) -> Option<String> {
    let path = target.split(['?', '#']).next()?;
    let path = String::from_utf8(unquote(path.as_bytes()).ok()?).ok()?;
    path.strip_prefix(".markitai/assets/")
        .or_else(|| path.strip_prefix("assets/"))
        .map(str::to_owned)
}

/// Which image references become owned assets.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Scope {
    /// Inline data images only; nothing is read or downloaded.
    Data,
    /// Data, local and HTTP(S) images, for image enrichment.
    All,
}

pub(super) fn prepare(
    doc: &mut Document,
    source: &str,
    cfg: &serde_json::Value,
    scope: Scope,
) -> Result<()> {
    let references = output_profiles::image_references(&doc.markdown);
    if references.len() > MAX_REFERENCES {
        return Err(failure(
            "more than 1024 image references; no resources were downloaded",
        ));
    }
    let owned: HashSet<_> = doc.assets.iter().map(|asset| asset.name.clone()).collect();
    let mut names = owned.clone();
    let mut replacements = HashMap::new();
    let mut attempted = HashSet::new();
    let mut total = 0usize;
    let base = Url::parse(source)
        .ok()
        .filter(|url| matches!(url.scheme(), "http" | "https"));
    let allow_private = base.as_ref().is_some_and(private_origin);
    for target in references {
        if scope == Scope::Data && !target.starts_with("data:image/") {
            continue;
        }
        if elided_data(&target) {
            // A converter's placeholder for an inline image whose payload it
            // did not keep; there is nothing to localize and nothing wrong.
            continue;
        }
        if asset_name(&target).is_some_and(|name| owned.contains(&name))
            || !attempted.insert(target.clone())
        {
            continue;
        }
        let result = if target.starts_with("data:") {
            data_image(&target)
        } else if let Some(base) = &base {
            base.join(&target)
                .map_err(|_| failure("invalid image URL"))
                .and_then(|url| download(url, allow_private))
        } else if matches!(Url::parse(&target), Ok(url) if matches!(url.scheme(), "http" | "https"))
        {
            download(Url::parse(&target).unwrap(), false)
        } else {
            local_image(
                source,
                &target,
                config::enabled(cfg, "/output/allow_symlinks"),
            )
        };
        match result {
            Ok(bytes) => {
                total = total
                    .checked_add(bytes.len())
                    .ok_or_else(|| failure("image total overflow"))?;
                if total > MAX_ADDED {
                    return Err(failure(
                        "downloaded/localized image bytes exceed the 100 MiB document budget",
                    ));
                }
                let suffix = image::guess_format(&bytes)
                    .ok()
                    .and_then(|format| format.extensions_str().first().copied())
                    .unwrap_or("img");
                let digest = crate::hex(Sha256::digest(&bytes));
                let mut index = 0;
                let name = loop {
                    let name = format!("image-{}-{index}.{suffix}", &digest[..20]);
                    if names.insert(name.clone()) {
                        break name;
                    }
                    index += 1;
                };
                replacements.insert(target, format!(".markitai/assets/{name}"));
                doc.assets.push(Asset { name, bytes });
            }
            Err(error) => {
                let label = if target.starts_with("data:") {
                    "embedded data image".into()
                } else {
                    crate::output::redact_url(&target)
                };
                doc.warnings.push(format!("Image resource {label} could not be localized: {error}; the original reference was retained."));
            }
        }
    }
    doc.markdown = output_profiles::rewrite_image_targets(&doc.markdown, &replacements);
    Ok(())
}
fn unquote(input: &[u8]) -> Result<Vec<u8>> {
    let mut decoded = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] == b'%' && i + 2 < input.len() {
            let hex = |byte: u8| char::from(byte).to_digit(16).map(|n| n as u8);
            if let (Some(a), Some(b)) = (hex(input[i + 1]), hex(input[i + 2])) {
                decoded.push(a * 16 + b);
                i += 3;
                continue;
            }
        }
        decoded.push(input[i]);
        i += 1;
    }
    Ok(decoded)
}
/// `data:image/png;base64...`: the media type with the payload left out.
fn elided_data(target: &str) -> bool {
    target.starts_with("data:") && !target.contains(',') && target.ends_with("...")
}
fn data_image(target: &str) -> Result<Vec<u8>> {
    let (header, body) = target
        .split_once(',')
        .ok_or_else(|| failure("malformed data URI"))?;
    if !header.starts_with("data:image/") || target.len() > MAX_IMAGE * 4 / 3 + 1024 {
        return Err(failure("data URI is not an image or exceeds 64 MiB"));
    }
    let bytes = if header.ends_with(";base64") {
        base64::engine::general_purpose::STANDARD
            .decode(body)
            .map_err(|_| failure("invalid base64 image"))?
    } else {
        unquote(body.as_bytes())?
    };
    if bytes.len() > MAX_IMAGE {
        return Err(failure("data image exceeds 64 MiB"));
    }
    Ok(bytes)
}
fn local_image(source: &str, target: &str, allow_symlinks: bool) -> Result<Vec<u8>> {
    if Url::parse(target).is_ok()
        || target.contains("://")
        || target.starts_with('#')
        || target.starts_with('/')
    {
        return Err(failure("unsupported or absolute local image target"));
    }
    let text = String::from_utf8(unquote(
        target.split(['?', '#']).next().unwrap_or("").as_bytes(),
    )?)
    .map_err(|_| failure("image path is not UTF-8"))?;
    let root = Path::new(source)
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let path = root.join(text);
    crate::output::check_path(&path, allow_symlinks)?;
    let path = path.canonicalize()?;
    if !path.starts_with(root) {
        return Err(failure("image path escapes the source directory"));
    }
    let meta = std::fs::metadata(&path)?;
    if !meta.is_file() || meta.len() > MAX_IMAGE as u64 {
        return Err(failure("local image is not a file within the 64 MiB limit"));
    }
    crate::platform::read_limited(std::fs::File::open(path)?, MAX_IMAGE as u64)?
        .ok_or_else(|| failure("local image grew beyond 64 MiB"))
}
fn private_origin(url: &Url) -> bool {
    match url.host() {
        Some(Host::Domain(host)) => {
            host.eq_ignore_ascii_case("localhost") || host.ends_with(".localhost")
        }
        Some(Host::Ipv4(address)) => is_private(address.into()),
        Some(Host::Ipv6(address)) => is_private(address.into()),
        None => false,
    }
}
fn download(mut url: Url, allow_private: bool) -> Result<Vec<u8>> {
    for _ in 0..=5 {
        if !matches!(url.scheme(), "http" | "https")
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(failure(
                "image URL must be HTTP(S) without embedded credentials",
            ));
        }
        let host = url
            .host_str()
            .ok_or_else(|| failure("image URL has no hostname"))?;
        let addresses = url
            .socket_addrs(|| None)
            .map_err(|_| failure("image hostname resolution failed"))?;
        if addresses.is_empty()
            || !allow_private && addresses.iter().any(|address| is_private(address.ip()))
        {
            return Err(failure(
                "public/local-file documents cannot fetch private image targets",
            ));
        }
        // Pin validated DNS addresses and use direct transport: inherited proxy
        // credentials and browser cookies must not be sent to document images.
        let client = Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .resolve_to_addrs(host, &addresses)
            .user_agent(concat!("markitai/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|_| failure("cannot initialize image HTTP client"))?;
        let response = client
            .get(url.clone())
            .send()
            .map_err(|_| failure("image request failed"))?;
        if response.status().is_redirection() {
            let target = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .ok_or_else(|| failure("image redirect has no valid Location"))?;
            url = url
                .join(target)
                .map_err(|_| failure("invalid image redirect"))?;
            continue;
        }
        if !response.status().is_success() {
            return Err(failure(&format!(
                "image HTTP status {}",
                response.status().as_u16()
            )));
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_IMAGE as u64)
        {
            return Err(failure("image response exceeds 64 MiB"));
        }
        let mut bytes = Vec::new();
        response
            .take(MAX_IMAGE as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| failure("cannot read image body"))?;
        if bytes.len() > MAX_IMAGE {
            return Err(failure("image body exceeds 64 MiB"));
        }
        return Ok(bytes);
    }
    Err(failure("image redirect limit exceeded"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn elided_inline_images_are_placeholders_not_malformed_data() {
        let markdown =
            "![kept](data:image/png;base64...)\n\n![broken](data:image/png;base64)\n".to_owned();
        let mut doc = Document {
            markdown: markdown.clone(),
            ..Default::default()
        };
        prepare(&mut doc, "page.html", &serde_json::json!({}), Scope::Data).unwrap();
        assert_eq!(doc.markdown, markdown);
        assert!(doc.assets.is_empty());
        assert_eq!(doc.warnings.len(), 1, "{:?}", doc.warnings);
        assert!(doc.warnings[0].contains("malformed data URI"));
    }
    #[test]
    fn owned_paths_decode_once_and_keep_query_separate() {
        assert_eq!(
            asset_name(".markitai/assets/a%20b.png?raw=1"),
            Some("a b.png".into())
        );
        assert_eq!(
            asset_name(".markitai/assets/a%2520b.png"),
            Some("a%20b.png".into())
        );
        assert_eq!(asset_name("https://example.test/a.png"), None);
    }
    #[test]
    fn local_resource_cannot_escape_source_tree_or_read_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let inner = dir.path().join("inner");
        std::fs::create_dir(&inner).unwrap();
        std::fs::write(dir.path().join("outside.png"), b"private").unwrap();
        let source = inner.join("doc.md");
        assert!(local_image(source.to_str().unwrap(), "../outside.png", false).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(dir.path().join("outside.png"), inner.join("link.png"))
                .unwrap();
            assert!(local_image(source.to_str().unwrap(), "link.png", false).is_err());
        }
    }
    #[test]
    #[cfg(unix)]
    fn unresolved_cid_cannot_read_an_existing_same_named_local_file() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("message.eml");
        std::fs::write(
            dir.path().join("cid:photo"),
            b"must not become the missing MIME image",
        )
        .unwrap();
        assert!(local_image(source.to_str().unwrap(), "cid:photo", false).is_err());
        assert!(local_image(source.to_str().unwrap(), "CID:photo", false).is_err());
        assert!(local_image(source.to_str().unwrap(), "file:photo", false).is_err());
        // An explicitly encoded colon is a local filename, not a URI scheme.
        assert_eq!(
            local_image(source.to_str().unwrap(), "cid%3Aphoto", false).unwrap(),
            b"must not become the missing MIME image"
        );
    }

    #[test]
    fn ipv6_literal_hosts_are_classified_instead_of_failing_to_resolve() {
        for (page, private) in [
            ("http://[::1]:8080/page", true),
            ("http://[fd00::1]/page", true),
            ("http://[64:ff9b::7f00:1]/page", true),
            ("http://[2606:4700:4700::1111]/page", false),
            ("http://192.0.2.1/page", true),
        ] {
            assert_eq!(
                private_origin(&Url::parse(page).unwrap()),
                private,
                "{page}"
            );
        }
        let error = download(Url::parse("http://[::1]:9/image").unwrap(), false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("private image targets"), "{error}");
    }
    #[test]
    fn private_networks_and_embedded_credentials_cannot_bypass_download_policy() {
        for address in [
            "127.0.0.1",
            "169.254.169.254",
            "10.0.0.1",
            "100.64.0.1",
            "::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(is_private(address.parse().unwrap()));
        }
        assert!(
            download(
                Url::parse("http://user:secret@127.0.0.1/image").unwrap(),
                true
            )
            .is_err()
        );
        assert!(download(Url::parse("http://127.0.0.1/image").unwrap(), false).is_err());
    }
}
