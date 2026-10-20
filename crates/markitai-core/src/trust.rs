//! The certificate authorities markitai's HTTPS clients trust. Every client
//! starts from [`client_builder`]: the Mozilla roots built into markitai,
//! plus the operating system's store (the Windows and macOS stores, or the
//! OpenSSL bundle and directory on Linux and other Unix systems). When
//! `SSL_CERT_FILE` or `SSL_CERT_DIR` is set, the PEM certificates they name
//! are read instead of the system store, as OpenSSL reads them. A company
//! proxy that re-signs HTTPS with its own authority is then trusted wherever
//! that authority is installed, while every site the built-in roots verify
//! keeps working when the system store cannot be read.
use std::sync::OnceLock;

/// A blocking client builder trusting the built-in roots and, beside them,
/// the system's or the `SSL_CERT_FILE`/`SSL_CERT_DIR` certificates.
pub(crate) fn client_builder() -> reqwest::blocking::ClientBuilder {
    added()
        .iter()
        .cloned()
        .fold(reqwest::blocking::Client::builder(), |builder, root| {
            builder.add_root_certificate(root)
        })
}

/// The roots added to the built-in ones, read once per process.
fn added() -> &'static [reqwest::Certificate] {
    static ADDED: OnceLock<Vec<reqwest::Certificate>> = OnceLock::new();
    ADDED.get_or_init(|| {
        let loaded = rustls_native_certs::load_native_certs();
        // An unreadable system store is no failure (the built-in roots
        // remain); one the user named is worth a line.
        let named = ["SSL_CERT_FILE", "SSL_CERT_DIR"]
            .iter()
            .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()));
        if named {
            for error in &loaded.errors {
                eprintln!("markitai: certificates in SSL_CERT_FILE/SSL_CERT_DIR not read: {error}");
            }
        }
        usable(loaded.certs)
    })
}

/// The certificates a TLS client can use as roots. A store often carries
/// certificates rustls cannot read as a trust anchor (old roots without
/// extensions); one of them would make every client fail to build, so each
/// is checked and skipped alone.
fn usable(
    certificates: Vec<rustls::pki_types::CertificateDer<'static>>,
) -> Vec<reqwest::Certificate> {
    certificates
        .into_iter()
        .filter_map(|der| {
            rustls::RootCertStore::empty()
                .add(der.clone())
                .ok()
                .and_then(|()| reqwest::Certificate::from_der(&der).ok())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    /// A DER element: `tag`, its definite length and `content`.
    fn tlv(tag: u8, content: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        let len = content.len();
        if len < 0x80 {
            out.push(len as u8);
        } else {
            let bytes: Vec<u8> = len
                .to_be_bytes()
                .into_iter()
                .skip_while(|b| *b == 0)
                .collect();
            out.push(0x80 | bytes.len() as u8);
            out.extend(bytes);
        }
        out.extend_from_slice(content);
        out
    }

    fn seq(parts: &[Vec<u8>]) -> Vec<u8> {
        tlv(0x30, &parts.concat())
    }

    /// A name of one common name.
    fn name(common: &str) -> Vec<u8> {
        seq(&[tlv(
            0x31,
            &seq(&[tlv(0x06, &[0x55, 0x04, 0x03]), tlv(0x0C, common.as_bytes())]),
        )])
    }

    /// An Ed25519 key, as PKCS #8 and as its key pair.
    fn key() -> (Vec<u8>, ring::signature::Ed25519KeyPair) {
        let pkcs8 =
            ring::signature::Ed25519KeyPair::generate_pkcs8(&ring::rand::SystemRandom::new())
                .unwrap();
        let pair = ring::signature::Ed25519KeyPair::from_pkcs8(pkcs8.as_ref()).unwrap();
        (pkcs8.as_ref().to_vec(), pair)
    }

    /// An X.509 v3 certificate for `subject`'s key, signed by `issuer`.
    fn certificate(
        serial: u8,
        issuer: (&str, &ring::signature::Ed25519KeyPair),
        subject: (&str, &ring::signature::Ed25519KeyPair),
        extension: Vec<u8>,
    ) -> Vec<u8> {
        use ring::signature::KeyPair;
        let ed25519 = seq(&[tlv(0x06, &[0x2B, 0x65, 0x70])]);
        let mut key = vec![0];
        key.extend_from_slice(subject.1.public_key().as_ref());
        let tbs = seq(&[
            tlv(0xA0, &tlv(0x02, &[2])),
            tlv(0x02, &[serial]),
            ed25519.clone(),
            name(issuer.0),
            seq(&[tlv(0x17, b"200101000000Z"), tlv(0x18, b"20991231235959Z")]),
            name(subject.0),
            seq(&[ed25519.clone(), tlv(0x03, &key)]),
            tlv(0xA3, &seq(&[extension])),
        ]);
        let mut signature = vec![0];
        signature.extend_from_slice(issuer.1.sign(&tbs).as_ref());
        seq(&[tbs, ed25519, tlv(0x03, &signature)])
    }

    /// A test authority, and a server certificate it signed for
    /// `localhost` with that certificate's PKCS #8 key, all DER.
    fn authority() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (_, ca_key) = key();
        let (leaf_pkcs8, leaf_key) = key();
        let ca_name = "markitai test authority";
        // basicConstraints, critical: a certificate authority.
        let ca_extension = seq(&[
            tlv(0x06, &[0x55, 0x1D, 0x13]),
            tlv(0x01, &[0xFF]),
            tlv(0x04, &seq(&[tlv(0x01, &[0xFF])])),
        ]);
        // subjectAltName: dNSName localhost.
        let leaf_extension = seq(&[
            tlv(0x06, &[0x55, 0x1D, 0x11]),
            tlv(0x04, &seq(&[tlv(0x82, b"localhost")])),
        ]);
        let ca = certificate(1, (ca_name, &ca_key), (ca_name, &ca_key), ca_extension);
        let leaf = certificate(
            2,
            (ca_name, &ca_key),
            ("localhost", &leaf_key),
            leaf_extension,
        );
        (ca, leaf, leaf_pkcs8)
    }

    /// A TLS server on loopback answering `ok` to `requests` requests with
    /// `leaf`; its port.
    fn serve(leaf: Vec<u8>, pkcs8: Vec<u8>, requests: usize) -> (u16, std::thread::JoinHandle<()>) {
        let config = std::sync::Arc::new(
            rustls::ServerConfig::builder_with_provider(std::sync::Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(
                vec![leaf.into()],
                rustls::pki_types::PrivateKeyDer::Pkcs8(pkcs8.into()),
            )
            .unwrap(),
        );
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            for _ in 0..requests {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
                let mut connection = rustls::ServerConnection::new(config.clone()).unwrap();
                let mut tls = rustls::Stream::new(&mut connection, &mut stream);
                let mut request = [0u8; 1024];
                if tls.read(&mut request).is_ok() {
                    let _ = tls.write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                    );
                    let _ = tls.flush();
                }
            }
        });
        (port, server)
    }

    #[test]
    fn unusable_store_entries_are_skipped_not_fatal() {
        let (ca, _, _) = authority();
        let garbage = rustls::pki_types::CertificateDer::from(b"not a certificate".to_vec());
        let roots = usable(vec![garbage, ca.into()]);
        assert_eq!(roots.len(), 1);
        // A builder carrying them still builds.
        roots
            .into_iter()
            .fold(reqwest::blocking::Client::builder(), |builder, root| {
                builder.add_root_certificate(root)
            })
            .build()
            .unwrap();
    }

    /// A server whose certificate only the test authority vouches for is
    /// refused by the built-in roots alone and reached once that authority
    /// is among the added roots.
    #[test]
    fn an_added_authority_is_trusted_beside_the_built_in_roots() {
        let (ca, leaf, pkcs8) = authority();
        let (port, server) = serve(leaf, pkcs8, 2);
        let url = format!("https://localhost:{port}/");
        let bundled = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        let error = bundled.get(&url).send().unwrap_err();
        assert!(format!("{error:?}").contains("UnknownIssuer"), "{error:?}");
        let trusted = usable(vec![ca.into()])
            .into_iter()
            .fold(reqwest::blocking::Client::builder(), |builder, root| {
                builder.add_root_certificate(root)
            })
            .no_proxy()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        assert_eq!(trusted.get(&url).send().unwrap().text().unwrap(), "ok");
        server.join().unwrap();
    }

    /// `SSL_CERT_FILE` naming the authority makes [`client_builder`]'s
    /// clients trust it, in a child process whose environment has it from
    /// the start (the roots are read once per process).
    #[test]
    fn ssl_cert_file_adds_its_authority_to_every_client() {
        const CASE: &str = "MARKITAI_TRUST_TEST_URL";
        if let Ok(url) = std::env::var(CASE) {
            let client = client_builder()
                .no_proxy()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap();
            assert_eq!(client.get(&url).send().unwrap().text().unwrap(), "ok");
            return;
        }
        let (ca, leaf, pkcs8) = authority();
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("authority.pem");
        let pem = {
            use base64::Engine as _;
            let body = base64::engine::general_purpose::STANDARD.encode(&ca);
            let lines: Vec<&str> = body
                .as_bytes()
                .chunks(64)
                .map(|line| std::str::from_utf8(line).unwrap())
                .collect();
            format!(
                "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
                lines.join("\n")
            )
        };
        std::fs::write(&file, pem).unwrap();
        let (port, server) = serve(leaf, pkcs8, 1);
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "trust::tests::ssl_cert_file_adds_its_authority_to_every_client",
                "--nocapture",
            ])
            .env(CASE, format!("https://localhost:{port}/"))
            .env("SSL_CERT_FILE", &file)
            .env_remove("SSL_CERT_DIR")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        server.join().unwrap();
    }

    /// Every HTTP client is built through [`client_builder`], so none of them
    /// trusts the built-in roots alone.
    #[test]
    fn every_client_starts_from_the_shared_builder() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut pending = vec![root];
        let mut offenders = Vec::new();
        while let Some(directory) = pending.pop() {
            for entry in std::fs::read_dir(&directory).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    // A `tests` directory is test code throughout.
                    if path.file_name().is_none_or(|name| name != "tests") {
                        pending.push(path);
                    }
                    continue;
                }
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                if !name.ends_with(".rs") || name.ends_with("tests.rs") || name == "trust.rs" {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                // Only the code before a file's own test module counts.
                let code = text.split("#[cfg(test)]\nmod ").next().unwrap_or_default();
                if code.contains("Client::builder()") || code.contains("ClientBuilder::new()") {
                    offenders.push(path.display().to_string());
                }
            }
        }
        assert!(offenders.is_empty(), "{offenders:?}");
    }
}
