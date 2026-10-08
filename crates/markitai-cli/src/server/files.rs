use super::{
    State, http,
    jobs::{self, Job},
    store,
    types::{ApiError, ApiResult},
};
use axum::{
    Json,
    body::Body,
    extract::{Path, State as ExtractState},
    http::{HeaderValue, header},
    response::Response,
};
use futures_util::stream;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    sync::Arc,
};
use tokio::io::AsyncReadExt;

const MAX_RESULT: u64 = 64 * 1024 * 1024;
fn encoded(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes())
        .collect::<String>()
        .replace('+', "%20")
}
fn body(
    file: File,
    temporary: Option<tempfile::NamedTempFile>,
    content_type: &str,
    filename: &str,
) -> ApiResult<Response> {
    let size = file.metadata().map_err(ApiError::internal)?.len();
    let stream = stream::try_unfold(
        (tokio::fs::File::from_std(file), temporary),
        |(mut file, temporary)| async move {
            let mut bytes = vec![0u8; 64 * 1024];
            let count = file.read(&mut bytes).await?;
            if count == 0 {
                Ok::<_, std::io::Error>(None)
            } else {
                bytes.truncate(count);
                Ok(Some((bytes, (file, temporary))))
            }
        },
    );
    let mut response = Response::new(Body::from_stream(stream));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(content_type).unwrap(),
    );
    response
        .headers_mut()
        .insert(header::CONTENT_LENGTH, HeaderValue::from(size));
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_str(&format!(
            "attachment; filename*=UTF-8''{}",
            encoded(filename)
        ))
        .map_err(ApiError::internal)?,
    );
    response.headers_mut().insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    Ok(response)
}
fn content_type(path: &str) -> &'static str {
    match path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "md" | "txt" => "text/plain; charset=utf-8",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "avif" => "image/avif",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        _ => "application/octet-stream",
    }
}

pub(super) async fn download(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, relative)): Path<(String, String)>,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    let filename = relative.clone();
    let file = crate::task::blocking(move || {
        let _guard = job.access.lock().unwrap();
        if !public_member(&relative) {
            return Err(ApiError::new(404, "file_not_found", "file not found"));
        }
        let path = store::safe_file(&job.folder.join("out"), &relative)?;
        File::open(path).map_err(ApiError::internal)
    })
    .await
    .map_err(ApiError::internal)??;
    body(
        file,
        None,
        content_type(&filename),
        filename.rsplit('/').next().unwrap_or("download"),
    )
}

pub(super) async fn result(
    ExtractState(state): ExtractState<Arc<State>>,
    Path((id, item_id)): Path<(String, String)>,
) -> ApiResult<Json<Value>> {
    let job = jobs::get(&state, &id)?;
    crate::task::blocking(move||{
        let _guard=job.access.lock().unwrap();
        // Copy this item's row; reading its files must not hold the data lock.
        let (item,base,assets)={
            let data=job.data.lock().unwrap();
            let item=data.items.iter().find(|item|item.item_id==item_id).ok_or_else(||ApiError::new(404,"item_not_found","item not found"))?.clone();
            let base=item_base(&data,&item);
            (item,base,data.assets.get(&item_id).cloned())
        };
        let selected=item.output.as_deref().filter(|_|item.status=="done").ok_or_else(||ApiError::new(404,"result_unavailable","item result not available"))?;
        let out=job.folder.join("out");store::safe_file(&out,selected)?;
        let base=base?;
        let base_name=format!("{base}.md");let enhanced_name=format!("{base}.llm.md");
        let base_path=store::safe_file(&out,&base_name).ok();let enhanced_path=store::safe_file(&out,&enhanced_name).ok();
        let (variant,path)=if item.llm_enhanced&&enhanced_path.is_some(){("llm",enhanced_path.clone().unwrap())}else if let Some(path)=base_path.clone(){("base",path)}else if let Some(path)=enhanced_path.clone(){("llm",path)}else{return Err(ApiError::new(404,"result_unavailable","item result not available"));};
        if fs::metadata(&path).map_err(ApiError::internal)?.len()>MAX_RESULT{return Err(ApiError::new(413,"result_too_large","result is too large for JSON; use the file download endpoint"));}
        let markdown=fs::read_to_string(&path).map_err(ApiError::internal)?;
        let mut artifacts=Vec::new();let mut seen=HashSet::new();
        let mut add=|relative:String|->ApiResult<()> {
            if public_member(&relative)&&seen.insert(relative.clone())&&let Ok(path)=store::safe_file(&out,&relative){artifacts.push(json!({"relpath":relative,"size":fs::metadata(path).map_err(ApiError::internal)?.len()}));}Ok(())
        };
        if base_path.is_some() {
            add(base_name)?;
        }
        if item.llm_enhanced && enhanced_path.is_some() {
            add(enhanced_name)?;
        }
        if let Some(assets)=assets{for asset in assets{add(asset)?;}}
        else {
            // Older CLI/server histories lack an item asset index. Recover ownership
            // through actual destinations rather than exposing unrelated job assets.
            let candidates=store::files(&out).map_err(ApiError::internal)?.into_iter().filter(|(name,_)|name.starts_with(".markitai/assets/")||name.starts_with(".markitai/screenshots/")||name.starts_with("assets/")).map(|(name,_)|name).collect::<Vec<_>>();
            let marker=format!("MARKITAI-ARTIFACT-{}-",uuid::Uuid::new_v4().simple());
            let replacements=candidates.iter().enumerate().map(|(i,name)|(name.clone(),format!("{marker}{i}-END"))).collect::<HashMap<_,_>>();
            let mut text=markdown.clone();
            if item.llm_enhanced&&let Some(base_path)=base_path&&base_path!=path&&fs::metadata(&base_path).map_err(ApiError::internal)?.len()<=MAX_RESULT{text.push_str(&fs::read_to_string(base_path).map_err(ApiError::internal)?);}
            let rewritten=markitai_core::output::rewrite_asset_references(&text,&replacements);
            for (index,name) in candidates.into_iter().enumerate(){if rewritten.contains(&format!("{marker}{index}-END")){add(name)?;}}
        }
        Ok(Json(json!({"name":item.name,"variant":variant,"markdown":markdown,"artifacts":artifacts})))
    }).await.map_err(ApiError::internal)?
}

fn zip_jobs(
    root: &std::path::Path,
    jobs: Vec<Arc<Job>>,
) -> ApiResult<(File, tempfile::NamedTempFile)> {
    let temporary = tempfile::Builder::new()
        .prefix(".archive-")
        .suffix(".tmp")
        .tempfile_in(root)
        .map_err(ApiError::internal)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        temporary
            .as_file()
            .set_permissions(fs::Permissions::from_mode(0o600))
            .map_err(ApiError::internal)?;
    }
    let mut writer = zip::ZipWriter::new(temporary.reopen().map_err(ApiError::internal)?);
    let multiple = jobs.len() > 1;
    let mut directories = HashSet::new();
    let mut count = 0usize;
    for job in jobs {
        let _guard = job.access.lock().unwrap();
        let data = job.data.lock().unwrap();
        if data.status == "running" {
            return Err(ApiError::new(409, "job_running", "job is still running"));
        }
        let prefix = if multiple {
            let raw = data
                .items
                .iter()
                .find_map(|item| item.output.as_deref())
                .map(|name| {
                    let filename = name.rsplit('/').next().unwrap_or(name);
                    filename
                        .strip_suffix(".llm.md")
                        .or_else(|| filename.strip_suffix(".md"))
                        .unwrap_or(filename)
                        .to_owned()
                })
                .unwrap_or_else(|| format!("job-{}", data.id));
            let base = jobs::sanitize_name(&raw);
            let mut name = base.clone();
            let mut count = 2;
            while !directories.insert(caseless::default_case_fold_str(&name)) {
                name = format!("{base} ({count})");
                count += 1;
            }
            format!("{name}/")
        } else {
            String::new()
        };
        drop(data);
        for (relative, path) in store::files(&job.folder.join("out")).map_err(ApiError::internal)? {
            if !public_member(&relative) {
                continue;
            }
            count += 1;
            if count > 100_000 {
                return Err(ApiError::new(
                    413,
                    "archive_too_large",
                    "archive has too many files",
                ));
            }
            writer
                .start_file(
                    format!("{prefix}{relative}"),
                    zip::write::SimpleFileOptions::default()
                        .compression_method(zip::CompressionMethod::Deflated)
                        .unix_permissions(0o600),
                )
                .map_err(ApiError::internal)?;
            let mut input = File::open(path).map_err(ApiError::internal)?;
            std::io::copy(&mut input, &mut writer).map_err(ApiError::internal)?;
        }
    }
    writer
        .finish()
        .map_err(ApiError::internal)?
        .sync_all()
        .map_err(ApiError::internal)?;
    let file = temporary.reopen().map_err(ApiError::internal)?;
    Ok((file, temporary))
}
pub(super) async fn job_archive(
    ExtractState(state): ExtractState<Arc<State>>,
    Path(id): Path<String>,
) -> ApiResult<Response> {
    let job = jobs::get(&state, &id)?;
    let root = state.root.clone();
    let (file, temp) = crate::task::blocking(move || zip_jobs(&root, vec![job]))
        .await
        .map_err(ApiError::internal)??;
    body(
        file,
        Some(temp),
        "application/zip",
        &format!("markitai-{id}.zip"),
    )
}
pub(super) async fn history_archive(
    ExtractState(state): ExtractState<Arc<State>>,
) -> ApiResult<Response> {
    http::refresh(&state).await?;
    let mut jobs = http::registered_jobs(&state);
    jobs.retain(|job| job.data.lock().unwrap().status == "done");
    crate::sort::by_key(&mut jobs, |job| {
        super::types::instant(&job.data.lock().unwrap().created_at)
    });
    if jobs.is_empty() {
        return Err(ApiError::new(404, "history_empty", "history is empty"));
    }
    let root = state.root.clone();
    let (file, temp) = crate::task::blocking(move || zip_jobs(&root, jobs))
        .await
        .map_err(ApiError::internal)??;
    body(file, Some(temp), "application/zip", "markitai-all.zip")
}

pub(super) fn item_base(
    data: &super::jobs::JobData,
    item: &super::types::Item,
) -> ApiResult<String> {
    let invalid = || {
        ApiError::new(
            409,
            "output_identity_conflict",
            "saved output identity is inconsistent or unsafe",
        )
    };
    let base = if let Some(base) = data.bases.get(&item.item_id) {
        base.clone()
    } else if let Some(name) = &item.output_name {
        name.strip_suffix(".md").unwrap_or(name).to_owned()
    } else if let Some(output) = &item.output {
        let name = if item.llm_enhanced {
            output.strip_suffix(".llm.md")
        } else {
            output.strip_suffix(".md")
        };
        name.or_else(|| output.strip_suffix(".md"))
            .unwrap_or(output)
            .to_owned()
    } else {
        match item.kind.as_str() {
            "file" => item
                .name
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or("")
                .to_owned(),
            "url" => markitai_core::output::url_name(&item.name, &Default::default()),
            _ => return Err(invalid()),
        }
    };
    if base.is_empty() || base.contains(['/', '\\', '\0']) || matches!(base.as_str(), "." | "..") {
        return Err(invalid());
    }
    let base_name = format!("{base}.md");
    let enhanced_name = format!("{base}.llm.md");
    if item
        .output_name
        .as_deref()
        .is_some_and(|name| name != base_name && name != base)
        || item
            .output
            .as_deref()
            .is_some_and(|name| name.ends_with(".md") && name != base_name && name != enhanced_name)
    {
        return Err(invalid());
    }
    Ok(base)
}

/// A filename match may locate a result, but cannot grant two items the same
/// mutable Markdown member. Asset sharing is handled separately by owned_files.
pub(super) fn exclusive_item_base(
    data: &super::jobs::JobData,
    item: &super::types::Item,
) -> ApiResult<String> {
    let base = item_base(data, item)?;
    let family = [format!("{base}.md"), format!("{base}.llm.md")]
        .map(|name| caseless::default_case_fold_str(&name));
    for sibling in data
        .items
        .iter()
        .filter(|other| other.item_id != item.item_id)
    {
        let other = item_base(data, sibling)?;
        if [format!("{other}.md"), format!("{other}.llm.md")]
            .iter()
            .any(|name| family.contains(&caseless::default_case_fold_str(name)))
        {
            return Err(ApiError::new(
                409,
                "output_identity_conflict",
                "saved output family overlaps another item",
            ));
        }
    }
    Ok(base)
}

/// Native indexes are authoritative. Legacy histories additionally use exact
/// Markdown destinations and converter filename suffixes, never substring globs.
pub(super) fn owned_files(
    folder: &std::path::Path,
    data: &super::jobs::JobData,
    item: &super::types::Item,
) -> ApiResult<HashSet<String>> {
    let base = item_base(data, item)?;
    let out = folder.join("out");
    let mut names = HashSet::new();
    let mut text = String::new();
    for name in [format!("{base}.md"), format!("{base}.llm.md")] {
        if let Ok(path) = store::safe_file(&out, &name) {
            names.insert(name);
            if fs::metadata(&path).map_err(ApiError::internal)?.len() <= MAX_RESULT
                && let Ok(content) = fs::read_to_string(path)
            {
                text.push_str(&content);
                text.push('\n');
            }
        }
    }
    if let Some(output) = &item.output
        && store::safe_file(&out, output).is_ok()
    {
        names.insert(output.clone());
    }
    if let Some(assets) = data.assets.get(&item.item_id) {
        for asset in assets {
            if store::safe_file(&out, asset).is_ok() {
                names.insert(asset.clone());
            }
        }
    } else {
        let candidates = store::files(&out)
            .map_err(ApiError::internal)?
            .into_iter()
            .filter(|(name, _)| {
                name.starts_with(".markitai/assets/")
                    || name.starts_with(".markitai/screenshots/")
                    || name.starts_with("assets/")
            })
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        let marker = format!("MARKITAI-OWNED-{}-", uuid::Uuid::new_v4().simple());
        let replacements = candidates
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), format!("{marker}{i}-END")))
            .collect();
        let rewritten = markitai_core::output::rewrite_asset_references(&text, &replacements);
        for (i, name) in candidates.into_iter().enumerate() {
            if rewritten.contains(&format!("{marker}{i}-END"))
                || legacy_asset(&base, name.rsplit('/').next().unwrap_or(""))
            {
                names.insert(name);
            }
        }
    }
    names.retain(|name| public_member(name));
    Ok(names)
}
fn legacy_asset(base: &str, name: &str) -> bool {
    let Some((stem, extension)) = name.rsplit_once('.') else {
        return false;
    };
    if ![
        "png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif", "svg", "ico", "avif", "heic",
        "heif",
    ]
    .contains(&extension.to_ascii_lowercase().as_str())
    {
        return false;
    }
    let digits = |s: &str, max: usize| {
        !s.is_empty() && s.len() <= max && s.bytes().all(|b| b.is_ascii_digit())
    };
    if let Some(suffix) = stem.strip_prefix(base).and_then(|s| s.strip_prefix('.')) {
        if suffix == "full" || suffix.strip_prefix("full--").is_some_and(|s| digits(s, 4)) {
            return true;
        }
        return digits(suffix, 6)
            || suffix
                .strip_prefix("page")
                .or_else(|| suffix.strip_prefix("slide"))
                .is_some_and(|s| digits(s, 6));
    }
    if let Some(suffix) = stem.strip_prefix(base).and_then(|s| s.strip_prefix('-')) {
        return suffix
            .split_once('-')
            .map_or_else(|| digits(suffix, 6), |(a, b)| digits(a, 6) && digits(b, 6));
    }
    false
}

pub(super) fn public_member(name: &str) -> bool {
    name.rsplit('/').next() != Some(".images.lock")
}
