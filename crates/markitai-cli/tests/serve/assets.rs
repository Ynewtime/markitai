//! `GET /api/jobs/{job_id}/items/{item_id}/assets/{*relpath}`: the images of an
//! item's own result, inline, and nothing else of the job or the machine.
use super::*;

/// Two different PNGs, one pixel and two pixels wide.
const PIXEL: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
const PIXELS: &str = "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4IScHRAAIdwIJkXAbPAAAAABJRU5ErkJggg==";

/// A message with a PNG and a text attachment, both extracted as its assets.
fn letter(subject: &str, png: &str) -> Vec<u8> {
    format!(
        "From: sender@example.test\r\nTo: receiver@example.test\r\nSubject: {subject}\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary=asset-route\r\n\r\n--asset-route\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{subject} body.\r\n--asset-route\r\nContent-Type: image/png; name=pixel.png\r\nContent-Disposition: attachment; filename=pixel.png\r\nContent-Transfer-Encoding: base64\r\n\r\n{png}\r\n--asset-route\r\nContent-Type: text/plain; name=notes.txt\r\nContent-Disposition: attachment; filename=notes.txt\r\n\r\n{subject} notes\r\n--asset-route--\r\n"
    )
    .into_bytes()
}

fn asset(job: &str, item: &str, relpath: &str) -> String {
    format!(
        "/api/jobs/{job}/items/{item}/assets/{}",
        encode_path(relpath)
    )
}

fn relpaths(result: &Value) -> Vec<String> {
    result["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|artifact| artifact["relpath"].as_str().unwrap().to_owned())
        .collect()
}

#[track_caller]
fn refused(reply: &Reply, reason: &str) {
    assert_eq!(reply.status, 404, "{}", reply.text());
    assert_eq!(reply.json()["reason"], reason);
}

#[track_caller]
fn inline(reply: &Reply, media: &str, filename: &str) {
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.headers["content-type"], media);
    assert_eq!(
        reply.headers["content-disposition"],
        format!("inline; filename*=UTF-8''{}", encode_path(filename))
    );
    assert_eq!(reply.headers["x-content-type-options"], "nosniff");
    assert_eq!(
        reply.headers["content-security-policy"],
        "default-src 'none'; sandbox"
    );
    assert_eq!(reply.headers["cache-control"], "private, no-cache");
    assert!(
        reply.headers["etag"].starts_with("W/\""),
        "{:?}",
        reply.headers
    );
    assert_eq!(
        reply.headers["content-length"],
        reply.body.len().to_string()
    );
}

/// A saved job written the way the server keeps one: items, their output bases
/// and, unless `assets` is null, the item asset index.
fn saved_job(root: &Path, id: &str, items: Value, bases: Value, assets: Value) -> PathBuf {
    configure(root);
    let folder = root.join("home/serve/jobs").join(id);
    for directory in [
        "out/.markitai/assets",
        "out/.markitai/screenshots",
        "out/assets",
        "uploads",
    ] {
        std::fs::create_dir_all(folder.join(directory)).unwrap();
    }
    let mut meta = json!({"job_id":id,"created_at":hours_ago(2),"finished_at":hours_ago(1),
        "status":"done","version":2,"options":{},"items":items,"native_bases":bases});
    if !assets.is_null() {
        meta["native_assets"] = assets;
    }
    std::fs::write(folder.join("meta.json"), meta.to_string()).unwrap();
    folder
}

fn done_item(id: &str, output: &str) -> Value {
    json!({"item_id":id,"name":format!("{id}.docx"),"kind":"file","status":"done","error":null,
        "output":output,"output_name":output,"llm_enhanced":false,"operation":"convert",
        "skipped":false,"skip_reason":null,"warnings":[]})
}

/// A conversion's extracted picture is served inline from the item's asset
/// index; the item's Markdown, its other attachments and a sibling item's
/// picture in the same job are not.
#[test]
fn an_items_own_extracted_images_are_served_inline_and_nothing_else_of_the_job() {
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let created = server.submit(
        &[
            ("first.eml", &letter("First", PIXEL)),
            ("second.eml", &letter("Second", PIXELS)),
        ],
        json!([]),
        json!({}),
    );
    let id = created["job_id"].as_str().unwrap();
    assert_eq!(server.done(id)["done"], 2);
    let first = relpaths(&server.json(&format!("/api/jobs/{id}/items/i1/result")));
    let second = relpaths(&server.json(&format!("/api/jobs/{id}/items/i2/result")));
    let find = |names: &[String], suffix: &str| {
        names
            .iter()
            .find(|name| name.ends_with(suffix))
            .unwrap_or_else(|| panic!("no {suffix} in {names:?}"))
            .clone()
    };
    let (mine, theirs) = (find(&first, ".png"), find(&second, ".png"));
    assert_ne!(mine, theirs);
    let path = asset(id, "i1", &mine);
    let reply = server.request("GET", &path, &[], &[]);
    inline(&reply, "image/png", mine.rsplit('/').next().unwrap());
    assert!(reply.body.starts_with(b"\x89PNG\r\n\x1a\n"));
    let download = server.request(
        "GET",
        &format!("/api/jobs/{id}/files/{}", encode_path(&mine)),
        &[],
        &[],
    );
    assert_eq!(reply.body, download.body);
    // The client's copy is current under its tag, weak or strong, in a list
    // or as `*`; any other tag gets the bytes again.
    let tag = reply.headers["etag"].clone();
    for condition in [
        tag.clone(),
        tag.trim_start_matches("W/").to_owned(),
        format!("\"other\", {tag}"),
        "*".to_owned(),
    ] {
        let cached = server.request("GET", &path, &[("If-None-Match", &condition)], &[]);
        assert_eq!(cached.status, 304, "{condition}");
        assert!(cached.body.is_empty());
        assert_eq!(cached.headers["etag"], tag);
        assert_eq!(cached.headers["cache-control"], "private, no-cache");
    }
    let stale = server.request("GET", &path, &[("If-None-Match", "W/\"0-0\"")], &[]);
    assert_eq!((stale.status, &stale.body), (200, &reply.body));
    // The sibling's picture is its own, and only its own.
    inline(
        &server.request("GET", &asset(id, "i2", &theirs), &[], &[]),
        "image/png",
        theirs.rsplit('/').next().unwrap(),
    );
    for (item, name) in [
        ("i1", theirs.clone()),
        ("i2", mine.clone()),
        ("i1", find(&first, ".md")),
        ("i1", find(&first, ".txt")),
    ] {
        refused(
            &server.request("GET", &asset(id, item, &name), &[], &[]),
            "file_not_found",
        );
    }
    refused(
        &server.request("GET", &asset(id, "i9", &mine), &[], &[]),
        "item_not_found",
    );
    refused(
        &server.request("GET", &asset("0123456789ab", "i1", &mine), &[], &[]),
        "job_not_found",
    );
    server.stop();
}

/// Owned raster images of every kind are served under their names, CJK, spaces
/// and parentheses included; an indexed name that is not such an image, is
/// missing, names another item or leaves the job's output through traversal,
/// an absolute path, a backslash or a symbolic link is refused the same way.
#[test]
fn item_assets_are_only_owned_raster_images_resolved_inside_the_job_output() {
    let temp = tempfile::tempdir().unwrap();
    let id = "0123456789ab";
    let base = "报告 (终稿)";
    let owned = [
        (".markitai/assets/图 1 (封面).png", "image/png"),
        (".markitai/assets/报告 (终稿).0002.JPEG", "image/jpeg"),
        (".markitai/assets/chart.jpg", "image/jpeg"),
        (".markitai/assets/chart.gif", "image/gif"),
        (".markitai/assets/chart.webp", "image/webp"),
        (".markitai/assets/chart.avif", "image/avif"),
        (
            ".markitai/screenshots/报告 (终稿).page0001.jpg",
            "image/jpeg",
        ),
        ("assets/legacy visible.png", "image/png"),
    ];
    let refused_names = [
        format!("{base}.md"),
        ".markitai/assets/drawing.svg".to_owned(),
        ".markitai/assets/table.csv".to_owned(),
        ".markitai/assets/missing.png".to_owned(),
        ".markitai\\assets\\back.png".to_owned(),
        ".markitai/assets/escape.png".to_owned(),
        ".markitai/linked/inside.png".to_owned(),
        "../uploads/secret.png".to_owned(),
    ];
    // An index is trusted for ownership only: every entry still has to resolve
    // inside the output, so a tampered one naming an absolute path is refused.
    let secret = temp
        .path()
        .join("home/serve/jobs")
        .join(id)
        .join("uploads/secret.png");
    let absolute = secret.to_str().unwrap().to_owned();
    let mut index: Vec<String> = owned.iter().map(|(name, _)| (*name).to_owned()).collect();
    index.extend(refused_names.iter().cloned());
    index.push(absolute.clone());
    let mut failed = done_item("i3", "failed.md");
    failed["status"] = json!("error");
    failed["output"] = Value::Null;
    let folder = saved_job(
        temp.path(),
        id,
        json!([
            done_item("i1", &format!("{base}.md")),
            done_item("i2", "other.md"),
            failed
        ]),
        json!({"i1":base,"i2":"other","i3":"failed"}),
        json!({"i1":index,"i2":[".markitai/assets/other.png"],"i3":[owned[0].0]}),
    );
    let out = folder.join("out");
    std::fs::write(out.join(format!("{base}.md")), "# Report\n").unwrap();
    std::fs::write(out.join("other.md"), "# Other\n").unwrap();
    for (name, _) in owned {
        std::fs::write(out.join(name), format!("bytes of {name}")).unwrap();
    }
    for name in [
        ".markitai/assets/drawing.svg",
        ".markitai/assets/table.csv",
        ".markitai/assets/other.png",
        ".markitai/assets/stray.png",
        // One file whose name holds the backslashes, as indexed.
        ".markitai\\assets\\back.png",
    ] {
        std::fs::write(out.join(name), "not this item's to show").unwrap();
    }
    std::fs::write(folder.join("uploads/secret.png"), "an upload").unwrap();
    std::fs::write(temp.path().join("outside.png"), "outside the job").unwrap();
    let server = Server::start(temp.path());
    // Links made after the history loaded: one to a file outside, one through
    // a linked directory.
    std::os::unix::fs::symlink(
        temp.path().join("outside.png"),
        out.join(".markitai/assets/escape.png"),
    )
    .unwrap();
    std::fs::create_dir(temp.path().join("linked")).unwrap();
    std::fs::write(temp.path().join("linked/inside.png"), "outside the job").unwrap();
    std::os::unix::fs::symlink(temp.path().join("linked"), out.join(".markitai/linked")).unwrap();

    let listed = relpaths(&server.json(&format!("/api/jobs/{id}/items/i1/result")));
    for (name, media) in owned {
        assert!(
            listed.iter().any(|entry| entry == name),
            "{name}: {listed:?}"
        );
        let reply = server.request("GET", &asset(id, "i1", name), &[], &[]);
        inline(&reply, media, name.rsplit('/').next().unwrap());
        assert_eq!(reply.body, format!("bytes of {name}").into_bytes());
    }
    // A browser leaves parentheses unencoded; the name is the same.
    let reply = server.request(
        "GET",
        &format!(
            "/api/jobs/{id}/items/i1/assets/.markitai/assets/%E5%9B%BE%201%20(%E5%B0%81%E9%9D%A2).png"
        ),
        &[],
        &[],
    );
    assert_eq!(reply.body, format!("bytes of {}", owned[0].0).into_bytes());
    inline(
        &server.request(
            "GET",
            &asset(id, "i2", ".markitai/assets/other.png"),
            &[],
            &[],
        ),
        "image/png",
        "other.png",
    );
    // Every refusal looks the same. Indexed names that are not owned raster
    // images inside the output, a sibling's and nobody's image, an index entry
    // of an item without a result, and paths that would leave the output, sent
    // as a client might.
    let mut paths: Vec<String> = refused_names
        .iter()
        .chain(&[
            ".markitai/assets/other.png".to_owned(),
            ".markitai/assets/stray.png".to_owned(),
        ])
        .map(|name| asset(id, "i1", name))
        .collect();
    paths.push(asset(id, "i3", owned[0].0));
    paths.extend(
        [
            "../uploads/secret.png".to_owned(),
            ".markitai/assets/../../uploads/secret.png".to_owned(),
            ".markitai/assets/../assets/chart.gif".to_owned(),
            "%2e%2e/uploads/secret.png".to_owned(),
            "%2E%2E%2Fuploads%2Fsecret.png".to_owned(),
            format!("/{}", encode_path(absolute.trim_start_matches('/'))),
            encode_path(&absolute).replace('/', "%2F"),
            ".markitai%5Cassets%5Cchart.gif".to_owned(),
            "./.markitai/assets/chart.gif".to_owned(),
        ]
        .iter()
        .map(|raw| format!("/api/jobs/{id}/items/i1/assets/{raw}")),
    );
    let answers: Vec<(String, u16, Value)> = paths
        .into_iter()
        .map(|path| {
            let reply = server.request("GET", &path, &[], &[]);
            let reason = serde_json::from_slice::<Value>(&reply.body)
                .map(|body| body["reason"].clone())
                .unwrap_or(Value::Null);
            (path, reply.status, reason)
        })
        .collect();
    let wrong: Vec<_> = answers
        .iter()
        .filter(|(_, status, reason)| (*status, reason) != (404, &json!("file_not_found")))
        .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
    server.stop();
}

/// Without an asset index, an older history's item owns the images its
/// Markdown (and, when enhanced, its base Markdown) references, exactly as its
/// result lists them, and no other image of the job.
#[test]
fn a_history_without_an_asset_index_serves_only_the_images_its_markdown_references() {
    let temp = tempfile::tempdir().unwrap();
    let id = "00000000abcd";
    let mut enhanced = done_item("i1", "page.llm.md");
    enhanced["llm_enhanced"] = json!(true);
    enhanced["output_name"] = json!("page.md");
    let folder = saved_job(
        temp.path(),
        id,
        json!([enhanced, done_item("i2", "notes.md")]),
        json!({"i1":"page","i2":"notes"}),
        Value::Null,
    );
    let out = folder.join("out");
    for (name, body) in [
        ("page.md", "Base. ![only in base](assets/base-only.png)\n"),
        (
            "page.llm.md",
            "Enhanced. ![图](assets/%E5%9B%BE%201.png)\n\n![shot](.markitai/screenshots/page.full.jpg)\n",
        ),
        ("notes.md", "Notes. ![theirs](assets/theirs.png)\n"),
    ] {
        std::fs::write(out.join(name), body).unwrap();
    }
    for name in [
        "assets/base-only.png",
        "assets/图 1.png",
        ".markitai/screenshots/page.full.jpg",
        "assets/theirs.png",
        // Named after the item but referenced by no Markdown of it.
        "assets/page.0001.png",
        "assets/stray.png",
    ] {
        std::fs::write(out.join(name), format!("bytes of {name}")).unwrap();
    }
    let server = Server::start(temp.path());
    let listed = relpaths(&server.json(&format!("/api/jobs/{id}/items/i1/result")));
    let images: Vec<&String> = listed
        .iter()
        .filter(|name| !name.ends_with(".md"))
        .collect();
    let mut expected = vec![
        "assets/base-only.png",
        "assets/图 1.png",
        ".markitai/screenshots/page.full.jpg",
    ];
    let mut served = images.iter().map(|name| name.as_str()).collect::<Vec<_>>();
    expected.sort_unstable();
    served.sort_unstable();
    assert_eq!(served, expected, "{listed:?}");
    for name in images {
        let reply = server.request("GET", &asset(id, "i1", name), &[], &[]);
        let media = if name.ends_with(".jpg") {
            "image/jpeg"
        } else {
            "image/png"
        };
        inline(&reply, media, name.rsplit('/').next().unwrap());
        assert_eq!(reply.body, format!("bytes of {name}").into_bytes());
    }
    for name in [
        "assets/theirs.png",
        "assets/page.0001.png",
        "assets/stray.png",
        "page.md",
    ] {
        refused(
            &server.request("GET", &asset(id, "i1", name), &[], &[]),
            "file_not_found",
        );
    }
    assert_eq!(
        server
            .request("GET", &asset(id, "i2", "assets/theirs.png"), &[], &[])
            .status,
        200
    );
    server.stop();
}

/// The route is an ordinary API route: the startup token is required, and no
/// download ticket can be issued for it or stand in for the token.
#[test]
fn item_assets_need_the_startup_token_and_no_ticket_names_them() {
    let temp = tempfile::tempdir().unwrap();
    let id = "0000000000ff";
    let folder = saved_job(
        temp.path(),
        id,
        json!([done_item("i1", "a.md")]),
        json!({"i1":"a"}),
        json!({"i1":[".markitai/assets/a.0001.png"]}),
    );
    std::fs::write(
        folder.join("out/a.md"),
        "![a](.markitai/assets/a.0001.png)\n",
    )
    .unwrap();
    std::fs::write(folder.join("out/.markitai/assets/a.0001.png"), "png bytes").unwrap();
    let token = "authored-asset-route-test-token";
    let server = Server::start_with(
        temp.path(),
        &[("MARKITAI_SERVE_TOKEN", std::ffi::OsStr::new(token))],
    );
    let path = asset(id, "i1", ".markitai/assets/a.0001.png");
    let bearer = format!("Bearer {token}");
    let anonymous = server.request("GET", &path, &[], &[]);
    assert_eq!(
        (anonymous.status, anonymous.json()["reason"].clone()),
        (401, json!("token_required"))
    );
    assert_eq!(
        server
            .request("GET", &path, &[("Authorization", "Bearer wrong")], &[])
            .status,
        401
    );
    let reply = server.request("GET", &path, &[("Authorization", &bearer)], &[]);
    inline(&reply, "image/png", "a.0001.png");
    assert_eq!(reply.body, b"png bytes");
    assert_eq!(
        server
            .request("GET", &format!("{path}?token={token}"), &[], &[])
            .status,
        200
    );
    let ticket = server.request(
        "POST",
        "/api/download-tickets",
        &[
            ("Authorization", &bearer),
            ("Content-Type", "application/json"),
        ],
        json!({ "path": path }).to_string().as_bytes(),
    );
    assert_eq!(
        (ticket.status, ticket.json()["reason"].clone()),
        (422, json!("invalid_ticket_path"))
    );
    assert_eq!(
        server
            .request("GET", &format!("{path}?ticket=made-up"), &[], &[])
            .json()["reason"],
        "ticket_invalid"
    );
    // A wrong Host is refused before the route, as for every API path.
    let foreign = server.request(
        "GET",
        &path,
        &[("Authorization", &bearer), ("Host", "untrusted.invalid")],
        &[],
    );
    assert_eq!(foreign.status, 400);
    server.stop();
}
