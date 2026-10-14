//! Workbench contract v1.14: how usable a URL item's kept page and its
//! screenshots were (`capture`), where the screenshots show the page's
//! blocks (`…/original/blocks`), and the compare screenshots a browser load
//! keeps when no screenshots were asked for. Pages are invented and served
//! on loopback; nothing here reaches the network.
use super::*;

const JPEG: &[u8] =
    include_bytes!("../fixtures/python-batch/anthropic/.markitai/assets/deck.pptx.0001.jpg");
const BLOCKS: &str = r#"{"version":1,"width":1280,"height":4000,"blocks":[{"tag":"h1","text":"Local article","bbox":[8,8,600,40]}]}"#;

fn snapshot_files(folder: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let snapshots = folder.join("snapshots");
    for entry in std::fs::read_dir(&snapshots).into_iter().flatten() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type().unwrap().is_dir() {
            for inner in std::fs::read_dir(entry.path()).unwrap() {
                found.push(format!(
                    "{name}/{}",
                    inner.unwrap().file_name().to_string_lossy()
                ));
            }
            if std::fs::read_dir(entry.path()).unwrap().next().is_none() {
                found.push(format!("{name}/"));
            }
        } else {
            found.push(name);
        }
    }
    found.sort();
    found
}

/// Compare screenshots and block boxes as a browser load leaves them.
fn plant_shots(folder: &Path, item: &str, count: usize) {
    let shots = folder.join("snapshots").join(format!("{item}.shots"));
    std::fs::create_dir_all(&shots).unwrap();
    for number in 1..=count {
        std::fs::write(shots.join(format!("{number}.jpg")), JPEG).unwrap();
    }
    std::fs::write(
        folder.join("snapshots").join(format!("{item}.blocks.json")),
        BLOCKS,
    )
    .unwrap();
}

#[test]
fn a_url_item_records_its_page_and_its_kept_files_go_with_it() {
    let origin = Origin::start();
    let temp = tempfile::tempdir().unwrap();
    let server = Server::start(temp.path());
    let id = server.submit(
        &[("notes.md", b"# Notes\n".as_slice())],
        json!([origin.url("/first"), origin.url("/second")]),
        json!({"strategy":"static"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    server.done(&id);
    let folder = server.jobdir(&id);
    // The upload is i1; the URLs are i2 and i3. A static conversion keeps
    // the page and how it reads, and no screenshot of any kind.
    let (first, second) = ("i2", "i3");
    assert_eq!(
        snapshot_files(&folder),
        ["i2.capture.json", "i2.html", "i3.capture.json", "i3.html"]
    );
    let record: Value =
        serde_json::from_slice(&std::fs::read(folder.join("snapshots/i2.capture.json")).unwrap())
            .unwrap();
    // The invented page holds 40 characters of text: too few to be content.
    let page = json!({"state":"empty","text_length":40,"title":"Local page","reason":"blank"});
    assert_eq!(record, json!({"version":1,"page":page,"screenshot":null}));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(folder.join("snapshots/i2.capture.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o077, 0);
    }
    let describe = |item: &str| server.json(&format!("/api/jobs/{id}/items/{item}/original"));
    let original = describe(first);
    assert_eq!(
        (
            original["capture"].clone(),
            original["blocks"].clone(),
            original["page_count"].clone()
        ),
        (
            json!({"page":page,"screenshot":null}),
            Value::Null,
            json!(0)
        )
    );
    // An upload has no capture.
    let upload = describe("i1");
    assert_eq!(
        (upload["capture"].clone(), upload["blocks"].clone()),
        (Value::Null, Value::Null)
    );
    let blocks_path = |item: &str| format!("/api/jobs/{id}/items/{item}/original/blocks");
    for item in ["i1", first, "i9"] {
        let reply = server.request("GET", &blocks_path(item), &[], &[]);
        let expected = if item == "i9" {
            "item_not_found"
        } else {
            "original_unavailable"
        };
        assert_eq!(
            (reply.status, reply.json()["reason"].clone()),
            (404, json!(expected)),
            "{item}"
        );
    }

    // Compare screenshots stand in for pages, with their block boxes.
    plant_shots(&folder, first, 2);
    let original = describe(first);
    assert_eq!(
        (
            original["kind"].clone(),
            original["page_count"].clone(),
            original["blocks"].clone()
        ),
        (json!("html"), json!(2), json!(blocks_path(first)))
    );
    let blocks = server.request("GET", &blocks_path(first), &[], &[]);
    assert_eq!(blocks.status, 200, "{}", blocks.text());
    assert_eq!(blocks.headers["content-type"], "application/json");
    assert_eq!(blocks.headers["cache-control"], "no-store");
    assert_eq!(blocks.headers["x-content-type-options"], "nosniff");
    assert_eq!(blocks.body, BLOCKS.as_bytes());
    let shot = server.request(
        "GET",
        &format!("/api/jobs/{id}/items/{first}/original/pages/2?width=2400"),
        &[],
        &[],
    );
    assert_eq!((shot.status, shot.body.as_slice()), (200, JPEG));
    assert_eq!(
        server
            .request(
                "GET",
                &format!("/api/jobs/{id}/items/{first}/original/pages/3"),
                &[],
                &[]
            )
            .status,
        404
    );
    // A gap ends the screenshots: the page is cut from its top.
    std::fs::remove_file(folder.join("snapshots/i2.shots/1.jpg")).unwrap();
    assert_eq!(describe(first)["page_count"], 0);
    assert!(describe(first)["blocks"].is_null());
    std::fs::write(folder.join("snapshots/i2.shots/1.jpg"), JPEG).unwrap();
    // Compare screenshots are not results: not listed, not archived.
    let snapshot = server.json(&format!("/api/jobs/{id}"));
    assert!(!snapshot.to_string().contains(".shots"), "{snapshot}");
    let archive = server.request("GET", &format!("/api/jobs/{id}/archive"), &[], &[]);
    assert_eq!(archive.status, 200);
    let members = zip_contents(&archive.body);
    assert!(
        members
            .keys()
            .all(|name| !name.contains("shots") && !name.contains("blocks")),
        "{:?}",
        members.keys()
    );

    // A retry keeps what its own load left and drops the rest.
    let reply = server.request(
        "POST",
        &format!("/api/jobs/{id}/items/{first}/retry"),
        &[("Content-Type", "application/json")],
        b"{}",
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
    server.done(&id);
    assert_eq!(
        snapshot_files(&folder),
        ["i2.capture.json", "i2.html", "i3.capture.json", "i3.html"]
    );
    assert!(describe(first)["blocks"].is_null());

    // Deleting an item deletes every file it keeps.
    plant_shots(&folder, second, 3);
    let reply = server.request(
        "DELETE",
        &format!("/api/jobs/{id}/items/{second}"),
        &[],
        &[],
    );
    assert_eq!(reply.status, 204, "{}", reply.text());
    assert_eq!(snapshot_files(&folder), ["i2.capture.json", "i2.html"]);
    // So does a batch deletion.
    plant_shots(&folder, first, 1);
    let reply = server.request(
        "POST",
        "/api/history/delete",
        &[("Content-Type", "application/json")],
        json!({"items":[{"job_id":id,"item_id":first}]})
            .to_string()
            .as_bytes(),
    );
    assert_eq!(reply.status, 200, "{}", reply.text());
    assert_eq!(reply.json()["deleted"], 1, "{}", reply.text());
    assert!(
        snapshot_files(&folder).is_empty(),
        "{:?}",
        snapshot_files(&folder)
    );
    server.stop();
}

#[test]
fn an_older_job_has_its_kept_page_judged_when_asked() {
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let id = "00000000abcf";
    let folder = temp.path().join("home/serve/jobs").join(id);
    std::fs::create_dir_all(folder.join("snapshots")).unwrap();
    std::fs::create_dir_all(folder.join("out")).unwrap();
    let challenge = "<!doctype html><html><head><title>Just a moment...</title></head><body><div class=\"main-wrapper\"><div id=\"cf-chl-widget-a1\"></div><noscript>Enable JavaScript and cookies to continue</noscript></div><script src=\"/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1\"></script></body></html>";
    let captcha = "<html><head><title>Video</title></head><body><div id=\"risk-captcha-app\"></div></body></html>";
    let article = format!(
        "<html><head><title>An invented article</title></head><body><article><h1>Heading</h1><p>{}</p></article></body></html>",
        "Plain words of an invented article. ".repeat(5)
    );
    let notice = "<html><body><p>JavaScript is not available.</p></body></html>";
    let pages = [
        ("i1", "https://example.org/a", Some(challenge.to_owned())),
        (
            "i2",
            "https://www.bilibili.com/video/BV0000000000",
            Some(captcha.to_owned()),
        ),
        ("i3", "https://example.com/post", Some(article)),
        (
            "i4",
            "https://example.com/status/1",
            Some(notice.to_owned()),
        ),
        ("i5", "https://example.com/none", None),
    ];
    let mut items = Vec::new();
    let mut bases = serde_json::Map::new();
    for (item, url, page) in &pages {
        if let Some(page) = page {
            std::fs::write(folder.join(format!("snapshots/{item}.html")), page).unwrap();
        }
        std::fs::write(folder.join(format!("out/{item}.md")), "# Page\n").unwrap();
        items.push(
            json!({"item_id":item,"name":url,"kind":"url","status":"done",
            "output":format!("{item}.md"),"output_name":format!("{item}.md")}),
        );
        bases.insert((*item).into(), json!(item));
    }
    let meta = json!({"job_id":id,"created_at":hours_ago(1),"finished_at":hours_ago(1),"status":"done",
        "version":2,"options":{},"items":items,"native_bases":bases,
        "native_assets":{"i1":[],"i2":[],"i3":[],"i4":[],"i5":[]}});
    std::fs::write(folder.join("meta.json"), meta.to_string()).unwrap();
    let server = Server::start(temp.path());
    let capture = |item: &str| {
        server.json(&format!("/api/jobs/{id}/items/{item}/original"))["capture"].clone()
    };
    let judged = |item: &str| {
        let value = capture(item);
        assert!(value["screenshot"].is_null(), "{item}: {value}");
        (
            value["page"]["state"].clone(),
            value["page"]["reason"].clone(),
        )
    };
    assert_eq!(judged("i1"), (json!("challenge"), json!("challenge")));
    assert_eq!(
        capture("i1")["page"],
        json!({"state":"challenge","text_length":41,"title":"Just a moment...","reason":"challenge"})
    );
    assert_eq!(judged("i2"), (json!("challenge"), json!("verification")));
    assert_eq!(judged("i3"), (json!("ok"), Value::Null));
    assert_eq!(judged("i4"), (json!("empty"), json!("javascript")));
    // Without a kept page there is nothing to judge.
    assert_eq!(capture("i5"), json!({"page":null,"screenshot":null}));
    // Asking again gives the same answer, and judging writes nothing.
    assert_eq!(judged("i1"), (json!("challenge"), json!("challenge")));
    assert!(!folder.join("snapshots/i1.capture.json").exists());
    // A record, when there is one, is what the descriptor reports.
    let screenshot = json!({"state":"ok","text_length":900,"title":"Signed in view","reason":null});
    std::fs::write(
        folder.join("snapshots/i1.capture.json"),
        json!({"version":1,"page":null,"screenshot":screenshot}).to_string(),
    )
    .unwrap();
    assert_eq!(capture("i1"), json!({"page":null,"screenshot":screenshot}));
    server.stop();
}

/// Chromium for a test: `MARKITAI_BROWSER_EXECUTABLE`, else Playwright's
/// cache (`PLAYWRIGHT_BROWSERS_PATH`, else `~/.cache/ms-playwright`).
fn chromium() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("MARKITAI_BROWSER_EXECUTABLE") {
        return Some(path.into());
    }
    let cache = std::env::var_os("PLAYWRIGHT_BROWSERS_PATH")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| Path::new(&home).join(".cache/ms-playwright"))
        })?;
    let mut found: Vec<PathBuf> = std::fs::read_dir(cache)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("chromium-"))
        })
        .flat_map(|path| {
            [
                "chrome-linux64/chrome",
                "chrome-linux-arm64/chrome",
                "chrome-linux/chrome",
                "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
                "chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
            ]
            .map(|suffix| path.join(suffix))
        })
        .filter(|path| path.is_file())
        .collect();
    found.sort();
    found.pop()
}

/// Invented pages on loopback: a 15 000 px article, a short one and a
/// challenge page.
struct Pages {
    port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl Pages {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let thread = std::thread::spawn(move || {
            while !stopping.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(_) => break,
                };
                stream.set_nonblocking(false).unwrap();
                stream.set_read_timeout(Some(WAIT)).unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut start = String::new();
                if reader.read_line(&mut start).is_err() {
                    continue;
                }
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
                        break;
                    }
                }
                let path = start.split_whitespace().nth(1).unwrap_or("/").to_owned();
                let body = page(&path);
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        Self {
            port,
            stop,
            thread: Some(thread),
        }
    }
    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }
}
impl Drop for Pages {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn page(path: &str) -> String {
    let style = "<style>body{margin:0;font:16px/1.5 sans-serif;color:#222}section{height:100px;padding:0 24px;box-sizing:border-box;border-top:1px solid #ddd}h2{margin:8px 0 4px;font-size:18px}p{margin:0}</style>";
    match path {
        "/long" => {
            let sections: String = (1..=150)
                .map(|number| {
                    format!(
                        "<section><h2>Section {number}</h2><p>Section {number} of an invented long page: its sentences say nothing but take room, as the text of a real article would, so the screenshot tiles hold lines of words.</p></section>"
                    )
                })
                .collect();
            format!(
                "<!doctype html><html><head><title>An invented long page</title>{style}</head><body><main>{sections}</main></body></html>"
            )
        }
        "/challenge" => "<!doctype html><html><head><title>Just a moment...</title></head><body><div class=\"main-wrapper\"><h1>127.0.0.1</h1><p>Verify you are human by completing the action below.</p><div id=\"cf-chl-widget-x1\"></div></div></body></html>".to_owned(),
        _ => format!(
            "<!doctype html><html><head><title>An invented short page</title>{style}</head><body><article><h1>A short invented page</h1><p>{}</p><p>{}</p></article></body></html>",
            "The first paragraph of a short invented page holds a few sentences of plain words. ".repeat(3),
            "The second paragraph says a little more, so there is something to compare. ".repeat(2)
        ),
    }
}

/// The real browser path: a URL converted by the local browser keeps compare
/// screenshots (JPEG quality 70, at most 20 tiles of the page from its top),
/// the page's block boxes and how page and screenshot read; screenshots the
/// user asked for are used instead of a compare copy. Prints the bytes the
/// compare screenshots take.
#[test]
#[ignore = "requires installed Chromium; loopback pages only"]
fn a_browser_load_keeps_compare_screenshots_blocks_and_qualities() {
    let browser = chromium().expect("Chromium for this test");
    let pages = Pages::start();
    let temp = tempfile::tempdir().unwrap();
    configure(temp.path());
    let path = temp.path().join("config.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["fetch"]["playwright"] =
        json!({"extra_wait_ms":0,"skip_auto_scroll":true,"timeout":20000});
    std::fs::write(&path, config.to_string()).unwrap();
    let server = Server::start_with(
        temp.path(),
        &[("MARKITAI_BROWSER_EXECUTABLE", browser.as_os_str())],
    );
    let id = server.submit(
        &[],
        json!([
            pages.url("/long"),
            pages.url("/short"),
            pages.url("/challenge")
        ]),
        json!({"strategy":"playwright"}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let job = server.done(&id);
    for item in job["items"].as_array().unwrap() {
        assert_eq!(item["status"], "done", "{item}");
    }
    let folder = server.jobdir(&id);
    // The kept page keeps its doctype: without it the frame would lay the page out in quirks
    // mode (and a bare `:hover` there matches only links).
    let kept = std::fs::read_to_string(folder.join("snapshots/i2.html")).unwrap();
    assert!(
        kept.starts_with("<!DOCTYPE html>"),
        "{}",
        &kept[..kept.len().min(80)]
    );
    let shots = |item: &str| -> Vec<u64> {
        (1..=30)
            .map_while(|number| {
                std::fs::metadata(folder.join(format!("snapshots/{item}.shots/{number}.jpg")))
                    .ok()
                    .map(|meta| meta.len())
            })
            .collect()
    };
    let describe = |item: &str| server.json(&format!("/api/jobs/{id}/items/{item}/original"));
    let blocks = |item: &str| {
        let reply = server.request(
            "GET",
            &format!("/api/jobs/{id}/items/{item}/original/blocks"),
            &[],
            &[],
        );
        assert_eq!(reply.status, 200, "{item}: {}", reply.text());
        reply.json()
    };
    // The long page: 15 000 CSS pixels in 2000-pixel tiles.
    let long = shots("i1");
    assert_eq!(long.len(), 8, "{long:?}");
    let original = describe("i1");
    assert_eq!(original["kind"], "html");
    assert_eq!(original["page_count"], 8);
    assert_eq!(
        original["blocks"],
        format!("/api/jobs/{id}/items/i1/original/blocks")
    );
    let width = original["pages"][0]["width"].as_f64().unwrap();
    let boxes = blocks("i1");
    assert_eq!(boxes["version"], 1);
    assert_eq!(boxes["width"].as_f64().unwrap(), width);
    assert!(
        boxes["height"].as_f64().unwrap() >= 15_000.0,
        "{}",
        boxes["height"]
    );
    let headings: Vec<_> = boxes["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["tag"] == "h2")
        .collect();
    assert_eq!(headings.len(), 150);
    // Section 101 starts 10 000 CSS pixels down, in the sixth tile.
    assert_eq!(headings[100]["text"], "Section 101");
    let top = headings[100]["bbox"][1].as_f64().unwrap();
    assert!((10_000.0..10_100.0).contains(&top), "{top}");
    assert_eq!(
        original["capture"]["screenshot"]["state"], "ok",
        "{}",
        original["capture"]
    );
    assert_eq!(original["capture"]["page"]["state"], "ok");
    assert_eq!(
        original["capture"]["screenshot"]["title"],
        "An invented long page"
    );
    // The short page: one tile.
    let short = shots("i2");
    assert_eq!(short.len(), 1, "{short:?}");
    assert_eq!(describe("i2")["capture"]["screenshot"]["state"], "ok");
    // The challenge page reads as one in both views.
    let challenge = describe("i3")["capture"].clone();
    assert_eq!(challenge["screenshot"]["state"], "challenge", "{challenge}");
    assert_eq!(challenge["page"]["state"], "challenge", "{challenge}");
    eprintln!(
        "compare screenshots: long page {} tiles, {} bytes ({:?}); short page {} tile, {} bytes",
        long.len(),
        long.iter().sum::<u64>(),
        long,
        short.len(),
        short.iter().sum::<u64>()
    );
    // A retry loads the page again and keeps its new compare screenshots.
    let reply = server.request(
        "POST",
        &format!("/api/jobs/{id}/items/i2/retry"),
        &[("Content-Type", "application/json")],
        b"{}",
    );
    assert_eq!(reply.status, 202, "{}", reply.text());
    server.done(&id);
    assert_eq!(shots("i2").len(), 1);
    assert!(folder.join("snapshots/i2.blocks.json").is_file());

    // Screenshots the user asked for stand in; no compare copy is kept.
    let asked = server.submit(
        &[],
        json!([pages.url("/short")]),
        json!({"strategy":"playwright","screenshot":true}),
    )["job_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let job = server.done(&asked);
    assert_eq!(job["items"][0]["status"], "done", "{job}");
    let folder = server.jobdir(&asked);
    assert!(!folder.join("snapshots/i1.shots").exists());
    let original = server.json(&format!("/api/jobs/{asked}/items/i1/original"));
    assert!(original["page_count"].as_u64().unwrap() >= 1, "{original}");
    assert_eq!(
        original["blocks"],
        format!("/api/jobs/{asked}/items/i1/original/blocks")
    );
    assert_eq!(original["capture"]["screenshot"]["state"], "ok");
    server.stop();
}
