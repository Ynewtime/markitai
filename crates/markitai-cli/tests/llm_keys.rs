//! A model whose API key is not set is left out before anything is sent:
//! one warning names the variable, and with no model left the run fails.
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};

use crate::console;

fn invoke(root: &Path, cfg: Value, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    console::detached(&mut command);
    console::english(command.env_clear());
    for key in ["PATH", "SYSTEMROOT", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .current_dir(root)
        .env("HOME", root.join("user"))
        .env("USERPROFILE", root.join("user"))
        .env("MARKITAI_HOME", root.join("home"))
        .envs(env.iter().copied())
        .arg("--config-json")
        .arg(cfg.to_string())
        .args(args)
        .output()
        .unwrap()
}

/// A chat completions server that answers every request and counts them.
fn answering() -> (String, Arc<Mutex<usize>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let count = Arc::new(Mutex::new(0));
    let seen = count.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let mut request = Vec::new();
            let mut buffer = [0; 8192];
            loop {
                let text = String::from_utf8_lossy(&request);
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let length = head
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|value| value.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if body.len() >= length {
                        break;
                    }
                }
                match stream.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => request.extend_from_slice(&buffer[..read]),
                }
            }
            *seen.lock().unwrap() += 1;
            let body = json!({"choices":[{"message":{"content":"# Note\n\nCleaned text."},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":10,"completion_tokens":5}})
            .to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (address, count)
}

#[test]
fn no_model_with_its_key_fails_before_converting_and_names_the_variable() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("note.md"), "# Note\n\nText.\n").unwrap();
    let cfg = json!({"cache":{"enabled":false},"llm":{"model_list":[
        {"model_name":"default","litellm_params":{"model":"deepseek/deepseek-flash"}}]}});
    let output = invoke(
        root.path(),
        cfg.clone(),
        &["note.md", "-o", "out", "--llm"],
        &[],
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(1), "{stderr}");
    // This system's shell: `export` here, PowerShell's `$env:` on Windows.
    let set = if cfg!(windows) {
        "$env:DEEPSEEK_API_KEY=\"your_key\""
    } else {
        "export DEEPSEEK_API_KEY=your_key"
    };
    assert!(
        stderr.contains(&format!(
            "DEEPSEEK_API_KEY is not set, and deepseek/deepseek-flash needs it; set it with: {set} (or add DEEPSEEK_API_KEY=your_key to "
        )),
        "{stderr}"
    );
    assert!(!root.path().join("out").exists());
    // In Chinese too.
    let output = invoke(
        root.path(),
        cfg,
        &["note.md", "-o", "out", "--llm"],
        &[("MARKITAI_LANG", "zh")],
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("未设置 DEEPSEEK_API_KEY，deepseek/deepseek-flash 需要它；设置方法："),
        "{stderr}"
    );
}

#[test]
fn a_model_without_its_key_is_left_out_with_one_warning() {
    let root = tempfile::tempdir().unwrap();
    std::fs::create_dir(root.path().join("in")).unwrap();
    for name in ["a.md", "b.md"] {
        std::fs::write(root.path().join("in").join(name), "# Note\n\nText.\n").unwrap();
    }
    let (address, count) = answering();
    let cfg = json!({"cache":{"enabled":false},"llm":{"model_list":[
        {"model_name":"default","litellm_params":{"model":"gemini/gemini-flash-lite-latest"}},
        {"model_name":"default","litellm_params":{"model":"openai/served","api_key":"fake-test-key",
            "api_base":format!("http://{address}/v1")}}]}});
    let output = invoke(root.path(), cfg, &["in", "-o", "out", "--llm"], &[]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    assert_eq!(
        stderr
            .matches("gemini/gemini-flash-lite-latest is left out of the model pool: GEMINI_API_KEY is not set")
            .count(),
        1,
        "{stderr}"
    );
    // Every request went to the model that has its key.
    assert!(*count.lock().unwrap() >= 2);
}

#[test]
fn doctor_names_the_variable_a_model_lacks() {
    let root = tempfile::tempdir().unwrap();
    let cfg = json!({"llm":{"model_list":[
        {"model_name":"default","litellm_params":{"model":"deepseek/deepseek-flash"}},
        {"model_name":"default","litellm_params":{"model":"openai/x","api_key":"fake-test-key"}}]}});
    let output = invoke(root.path(), cfg, &["doctor", "--json"], &[]);
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let text = report.to_string();
    assert!(
        text.contains("Not set: DEEPSEEK_API_KEY; 1 model(s) left out, 1 still used"),
        "{text}"
    );
}
