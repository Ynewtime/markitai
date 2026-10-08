use std::path::Path;
use std::process::Command;

fn init(work: &Path, vars: &[(&str, &Path)]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
    command.env_clear();
    for key in ["PATH", "SYSTEMROOT", "TMPDIR"] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    for (key, value) in vars {
        command.env(key, value);
    }
    command
        .current_dir(work)
        .env("COPILOT_CLI_PATH", work.join("missing-copilot"))
        .env("CLAUDE_CLI_PATH", work.join("missing-claude"))
        .env("CODEX_CLI_PATH", work.join("missing-codex"))
        .args(["init", "--yes"])
        .output()
        .unwrap()
}

#[test]
fn an_empty_home_variable_is_unset_rather_than_the_current_directory() {
    let root = tempfile::tempdir().unwrap();
    let user = root.path().join("user");
    let work = root.path().join("work");
    std::fs::create_dir_all(&user).unwrap();
    std::fs::create_dir_all(&work).unwrap();
    let empty = Path::new("");
    for vars in [
        vec![
            ("MARKITAI_HOME", empty),
            ("HOME", user.as_path()),
            ("USERPROFILE", user.as_path()),
        ],
        vec![("HOME", empty), ("USERPROFILE", user.as_path())],
    ] {
        let output = init(&work, &vars);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(user.join(".markitai/config.json").is_file(), "{vars:?}");
        let left: Vec<_> = std::fs::read_dir(&work).unwrap().collect();
        assert!(left.is_empty(), "{vars:?} wrote {left:?}");
        std::fs::remove_dir_all(user.join(".markitai")).unwrap();
    }
}
