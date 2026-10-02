//! The terminal side of remote-fallback consent (the core's
//! `RemoteFallback`): the question asked once per run, under
//! `fetch.remote_consent=ask`, before `auto` first sends a public URL that
//! local fetching could not read to a remote extraction service, and the two
//! notices the core asks for.
//!
//! The question needs a terminal on stdin and stderr and a run that is not
//! `--quiet`; otherwise `ask` counts as `never`, with one note. While it is
//! asked, stderr is held, so the status line cannot draw over it and other
//! conversions wait for the answer.

use super::i18n::{self, Lang};
use markitai_core::{ConsentRequest, RemoteFallback, RemoteNotice};
use serde_json::Value;
use std::io::{self, BufRead, IsTerminal, Write};

type Ask = Box<dyn Fn(&ConsentRequest) -> bool + Send + Sync>;

/// What the user's own configuration (the selected file merged with
/// `--config-json`, before defaults) says that the effective configuration
/// cannot: `fetch.remote_consent` written as `always`, which unlike the same
/// default opts in to remote fallback, and a `fetch.fallback_patterns` list
/// written at all, which unlike the default list makes its domains
/// browser-first.
pub(super) fn written(raw: &Value) -> (bool, bool) {
    (
        raw.pointer("/fetch/remote_consent").and_then(Value::as_str) == Some("always"),
        raw.pointer("/fetch/fallback_patterns")
            .is_some_and(|value| !value.is_null()),
    )
}

/// Install this process's consent facts from the user's own configuration
/// (see [`written`]).
pub(super) fn install(raw: &Value, quiet: bool) {
    let (explicitly_always, explicit_fallback_patterns) = written(raw);
    let interactive = !quiet && io::stdin().is_terminal() && io::stderr().is_terminal();
    let ask = interactive.then(|| {
        Box::new(|request: &ConsentRequest| {
            let mut stderr = io::stderr().lock();
            super::progress::clear();
            let stdin = io::stdin();
            let mut input = stdin.lock();
            question(request, &mut input, &mut stderr, i18n::lang())
        }) as Ask
    });
    let notify = Box::new(move |notice: &RemoteNotice| {
        if quiet {
            return false;
        }
        let english = notice_text(notice, Lang::En);
        let console = notice_text(notice, i18n::lang());
        super::logging::diagnostic_as(format_args!("{english}"), format_args!("{console}"));
        true
    });
    markitai_core::set_remote_fallback(RemoteFallback {
        explicitly_always,
        explicit_fallback_patterns,
        ask,
        notify,
    });
}

fn names(services: &[&str], lang: Lang) -> String {
    services
        .iter()
        .map(|service| match (*service, lang) {
            ("defuddle", _) => "defuddle.md",
            ("jina", _) => "Jina Reader",
            ("cloudflare", Lang::En) => "Cloudflare (your account)",
            ("cloudflare", Lang::Zh) => "Cloudflare（你的账户）",
            (other, _) => other,
        })
        .collect::<Vec<_>>()
        .join(match lang {
            Lang::En => ", ",
            Lang::Zh => "、",
        })
}

/// Ask on `output`, read one line from `input`: yes only for `y`, `yes` (or
/// `是`, `好`, `允许`); an empty line, anything else or the end of input is no.
pub(super) fn question(
    request: &ConsentRequest,
    input: &mut dyn BufRead,
    output: &mut dyn Write,
    lang: Lang,
) -> bool {
    let url = &request.url;
    let services = names(&request.services, lang);
    let text = text!(lang =>
        "Local fetching could not read {url}.\nThis run can send public URLs that local fetching cannot read to remote extraction services, one at a time until one succeeds: {services}.\nAllow that for this run? [y/N] ",
        "本地抓取无法读取 {url}。\n本次运行可以把本地无法读取的公开 URL 依次发送给远程抽取服务，直到其中一个成功：{services}。\n本次运行允许吗？[y/N] ",
    );
    let _ = output.write_all(text.as_bytes());
    let _ = output.flush();
    let mut line = String::new();
    if input.read_line(&mut line).unwrap_or(0) == 0 {
        // No answer: the next line of output starts on its own line.
        let _ = output.write_all(b"\n");
        let _ = output.flush();
        return false;
    }
    matches!(
        line.trim().to_lowercase().as_str(),
        "y" | "yes" | "是" | "好" | "允许"
    )
}

pub(super) fn notice_text(notice: &RemoteNotice, lang: Lang) -> String {
    match notice {
        RemoteNotice::Disclosure { services } => {
            let services = names(services, lang);
            text!(lang =>
                "Note: fetch.remote_consent is always, so public URLs that local fetching cannot read are sent to remote extraction services ({services}). Set it to ask or never, or pass --no-remote-fetch, to keep fetching local. This note is shown once.",
                "提示：fetch.remote_consent 为 always，本地无法读取的公开 URL 会发送给远程抽取服务（{services}）。设为 ask 或 never，或使用 --no-remote-fetch，即可只在本地抓取。此提示只显示一次。",
            )
        }
        RemoteNotice::NotAsked => text!(lang =>
            "Note: remote extraction services were skipped: fetch.remote_consent is ask and there is no terminal to ask. Set it to always to allow them, or to never to fetch locally without this note.",
            "提示：已跳过远程抽取服务：fetch.remote_consent 为 ask，但没有可以询问的终端。设为 always 可允许使用，设为 never 则只在本地抓取且不再提示。",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> ConsentRequest {
        ConsentRequest {
            url: "https://example.com/a".into(),
            services: vec!["defuddle", "jina", "cloudflare"],
        }
    }

    fn answer(typed: &str, lang: Lang) -> (bool, String) {
        let mut output = Vec::new();
        let yes = question(&request(), &mut typed.as_bytes(), &mut output, lang);
        (yes, String::from_utf8(output).unwrap())
    }

    #[test]
    fn only_a_yes_allows_and_the_question_names_the_page_and_services() {
        for typed in ["y\n", "Yes\n", " YES \n"] {
            assert!(answer(typed, Lang::En).0, "{typed:?}");
        }
        for typed in ["\n", "n\n", "no\n", "sure\n"] {
            assert!(!answer(typed, Lang::En).0, "{typed:?}");
        }
        let (yes, shown) = answer("", Lang::En);
        assert!(!yes);
        assert!(shown.ends_with("[y/N] \n"), "{shown}");
        assert!(shown.starts_with("Local fetching could not read https://example.com/a."));
        assert!(
            shown.contains("defuddle.md, Jina Reader, Cloudflare (your account)"),
            "{shown}"
        );
        let (yes, shown) = answer("是\n", Lang::Zh);
        assert!(yes);
        assert!(
            shown.contains("defuddle.md、Jina Reader、Cloudflare（你的账户）"),
            "{shown}"
        );
    }

    #[test]
    fn only_values_the_user_wrote_count_as_written() {
        use serde_json::json;
        assert_eq!(written(&json!({})), (false, false));
        // The effective configuration always holds both; only the raw one tells.
        let effective = markitai_core::config::defaults();
        assert_eq!(effective["fetch"]["remote_consent"], "always");
        assert!(effective["fetch"]["fallback_patterns"].is_array());
        assert_eq!(
            written(&json!({"fetch": {"remote_consent": "always", "fallback_patterns": []}})),
            (true, true)
        );
        assert_eq!(
            written(&json!({"fetch": {"remote_consent": "ask", "fallback_patterns": ["x.com"]}})),
            (false, true)
        );
        assert_eq!(
            written(&json!({"fetch": {"strategy": "auto"}})),
            (false, false)
        );
    }

    #[test]
    fn notices_say_which_setting_decides_in_both_languages() {
        let disclosure = RemoteNotice::Disclosure {
            services: vec!["defuddle", "jina"],
        };
        for lang in [Lang::En, Lang::Zh] {
            for notice in [&disclosure, &RemoteNotice::NotAsked] {
                let text = notice_text(notice, lang);
                assert!(text.contains("fetch.remote_consent"), "{text}");
            }
        }
        let english = notice_text(&disclosure, Lang::En);
        assert!(english.contains("(defuddle.md, Jina Reader)"), "{english}");
        assert!(english.contains("--no-remote-fetch"), "{english}");
        assert!(notice_text(&RemoteNotice::NotAsked, Lang::En).starts_with("Note: "));
    }
}
