//! `auto` as an ordered list of attempts ([`super::policy::order`]).
//!
//! The local steps keep the rules the static-then-browser path always had:
//! a page that needs JavaScript, a challenge, a verification page or an empty
//! page goes on to the browser; a script-rendered shell keeps its static text
//! when nothing better comes; any other static failure (an HTTP status, a
//! transport error) ends the local steps. Remote steps run only when consent
//! and the target allow them, and only for a failure a remote service could
//! repair: never after a 404 or 410, or a configuration or input error.

use super::policy::Step;
use super::remote::Service;
use super::{
    FetchOutcome, JS_REQUIRED, JS_SHELL, NO_BROWSER_FOR_JAVASCRIPT, NO_CONTENT,
    browser_quality_failure, needs_javascript, sites,
};
use crate::{Error, Result};

/// What a remote step may do once it is reached.
pub(super) enum Readiness {
    Ready,
    /// This service cannot run (Cloudflare without credentials).
    SkipService,
    /// No remote service may run for this URL: no consent (silently), or a
    /// target that may not leave the machine (with the reason).
    SkipAll(Option<String>),
}

pub(super) struct Attempts<'a> {
    pub static_fetch: &'a mut dyn FnMut() -> Result<FetchOutcome>,
    pub browser_ready: &'a dyn Fn() -> bool,
    /// Renders with the local browser; the flag says the static page asked
    /// for JavaScript in its own text (a route worth learning).
    pub render: &'a mut dyn FnMut(bool) -> Result<FetchOutcome>,
    pub remote_ready: &'a mut dyn FnMut(Service) -> Readiness,
    pub remote: &'a mut dyn FnMut(Service) -> Result<FetchOutcome>,
}

const NO_BROWSER_FOR_ROUTE: &str = "No local browser (Chrome or Chromium) was found for a strategy order that needs one; install one or set MARKITAI_BROWSER_EXECUTABLE (see 'markitai doctor')";

fn is_shell(outcome: &FetchOutcome) -> bool {
    matches!(&outcome.content, super::FetchContent::Document(document)
        if document.warnings.iter().any(|warning| warning == JS_SHELL))
}

/// Whether a remote service could read a page that failed this way.
fn remote_may_help(error: &Error) -> bool {
    match error {
        Error::Fetch(message) => {
            !(message.starts_with("HTTP 404 ") || message.starts_with("HTTP 410 "))
        }
        Error::Conversion(message) => message == NO_CONTENT,
        _ => false,
    }
}

fn appended(error: Error, text: &str) -> Error {
    match error {
        Error::Fetch(message) => Error::Fetch(format!("{message}; {text}")),
        Error::Conversion(message) => Error::Conversion(format!("{message}; {text}")),
        other => other,
    }
}

pub(super) fn run(steps: &[Step], attempts: Attempts<'_>) -> Result<FetchOutcome> {
    // The local failure to report, and the static text of a shell to keep.
    let mut failure: Option<Error> = None;
    let mut kept: Option<FetchOutcome> = None;
    let mut javascript_said = false;
    let mut local_over = false;
    let mut remote_blocked = false;
    let mut remote_skipped = false;
    let mut browser_missing = false;
    let mut browser_failed: Option<String> = None;
    let mut remote_failures: Vec<String> = Vec::new();
    for step in steps {
        match *step {
            Step::Static => {
                if local_over {
                    continue;
                }
                match (attempts.static_fetch)() {
                    Ok(outcome) if is_shell(&outcome) => kept = Some(outcome),
                    Ok(outcome) => return Ok(outcome),
                    Err(error) if browser_quality_failure(&error) => {
                        javascript_said =
                            matches!(&error, Error::Fetch(reason) if reason == JS_REQUIRED);
                        failure.get_or_insert(error);
                    }
                    Err(error) => {
                        remote_blocked |= !remote_may_help(&error);
                        failure.get_or_insert(error);
                        local_over = true;
                    }
                }
            }
            Step::Browser => {
                if local_over {
                    continue;
                }
                if !(attempts.browser_ready)() {
                    browser_missing = true;
                    continue;
                }
                match (attempts.render)(javascript_said) {
                    Ok(outcome) => return Ok(outcome),
                    Err(Error::Fetch(after)) => {
                        if kept.is_none() {
                            // A site's verification page is what the reader
                            // needs to hear about, not only that the browser
                            // then failed in its own way.
                            failure = Some(match failure.take() {
                                Some(Error::Fetch(before))
                                    if before.contains(sites::VERIFICATION_PAGE)
                                        && !after.contains(sites::VERIFICATION_PAGE)
                                        && !after.starts_with("HTTP ") =>
                                {
                                    Error::Fetch(format!(
                                        "{before}; the local browser failed as well ({after})"
                                    ))
                                }
                                _ => Error::Fetch(after.clone()),
                            });
                        }
                        browser_failed = Some(after);
                    }
                    // Configuration and input errors are not swallowed.
                    Err(error) => return Err(error),
                }
            }
            Step::Remote(service) => {
                if remote_blocked || remote_skipped {
                    continue;
                }
                match (attempts.remote_ready)(service) {
                    Readiness::Ready => {}
                    Readiness::SkipService => continue,
                    Readiness::SkipAll(reason) => {
                        remote_skipped = true;
                        remote_failures.extend(reason);
                        continue;
                    }
                }
                match (attempts.remote)(service) {
                    Ok(outcome) => return Ok(outcome),
                    Err(error) => remote_failures.push(error.to_string()),
                }
            }
        }
    }
    if let Some(mut outcome) = kept {
        let warnings = outcome.content.warnings_mut();
        if let Some(reason) = browser_failed {
            warnings.push(format!(
                "Browser rendering failed ({reason}); the static text was kept."
            ));
        }
        if !remote_failures.is_empty() {
            warnings.push(format!(
                "Remote extraction failed ({}); the static text was kept.",
                remote_failures.join("; ")
            ));
        }
        return Ok(outcome);
    }
    let error = match failure {
        Some(error) if browser_missing && needs_javascript(&error) => {
            Error::Fetch(NO_BROWSER_FOR_JAVASCRIPT.into())
        }
        Some(error) => error,
        None if !remote_failures.is_empty() => {
            return Err(Error::Fetch(format!(
                "No strategy could read the page: {}",
                remote_failures.join("; ")
            )));
        }
        None if browser_missing => Error::Fetch(NO_BROWSER_FOR_ROUTE.into()),
        None => Error::Fetch(
            "No fetch strategy could run for this URL; check fetch.policy.strategy_priority and fetch.domain_profiles".into(),
        ),
    };
    if remote_failures.is_empty() {
        Err(error)
    } else {
        Err(appended(
            error,
            &format!(
                "remote services failed as well ({})",
                remote_failures.join("; ")
            ),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Document;
    use crate::fetch::FetchContent;
    use std::cell::{Cell, RefCell};

    fn page(markdown: &str) -> FetchOutcome {
        FetchOutcome {
            content: FetchContent::Document(Document {
                markdown: markdown.into(),
                ..Default::default()
            }),
            cache_hit: false,
            screenshots: Vec::new(),
        }
    }

    fn shell() -> FetchOutcome {
        let mut outcome = page("# Shell");
        outcome.content.warnings_mut().push(JS_SHELL.into());
        outcome
    }

    fn text(outcome: &FetchOutcome) -> &str {
        match &outcome.content {
            FetchContent::Document(document) => &document.markdown,
            FetchContent::Pdf(_) => panic!("not a document"),
        }
    }

    const DEFAULT: [Step; 5] = [
        Step::Static,
        Step::Browser,
        Step::Remote(Service::Defuddle),
        Step::Remote(Service::Jina),
        Step::Remote(Service::Cloudflare),
    ];

    /// Runs `steps` with a static answer, an optional browser answer and
    /// remote answers, recording the services asked.
    struct Run {
        static_answer: RefCell<Option<Result<FetchOutcome>>>,
        browser: Option<RefCell<Option<Result<FetchOutcome>>>>,
        remote_ok: Option<Service>,
        ready: fn(Service) -> Readiness,
        asked: RefCell<Vec<Service>>,
        rendered: Cell<bool>,
    }

    impl Run {
        fn new(static_answer: Result<FetchOutcome>) -> Self {
            Self {
                static_answer: RefCell::new(Some(static_answer)),
                browser: None,
                remote_ok: None,
                ready: |_| Readiness::Ready,
                asked: RefCell::new(Vec::new()),
                rendered: Cell::new(false),
            }
        }
        fn browser(mut self, answer: Result<FetchOutcome>) -> Self {
            self.browser = Some(RefCell::new(Some(answer)));
            self
        }
        fn remote_ok(mut self, service: Service) -> Self {
            self.remote_ok = Some(service);
            self
        }
        fn ready(mut self, ready: fn(Service) -> Readiness) -> Self {
            self.ready = ready;
            self
        }
        fn go(&self, steps: &[Step]) -> Result<FetchOutcome> {
            let ready = self.ready;
            run(
                steps,
                Attempts {
                    static_fetch: &mut || {
                        self.static_answer
                            .borrow_mut()
                            .take()
                            .expect("one static request")
                    },
                    browser_ready: &|| self.browser.is_some(),
                    render: &mut |_| {
                        self.rendered.set(true);
                        self.browser
                            .as_ref()
                            .expect("a browser")
                            .borrow_mut()
                            .take()
                            .expect("one render")
                    },
                    remote_ready: &mut |service| ready(service),
                    remote: &mut |service| {
                        self.asked.borrow_mut().push(service);
                        if self.remote_ok == Some(service) {
                            Ok(page(&format!("from {}", service.name())))
                        } else {
                            Err(Error::Fetch(format!(
                                "HTTP 503 from the {} service",
                                service.name()
                            )))
                        }
                    },
                },
            )
        }
    }

    fn refused() -> Error {
        Error::Fetch("HTTP 403 for https://example.com/a: the site refused access".into())
    }

    #[test]
    fn a_refusal_skips_the_browser_and_goes_to_the_remote_services_in_order() {
        let run = Run::new(Err(refused())).remote_ok(Service::Jina);
        let outcome = run.go(&DEFAULT).unwrap();
        assert_eq!(text(&outcome), "from jina");
        assert!(!run.rendered.get());
        assert_eq!(*run.asked.borrow(), [Service::Defuddle, Service::Jina]);
    }

    #[test]
    fn a_missing_page_or_a_local_error_is_never_sent_anywhere() {
        for error in [
            Error::Fetch("HTTP 404 for https://example.com/a: gone".into()),
            Error::Fetch("HTTP 410 for https://example.com/a: gone".into()),
            Error::Unsupported("Unsupported URL content type: application/zip".into()),
        ] {
            let shown = error.to_string();
            let run = Run::new(Err(error)).remote_ok(Service::Defuddle);
            assert_eq!(run.go(&DEFAULT).err().unwrap().to_string(), shown);
            assert!(run.asked.borrow().is_empty());
        }
    }

    #[test]
    fn without_consent_the_local_answer_is_unchanged() {
        let run = Run::new(Err(refused())).ready(|_| Readiness::SkipAll(None));
        assert_eq!(
            run.go(&DEFAULT).err().unwrap().to_string(),
            refused().to_string()
        );
        assert!(run.asked.borrow().is_empty());
        // A target that may not leave the machine says why.
        let run = Run::new(Err(refused())).ready(|_| {
            Readiness::SkipAll(Some(
                "Private URLs cannot be sent to remote extraction services".into(),
            ))
        });
        let message = run.go(&DEFAULT).err().unwrap().to_string();
        assert!(message.starts_with("HTTP 403 for "), "{message}");
        assert!(message.ends_with("remote services failed as well (Private URLs cannot be sent to remote extraction services)"), "{message}");
    }

    #[test]
    fn every_failure_is_reported_with_the_local_one_first() {
        let run = Run::new(Err(refused()));
        let message = run.go(&DEFAULT).err().unwrap().to_string();
        assert_eq!(
            message,
            format!(
                "{}; remote services failed as well (HTTP 503 from the defuddle service; HTTP 503 from the jina service; HTTP 503 from the cloudflare service)",
                refused()
            )
        );
        // A service without credentials is passed over without a word.
        let run = Run::new(Err(refused())).ready(|service| match service {
            Service::Cloudflare => Readiness::SkipService,
            _ => Readiness::Ready,
        });
        let message = run.go(&DEFAULT).err().unwrap().to_string();
        assert!(!message.contains("cloudflare"), "{message}");
        assert_eq!(*run.asked.borrow(), [Service::Defuddle, Service::Jina]);
    }

    #[test]
    fn javascript_pages_try_the_browser_before_any_remote_service() {
        let needs = || Error::Fetch(JS_REQUIRED.into());
        let run = Run::new(Err(needs()))
            .browser(Ok(page("rendered")))
            .remote_ok(Service::Defuddle);
        assert_eq!(text(&run.go(&DEFAULT).unwrap()), "rendered");
        assert!(run.asked.borrow().is_empty());
        // No browser: remote, and the no-browser advice when that fails too.
        let run = Run::new(Err(needs())).remote_ok(Service::Defuddle);
        assert_eq!(text(&run.go(&DEFAULT).unwrap()), "from defuddle");
        let run = Run::new(Err(needs())).ready(|_| Readiness::SkipAll(None));
        assert_eq!(
            run.go(&DEFAULT).err().unwrap().to_string(),
            NO_BROWSER_FOR_JAVASCRIPT
        );
        // The browser fails: remote services after it.
        let run = Run::new(Err(needs()))
            .browser(Err(Error::Fetch("Browser navigation timed out".into())))
            .remote_ok(Service::Jina);
        assert_eq!(text(&run.go(&DEFAULT).unwrap()), "from jina");
    }

    #[test]
    fn a_shell_is_replaced_by_a_remote_reading_or_kept_with_every_reason() {
        let run = Run::new(Ok(shell())).remote_ok(Service::Defuddle);
        assert_eq!(text(&run.go(&DEFAULT).unwrap()), "from defuddle");
        let run =
            Run::new(Ok(shell())).browser(Err(Error::Fetch("Browser navigation timed out".into())));
        let kept = run.go(&DEFAULT).unwrap();
        assert_eq!(text(&kept), "# Shell");
        let FetchContent::Document(document) = &kept.content else {
            panic!()
        };
        assert_eq!(document.warnings.len(), 3);
        assert_eq!(document.warnings[0], JS_SHELL);
        assert!(
            document.warnings[1]
                .starts_with("Browser rendering failed (Browser navigation timed out)")
        );
        assert!(
            document.warnings[2]
                .starts_with("Remote extraction failed (HTTP 503 from the defuddle service;")
        );
        // Without consent the warnings are what they always were.
        let run = Run::new(Ok(shell())).ready(|_| Readiness::SkipAll(None));
        let kept = run.go(&DEFAULT).unwrap();
        let FetchContent::Document(document) = &kept.content else {
            panic!()
        };
        assert_eq!(document.warnings, [JS_SHELL]);
    }

    #[test]
    fn a_browser_first_order_falls_back_to_static_then_remote() {
        let order = [Step::Browser, Step::Static, Step::Remote(Service::Defuddle)];
        let run = Run::new(Ok(page("static text"))).browser(Err(Error::Fetch(
            "HTTP 403 for https://x.com/a: refused".into(),
        )));
        assert_eq!(text(&run.go(&order).unwrap()), "static text");
        // Both local steps fail: the browser's failure leads.
        let run = Run::new(Err(refused()))
            .browser(Err(Error::Fetch("Browser navigation timed out".into())))
            .remote_ok(Service::Defuddle);
        assert_eq!(text(&run.go(&order).unwrap()), "from defuddle");
        let run = Run::new(Err(refused()))
            .browser(Err(Error::Fetch("Browser navigation timed out".into())))
            .ready(|_| Readiness::SkipAll(None));
        assert_eq!(
            run.go(&order).err().unwrap().to_string(),
            "Browser navigation timed out"
        );
        // A configured order may put a remote service first.
        let run = Run::new(Ok(page("static text"))).remote_ok(Service::Jina);
        assert_eq!(
            text(
                &run.go(&[Step::Remote(Service::Jina), Step::Static])
                    .unwrap()
            ),
            "from jina"
        );
        // An order whose only step cannot run says so.
        let run = Run::new(Ok(page("unused")));
        let message = run.go(&[Step::Browser]).err().unwrap().to_string();
        assert!(message.starts_with("No local browser"), "{message}");
    }
}
