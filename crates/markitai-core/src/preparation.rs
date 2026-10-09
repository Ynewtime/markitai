//! Own conversion work without acknowledging document publication.
use super::*;

/// A provisional conversion is not an externally visible successful result.
/// Assets may already exist; document members and their image sidecar remain pending.
#[doc(hidden)]
#[must_use = "prepared conversions must be published or explicitly failed"]
pub struct PreparedConversion {
    outcome: DetailedResult<ConversionOutput>,
    plan: output::PreparedOutput,
    usage: ConversionUsage,
    /// The item's own conversion and publication time so far; time spent
    /// waiting for a publication group to commit is not part of it.
    worked: std::time::Duration,
}

/// Prepare the exact document bytes for an already acquired native claim.
/// This path never calls Publication::publish and never returns a completed result.
#[doc(hidden)]
pub fn prepare_with_publication(
    source: &str,
    options: ConvertOptions,
    context: ConvertContext<'_>,
    publication: &dyn output::Publication,
) -> PreparedConversion {
    let started = Instant::now();
    let mut scope = None;
    let mut plan = output::PreparedOutput::default();
    let result = convert_inner(
        source,
        options,
        context,
        Some(publication),
        &mut scope,
        Some(&mut plan),
    );
    let usage = scope
        .as_ref()
        .map(llm::DocumentScope::usage)
        .unwrap_or_default();
    let outcome = result.map_err(|error| ConversionFailure {
        error,
        usage: usage.clone(),
    });
    PreparedConversion {
        outcome,
        plan,
        usage,
        worked: started.elapsed(),
    }
}

impl PreparedConversion {
    pub fn members(&self) -> &[output::RenderedMember] {
        &self.plan.members
    }

    /// Transfer the pending document bytes to an owned publication plan.
    /// The caller must commit every member before finish_after_publication.
    pub fn take_members(&mut self) -> Vec<output::RenderedMember> {
        std::mem::take(&mut self.plan.members)
    }

    /// Complete side effects after all transferred document members are durable.
    /// A retained-base model failure remains a failure after its base is published.
    pub fn finish_after_publication(self) -> DetailedResult<ConversionOutput> {
        if !self.plan.members.is_empty() {
            return Err(self.fail_publication(Error::InvalidInput(
                "Prepared document members have not been transferred for publication".into(),
            )));
        }
        let Self {
            mut outcome,
            plan,
            usage,
            worked,
        } = self;
        let finalizing = Instant::now();
        plan.finalize()
            .map_err(|error| ConversionFailure { error, usage })?;
        if let Ok(output) = &mut outcome {
            output.duration = (worked + finalizing.elapsed()).as_secs_f64();
        }
        outcome
    }

    /// Replace provisional success/failure with a publication error, keeping usage.
    pub fn fail_publication(self, error: Error) -> ConversionFailure {
        ConversionFailure {
            error,
            usage: self.usage,
        }
    }

    /// Publish an empty or oversized plan without rerunning conversion or model work.
    /// The caller retains its claim through this operation and result recording.
    pub fn publish_immediately(
        mut self,
        publication: &dyn output::Publication,
    ) -> DetailedResult<ConversionOutput> {
        let publishing = Instant::now();
        for member in self.take_members() {
            if let Err(error) = publication.publish(&member.path, &member.bytes) {
                return Err(self.fail_publication(error));
            }
        }
        self.worked += publishing.elapsed();
        self.finish_after_publication()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::Publication;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Publisher {
        calls: AtomicUsize,
        fail: bool,
    }
    impl output::Publication for Publisher {
        fn skip_existing(&self) -> bool {
            false
        }
        fn publish(&self, path: &Path, bytes: &[u8]) -> Result<()> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                return Err(Error::Conversion("authored publication failure".into()));
            }
            std::fs::write(path, bytes)?;
            Ok(())
        }
    }
    fn prepare(root: &Path, publication: &Publisher) -> PreparedConversion {
        let source = root.join("source.txt");
        std::fs::write(&source, "Complete authored text 🚀.\n").unwrap();
        prepare_with_publication(
            source.to_str().unwrap(),
            ConvertOptions {
                output_dir: Some(root.join("out")),
                config: Some(json!({"llm":{"enabled":false},"ocr":{"enabled":false},
                "screenshot":{"enabled":false},"image":{"alt_enabled":false,"desc_enabled":false},
                "log":{"dir":null},"prompts":{"dir":root.join("prompts")},
                "cache":{"enabled":false,"global_dir":root.join("cache")}})),
                ..Default::default()
            },
            ConvertContext::default(),
            publication,
        )
    }
    #[test]
    fn preparation_does_not_publish_and_is_owned_send_without_claim_borrows() {
        fn is_send<T: Send>() {}
        is_send::<PreparedConversion>();
        let root = tempfile::tempdir().unwrap();
        let publication = Publisher {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        let prepared = prepare(root.path(), &publication);
        assert_eq!(publication.calls.load(Ordering::SeqCst), 0);
        assert_eq!(prepared.members().len(), 1);
        let path = prepared.members()[0].path.clone();
        assert!(!path.exists());
        assert!(
            String::from_utf8_lossy(&prepared.members()[0].bytes)
                .contains("Complete authored text 🚀.")
        );
        let failure = prepared.finish_after_publication().unwrap_err();
        assert!(failure.error.to_string().contains("not been transferred"));
        assert!(!path.exists());
    }
    #[test]
    fn transferred_plan_finishes_after_publication_and_drop_has_no_success_side_effect() {
        let root = tempfile::tempdir().unwrap();
        let publication = Publisher {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        let mut prepared = prepare(root.path(), &publication);
        let members = prepared.take_members();
        for member in &members {
            publication.publish(&member.path, &member.bytes).unwrap();
        }
        let output = prepared.finish_after_publication().unwrap();
        assert_eq!(output.output_path.as_ref(), Some(&members[0].path));
        assert_eq!(std::fs::read(&members[0].path).unwrap(), members[0].bytes);
        assert!(output.duration > 0.0);

        let other = tempfile::tempdir().unwrap();
        let prepared = prepare(other.path(), &publication);
        let path = prepared.members()[0].path.clone();
        drop(prepared);
        assert!(!path.exists());
    }
    #[test]
    fn waiting_for_a_publication_group_is_not_the_items_duration() {
        let root = tempfile::tempdir().unwrap();
        let publication = Publisher {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        let mut prepared = prepare(root.path(), &publication);
        let converted = prepared.worked;
        let members = prepared.take_members();
        // Other items of a serial batch join the group before it commits.
        std::thread::sleep(std::time::Duration::from_millis(400));
        for member in &members {
            publication.publish(&member.path, &member.bytes).unwrap();
        }
        let output = prepared.finish_after_publication().unwrap();
        assert!(output.duration >= converted.as_secs_f64());
        assert!(
            output.duration < converted.as_secs_f64() + 0.3,
            "{}",
            output.duration
        );
    }
    #[test]
    fn immediate_fallback_uses_the_prepared_bytes_and_preserves_failure() {
        let root = tempfile::tempdir().unwrap();
        let publication = Publisher {
            calls: AtomicUsize::new(0),
            fail: false,
        };
        let prepared = prepare(root.path(), &publication);
        let expected = prepared.members()[0].bytes.clone();
        let output = prepared.publish_immediately(&publication).unwrap();
        assert_eq!(publication.calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            std::fs::read(output.output_path.unwrap()).unwrap(),
            expected
        );

        let other = tempfile::tempdir().unwrap();
        let denied = Publisher {
            calls: AtomicUsize::new(0),
            fail: true,
        };
        let prepared = prepare(other.path(), &denied);
        let path = prepared.members()[0].path.clone();
        assert!(
            prepared
                .publish_immediately(&denied)
                .unwrap_err()
                .error
                .to_string()
                .contains("authored publication failure")
        );
        assert!(!path.exists());
    }
}
