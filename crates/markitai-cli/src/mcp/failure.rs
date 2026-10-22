//! Application failures keep their observed work until the wire boundary.
use crate::diagnostics::{AttemptDiagnostics, Operation};
use markitai_core::{ConversionFailure, ConversionUsage};

#[derive(Debug)]
pub(super) struct Failure {
    pub message: String,
    pub diagnostics: Option<AttemptDiagnostics>,
}

impl Failure {
    pub fn observed(message: String, usage: ConversionUsage) -> Self {
        let diagnostics = AttemptDiagnostics::failed(Operation::Convert, &message, usage);
        Self {
            message,
            diagnostics,
        }
    }
}

impl From<ConversionFailure> for Failure {
    fn from(failure: ConversionFailure) -> Self {
        let message = if matches!(failure.error, markitai_core::Error::NoModelConfigured) {
            format!(
                "{}. For this MCP server, set MODEL and a provider API key in the env block of its mcpServers entry, or configure llm.model_list.",
                failure.error
            )
        } else if matches!(failure.error, markitai_core::Error::MissingApiKey(_)) {
            format!(
                "{}. For this MCP server, the variable can also be set in the env block of its mcpServers entry.",
                failure.error
            )
        } else {
            failure.error.to_string()
        };
        Self::observed(message, failure.usage)
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self {
            message,
            diagnostics: None,
        }
    }
}

impl From<&str> for Failure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}
