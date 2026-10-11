//! OpenAI Batch and Anthropic Message Batches transport; persistence and output
//! publication belong to callers.
mod input;
mod transport;
mod types;

pub use crate::llm::batch::{DecodedDocument, Endpoint, Plan, Prepared, Session};
pub use crate::llm::python_batch::{
    FrozenPlan, PythonExtras, PythonItem, PythonKind, PythonPlan, PythonState,
};
pub use transport::Client;
pub use types::{
    Api, Batch, BatchStatus, DownloadFailure, Error, INLINE_INPUT, Limits, ReconcileLimits,
    Reconciliation, RemoteIdentity, ResultItem, Results, UploadedInput, batch_id, request_id,
};

#[cfg(test)]
mod anthropic_tests;
#[cfg(test)]
mod tests;
