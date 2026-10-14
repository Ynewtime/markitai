//! Caller-owned Chromium processes and ephemeral web sessions.
use crate::{Result, browser::pool::Pool};

/// A bounded owner for browser reuse within one trusted caller or conversion run.
/// It never shares sessions with another instance or writes reusable login state.
pub struct BrowserRuntime {
    pub(crate) pool: Pool,
    /// Whether a page this runtime loads without screenshots of its own is
    /// also captured as compare screenshots (see `PageCapture`).
    pub(crate) compare_screenshots: bool,
}

impl BrowserRuntime {
    /// Capacity counts starting, active, idle and retiring Chromium processes together.
    pub fn new(capacity: usize) -> Result<Self> {
        Ok(Self {
            pool: Pool::new(capacity)?,
            compare_screenshots: false,
        })
    }

    /// Also capture each page this runtime loads for its content, when the
    /// conversion asked for no screenshots, as compare screenshots (JPEG
    /// quality 70, at most 20 tiles) with the page's block boxes, for
    /// showing the page beside its Markdown. They reach a caller only through
    /// [`crate::convert_with_page_capture`]: they are never conversion output
    /// and never sent to a model or OCR.
    #[must_use]
    pub fn keeping_compare_screenshots(mut self) -> Self {
        self.compare_screenshots = true;
        self
    }

    /// Stop admission and dispose idle sessions. Active calls keep their normal
    /// deadlines, then dispose their processes instead of returning them to the pool.
    pub fn close(&self) {
        self.pool.close();
    }
}

impl std::fmt::Debug for BrowserRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BrowserRuntime")
            .finish_non_exhaustive()
    }
}
