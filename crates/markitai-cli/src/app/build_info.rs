//! The build identity `--help` ends with: the version, the commit this binary
//! was built from (marked when the worktree differed from it), the target and
//! profile, and when the compiler ran.
//!
//! Only `--help` carries this: `-h` stays as it was, and scripts read the
//! version from the last word of the `--version` line.
use crate::app::i18n::Lang;
use chrono::{DateTime, Utc};
use markitai_core::build_info;

/// The commit as one reads it: a hash, with a marker when the build included
/// changes the commit does not hold.
fn commit(lang: Lang) -> String {
    if !build_info::dirty() {
        return build_info::COMMIT.to_owned();
    }
    match lang {
        Lang::En => format!("{} (local changes)", build_info::COMMIT),
        Lang::Zh => format!("{}（含本地改动）", build_info::COMMIT),
    }
}

/// Compiler start time in UTC, or `unknown` when no clock and no
/// `SOURCE_DATE_EPOCH` were available.
fn built() -> String {
    match DateTime::<Utc>::from_timestamp(build_info::epoch(), 0) {
        Some(at) => at.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
        None => "unknown".to_owned(),
    }
}

/// The identity block, without a leading blank line.
pub(crate) fn block(lang: Lang) -> String {
    let fields = (
        markitai_core::VERSION,
        commit(lang),
        build_info::TARGET,
        build_info::PROFILE,
        built(),
    );
    match lang {
        Lang::En => format!(
            "Build:\n  version  {}\n  commit   {}\n  target   {}, {}\n  built    {}",
            fields.0, fields.1, fields.2, fields.3, fields.4,
        ),
        Lang::Zh => format!(
            "构建信息:\n  版本  {}\n  提交  {}\n  目标  {}, {}\n  构建  {}",
            fields.0, fields.1, fields.2, fields.3, fields.4,
        ),
    }
}

/// The long help of a command, with the identity appended to what it already
/// ends with. A command without closing text still gets the block.
pub(crate) fn append(lang: Lang, existing: Option<String>) -> String {
    let block = block(lang);
    match existing {
        Some(text) if !text.trim().is_empty() => format!("{}\n\n{block}", text.trim_end()),
        _ => block,
    }
}
