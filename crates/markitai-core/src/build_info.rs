//! The build identity of this build: the commit the sources came from, whether
//! the worktree differed from it, the target and profile, and when the build
//! script ran.
//!
//! `build.rs` reads these from git and the environment and publishes them with
//! `cargo:rustc-env`, so every package that runs it (the core, the CLI and the
//! language bindings) carries its own copy. `SOURCE_DATE_EPOCH` fixes the
//! timestamp, and a source archive without a checkout reports `unknown`.

/// The commit these sources came from, or `unknown` outside a checkout.
pub const COMMIT: &str = env!("MARKITAI_BUILD_COMMIT");
/// `1` when the worktree differed from that commit as this build ran.
pub const DIRTY: &str = env!("MARKITAI_BUILD_DIRTY");
/// Seconds since the Unix epoch when the build script ran.
pub const EPOCH: &str = env!("MARKITAI_BUILD_EPOCH");
/// The target triple this build was compiled for.
pub const TARGET: &str = env!("MARKITAI_BUILD_TARGET");
/// The cargo profile this build used, `debug` or `release`.
pub const PROFILE: &str = env!("MARKITAI_BUILD_PROFILE");

/// Whether the worktree differed from [`COMMIT`] when this build ran.
pub const fn dirty() -> bool {
    matches!(DIRTY.as_bytes(), b"1")
}

/// [`EPOCH`] as a number, or `0` when the build script could not read a clock.
pub fn epoch() -> i64 {
    EPOCH.trim().parse().unwrap_or(0)
}
