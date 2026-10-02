use super::*;
use crate::output_claims::{MemberLeases, Policy};

fn item(number: usize) -> ItemKey {
    ItemKey::File(format!("{number}.txt"))
}
fn names(base: &str) -> Vec<String> {
    vec![format!("{base}.md"), format!("{base}.llm.md")]
}

#[test]
fn eviction_rebuilds_every_missing_completed_family_and_alias() {
    let root = tempfile::tempdir().unwrap();
    let reservations = Reservations::new();
    let first = root.path().join("0");
    std::fs::create_dir(&first).unwrap();
    reservations
        .reserve(&first, &names("x"), item(0), false)
        .unwrap();
    reservations
        .reserve(&first, &names("literal.llm"), item(1), false)
        .unwrap();
    for index in 0..=IDLE_PARENTS {
        let parent = root.path().join(index.to_string());
        if index != 0 {
            std::fs::create_dir(&parent).unwrap();
        }
        drop(reservations.epoch(&parent, false).unwrap());
    }
    assert!(reservations.inner.lock().unwrap().hot.len() <= IDLE_PARENTS);
    let id = platform::status(&first).unwrap().id();
    assert!(!reservations.inner.lock().unwrap().hot.contains_key(&id));
    assert_eq!(
        std::fs::read_dir(first.join(".markitai/ownership/names-v2"))
            .unwrap()
            .count(),
        0
    );
    assert!(
        reservations
            .blocked(&first, &names("x.llm"), &item(2), false)
            .unwrap()
    );
    assert!(
        reservations
            .blocked(&first, &names("literal.llm"), &item(2), false)
            .unwrap()
    );
    assert!(
        !reservations
            .blocked(&first, &names("x"), &item(0), false)
            .unwrap()
    );
    drop(reservations);
    assert_eq!(
        std::fs::read_dir(first.join(".markitai/ownership/names-v2"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn retained_claims_keep_epochs_and_terminal_names_without_member_fds() {
    let root = tempfile::tempdir().unwrap();
    let reservations = Reservations::new();
    let epoch = reservations.epoch(root.path(), false).unwrap();
    let leases = MemberLeases::acquire_with_epoch(root.path(), &names("x"), false, epoch).unwrap();
    let claim = Claim::new(leases, None, Policy::NoClobber, false).unwrap();
    reservations.retain(&claim, item(0)).unwrap();
    for index in 0..20 {
        let parent = root.path().join(index.to_string());
        std::fs::create_dir(&parent).unwrap();
        drop(reservations.epoch(&parent, false).unwrap());
    }
    let id = platform::status(root.path()).unwrap().id();
    assert!(reservations.inner.lock().unwrap().hot.contains_key(&id));
    assert!(
        reservations
            .blocked(root.path(), &names("x.llm"), &item(1), false)
            .unwrap()
    );
    drop(claim);
    assert!(
        reservations
            .blocked(root.path(), &names("x.llm"), &item(1), false)
            .unwrap()
    );
}

#[cfg(unix)]
fn fds() -> usize {
    // SAFETY: F_GETFD observes an integer descriptor without retaining it.
    (0..1024)
        .filter(|&fd| unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0)
        .count()
}

#[test]
#[ignore = "10k/100k coordination benchmark with an isolated caller-owned root"]
fn coordination_benchmark_helper() {
    use std::time::Instant;
    let Some(root) = std::env::var_os("MARKITAI_V2_BENCH_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let n: usize = std::env::var("MARKITAI_V2_BENCH_N")
        .unwrap()
        .parse()
        .unwrap();
    let legacy = std::env::var("MARKITAI_V2_BENCH_LEGACY").ok().as_deref() == Some("1");
    if legacy {
        for name in [
            ".markitai",
            ".markitai/ownership",
            ".markitai/ownership/members",
        ] {
            platform::private_directory()
                .create(root.join(name))
                .unwrap();
        }
    }
    let reservations = Reservations::new();
    let all = Instant::now();
    for i in 0..n {
        reservations
            .reserve(&root, &names(&format!("item-{i}")), item(i), false)
            .unwrap();
    }
    let logical = all.elapsed().as_secs_f64();
    let activation = Instant::now();
    drop(reservations.epoch(&root, false).unwrap());
    let activation = activation.elapsed().as_secs_f64();
    #[cfg(unix)]
    let mut peak_fds = fds();
    let concurrency: usize = std::env::var("MARKITAI_V2_BENCH_C")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    let work = Instant::now();
    for start in (0..n).step_by(concurrency) {
        let mut claims = Vec::new();
        for i in start..(start + concurrency).min(n) {
            let epoch = reservations.epoch(&root, false).unwrap();
            let leases =
                MemberLeases::acquire_with_epoch(&root, &names(&format!("item-{i}")), false, epoch)
                    .unwrap();
            let claim = Claim::new(leases, None, Policy::NoClobber, false).unwrap();
            assert!(
                !reservations
                    .blocks_keys(claim.parent(), &claim.keys(), &item(i))
                    .unwrap()
            );
            claims.push(claim);
        }
        #[cfg(unix)]
        if start % 100 == 0 {
            peak_fds = peak_fds.max(fds());
        }
        drop(claims);
    }
    let work = work.elapsed().as_secs_f64();
    let cleanup = Instant::now();
    drop(reservations);
    let cleanup = cleanup.elapsed().as_secs_f64();
    let total = all.elapsed().as_secs_f64();
    let (gate_calls, gate_wait_ns) = v2::gate_metrics();
    let mut metrics = serde_json::json!({"n":n,"legacy":legacy,"logical_s":logical,"activation_s":activation,"claims_s":work,"cleanup_s":cleanup,"total_s":total});
    metrics["concurrency"] = serde_json::json!(concurrency);
    metrics["gate_calls"] = serde_json::json!(gate_calls);
    metrics["gate_wait_s"] = serde_json::json!(gate_wait_ns as f64 / 1e9);
    #[cfg(unix)]
    {
        // SAFETY: getrusage initializes the provided struct on success.
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            metrics["maxrss_native_units"] =
                serde_json::json!(unsafe { usage.assume_init() }.ru_maxrss);
        }
        metrics["peak_fds"] = serde_json::json!(peak_fds);
    }
    if !legacy {
        {
            let name = "names-v2";
            assert_eq!(
                std::fs::read_dir(root.join(".markitai/ownership").join(name))
                    .unwrap()
                    .count(),
                0
            );
        }
    }
    std::fs::write(
        root.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics).unwrap(),
    )
    .unwrap();
}

fn parent_alias_cold_activation(original_name: &str, alias_name: &str) {
    let root = tempfile::tempdir().unwrap();
    let original = root.path().join(original_name);
    std::fs::create_dir(&original).unwrap();
    let alias = root.path().join(alias_name);
    if platform::status(&alias).ok().map(|s| s.id())
        != platform::status(&original).ok().map(|s| s.id())
    {
        return;
    }
    let reserved = Reservations::new();
    reserved
        .reserve(&original, &names("x"), item(0), false)
        .unwrap();
    reserved
        .reserve(&original, &names("literal.llm"), item(1), false)
        .unwrap();
    drop(reserved.epoch(&original, false).unwrap());
    for n in 0..17 {
        let parent = root.path().join(n.to_string());
        std::fs::create_dir(&parent).unwrap();
        drop(reserved.epoch(&parent, false).unwrap());
    }
    assert!(
        reserved
            .blocked(&alias, &names("x.llm"), &item(2), false)
            .unwrap()
    );
    assert!(
        reserved
            .blocked(&alias, &names("literal.llm"), &item(2), false)
            .unwrap()
    );
    assert!(
        !reserved
            .blocked(&alias, &names("x"), &item(0), false)
            .unwrap()
    );
    let epoch = reserved.epoch(&alias, false).unwrap();
    let lease = MemberLeases::acquire_with_epoch(&alias, &names("next"), false, epoch).unwrap();
    lease.validate_member(&alias.join("next.md")).unwrap();
}
#[test]
fn parent_case_alias_rebuilds_all_completed_missing_names() {
    parent_alias_cold_activation("Out", "out");
}
#[test]
fn parent_unicode_alias_rebuilds_all_completed_missing_names() {
    parent_alias_cold_activation("caf\u{e9}", "cafe\u{301}");
}
