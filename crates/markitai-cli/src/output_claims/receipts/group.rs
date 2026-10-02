//! Bounded document preparation with three ordered phases and one durable acknowledgement.
use super::super::{Claim, sync_group::SyncGroup};
use super::*;
use std::collections::BTreeSet;
use std::sync::Arc;

pub(crate) const MAX_DOCUMENTS: usize = 16;
pub(crate) const MAX_DOCUMENT_BYTES: usize = 64 * 1024 * 1024;

/// A final rendered Markdown member; paths must already belong to the claim.
pub(crate) struct RenderedMember {
    pub(crate) path: PathBuf,
    pub(crate) bytes: Vec<u8>,
}

/// Fixed-size receipts and prepared bytes must not cause an unbounded read if a
/// noncooperating writer substitutes a much larger object during verification.
fn observe_regular_bounded(path: &Path, limit: u64) -> Result<Option<Proof>> {
    let before = match platform::status(path) {
        Ok(status) => status,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let metadata = before.metadata();
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > limit {
        return Err(mismatch("prepared regular file exceeds its expected bound"));
    }
    // A substituted FIFO must not block before the descriptor can be checked.
    let file = platform::open_read(path, false)?;
    if !same_observation(&before, &platform::file_status(&file)?) {
        return Err(mismatch("prepared file changed while opening"));
    }
    let mut reader = (&file).take(limit.saturating_add(1));
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 32 * 1024];
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(count) => count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        };
        if count == 0 {
            break;
        }
        bytes += count as u64;
        digest.update(&buffer[..count]);
    }
    if bytes != before.metadata().len()
        || bytes > limit
        || !same_observation(&before, &platform::file_status(&file)?)
        || !same_observation(&before, &platform::status(path)?)
    {
        return Err(mismatch("prepared file changed while checking its proof"));
    }
    Ok(Some(Proof {
        identity: identity(&before),
        kind: Kind::Regular,
        bytes,
        sha256: markitai_core::hex(digest.finalize()),
    }))
}
fn record_proof(leases: &MemberLeases, path: &Path) -> Result<Option<Proof>> {
    let Some(directory) = records_directory(leases, false)? else {
        return Ok(None);
    };
    match platform::status(path) {
        Ok(status) => check_record(&status, &platform::followed_status(&directory)?)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    }
    observe_regular_bounded(path, RECEIPT_LIMIT as u64)
}

/// Owns a temporary name only while its exact original object still exists.
struct StagedFile {
    file: Option<tempfile::NamedTempFile>,
    proof: Proof,
}
impl StagedFile {
    /// Documents follow the umask like other outputs; receipts stay private.
    fn create(parent: &Path, prefix: &str, bytes: &[u8], deliverable: bool) -> Result<Self> {
        let mut builder = if deliverable {
            markitai_core::output::deliverable_builder()
        } else {
            tempfile::Builder::new()
        };
        let mut file = builder.prefix(prefix).tempfile_in(parent)?;
        file.write_all(bytes)?;
        let proof = observe_regular_bounded(file.path(), bytes.len() as u64)?
            .ok_or_else(|| mismatch("prepared file disappeared"))?;
        if proof.kind != Kind::Regular
            || proof.bytes != bytes.len() as u64
            || proof.sha256 != markitai_core::hex(Sha256::digest(bytes))
        {
            return Err(mismatch("prepared file differs from its rendered bytes"));
        }
        Ok(Self {
            file: Some(file),
            proof,
        })
    }
    fn file(&self) -> &tempfile::NamedTempFile {
        self.file
            .as_ref()
            .expect("prepared file has not been installed")
    }
    fn verify(&self) -> Result<()> {
        if observe_regular_bounded(self.file().path(), self.proof.bytes)?.as_ref()
            != Some(&self.proof)
        {
            return Err(mismatch("prepared file changed before publication"));
        }
        Ok(())
    }
    fn install(&mut self, target: &Path, replace: bool) -> Result<File> {
        self.verify()?;
        let file = self.file.take().expect("verified prepared file exists");
        let result = if replace {
            platform::persist(file, target)
        } else {
            platform::persist_noclobber(file, target)
        };
        match result {
            Ok(file) => Ok(file),
            Err(error) => {
                // Retain the temporary name so Drop applies our identity check.
                self.file = Some(error.file);
                Err(error.error.into())
            }
        }
    }
}
impl Drop for StagedFile {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            if observe_regular_bounded(file.path(), self.proof.bytes)
                .ok()
                .flatten()
                .as_ref()
                == Some(&self.proof)
            {
                drop(file);
            } else {
                let _ = file.keep();
            }
        }
    }
}
struct StagedMember {
    target: PathBuf,
    expected: Option<Proof>,
    stage: StagedFile,
}

/// Prepared means neither a durable receipt nor a published document. The claim
/// remains owned until commit returns or this object is dropped.
#[must_use = "a prepared document is not a completed conversion"]
pub(crate) struct PreparedDocument {
    claim: Arc<Claim>,
    members: Vec<StagedMember>,
    receipt_target: PathBuf,
    receipt_expected: Option<Proof>,
    receipt: StagedFile,
    bytes: usize,
}
impl PreparedDocument {
    pub(crate) fn prepare(claim: Arc<Claim>, rendered: Vec<RenderedMember>) -> Result<Self> {
        if claim.skip || rendered.is_empty() || rendered.len() > 2 {
            return Err(Error::Invalid(
                "group preparation requires one or two non-skipped members".into(),
            ));
        }
        let bytes = rendered
            .iter()
            .try_fold(0_usize, |sum, member| sum.checked_add(member.bytes.len()))
            .filter(|bytes| *bytes <= MAX_DOCUMENT_BYTES)
            .ok_or_else(|| {
                Error::Invalid("document exceeds the group preparation byte limit".into())
            })?;
        let owner = claim
            .owner
            .as_ref()
            .ok_or_else(|| Error::Invalid("group preparation requires a native owner".into()))?;
        let leases = &claim.leases;
        validate_owner(owner, leases.parent())?;
        let paths = member_paths(leases)?;
        let receipt_target = receipt_path(leases, owner, &paths)?;
        let receipt_expected = record_proof(leases, &receipt_target)?;
        let existing = read_receipt(leases, &receipt_target)?;
        if record_proof(leases, &receipt_target)? != receipt_expected {
            return Err(mismatch("publication receipt changed during preparation"));
        }
        if let Some(receipt) = &existing {
            validate_receipt(receipt, leases.parent(), owner, &paths)?;
        }
        if claim.policy == Policy::RetryOwned {
            verify_members(&paths, existing.as_ref())?;
        }
        let mut receipt = existing.unwrap_or_else(|| Receipt {
            version: 1,
            owner: owner.clone(),
            parent: leases.parent().to_owned(),
            members: crate::sort::map(paths.iter().map(|(name, target)| {
                (
                    name.clone(),
                    Member {
                        target: target.clone(),
                        prior: None,
                        prepared: None,
                    },
                )
            })),
        });
        let mut members = Vec::with_capacity(rendered.len());
        let mut seen = BTreeSet::new();
        for rendered in rendered {
            let target = leases.validate_member(&rendered.path)?;
            if !seen.insert(target.clone()) {
                return Err(Error::Invalid(
                    "prepared document repeats a claimed member".into(),
                ));
            }
            let name = target
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| Error::Invalid("publication member name is not Unicode".into()))?;
            let member = receipt
                .members
                .get_mut(name)
                .ok_or_else(|| Error::Invalid("publication member was not claimed".into()))?;
            let expected = observe(&target)?;
            if claim.policy == Policy::RetryOwned
                && expected
                    .as_ref()
                    .is_some_and(|proof| !authorizes(member, proof))
            {
                return Err(mismatch(
                    "output changed after native ownership was checked",
                ));
            }
            if claim.policy == Policy::NoClobber && expected.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "output member already exists",
                )
                .into());
            }
            let stage = StagedFile::create(leases.parent(), STAGE_PREFIX, &rendered.bytes, true)?;
            let authority = if claim.policy == Policy::Overwrite
                || expected.as_ref().is_some_and(|proof| {
                    member.prior.as_ref().is_some_and(|prior| {
                        prior.proof == *proof && prior.authority == Authority::ExplicitOverwrite
                    })
                }) {
                Authority::ExplicitOverwrite
            } else {
                Authority::NativeOwned
            };
            member.prior = expected.clone().map(|proof| Prior { authority, proof });
            member.prepared = Some(Prepared {
                temporary: stage
                    .file()
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| Error::Invalid("temporary member name is not Unicode".into()))?
                    .to_owned(),
                proof: stage.proof.clone(),
            });
            members.push(StagedMember {
                target,
                expected,
                stage,
            });
        }
        let directory = records_directory(leases, true)?
            .ok_or_else(|| mismatch("publication receipt directory disappeared"))?;
        let mut encoded = Bounded::new(RECEIPT_LIMIT);
        serde_json::to_writer(&mut encoded, &receipt)
            .map_err(|_| mismatch("publication receipt exceeds its size limit"))?;
        let stage = StagedFile::create(&directory, ".markitai-receipt-", &encoded.bytes, false)?;
        check_record(
            &platform::status(stage.file().path())?,
            &platform::followed_status(&directory)?,
        )?;
        Ok(Self {
            claim,
            members,
            receipt_target,
            receipt_expected,
            receipt: stage,
            bytes,
        })
    }
    pub(crate) fn staged_bytes(&self) -> usize {
        self.bytes
    }

    fn validate(&self) -> Result<()> {
        let leases = &self.claim.leases;
        for member in &self.members {
            let path = leases.validate_member(&member.target)?;
            if observe(&path)? != member.expected {
                return Err(mismatch("output changed after group preparation"));
            }
            member.stage.verify()?;
        }
        let directory = records_directory(leases, false)?
            .ok_or_else(|| mismatch("publication receipt directory disappeared"))?;
        if self.receipt_target.parent() != Some(directory.as_path())
            || record_proof(leases, &self.receipt_target)? != self.receipt_expected
        {
            return Err(mismatch(
                "publication receipt changed after group preparation",
            ));
        }
        if let Ok(status) = platform::status(&self.receipt_target) {
            check_record(&status, &platform::followed_status(&directory)?)?;
        }
        self.receipt.verify()
    }
}

/// Bound both outstanding file claims and staged Markdown bytes. This is an
/// internal batch window, not a new public durability or benchmark option.
#[must_use = "dropping a publication group does not publish or acknowledge its documents"]
pub(crate) struct PublicationGroup {
    documents: Vec<PreparedDocument>,
    bytes: usize,
}
impl PublicationGroup {
    pub(crate) fn new() -> Self {
        Self {
            documents: Vec::new(),
            bytes: 0,
        }
    }
    pub(crate) fn can_fit(&self, bytes: usize) -> bool {
        self.documents.len() < MAX_DOCUMENTS
            && bytes <= MAX_DOCUMENT_BYTES.saturating_sub(self.bytes)
    }
    pub(crate) fn push(&mut self, document: PreparedDocument) -> Result<()> {
        if !self.can_fit(document.bytes) {
            return Err(Error::Invalid(
                "publication group is full; commit before preparing another document".into(),
            ));
        }
        let keys = document.claim.keys();
        if self
            .documents
            .iter()
            .any(|old| old.claim.keys().iter().any(|key| keys.contains(key)))
        {
            return Err(Error::Invalid(
                "publication group repeats an output member claim".into(),
            ));
        }
        self.bytes += document.bytes;
        self.documents.push(document);
        Ok(())
    }

    /// A success is returned only after all three phases. On error earlier
    /// renames may be visible; their receipts, which no document rename can
    /// outlive on the media, remain the retry authority.
    pub(crate) fn commit(self) -> Result<Vec<Arc<Claim>>> {
        self.commit_with(|_| Ok(()))
    }

    fn commit_with(
        mut self,
        mut boundary: impl FnMut(Boundary) -> Result<()>,
    ) -> Result<Vec<Arc<Claim>>> {
        if self.documents.is_empty() {
            return Ok(Vec::new());
        }
        for document in &self.documents {
            document.validate()?;
        }
        let mut data = SyncGroup::new();
        let mut stage_parents = BTreeSet::new();
        for document in &self.documents {
            for member in &document.members {
                data.stage(member.stage.file().as_file())?;
            }
            data.stage(document.receipt.file().as_file())?;
            stage_parents.insert(document.claim.parent().to_owned());
            stage_parents.insert(
                document
                    .receipt_target
                    .parent()
                    .expect("receipt has parent")
                    .to_owned(),
            );
        }
        // Order temporary names too; recovery never depends on a lucky
        // background directory write between the data and receipt phases.
        // Ordering suffices: no receipt can reach the media before these.
        for parent in stage_parents {
            data.stage_directory(&parent)?;
        }
        data.commit_ordered()?;
        boundary(Boundary::DataOrdered)?;

        let mut receipts = SyncGroup::new();
        let mut receipt_parents = BTreeSet::new();
        for (index, document) in self.documents.iter_mut().enumerate() {
            document.validate()?;
            let installed = document.receipt.install(
                &document.receipt_target,
                document.receipt_expected.is_some(),
            )?;
            receipts.stage_renamed(&installed)?;
            receipt_parents.insert(
                document
                    .receipt_target
                    .parent()
                    .expect("receipt has parent")
                    .to_owned(),
            );
            boundary(Boundary::ReceiptInstalled(index))?;
        }
        // No document name can reach the media before its installed receipt.
        for parent in receipt_parents {
            receipts.stage_directory(&parent)?;
        }
        receipts.commit_ordered()?;
        boundary(Boundary::ReceiptsOrdered)?;

        let mut outputs = SyncGroup::new();
        let mut output_parents = BTreeSet::new();
        for (document_index, document) in self.documents.iter_mut().enumerate() {
            for (member_index, member) in document.members.iter_mut().enumerate() {
                let target = document.claim.leases.validate_member(&member.target)?;
                if observe(&target)? != member.expected {
                    return Err(mismatch("output changed before group installation"));
                }
                let installed = member.stage.install(&target, member.expected.is_some())?;
                outputs.stage_renamed(&installed)?;
                output_parents.insert(document.claim.parent().to_owned());
                boundary(Boundary::OutputInstalled(document_index, member_index))?;
            }
        }
        // The acknowledgement fence. Each claim's metadata and staged files share
        // its output parent's file system, so this durable flush per volume also
        // persists everything the two ordered phases staged.
        for parent in output_parents {
            outputs.stage_directory(&parent)?;
        }
        outputs.commit()?;
        boundary(Boundary::OutputsDurable)?;
        Ok(self
            .documents
            .into_iter()
            .map(|document| document.claim)
            .collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Boundary {
    DataOrdered,
    ReceiptInstalled(usize),
    ReceiptsOrdered,
    OutputInstalled(usize, usize),
    OutputsDurable,
}

/// Group phases, interruption and kill recovery run on every platform; the
/// substituted-link test is Unix-only.
#[cfg(test)]
mod tests {
    use super::*;

    fn claim(parent: &Path, stem: &str, owner: Option<Owner>, policy: Policy) -> Arc<Claim> {
        let leases = MemberLeases::acquire(
            parent,
            &[format!("{stem}.md"), format!("{stem}.llm.md")],
            false,
        )
        .unwrap();
        let owner = owner.unwrap_or_else(|| Owner {
            generation: uuid::Uuid::new_v4().to_string(),
            mode: "directory".into(),
            input: leases.parent().join("input"),
            output: leases.parent().to_owned(),
            kind: "file".into(),
            key: format!("{stem}.txt"),
        });
        Arc::new(Claim::new(leases, Some(owner), policy, false).unwrap())
    }
    fn rendered(claim: &Claim, name: &str, bytes: &[u8]) -> RenderedMember {
        RenderedMember {
            path: claim.parent().join(name),
            bytes: bytes.to_vec(),
        }
    }
    fn prepared(claim: Arc<Claim>, stem: &str) -> PreparedDocument {
        let rendered = vec![
            rendered(&claim, &format!("{stem}.md"), b"base"),
            rendered(&claim, &format!("{stem}.llm.md"), b"enhanced"),
        ];
        PreparedDocument::prepare(claim, rendered).unwrap()
    }
    fn record(claim: &Claim) -> PathBuf {
        receipt_path(
            &claim.leases,
            claim.owner.as_ref().unwrap(),
            &member_paths(&claim.leases).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn preparation_keeps_formal_names_absent_and_drop_releases_claims() {
        let directory = tempfile::tempdir().unwrap();
        let claim = claim(directory.path(), "a", None, Policy::NoClobber);
        let document = prepared(claim.clone(), "a");
        let stages: Vec<_> = document
            .members
            .iter()
            .map(|member| member.stage.file().path().to_owned())
            .collect();
        assert!(!claim.parent().join("a.md").exists());
        assert!(!record(&claim).exists());
        assert!(matches!(
            MemberLeases::acquire(directory.path(), &["a.md".into()], false),
            Err(Error::Busy)
        ));
        drop(claim);
        drop(document);
        assert!(stages.iter().all(|path| !path.exists()));
        MemberLeases::acquire(directory.path(), &["a.md".into()], false).unwrap();
    }

    #[test]
    fn two_document_two_member_commit_keeps_combined_receipts_and_exact_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let a = claim(directory.path(), "a", None, Policy::NoClobber);
        let b = claim(directory.path(), "b", None, Policy::NoClobber);
        let mut group = PublicationGroup::new();
        group.push(prepared(a.clone(), "a")).unwrap();
        group.push(prepared(b.clone(), "b")).unwrap();
        let mut phases = Vec::new();
        let committed = group
            .commit_with(|phase| {
                phases.push(phase);
                Ok(())
            })
            .unwrap();
        assert_eq!(committed.len(), 2);
        assert_eq!(
            phases,
            [
                Boundary::DataOrdered,
                Boundary::ReceiptInstalled(0),
                Boundary::ReceiptInstalled(1),
                Boundary::ReceiptsOrdered,
                Boundary::OutputInstalled(0, 0),
                Boundary::OutputInstalled(0, 1),
                Boundary::OutputInstalled(1, 0),
                Boundary::OutputInstalled(1, 1),
                Boundary::OutputsDurable
            ]
        );
        for (claim, stem) in [(a, "a"), (b, "b")] {
            verify_owned(&claim.leases, claim.owner.as_ref().unwrap()).unwrap();
            assert_eq!(
                fs::read(claim.parent().join(format!("{stem}.md"))).unwrap(),
                b"base"
            );
            assert_eq!(
                fs::read(claim.parent().join(format!("{stem}.llm.md"))).unwrap(),
                b"enhanced"
            );
            let receipt = read_receipt(&claim.leases, &record(&claim))
                .unwrap()
                .unwrap();
            assert!(
                receipt
                    .members
                    .values()
                    .all(|member| member.prepared.is_some())
            );
        }
    }

    #[test]
    fn two_ordered_phases_precede_one_durable_acknowledgement_fence() {
        use crate::output_claims::sync_group::take_commits;
        let directory = tempfile::tempdir().unwrap();
        let a = claim(directory.path(), "a", None, Policy::NoClobber);
        let b = claim(directory.path(), "b", None, Policy::NoClobber);
        let mut group = PublicationGroup::new();
        group.push(prepared(a, "a")).unwrap();
        group.push(prepared(b, "b")).unwrap();
        take_commits();
        group.commit().unwrap();
        assert_eq!(take_commits(), ["ordered", "ordered", "durable"]);
        // A failure in an ordered phase never reaches the acknowledgement fence.
        let c = claim(directory.path(), "c", None, Policy::NoClobber);
        let mut group = PublicationGroup::new();
        group.push(prepared(c, "c")).unwrap();
        take_commits();
        assert!(
            group
                .commit_with(|phase| if phase == Boundary::ReceiptsOrdered {
                    Err(io::Error::other("stop after receipts").into())
                } else {
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(take_commits(), ["ordered", "ordered"]);
    }

    #[cfg(unix)]
    #[test]
    fn an_ancestor_swapped_for_a_symlink_after_preparation_installs_nothing() {
        let temp = tempfile::tempdir().unwrap();
        let root = platform::canonicalize(temp.path()).unwrap();
        let parent = root.join("tree/out");
        let claim = claim(&parent, "a", None, Policy::NoClobber);
        let receipt = record(&claim);
        let mut group = PublicationGroup::new();
        group.push(prepared(claim.clone(), "a")).unwrap();
        fs::rename(root.join("tree"), root.join("moved")).unwrap();
        std::os::unix::fs::symlink(root.join("moved"), root.join("tree")).unwrap();
        let Err(error) = group.commit() else {
            panic!("a swapped ancestor must fail the group");
        };
        let error = error.to_string();
        assert!(error.contains("symlink policy"), "{error}");
        let moved = root.join("moved/out");
        assert!(!moved.join("a.md").exists());
        assert!(!moved.join("a.llm.md").exists());
        assert!(!moved.join(receipt.strip_prefix(&parent).unwrap()).exists());
    }

    #[test]
    fn every_interrupted_boundary_retains_old_or_provably_owned_new_members() {
        for stop in [
            Boundary::DataOrdered,
            Boundary::ReceiptInstalled(0),
            Boundary::ReceiptsOrdered,
            Boundary::OutputInstalled(0, 0),
            Boundary::OutputInstalled(0, 1),
            Boundary::OutputsDurable,
        ] {
            let directory = tempfile::tempdir().unwrap();
            let initial = claim(directory.path(), "a", None, Policy::NoClobber);
            let owner = initial.owner.clone().unwrap();
            for (name, bytes) in [
                ("a.md", b"old base".as_slice()),
                ("a.llm.md", b"old enhanced".as_slice()),
            ] {
                publish(
                    &initial.leases,
                    Some(&owner),
                    Policy::NoClobber,
                    &initial.parent().join(name),
                    bytes,
                )
                .unwrap();
            }
            let old_receipt = fs::read(record(&initial)).unwrap();
            drop(initial);
            let retry = claim(directory.path(), "a", Some(owner), Policy::RetryOwned);
            let mut group = PublicationGroup::new();
            group.push(prepared(retry.clone(), "a")).unwrap();
            assert!(
                group
                    .commit_with(|phase| if phase == stop {
                        Err(io::Error::other("interrupted at boundary").into())
                    } else {
                        Ok(())
                    })
                    .is_err()
            );
            verify_owned(&retry.leases, retry.owner.as_ref().unwrap()).unwrap();
            if stop == Boundary::DataOrdered {
                assert_eq!(fs::read(record(&retry)).unwrap(), old_receipt);
            }
            for (name, old, new) in [
                ("a.md", b"old base".as_slice(), b"base".as_slice()),
                (
                    "a.llm.md",
                    b"old enhanced".as_slice(),
                    b"enhanced".as_slice(),
                ),
            ] {
                let bytes = fs::read(retry.parent().join(name)).unwrap();
                assert!(bytes == old || bytes == new);
            }
        }
    }

    #[test]
    fn foreign_replacement_after_receipt_fence_is_preserved() {
        let directory = tempfile::tempdir().unwrap();
        let claim = claim(directory.path(), "a", None, Policy::NoClobber);
        let path = claim.parent().join("a.md");
        let mut group = PublicationGroup::new();
        group.push(prepared(claim.clone(), "a")).unwrap();
        assert!(
            group
                .commit_with(|phase| {
                    if phase == Boundary::ReceiptsOrdered {
                        fs::write(&path, b"foreign")?;
                    }
                    Ok(())
                })
                .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), b"foreign");
        assert!(!claim.parent().join("a.llm.md").exists());
        assert!(verify_owned(&claim.leases, claim.owner.as_ref().unwrap()).is_err());
    }

    #[test]
    fn modified_stage_is_not_unlinked_and_no_receipt_is_installed() {
        let directory = tempfile::tempdir().unwrap();
        let claim = claim(directory.path(), "a", None, Policy::NoClobber);
        let document = prepared(claim.clone(), "a");
        let path = document.members[0].stage.file().path().to_owned();
        fs::write(&path, b"foreign replacement").unwrap();
        let mut group = PublicationGroup::new();
        group.push(document).unwrap();
        assert!(group.commit().is_err());
        assert_eq!(fs::read(path).unwrap(), b"foreign replacement");
        assert!(!record(&claim).exists());
    }

    #[test]
    fn group_bounds_and_duplicate_family_never_publish_early() {
        let directory = tempfile::tempdir().unwrap();
        let claim = claim(directory.path(), "a", None, Policy::NoClobber);
        let mut group = PublicationGroup::new();
        group.push(prepared(claim.clone(), "a")).unwrap();
        assert!(!group.can_fit(MAX_DOCUMENT_BYTES));
        assert!(group.push(prepared(claim.clone(), "a")).is_err());
        assert_eq!(group.documents.len(), 1);
        assert!(!claim.parent().join("a.md").exists());
        assert!(!record(&claim).exists());
        drop(group);
    }

    #[test]
    fn partial_receipt_installation_never_starts_document_installation() {
        let directory = tempfile::tempdir().unwrap();
        let a = claim(directory.path(), "a", None, Policy::NoClobber);
        let b = claim(directory.path(), "b", None, Policy::NoClobber);
        let mut group = PublicationGroup::new();
        group.push(prepared(a.clone(), "a")).unwrap();
        group.push(prepared(b.clone(), "b")).unwrap();
        assert!(
            group
                .commit_with(|phase| if phase == Boundary::ReceiptInstalled(0) {
                    Err(io::Error::other("stop between receipt replacements").into())
                } else {
                    Ok(())
                })
                .is_err()
        );
        assert!(record(&a).is_file());
        assert!(!record(&b).exists());
        for claim in [&a, &b] {
            for name in claim.leases.members() {
                assert!(!claim.parent().join(name).exists());
            }
            verify_owned(&claim.leases, claim.owner.as_ref().unwrap()).unwrap();
        }
    }

    #[test]
    fn killed_commit_after_first_rename_reopens_as_owned_retry() {
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};
        struct Child(std::process::Child);
        impl Drop for Child {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let initial = claim(directory.path(), "a", None, Policy::NoClobber);
        let owner = initial.owner.clone().unwrap();
        publish(
            &initial.leases,
            Some(&owner),
            Policy::NoClobber,
            &initial.parent().join("a.md"),
            b"old base",
        )
        .unwrap();
        publish(
            &initial.leases,
            Some(&owner),
            Policy::NoClobber,
            &initial.parent().join("a.llm.md"),
            b"old enhanced",
        )
        .unwrap();
        drop(initial);
        let ready = directory.path().join("child-ready");
        let mut child = Child(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "output_claims::receipts::group::tests::group_crash_child",
                    "--ignored",
                    "--nocapture",
                ])
                .env("MARKITAI_GROUP_TEST_PARENT", directory.path())
                .env(
                    "MARKITAI_GROUP_TEST_OWNER",
                    serde_json::to_string(&owner).unwrap(),
                )
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let until = Instant::now() + Duration::from_secs(20);
        while !ready.exists() {
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "child exited before its controlled boundary"
            );
            assert!(
                Instant::now() < until,
                "child never reached its controlled boundary"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(matches!(
            MemberLeases::acquire(directory.path(), &["a.md".into()], false),
            Err(Error::Busy)
        ));
        child.0.kill().unwrap();
        child.0.wait().unwrap();
        // Windows releases a killed owner's locks shortly after it exits.
        let until = Instant::now() + Duration::from_secs(10);
        while matches!(
            MemberLeases::acquire(directory.path(), &["a.md".into(), "a.llm.md".into()], false),
            Err(Error::Busy)
        ) && Instant::now() < until
        {
            std::thread::sleep(Duration::from_millis(20));
        }
        let retry = claim(directory.path(), "a", Some(owner), Policy::RetryOwned);
        assert_eq!(fs::read(retry.parent().join("a.md")).unwrap(), b"base");
        assert_eq!(
            fs::read(retry.parent().join("a.llm.md")).unwrap(),
            b"old enhanced"
        );
        verify_owned(&retry.leases, retry.owner.as_ref().unwrap()).unwrap();
        let mut group = PublicationGroup::new();
        group.push(prepared(retry.clone(), "a")).unwrap();
        group.commit().unwrap();
        assert_eq!(
            fs::read(retry.parent().join("a.llm.md")).unwrap(),
            b"enhanced"
        );
    }

    #[test]
    #[ignore = "private subprocess helper; parent supplies isolated temporary paths"]
    fn group_crash_child() {
        let Some(parent) = std::env::var_os("MARKITAI_GROUP_TEST_PARENT") else {
            return;
        };
        let parent = PathBuf::from(parent);
        let owner: Owner =
            serde_json::from_str(&std::env::var("MARKITAI_GROUP_TEST_OWNER").unwrap()).unwrap();
        let claim = claim(&parent, "a", Some(owner), Policy::RetryOwned);
        let mut group = PublicationGroup::new();
        group.push(prepared(claim, "a")).unwrap();
        group
            .commit_with(|phase| {
                if phase == Boundary::OutputInstalled(0, 0) {
                    fs::write(
                        parent.join("child-ready"),
                        b"receipt durable; first output renamed",
                    )?;
                    std::thread::sleep(std::time::Duration::from_secs(60));
                    return Err(
                        io::Error::other("parent did not terminate controlled child").into(),
                    );
                }
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn oversized_foreign_receipt_is_preserved_and_rejected_before_staging() {
        let directory = tempfile::tempdir().unwrap();
        let claim = claim(directory.path(), "a", None, Policy::NoClobber);
        records_directory(&claim.leases, true).unwrap();
        let path = record(&claim);
        let file = File::create(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        file.set_len(100 * 1024 * 1024).unwrap();
        let before = platform::file_status(&file).unwrap();
        assert!(
            PreparedDocument::prepare(claim.clone(), vec![rendered(&claim, "a.md", b"new")])
                .is_err()
        );
        assert!(same_observation(&before, &platform::status(&path).unwrap()));
        assert!(!claim.parent().join("a.md").exists());
        assert!(!fs::read_dir(claim.parent()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(STAGE_PREFIX)
        }));
    }
}
