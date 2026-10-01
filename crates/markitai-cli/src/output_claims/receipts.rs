//! A synced prepared receipt survives publication without a second metadata commit.
pub(super) mod group;
use super::{Error, MemberLeases, Owner, Policy, Result, sync_group::SyncGroup};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, Metadata};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

const RECEIPT_LIMIT: usize = 64 * 1024;
const STAGE_PREFIX: &str = ".markitai-stage-";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    device: u64,
    inode: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Regular,
    Symlink,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Proof {
    // None is used only by ordinary publication on platforms without native receipts.
    identity: Option<Identity>,
    kind: Kind,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Authority {
    ExplicitOverwrite,
    NativeOwned,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prior {
    authority: Authority,
    proof: Proof,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Prepared {
    temporary: String,
    proof: Proof,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    target: PathBuf,
    prior: Option<Prior>,
    prepared: Option<Prepared>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    version: u8,
    owner: Owner,
    parent: PathBuf,
    members: BTreeMap<String, Member>,
}

fn mismatch(message: &str) -> Error {
    Error::Ownership(message.into())
}

fn validate_owner(owner: &Owner, parent: &Path) -> Result<()> {
    if !cfg!(unix) {
        return Err(Error::Invalid(
            "native output receipts are not supported on this platform".into(),
        ));
    }
    let valid_mode = match owner.mode.as_str() {
        "directory" => matches!(owner.kind.as_str(), "file" | "url"),
        "url_list" | "single_url" => owner.kind == "url",
        "single_file" => owner.kind == "file",
        _ => false,
    };
    if uuid::Uuid::parse_str(&owner.generation).is_err()
        || !valid_mode
        || owner.key.is_empty()
        || owner.key.contains('\0')
        || !owner.input.is_absolute()
        || !owner.output.is_absolute()
        || [&owner.input, &owner.output].iter().any(|path| {
            path.components()
                .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
        })
        || !parent.starts_with(&owner.output)
    {
        return Err(Error::Invalid("native publication owner is invalid".into()));
    }
    if owner.kind == "file"
        && Path::new(&owner.key)
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(Error::Invalid(
            "native publication file key is invalid".into(),
        ));
    }
    Ok(())
}

fn member_paths(leases: &MemberLeases) -> Result<BTreeMap<String, PathBuf>> {
    if leases.members().is_empty() || leases.members().len() > 2 {
        return Err(Error::Invalid(
            "a document claim needs one or two members".into(),
        ));
    }
    let requested: Vec<_> = leases
        .members()
        .iter()
        .map(|name| leases.parent().join(name))
        .collect();
    let validated = leases.validate_members(requested.iter().map(PathBuf::as_path))?;
    let mut paths = BTreeMap::new();
    for (name, path) in leases.members().iter().zip(validated) {
        if paths.insert(name.clone(), path).is_some() {
            return Err(Error::Invalid("duplicate document claim member".into()));
        }
    }
    Ok(paths)
}

fn receipt_path(
    leases: &MemberLeases,
    owner: &Owner,
    paths: &BTreeMap<String, PathBuf>,
) -> Result<PathBuf> {
    locator_path(leases.parent(), owner, paths)
}

fn locator_path(
    parent: &Path,
    owner: &Owner,
    paths: &BTreeMap<String, PathBuf>,
) -> Result<PathBuf> {
    #[derive(Serialize)]
    struct Locator<'a> {
        owner: &'a Owner,
        parent: &'a Path,
        members: &'a BTreeMap<String, PathBuf>,
    }
    let mut encoded = Bounded::new(RECEIPT_LIMIT);
    serde_json::to_writer(
        &mut encoded,
        &Locator {
            owner,
            parent,
            members: paths,
        },
    )
    .map_err(|_| Error::Invalid("publication owner exceeds the receipt limit".into()))?;
    let name = format!(
        "{}.json",
        markitai_core::hex(Sha256::digest(&encoded.bytes))
    );
    Ok(parent.join(".markitai/ownership/records").join(name))
}

fn validate_proof(proof: &Proof) -> Result<()> {
    if proof.identity.is_none()
        || proof.sha256.len() != 64
        || !proof
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(mismatch(
            "publication receipt contains an invalid file proof",
        ));
    }
    Ok(())
}

fn validate_receipt(
    receipt: &Receipt,
    parent: &Path,
    owner: &Owner,
    paths: &BTreeMap<String, PathBuf>,
) -> Result<()> {
    if receipt.version != 1
        || &receipt.owner != owner
        || receipt.parent != parent
        || receipt.members.len() != paths.len()
    {
        return Err(mismatch(
            "publication receipt does not match this owner and family",
        ));
    }
    for (name, target) in paths {
        let member = receipt
            .members
            .get(name)
            .ok_or_else(|| mismatch("publication receipt member set differs"))?;
        if &member.target != target {
            return Err(mismatch("publication receipt target differs"));
        }
        if let Some(prior) = &member.prior {
            validate_proof(&prior.proof)?;
            if prior.proof.kind == Kind::Symlink && prior.authority != Authority::ExplicitOverwrite
            {
                return Err(mismatch(
                    "publication receipt has unauthorized symbolic-link evidence",
                ));
            }
        }
        if let Some(prepared) = &member.prepared {
            validate_proof(&prepared.proof)?;
            if prepared.proof.kind != Kind::Regular
                || !prepared.temporary.starts_with(STAGE_PREFIX)
                || prepared.temporary.len() <= STAGE_PREFIX.len()
                || Path::new(&prepared.temporary).components().count() != 1
                || !matches!(
                    Path::new(&prepared.temporary).components().next(),
                    Some(Component::Normal(_))
                )
            {
                return Err(mismatch(
                    "publication receipt has invalid staged-file evidence",
                ));
            }
        }
        if member.prior.is_some() && member.prepared.is_none() {
            return Err(mismatch("publication receipt has no prepared write"));
        }
    }
    Ok(())
}

/// Checking ownership has no receipt creation, migration, or cleanup side effects.
pub(crate) fn verify_owned(leases: &MemberLeases, owner: &Owner) -> Result<()> {
    validate_owner(owner, leases.parent())?;
    let paths = member_paths(leases)?;
    let path = receipt_path(leases, owner, &paths)?;
    let receipt = read_receipt(leases, &path)?;
    if let Some(receipt) = &receipt {
        validate_receipt(receipt, leases.parent(), owner, &paths)?;
    }
    verify_members(&paths, receipt.as_ref())
}

/// An identity digest of a validated receipt, not authority derived from a hash.
pub(crate) fn evidence_digest(leases: &MemberLeases, owner: &Owner) -> Result<String> {
    validate_owner(owner, leases.parent())?;
    let paths = member_paths(leases)?;
    let path = receipt_path(leases, owner, &paths)?;
    let receipt = read_receipt(leases, &path)?
        .ok_or_else(|| mismatch("publication has no native receipt"))?;
    validate_receipt(&receipt, leases.parent(), owner, &paths)?;
    verify_members(&paths, Some(&receipt))?;
    let bytes = serde_json::to_vec(&receipt)
        .map_err(|_| mismatch("publication receipt cannot be encoded"))?;
    Ok(markitai_core::hex(Sha256::digest(bytes)))
}

/// Persist a bare-URL adoption before the checkpoint switches to its named key.
/// Keeping the old receipt also preserves recovery if checkpoint replacement fails.
pub(crate) fn adopt_owner(leases: &MemberLeases, previous: &Owner, next: &Owner) -> Result<()> {
    let mut comparable = next.clone();
    comparable.key = previous.key.clone();
    if previous.kind != "url"
        || &comparable != previous
        || !markitai_core::is_url(&previous.key)
        || previous.key.chars().any(char::is_whitespace)
        || !next
            .key
            .strip_prefix(&previous.key)
            .and_then(|suffix| suffix.strip_prefix(' '))
            .is_some_and(|name| !name.is_empty() && !name.trim().is_empty())
    {
        return Err(Error::Invalid(
            "only a same-scope bare URL can adopt a named publication owner".into(),
        ));
    }
    validate_owner(next, leases.parent())?;
    verify_owned(leases, previous)?;
    let paths = member_paths(leases)?;
    let previous_path = receipt_path(leases, previous, &paths)?;
    let next_path = receipt_path(leases, next, &paths)?;
    let previous_receipt = read_receipt(leases, &previous_path)?;
    let next_receipt = read_receipt(leases, &next_path)?;
    if let Some(receipt) = &next_receipt {
        validate_receipt(receipt, leases.parent(), next, &paths)?;
    }
    let Some(mut receipt) = previous_receipt else {
        verify_members(&paths, None)?;
        if next_receipt.is_some() {
            return Err(mismatch(
                "named owner has evidence without a matching prior receipt",
            ));
        }
        return Ok(());
    };
    validate_receipt(&receipt, leases.parent(), previous, &paths)?;
    verify_members(&paths, Some(&receipt))?;
    if let Some(next_receipt) = next_receipt {
        if next_receipt.members != receipt.members {
            return Err(mismatch("named owner has different publication evidence"));
        }
        // An earlier attempt may have installed the receipt but failed its sync.
        // One durable fence covers the receipt and its directory entry.
        let mut durable = SyncGroup::new();
        durable.stage(&File::open(&next_path)?)?;
        durable.stage(&File::open(
            next_path.parent().expect("receipt has a parent"),
        )?)?;
        durable.commit()?;
        return Ok(());
    }
    receipt.owner = next.clone();
    // The merged checkpoint saved next may live on another volume, so this
    // copy is durable before returning rather than merely ordered.
    write_receipt(leases, &next_path, &receipt, SyncGroup::new(), true)
}

fn verify_members(paths: &BTreeMap<String, PathBuf>, receipt: Option<&Receipt>) -> Result<()> {
    for (name, path) in paths {
        if let Some(current) = observe(path)? {
            let member = receipt.and_then(|receipt| receipt.members.get(name));
            if !member.is_some_and(|member| authorizes(member, &current)) {
                return Err(mismatch(
                    "existing output has no matching native publication evidence",
                ));
            }
        }
    }
    Ok(())
}

fn authorizes(member: &Member, current: &Proof) -> bool {
    member
        .prepared
        .as_ref()
        .is_some_and(|prepared| &prepared.proof == current)
        || member
            .prior
            .as_ref()
            .is_some_and(|prior| &prior.proof == current)
}

/// Recover the reservation shape only; this never establishes write authority.
pub(crate) fn reservation_members(
    parent: &Path,
    owner: &Owner,
    output: &Path,
) -> Result<Option<Vec<String>>> {
    let parent = match fs::canonicalize(parent) {
        Ok(parent) => parent,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    validate_owner(owner, &parent)?;
    let Some(output_parent) = output.parent() else {
        return Ok(None);
    };
    if crate::report_store::resolve_path(output_parent)
        .map_err(|_| Error::Invalid("completed output parent cannot be resolved".into()))?
        != parent
    {
        return Err(mismatch("completed output is outside its claimed parent"));
    }
    let Some(name) = output.file_name().and_then(|name| name.to_str()) else {
        return Ok(None);
    };
    let Some(base_stem) = name.strip_suffix(".md").filter(|stem| !stem.is_empty()) else {
        return Ok(None);
    };
    let mut stems = vec![base_stem];
    if let Some(enhanced_stem) = base_stem
        .strip_suffix(".llm")
        .filter(|stem| !stem.is_empty())
    {
        stems.push(enhanced_stem);
    }
    let mut found = None;
    for stem in stems {
        let paths = crate::sort::map([format!("{stem}.md"), format!("{stem}.llm.md")].map(
            |member| {
                let target = parent.join(&member);
                (member, target)
            },
        ));
        let location = locator_path(&parent, owner, &paths)?;
        let Some(receipt) = read_receipt_at(&parent, &location)? else {
            continue;
        };
        validate_receipt(&receipt, &parent, owner, &paths)?;
        if found.is_some() {
            return Ok(None);
        }
        found = Some(paths.into_keys().collect());
    }
    Ok(found)
}

/// The prepublication receipt itself remains the durable publication evidence.
pub(crate) fn publish(
    leases: &MemberLeases,
    owner: Option<&Owner>,
    policy: Policy,
    path: &Path,
    bytes: &[u8],
) -> Result<()> {
    prepare(leases, owner, policy, path, bytes)?.finish()
}

struct Pending<'a> {
    leases: &'a MemberLeases,
    target: PathBuf,
    stage: Option<tempfile::NamedTempFile>,
    staged: Proof,
    expected: Option<Proof>,
}

impl Pending<'_> {
    fn install(mut self) -> Result<File> {
        let target = self.leases.validate_member(&self.target)?;
        if observe(&target)? != self.expected {
            return Err(mismatch("output changed after publication was prepared"));
        }
        let stage = self.stage.as_ref().expect("pending stage is present");
        if observe(stage.path())?.as_ref() != Some(&self.staged) {
            return Err(mismatch(
                "staged document changed after publication was prepared",
            ));
        }
        let stage = self.stage.take().expect("pending stage is present");
        let result = if self.expected.is_some() {
            stage.persist(&target)
        } else {
            stage.persist_noclobber(&target)
        };
        result.map_err(|error| Error::Io(error.error))
    }

    /// The acknowledgement fence: a durable flush of the output parent's volume.
    /// Claim metadata, receipts and the staged bytes share that file system,
    /// so it also persists everything the earlier ordered fences staged.
    fn finish(self) -> Result<()> {
        let parent = self.leases.parent().to_owned();
        let _published = self.install()?;
        let mut durable = SyncGroup::new();
        durable.stage(&File::open(&parent)?)?;
        durable.commit()?;
        Ok(())
    }
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        // Never unlink an externally replaced/modified object at our temporary name.
        if let Some(stage) = self.stage.take() {
            if observe(stage.path()).ok().flatten().as_ref() == Some(&self.staged) {
                drop(stage);
            } else {
                let _ = stage.keep();
            }
        }
    }
}

fn prepare<'a>(
    leases: &'a MemberLeases,
    owner: Option<&Owner>,
    policy: Policy,
    path: &Path,
    bytes: &[u8],
) -> Result<Pending<'a>> {
    if policy == Policy::RetryOwned && owner.is_none() {
        return Err(Error::Invalid(
            "native retry requires a publication owner".into(),
        ));
    }
    let target = leases.validate_member(path)?;
    let paths = member_paths(leases)?;
    let mut existing = None;
    let mut record_path = None;
    if let Some(owner) = owner {
        validate_owner(owner, leases.parent())?;
        let path = receipt_path(leases, owner, &paths)?;
        existing = read_receipt(leases, &path)?;
        if let Some(receipt) = &existing {
            validate_receipt(receipt, leases.parent(), owner, &paths)?;
        }
        if policy == Policy::RetryOwned {
            verify_members(&paths, existing.as_ref())?;
        }
        record_path = Some(path);
    }
    let expected = observe(&target)?;
    if policy == Policy::RetryOwned
        && expected.as_ref().is_some_and(|proof| {
            !existing
                .as_ref()
                .and_then(|receipt| {
                    target
                        .file_name()
                        .and_then(|name| name.to_str())
                        .and_then(|name| receipt.members.get(name))
                })
                .is_some_and(|member| authorizes(member, proof))
        })
    {
        return Err(mismatch(
            "output changed after native ownership was checked",
        ));
    }
    if expected.is_some() && policy == Policy::NoClobber {
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "output member already exists",
        )));
    }
    let mut stage = markitai_core::output::deliverable_builder()
        .prefix(STAGE_PREFIX)
        .tempfile_in(leases.parent())?;
    stage.write_all(bytes)?;
    // The staged bytes must reach the media before the rename that publishes
    // them; durability waits for the acknowledgement fence in finish().
    let mut ordered = SyncGroup::new();
    ordered.stage(stage.as_file())?;
    let staged = observe(stage.path())?.ok_or_else(|| mismatch("staged document disappeared"))?;
    if staged.kind != Kind::Regular
        || staged.bytes != bytes.len() as u64
        || staged.sha256 != markitai_core::hex(Sha256::digest(bytes))
    {
        return Err(mismatch(
            "staged document does not match the rendered bytes",
        ));
    }
    let pending = Pending {
        leases,
        target,
        stage: Some(stage),
        staged,
        expected,
    };
    if let (Some(owner), Some(record_path)) = (owner, record_path) {
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
        let name = pending
            .target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::Invalid("publication member name is not Unicode".into()))?;
        let member = receipt
            .members
            .get_mut(name)
            .ok_or_else(|| Error::Invalid("publication member was not claimed".into()))?;
        let authority = if policy == Policy::Overwrite
            || pending.expected.as_ref().is_some_and(|proof| {
                member.prior.as_ref().is_some_and(|prior| {
                    prior.proof == *proof && prior.authority == Authority::ExplicitOverwrite
                })
            }) {
            Authority::ExplicitOverwrite
        } else {
            Authority::NativeOwned
        };
        member.prior = pending
            .expected
            .clone()
            .map(|proof| Prior { authority, proof });
        member.prepared = Some(Prepared {
            temporary: pending
                .stage
                .as_ref()
                .expect("pending stage is present")
                .path()
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| Error::Invalid("temporary member name is not Unicode".into()))?
                .to_owned(),
            proof: pending.staged.clone(),
        });
        // The receipt's installation and directory entry are ordered before
        // the document rename; finish() makes all of them durable.
        write_receipt(leases, &record_path, &receipt, ordered, false)?;
    } else {
        ordered.commit_ordered()?;
    }
    Ok(pending)
}

fn identity(metadata: &Metadata) -> Option<Identity> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Some(Identity {
            device: metadata.dev(),
            inode: metadata.ino(),
        })
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        None
    }
}

fn same_observation(before: &Metadata, after: &Metadata) -> bool {
    let common = identity(before) == identity(after)
        && before.len() == after.len()
        && before.file_type() == after.file_type()
        && before.modified().ok() == after.modified().ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        common && before.ctime() == after.ctime() && before.ctime_nsec() == after.ctime_nsec()
    }
    #[cfg(not(unix))]
    {
        common
    }
}

fn observe(path: &Path) -> Result<Option<Proof>> {
    let before = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let (kind, bytes, digest) = if before.file_type().is_symlink() {
        let target = fs::read_link(path)?;
        #[cfg(unix)]
        let encoded = {
            use std::os::unix::ffi::OsStrExt;
            target.as_os_str().as_bytes().to_vec()
        };
        #[cfg(not(unix))]
        let encoded = target.as_os_str().as_encoded_bytes().to_vec();
        (
            Kind::Symlink,
            encoded.len() as u64,
            Sha256::digest(encoded).to_vec(),
        )
    } else if before.is_file() {
        let file = File::open(path)?;
        if !same_observation(&before, &file.metadata()?) {
            return Err(mismatch("output changed while checking its identity"));
        }
        let mut digest = Sha256::new();
        let mut reader = (&file).take(before.len().saturating_add(1));
        let mut buffer = [0_u8; 32 * 1024];
        let mut bytes = 0_u64;
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            bytes += count as u64;
            digest.update(&buffer[..count]);
        }
        if bytes != before.len() || !same_observation(&before, &file.metadata()?) {
            return Err(mismatch("output changed while checking its contents"));
        }
        (Kind::Regular, bytes, digest.finalize().to_vec())
    } else {
        return Err(mismatch(
            "output member is not a regular file or symbolic link",
        ));
    };
    let after = fs::symlink_metadata(path)?;
    if !same_observation(&before, &after) {
        return Err(mismatch(
            "output changed while checking publication evidence",
        ));
    }
    Ok(Some(Proof {
        identity: identity(&before),
        kind,
        bytes,
        sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
    }))
}

fn records_directory(leases: &MemberLeases, create: bool) -> Result<Option<PathBuf>> {
    records_directory_at(leases.parent(), create)
}

fn records_directory_at(parent: &Path, create: bool) -> Result<Option<PathBuf>> {
    let parent_meta = fs::metadata(parent)?;
    let mut path = parent.to_owned();
    for (index, part) in [".markitai", "ownership", "records"].iter().enumerate() {
        path.push(part);
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if !create {
                    return Ok(None);
                }
                #[cfg_attr(not(unix), allow(unused_mut))] // Only Unix sets a mode.
                let mut builder = fs::DirBuilder::new();
                #[cfg(unix)]
                {
                    use std::os::unix::fs::DirBuilderExt;
                    builder.mode(0o700);
                }
                match builder.create(&path) {
                    Ok(()) => sync_directory(path.parent().expect("metadata has parent"))?,
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
                    Err(error) => return Err(error.into()),
                }
                fs::symlink_metadata(&path)?
            }
            Err(error) => return Err(error.into()),
        };
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(mismatch(
                "publication metadata must use regular directories",
            ));
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if metadata.dev() != parent_meta.dev() || (index > 0 && metadata.mode() & 0o077 != 0) {
                return Err(mismatch(
                    "publication metadata directory is not private on this filesystem",
                ));
            }
        }
        #[cfg(not(unix))]
        let _ = (&parent_meta, index);
    }
    Ok(Some(path))
}

fn check_record(metadata: &Metadata, directory: &Metadata) -> Result<()> {
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err(mismatch("publication receipt is not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if metadata.dev() != directory.dev()
            || metadata.uid() != directory.uid()
            || metadata.mode() & 0o077 != 0
            || metadata.nlink() != 1
        {
            return Err(mismatch(
                "publication receipt is not private on this filesystem",
            ));
        }
    }
    #[cfg(not(unix))]
    let _ = directory;
    if metadata.len() > RECEIPT_LIMIT as u64 {
        return Err(mismatch("publication receipt exceeds its size limit"));
    }
    Ok(())
}

fn read_receipt(leases: &MemberLeases, path: &Path) -> Result<Option<Receipt>> {
    read_receipt_at(leases.parent(), path)
}

fn read_receipt_at(parent: &Path, path: &Path) -> Result<Option<Receipt>> {
    let Some(directory) = records_directory_at(parent, false)? else {
        return Ok(None);
    };
    let before = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    check_record(&before, &fs::metadata(directory)?)?;
    let file = File::open(path)?;
    if !same_observation(&before, &file.metadata()?) {
        return Err(mismatch("publication receipt changed while opening"));
    }
    let mut bytes = Vec::new();
    (&file)
        .take((RECEIPT_LIMIT + 1) as u64)
        .read_to_end(&mut bytes)?;
    if bytes.len() > RECEIPT_LIMIT
        || !same_observation(&before, &file.metadata()?)
        || !same_observation(&before, &fs::symlink_metadata(path)?)
    {
        return Err(mismatch("publication receipt changed while reading"));
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|_| mismatch("publication receipt is malformed"))
}

/// Install a receipt by atomic replacement. Its bytes, together with anything
/// already staged in `data` (the prepared document), are ordered before the
/// receipt name; the name is then ordered before whatever the caller does next,
/// or made durable when `durable` is set.
fn write_receipt(
    leases: &MemberLeases,
    path: &Path,
    receipt: &Receipt,
    mut data: SyncGroup,
    durable: bool,
) -> Result<()> {
    let mut bytes = Bounded::new(RECEIPT_LIMIT);
    serde_json::to_writer(&mut bytes, receipt)
        .map_err(|_| mismatch("publication receipt cannot be encoded within its size limit"))?;
    let directory = records_directory(leases, true)?
        .ok_or_else(|| mismatch("publication receipt directory disappeared"))?;
    match fs::symlink_metadata(path) {
        Ok(metadata) => check_record(&metadata, &fs::metadata(&directory)?)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    let mut stage = tempfile::NamedTempFile::new_in(&directory)?;
    stage.write_all(&bytes.bytes)?;
    data.stage(stage.as_file())?;
    data.commit_ordered()?;
    stage
        .persist(path)
        .map_err(|error| Error::Io(error.error))?;
    let mut name = SyncGroup::new();
    name.stage(&File::open(&directory)?)?;
    if durable {
        name.commit()?;
    } else {
        name.commit_ordered()?;
    }
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

struct Bounded {
    bytes: Vec<u8>,
    limit: usize,
}

impl Bounded {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
        }
    }
}

impl Write for Bounded {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("publication receipt size limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::{MetadataExt, PermissionsExt, symlink};

    fn setup() -> (tempfile::TempDir, MemberLeases, Owner) {
        let directory = tempfile::tempdir().unwrap();
        let leases = MemberLeases::acquire(
            directory.path(),
            &["note.md".into(), "note.llm.md".into()],
            false,
        )
        .unwrap();
        let owner = Owner {
            generation: uuid::Uuid::new_v4().to_string(),
            mode: "directory".into(),
            input: leases.parent().join("input"),
            output: leases.parent().to_owned(),
            kind: "file".into(),
            key: "note.txt".into(),
        };
        (directory, leases, owner)
    }

    fn record(leases: &MemberLeases, owner: &Owner) -> PathBuf {
        receipt_path(leases, owner, &member_paths(leases).unwrap()).unwrap()
    }

    #[test]
    fn publication_orders_each_phase_and_durably_fences_only_the_acknowledgement() {
        use crate::output_claims::sync_group::take_commits;
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        take_commits();
        // Without an owner: staged bytes ordered before the rename, then one
        // durable fence after it, before success is returned.
        publish(&leases, None, Policy::NoClobber, &path, b"plain").unwrap();
        assert_eq!(take_commits(), ["ordered", "durable"]);
        fs::remove_file(&path).unwrap();
        // With a receipt: bytes, then the receipt name, then the acknowledgement.
        publish(&leases, Some(&owner), Policy::NoClobber, &path, b"owned").unwrap();
        assert_eq!(take_commits(), ["ordered", "ordered", "durable"]);
        // A failed preparation never reaches the acknowledgement fence.
        assert!(publish(&leases, Some(&owner), Policy::NoClobber, &path, b"again").is_err());
        assert!(!take_commits().contains(&"durable"));
        // A copied adoption receipt is durable on its own: the checkpoint that
        // depends on it may be saved on another volume.
        let (previous, next) = url_owners(&owner);
        let enhanced = leases.parent().join("note.llm.md");
        publish(&leases, Some(&previous), Policy::NoClobber, &enhanced, b"e").unwrap();
        fs::remove_file(&path).unwrap();
        take_commits();
        adopt_owner(&leases, &previous, &next).unwrap();
        assert_eq!(take_commits(), ["ordered", "durable"]);
        adopt_owner(&leases, &previous, &next).unwrap();
        assert_eq!(take_commits(), ["durable"]);
    }

    #[test]
    fn an_ancestor_swapped_for_a_symlink_after_preparation_blocks_the_rename() {
        let temp = tempfile::tempdir().unwrap();
        let root = fs::canonicalize(temp.path()).unwrap();
        let parent = root.join("tree/out");
        let leases =
            MemberLeases::acquire(&parent, &["note.md".into(), "note.llm.md".into()], false)
                .unwrap();
        let path = parent.join("note.md");
        let pending = prepare(&leases, None, Policy::NoClobber, &path, b"generated").unwrap();
        // The same physical directory, now reached through a symbolic link.
        fs::rename(root.join("tree"), root.join("moved")).unwrap();
        symlink(root.join("moved"), root.join("tree")).unwrap();
        let error = pending.finish().unwrap_err().to_string();
        assert!(error.contains("symlink policy"), "{error}");
        assert!(!root.join("moved/out/note.md").exists());
        assert!(!fs::read_dir(root.join("moved/out")).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(STAGE_PREFIX)
        }));
    }

    #[test]
    fn ordinary_publication_does_not_create_a_receipt_or_clobber_existing_bytes() {
        let (_dir, leases, _owner) = setup();
        let path = leases.parent().join("note.md");
        publish(&leases, None, Policy::NoClobber, &path, b"first").unwrap();
        assert!(!leases.parent().join(".markitai/ownership/records").exists());
        assert!(publish(&leases, None, Policy::NoClobber, &path, b"second").is_err());
        assert_eq!(fs::read(path).unwrap(), b"first");
    }

    #[test]
    fn rename_is_proven_by_the_prepared_receipt_without_a_finalize_write() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        let pending = prepare(
            &leases,
            Some(&owner),
            Policy::NoClobber,
            &path,
            b"published",
        )
        .unwrap();
        let receipt_path = record(&leases, &owner);
        let before = fs::read(&receipt_path).unwrap();
        assert!(!path.exists());
        let staged_inode = pending.staged.identity.as_ref().unwrap().inode;
        pending.install().unwrap(); // Deliberately stop before the final directory sync.
        assert_eq!(fs::metadata(&path).unwrap().ino(), staged_inode);
        assert_eq!(fs::read(receipt_path).unwrap(), before);
        verify_owned(&leases, &owner).unwrap();
        publish(&leases, Some(&owner), Policy::RetryOwned, &path, b"retry").unwrap();
        assert_eq!(fs::read(path).unwrap(), b"retry");
    }

    #[test]
    fn interrupted_first_preparation_can_recreate_an_absent_member() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        let pending = prepare(
            &leases,
            Some(&owner),
            Policy::NoClobber,
            &path,
            b"never installed",
        )
        .unwrap();
        let temporary = pending.stage.as_ref().unwrap().path().to_owned();
        drop(pending);
        assert!(!temporary.exists());
        assert!(!path.exists());
        verify_owned(&leases, &owner).unwrap();
        publish(
            &leases,
            Some(&owner),
            Policy::RetryOwned,
            &path,
            b"restored",
        )
        .unwrap();
        assert_eq!(fs::read(path).unwrap(), b"restored");
    }

    #[test]
    fn interrupted_replacement_retains_prior_ownership_and_the_other_member() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        let enhanced = leases.parent().join("note.llm.md");
        publish(&leases, Some(&owner), Policy::NoClobber, &path, b"base").unwrap();
        publish(
            &leases,
            Some(&owner),
            Policy::NoClobber,
            &enhanced,
            b"enhanced",
        )
        .unwrap();
        drop(
            prepare(
                &leases,
                Some(&owner),
                Policy::RetryOwned,
                &path,
                b"interrupted",
            )
            .unwrap(),
        );
        verify_owned(&leases, &owner).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"base");
        publish(
            &leases,
            Some(&owner),
            Policy::RetryOwned,
            &path,
            b"new base",
        )
        .unwrap();
        verify_owned(&leases, &owner).unwrap();
        assert_eq!(fs::read(enhanced).unwrap(), b"enhanced");
    }

    #[test]
    fn replacement_with_identical_bytes_does_not_inherit_native_ownership() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        publish(
            &leases,
            Some(&owner),
            Policy::NoClobber,
            &path,
            b"identical",
        )
        .unwrap();
        let mut replacement = tempfile::NamedTempFile::new_in(leases.parent()).unwrap();
        replacement.write_all(b"identical").unwrap();
        replacement.persist(&path).unwrap();
        let replacement_inode = fs::metadata(&path).unwrap().ino();
        assert!(verify_owned(&leases, &owner).is_err());
        assert!(
            publish(
                &leases,
                Some(&owner),
                Policy::RetryOwned,
                &path,
                b"do not write"
            )
            .is_err()
        );
        assert_eq!(fs::metadata(&path).unwrap().ino(), replacement_inode);
        assert_eq!(fs::read(path).unwrap(), b"identical");
    }

    #[test]
    fn in_place_edit_is_preserved_and_missing_owned_output_can_be_recreated() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        publish(&leases, Some(&owner), Policy::NoClobber, &path, b"base").unwrap();
        let inode = fs::metadata(&path).unwrap().ino();
        fs::write(&path, b"edit").unwrap();
        assert_eq!(fs::metadata(&path).unwrap().ino(), inode);
        assert!(verify_owned(&leases, &owner).is_err());
        assert!(publish(&leases, Some(&owner), Policy::RetryOwned, &path, b"replace").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"edit");
        fs::remove_file(&path).unwrap();
        verify_owned(&leases, &owner).unwrap();
        publish(
            &leases,
            Some(&owner),
            Policy::RetryOwned,
            &path,
            b"recreated",
        )
        .unwrap();
        assert_eq!(fs::read(path).unwrap(), b"recreated");
    }

    #[test]
    fn an_owned_base_does_not_authorize_an_unrelated_enhanced_sibling() {
        let (_dir, leases, owner) = setup();
        let base = leases.parent().join("note.md");
        let sibling = leases.parent().join("note.llm.md");
        publish(&leases, Some(&owner), Policy::NoClobber, &base, b"base").unwrap();
        fs::write(&sibling, b"unrelated").unwrap();
        assert!(verify_owned(&leases, &owner).is_err());
        assert!(
            publish(
                &leases,
                Some(&owner),
                Policy::RetryOwned,
                &base,
                b"new base"
            )
            .is_err()
        );
        assert_eq!(fs::read(base).unwrap(), b"base");
        assert_eq!(fs::read(sibling).unwrap(), b"unrelated");
    }

    #[test]
    fn fresh_explicit_authority_survives_pre_rename_interruption() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        fs::write(&path, b"authorized old document").unwrap();
        drop(
            prepare(
                &leases,
                Some(&owner),
                Policy::Overwrite,
                &path,
                b"interrupted",
            )
            .unwrap(),
        );
        verify_owned(&leases, &owner).unwrap();
        publish(
            &leases,
            Some(&owner),
            Policy::RetryOwned,
            &path,
            b"replacement",
        )
        .unwrap();
        assert_eq!(fs::read(path).unwrap(), b"replacement");
    }

    #[test]
    fn changes_after_preparation_are_preserved_even_for_explicit_overwrite() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        fs::write(&path, b"old").unwrap();
        let pending = prepare(
            &leases,
            Some(&owner),
            Policy::Overwrite,
            &path,
            b"generated",
        )
        .unwrap();
        fs::write(&path, b"new user edit").unwrap();
        assert!(pending.finish().is_err());
        assert_eq!(fs::read(path).unwrap(), b"new user edit");
        assert!(verify_owned(&leases, &owner).is_err());
    }

    #[test]
    fn an_unrelated_replacement_at_the_stage_name_is_not_published_or_removed() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        let pending = prepare(
            &leases,
            Some(&owner),
            Policy::NoClobber,
            &path,
            b"generated",
        )
        .unwrap();
        let temporary = pending.stage.as_ref().unwrap().path().to_owned();
        let mut replacement = tempfile::NamedTempFile::new_in(leases.parent()).unwrap();
        replacement.write_all(b"unrelated temporary file").unwrap();
        replacement.persist(&temporary).unwrap();
        assert!(pending.finish().is_err());
        assert!(!path.exists());
        assert_eq!(fs::read(temporary).unwrap(), b"unrelated temporary file");
    }

    #[test]
    fn leaf_symlink_replacement_uses_link_identity_without_modifying_the_referent() {
        let (dir, original_leases, owner) = setup();
        drop(original_leases);
        let leases =
            MemberLeases::acquire(dir.path(), &["note.md".into(), "note.llm.md".into()], true)
                .unwrap();
        let target = leases.parent().join("private.txt");
        fs::write(&target, b"private referent").unwrap();
        let path = leases.parent().join("note.md");
        symlink(&target, &path).unwrap();
        assert!(verify_owned(&leases, &owner).is_err());
        drop(
            prepare(
                &leases,
                Some(&owner),
                Policy::Overwrite,
                &path,
                b"first attempt",
            )
            .unwrap(),
        );
        verify_owned(&leases, &owner).unwrap();
        publish(
            &leases,
            Some(&owner),
            Policy::RetryOwned,
            &path,
            b"document",
        )
        .unwrap();
        assert!(
            !fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(path).unwrap(), b"document");
        assert_eq!(fs::read(target).unwrap(), b"private referent");
    }

    #[test]
    fn equivalent_symlink_replacement_has_a_different_identity() {
        let (dir, original_leases, owner) = setup();
        drop(original_leases);
        let leases =
            MemberLeases::acquire(dir.path(), &["note.md".into(), "note.llm.md".into()], true)
                .unwrap();
        let path = leases.parent().join("note.md");
        symlink("missing-referent", &path).unwrap();
        drop(prepare(&leases, Some(&owner), Policy::Overwrite, &path, b"attempt").unwrap());
        let replacement = leases.parent().join("other-link");
        symlink("missing-referent", &replacement).unwrap();
        fs::rename(replacement, &path).unwrap();
        assert!(verify_owned(&leases, &owner).is_err());
        assert!(
            publish(
                &leases,
                Some(&owner),
                Policy::RetryOwned,
                &path,
                b"replacement"
            )
            .is_err()
        );
        assert_eq!(fs::read_link(path).unwrap(), Path::new("missing-referent"));
    }

    #[test]
    fn owner_locator_keeps_generation_kind_and_raw_named_url_key_distinct() {
        let (_dir, leases, owner) = setup();
        let mut other = owner.clone();
        let first = record(&leases, &owner);
        other.generation = uuid::Uuid::new_v4().to_string();
        assert_ne!(record(&leases, &other), first);
        other = owner.clone();
        other.kind = "url".into();
        other.key = "https://example.test/?secret=abc raw name".into();
        let named = record(&leases, &other);
        other.key = "https://example.test/?secret=abc".into();
        assert_ne!(record(&leases, &other), named);
        let path = leases.parent().join("note.md");
        publish(&leases, Some(&owner), Policy::NoClobber, &path, b"owned").unwrap();
        assert!(verify_owned(&leases, &other).is_err());
        assert!(
            !named
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains("secret")
        );
    }

    #[test]
    fn receipt_body_is_checked_independently_of_its_locator() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        publish(&leases, Some(&owner), Policy::NoClobber, &path, b"owned").unwrap();
        let location = record(&leases, &owner);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&location).unwrap()).unwrap();
        value["owner"]["key"] = serde_json::json!("https://example.test/?private=credential");
        fs::write(&location, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = verify_owned(&leases, &owner).unwrap_err().to_string();
        assert!(!error.contains("credential"));
        assert!(!error.contains("example.test"));
        assert!(
            publish(
                &leases,
                Some(&owner),
                Policy::RetryOwned,
                &path,
                b"replacement"
            )
            .is_err()
        );
        assert_eq!(fs::read(path).unwrap(), b"owned");
    }

    #[test]
    fn malformed_and_oversized_receipts_preserve_existing_output() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        publish(&leases, Some(&owner), Policy::NoClobber, &path, b"owned").unwrap();
        let location = record(&leases, &owner);
        for bytes in [
            b"{\"private=credential".to_vec(),
            vec![b' '; RECEIPT_LIMIT + 1],
        ] {
            fs::write(&location, bytes).unwrap();
            let error = verify_owned(&leases, &owner).unwrap_err().to_string();
            assert!(!error.contains("credential"));
            assert!(
                publish(
                    &leases,
                    Some(&owner),
                    Policy::RetryOwned,
                    &path,
                    b"replacement"
                )
                .is_err()
            );
            assert_eq!(fs::read(&path).unwrap(), b"owned");
        }
    }

    #[test]
    fn foreign_target_and_oversized_owner_fail_before_publication() {
        let (_dir, leases, mut owner) = setup();
        let unknown = leases.parent().join("unclaimed.md");
        assert!(publish(&leases, Some(&owner), Policy::Overwrite, &unknown, b"no").is_err());
        assert!(!unknown.exists());
        owner.key = "x".repeat(RECEIPT_LIMIT);
        let path = leases.parent().join("note.md");
        assert!(publish(&leases, Some(&owner), Policy::NoClobber, &path, b"no").is_err());
        assert!(!path.exists());
        assert!(!leases.parent().join(".markitai/ownership/records").exists());
    }

    #[test]
    fn receipt_symlinks_and_public_permissions_are_not_accepted_as_evidence() {
        let (_dir, leases, owner) = setup();
        let path = leases.parent().join("note.md");
        publish(&leases, Some(&owner), Policy::NoClobber, &path, b"owned").unwrap();
        let location = record(&leases, &owner);
        fs::set_permissions(&location, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(verify_owned(&leases, &owner).is_err());
        fs::set_permissions(&location, fs::Permissions::from_mode(0o600)).unwrap();
        let elsewhere = leases.parent().join("elsewhere.json");
        fs::rename(&location, &elsewhere).unwrap();
        symlink(&elsewhere, &location).unwrap();
        assert!(verify_owned(&leases, &owner).is_err());
        assert!(publish(&leases, Some(&owner), Policy::RetryOwned, &path, b"no").is_err());
        assert_eq!(fs::read(path).unwrap(), b"owned");
    }

    #[test]
    fn retry_without_an_owner_fails_even_when_the_target_is_absent() {
        let (_dir, leases, _owner) = setup();
        let path = leases.parent().join("note.md");
        assert!(publish(&leases, None, Policy::RetryOwned, &path, b"no").is_err());
        assert!(!path.exists());
    }

    #[test]
    fn completed_reservation_distinguishes_a_literal_llm_stem_without_taking_locks() {
        let (dir, leases, owner) = setup();
        let output = leases.parent().join("note.llm.md");
        assert!(
            reservation_members(leases.parent(), &owner, &output)
                .unwrap()
                .is_none()
        );
        publish(
            &leases,
            Some(&owner),
            Policy::NoClobber,
            &output,
            b"enhanced",
        )
        .unwrap();
        assert_eq!(
            reservation_members(leases.parent(), &owner, &output).unwrap(),
            Some(vec!["note.llm.md".into(), "note.md".into()])
        );
        fs::remove_file(&output).unwrap();
        // A completed entry keeps its reserved names even after its output is removed.
        assert!(
            reservation_members(leases.parent(), &owner, &output)
                .unwrap()
                .is_some()
        );
        drop(leases);
        let leases = MemberLeases::acquire(
            dir.path(),
            &["note.llm.md".into(), "note.llm.llm.md".into()],
            false,
        )
        .unwrap();
        let mut other = owner.clone();
        other.key = "different.txt".into();
        publish(
            &leases,
            Some(&other),
            Policy::NoClobber,
            &output,
            b"literal base",
        )
        .unwrap();
        assert_eq!(
            reservation_members(leases.parent(), &other, &output).unwrap(),
            Some(vec!["note.llm.llm.md".into(), "note.llm.md".into()])
        );
        // Two valid family records for the same owner cannot resolve the ambiguity.
        publish(
            &leases,
            Some(&owner),
            Policy::Overwrite,
            &output,
            b"literal base",
        )
        .unwrap();
        assert!(
            reservation_members(leases.parent(), &owner, &output)
                .unwrap()
                .is_none()
        );
    }

    fn url_owners(owner: &Owner) -> (Owner, Owner) {
        let mut previous = owner.clone();
        previous.kind = "url".into();
        previous.key = "https://example.test/page?token=private".into();
        let mut next = previous.clone();
        next.key.push_str(" raw name.md");
        (previous, next)
    }

    #[test]
    fn bare_url_adoption_is_durable_idempotent_and_preserves_the_previous_checkpoint_owner() {
        let (_dir, leases, owner) = setup();
        let (previous, next) = url_owners(&owner);
        let output = leases.parent().join("note.md");
        publish(
            &leases,
            Some(&previous),
            Policy::NoClobber,
            &output,
            b"partial base",
        )
        .unwrap();
        let before = fs::read(record(&leases, &previous)).unwrap();
        adopt_owner(&leases, &previous, &next).unwrap();
        let named_before = fs::read(record(&leases, &next)).unwrap();
        verify_owned(&leases, &previous).unwrap();
        verify_owned(&leases, &next).unwrap();
        assert_eq!(fs::read(record(&leases, &previous)).unwrap(), before);
        adopt_owner(&leases, &previous, &next).unwrap();
        assert_eq!(fs::read(record(&leases, &next)).unwrap(), named_before);
        publish(
            &leases,
            Some(&next),
            Policy::RetryOwned,
            &output,
            b"named retry",
        )
        .unwrap();
        assert_eq!(fs::read(output).unwrap(), b"named retry");
        assert_eq!(fs::read(record(&leases, &previous)).unwrap(), before);
    }

    #[test]
    fn bare_url_adoption_rejects_modified_outputs_and_different_named_evidence() {
        let (_dir, leases, owner) = setup();
        let (previous, next) = url_owners(&owner);
        let output = leases.parent().join("note.md");
        publish(
            &leases,
            Some(&previous),
            Policy::NoClobber,
            &output,
            b"base",
        )
        .unwrap();
        fs::write(&output, b"edit").unwrap();
        assert!(adopt_owner(&leases, &previous, &next).is_err());
        assert!(!record(&leases, &next).exists());
        assert_eq!(fs::read(&output).unwrap(), b"edit");
        fs::remove_file(&output).unwrap();
        publish(
            &leases,
            Some(&previous),
            Policy::RetryOwned,
            &output,
            b"restored",
        )
        .unwrap();
        adopt_owner(&leases, &previous, &next).unwrap();
        let named_path = record(&leases, &next);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&named_path).unwrap()).unwrap();
        value["members"]["note.md"]["prepared"]["proof"]["sha256"] =
            serde_json::json!("0".repeat(64));
        let conflicting = serde_json::to_vec(&value).unwrap();
        fs::write(&named_path, &conflicting).unwrap();
        assert!(adopt_owner(&leases, &previous, &next).is_err());
        assert_eq!(fs::read(named_path).unwrap(), conflicting);
        assert_eq!(fs::read(output).unwrap(), b"restored");
    }

    #[test]
    fn absent_bare_adoption_creates_no_evidence_and_cannot_change_scope_or_uri() {
        let (_dir, leases, owner) = setup();
        let (previous, next) = url_owners(&owner);
        adopt_owner(&leases, &previous, &next).unwrap();
        assert!(!record(&leases, &previous).exists());
        assert!(!record(&leases, &next).exists());
        let mut invalid = next.clone();
        invalid.generation = uuid::Uuid::new_v4().to_string();
        assert!(adopt_owner(&leases, &previous, &invalid).is_err());
        invalid = next.clone();
        invalid.key = "https://elsewhere.test/page raw name.md".into();
        assert!(adopt_owner(&leases, &previous, &invalid).is_err());
        invalid = next.clone();
        invalid.output = leases.parent().join("another-scope");
        assert!(adopt_owner(&leases, &previous, &invalid).is_err());
        fs::write(leases.parent().join("note.md"), b"unowned").unwrap();
        assert!(adopt_owner(&leases, &previous, &next).is_err());
        assert!(!record(&leases, &previous).exists());
    }
}
