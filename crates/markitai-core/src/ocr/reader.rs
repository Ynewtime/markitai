//! macOS Vision in a process of its own, so that a request that never returns (see
//! `docs/ocr.md`) costs a restart and not the program.
//!
//! A reader takes prepared images on stdin and answers with their lines on stdout,
//! each request on a thread of its own, and goes when its stdin closes, which it
//! sees even with a request stopped. It is this same program started again in a
//! Markitai executable ([`start`]); elsewhere it is `markitai-vision-reader` beside
//! the library holding this code, which the Python and Node packages carry. A host
//! with neither reads in process ([`Outcome::NoReader`]).
use super::{Line, OcrResult, Result, assemble, failure, pixels, trace};
use crate::Error;
use std::collections::HashMap;
use std::ffi::{CStr, OsStr, OsString};
use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::ffi::OsStrExt;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, mpsc};
use std::time::{Duration, Instant};

/// The first argument that makes a Markitai executable a reader.
const ARGUMENT: &str = "__markitai-vision-reader";
/// The reader beside a library in a program that is not a Markitai executable.
const BESIDE: &str = "markitai-vision-reader";
/// The least CPU per [`Limits::poll`] that counts as work, in the units of
/// `proc_pid_rusage`: a few milliseconds on Apple silicon.
const WORK: u64 = 100_000;
/// The most an answer can hold of a string; more means the reader is not ours.
const LONGEST_STRING: usize = 1 << 28;

/// When a reader counts as stopped.
#[derive(Clone, Copy)]
struct Limits {
    /// How long a reader with work may use no CPU, the first time: a stopped request
    /// uses none. Each restart in a row doubles it, up to sixteen times, for what
    /// merely waits on the Neural Engine.
    idle: Duration,
    /// How long a request may go unanswered, whatever the reader is doing.
    longest: Duration,
    /// How many readers a request is tried on, or one after a request that ran out.
    attempts: u32,
    /// How often a reader is looked at.
    poll: Duration,
    /// Whether the system is compiling models, for which a first use of a program
    /// waits without using any CPU of its own.
    compiling: fn() -> bool,
}

const LIMITS: Limits = Limits {
    idle: Duration::from_secs(5),
    longest: Duration::from_secs(900),
    attempts: 6,
    poll: Duration::from_millis(250),
    compiling,
};

static SERVING: AtomicBool = AtomicBool::new(false);
static OWN_PROGRAM: OnceLock<PathBuf> = OnceLock::new();

/// Whether this process is a reader, whose readings the program assembles.
pub(super) fn serving() -> bool {
    SERVING.load(Ordering::Relaxed)
}

/// For a Markitai executable, at its very start: be a reader when started as one
/// (never returning), and otherwise read through this program started again.
pub(super) fn start(will_read: impl FnOnce() -> bool) {
    if std::env::args_os()
        .nth(1)
        .is_some_and(|first| first == ARGUMENT)
    {
        serve();
    }
    if let Ok(program) = std::env::current_exe() {
        let _ = OWN_PROGRAM.set(program);
    }
    if will_read() {
        prestart();
    }
}

/// Have a reader starting while the image gets ready, which is the time its start
/// takes.
pub(super) fn prestart() {
    static ASKED: AtomicBool = AtomicBool::new(false);
    if !ASKED.swap(true, Ordering::Relaxed) {
        let _ = std::thread::Builder::new().spawn(|| {
            if let Some(client) = client() {
                let _ = client.reader();
            }
        });
    }
}

fn client() -> Option<&'static Arc<Client>> {
    static CLIENT: OnceLock<Option<Arc<Client>>> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            let (program, arguments) = match OWN_PROGRAM.get() {
                Some(program) => (program.clone(), vec![OsString::from(ARGUMENT)]),
                None => (beside()?, Vec::new()),
            };
            Some(Arc::new(Client::new(program, arguments, LIMITS)))
        })
        .as_ref()
}

/// The reader beside the executable or library holding this code, if there is one.
fn beside() -> Option<PathBuf> {
    let mut info = std::mem::MaybeUninit::<libc::Dl_info>::zeroed();
    // SAFETY: The address is of a function in the image to find, and the record is
    // filled in when the call does not return 0.
    let found = unsafe { libc::dladdr(beside as *const () as *const _, info.as_mut_ptr()) };
    if found == 0 {
        return None;
    }
    // SAFETY: Filled in by the call above, whose name lives as long as the image.
    let name = unsafe { CStr::from_ptr(info.assume_init().dli_fname) };
    let reader = Path::new(OsStr::from_bytes(name.to_bytes()))
        .parent()?
        .join(BESIDE);
    reader.is_file().then_some(reader)
}

// The reader's side.

/// Answer requests until stdin closes.
pub(super) fn serve() -> ! {
    SERVING.store(true, Ordering::Relaxed);
    // SAFETY: Standard descriptors, whichever way the calls go. What a framework
    // prints goes to stderr from here on, and not into the answers, whose descriptor
    // no process it starts gets to hold open.
    let answers = unsafe {
        let answers = libc::fcntl(1, libc::F_DUPFD_CLOEXEC, 3);
        if answers < 0 || libc::dup2(2, 1) < 0 {
            libc::_exit(1);
        }
        File::from_raw_fd(answers)
    };
    // What the first request does anyway, while the program gets its image ready.
    let _ = std::thread::Builder::new().spawn(|| {
        let _ = super::vision::languages();
    });
    let answers = Arc::new(Mutex::new(answers));
    let mut input = std::io::stdin().lock();
    while let Some((id, spelling, image)) = request(&mut input) {
        let answers = answers.clone();
        let spawned = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                let read = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    super::read_here(&image, &spelling)
                }));
                let read = read.unwrap_or_else(|_| Err(failure("a Vision reading panicked")));
                let reply = encode(id, &read);
                let mut answers = lock(&answers);
                if answers.write_all(&reply).is_err() {
                    // SAFETY: Ends the process, with nobody left to answer.
                    unsafe { libc::_exit(0) };
                }
            });
        if spawned.is_err() {
            break;
        }
    }
    // SAFETY: Ends the process; a stopped request must not hold up the exit.
    unsafe { libc::_exit(0) }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing these locks guard can be left half done by a panic.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn u32_of(input: &mut impl Read) -> Option<u32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes).ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn u64_of(input: &mut impl Read) -> Option<u64> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes).ok()?;
    Some(u64::from_le_bytes(bytes))
}

fn f32_of(input: &mut impl Read) -> Option<f32> {
    u32_of(input).map(f32::from_bits)
}

fn string_of(input: &mut impl Read) -> Option<String> {
    let length = u32_of(input)? as usize;
    if length > LONGEST_STRING {
        return None;
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes).ok()?;
    String::from_utf8(bytes).ok()
}

fn request(input: &mut impl Read) -> Option<(u64, String, pixels::Prepared)> {
    let id = u64_of(input)?;
    let spelling = string_of(input)?;
    let width = u32_of(input)?;
    let height = u32_of(input)?;
    let length = usize::try_from(u64_of(input)?).ok()?;
    let mut png = vec![0; length];
    input.read_exact(&mut png).ok()?;
    Some((id, spelling, pixels::Prepared { png, width, height }))
}

fn put_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

/// Requests are answered in the bits they were read as, so that a reading is the
/// same as in process.
fn encode(id: u64, read: &Result<OcrResult>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&id.to_le_bytes());
    match read {
        Ok(result) => {
            out.push(0);
            out.extend_from_slice(&result.scale.to_bits().to_le_bytes());
            put_string(&mut out, &result.language);
            out.push(u8::from(result.unread));
            out.extend_from_slice(&(result.raw.len() as u32).to_le_bytes());
            for line in &result.raw {
                put_string(&mut out, &line.text);
                out.extend_from_slice(&line.confidence.to_bits().to_le_bytes());
                for number in line.bounds.iter().chain(&line.direction) {
                    out.extend_from_slice(&number.to_bits().to_le_bytes());
                }
            }
        }
        Err(Error::Unsupported(message)) => {
            out.push(1);
            put_string(&mut out, message);
        }
        Err(Error::TextRecognition(message)) => {
            out.push(2);
            put_string(&mut out, message);
        }
        Err(Error::Config(message)) => {
            out.push(3);
            put_string(&mut out, message);
        }
        Err(error) => {
            out.push(2);
            put_string(&mut out, &error.to_string());
        }
    }
    out
}

// The program's side.

enum Reply {
    Lines(f32, String, bool, Vec<Line>),
    Failed(Error),
}

fn decode(input: &mut impl Read) -> Option<(u64, Reply)> {
    let id = u64_of(input)?;
    let mut kind = [0];
    input.read_exact(&mut kind).ok()?;
    let reply = match kind[0] {
        0 => {
            let scale = f32_of(input)?;
            let language = string_of(input)?;
            let mut unread = [0];
            input.read_exact(&mut unread).ok()?;
            let count = u32_of(input)? as usize;
            let mut lines = Vec::new();
            for _ in 0..count {
                lines.push(Line {
                    text: string_of(input)?,
                    confidence: f32_of(input)?,
                    bounds: [
                        f32_of(input)?,
                        f32_of(input)?,
                        f32_of(input)?,
                        f32_of(input)?,
                    ],
                    direction: [f32_of(input)?, f32_of(input)?],
                    marks: Default::default(),
                });
            }
            Reply::Lines(scale, language, unread[0] != 0, lines)
        }
        1 => Reply::Failed(Error::Unsupported(string_of(input)?)),
        3 => Reply::Failed(Error::Config(string_of(input)?)),
        _ => Reply::Failed(Error::TextRecognition(string_of(input)?)),
    };
    Some((id, reply))
}

/// One reader, and who waits for it.
struct Reader {
    input: Mutex<ChildStdin>,
    pending: Mutex<HashMap<u64, Waiting>>,
    /// Its answers ended, so no more are coming.
    dead: AtomicBool,
    /// [`Client::watch`] killed it, rather than it going by itself.
    killed: AtomicBool,
    child: Mutex<Child>,
    pid: i32,
}

struct Waiting {
    answer: mpsc::Sender<Reply>,
    since: Instant,
}

/// What became of a request.
enum Exchange {
    Answered(Reply),
    /// Every reader tried stopped or went without answering it.
    Stopped,
    /// There is no reader to be had here.
    Unavailable,
}

/// The readers of a program, one at a time.
struct Client {
    program: PathBuf,
    arguments: Vec<OsString>,
    limits: Limits,
    current: Mutex<Option<Arc<Reader>>>,
    next: AtomicU64,
    /// Readers killed since the last answer.
    restarts: AtomicU32,
    /// Readers that went by themselves, which is no use counting once one answered.
    gone: AtomicU32,
    answered: AtomicBool,
    unavailable: AtomicBool,
    /// A request ran out of readers and nothing has been answered since, so the next
    /// gets one: Vision not responding must not cost every image of a document the
    /// minutes it cost the first.
    failing: AtomicBool,
}

impl Client {
    fn new(program: PathBuf, arguments: Vec<OsString>, limits: Limits) -> Self {
        Client {
            program,
            arguments,
            limits,
            current: Mutex::new(None),
            next: AtomicU64::new(1),
            restarts: AtomicU32::new(0),
            gone: AtomicU32::new(0),
            answered: AtomicBool::new(false),
            unavailable: AtomicBool::new(false),
            failing: AtomicBool::new(false),
        }
    }

    /// The reader in use, started when there is none.
    fn reader(self: &Arc<Self>) -> std::io::Result<Arc<Reader>> {
        let mut current = lock(&self.current);
        if let Some(reader) = current.as_ref()
            && !reader.dead.load(Ordering::SeqCst)
        {
            return Ok(reader.clone());
        }
        let mut child = Command::new(&self.program)
            .args(&self.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let (Some(input), Some(output)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(std::io::Error::other("the reader has no pipes"));
        };
        // SAFETY: An open descriptor. A host may not ignore SIGPIPE, and a reader
        // that went is an answer not given rather than the end of the program.
        unsafe { libc::fcntl(input.as_raw_fd(), libc::F_SETNOSIGPIPE, 1) };
        let reader = Arc::new(Reader {
            input: Mutex::new(input),
            pending: Mutex::new(HashMap::new()),
            dead: AtomicBool::new(false),
            killed: AtomicBool::new(false),
            pid: child.id() as i32,
            child: Mutex::new(child),
        });
        let (client, listening) = (self.clone(), reader.clone());
        std::thread::Builder::new().spawn(move || client.listen(listening, output))?;
        let (client, watched) = (self.clone(), reader.clone());
        std::thread::Builder::new().spawn(move || client.watch(watched))?;
        *current = Some(reader.clone());
        Ok(reader)
    }

    fn retire(&self, reader: &Arc<Reader>) {
        let mut current = lock(&self.current);
        if current
            .as_ref()
            .is_some_and(|other| Arc::ptr_eq(other, reader))
        {
            *current = None;
        }
    }

    /// Hand a reader's answers to who waits for them, until it goes.
    fn listen(&self, reader: Arc<Reader>, output: ChildStdout) {
        let mut output = std::io::BufReader::with_capacity(1 << 16, output);
        while let Some((id, reply)) = decode(&mut output) {
            if let Some(waiting) = lock(&reader.pending).remove(&id) {
                self.restarts.store(0, Ordering::SeqCst);
                self.answered.store(true, Ordering::SeqCst);
                self.failing.store(false, Ordering::SeqCst);
                let _ = waiting.answer.send(reply);
            }
        }
        if !reader.killed.load(Ordering::SeqCst) {
            self.gone.fetch_add(1, Ordering::SeqCst);
        }
        reader.dead.store(true, Ordering::SeqCst);
        self.retire(&reader);
        lock(&reader.pending).clear();
        // Answers that made no sense leave it running, and nobody is listening.
        let mut child = lock(&reader.child);
        let _ = child.kill();
        let _ = child.wait();
    }

    /// Kill a reader that stopped: with work and no CPU, or too long on a request.
    /// Nothing here may panic or give up, for nothing else ends the wait for a reader
    /// that stopped.
    fn watch(&self, reader: Arc<Reader>) {
        let mut last = cpu(reader.pid).unwrap_or(0);
        let mut since = Instant::now();
        loop {
            std::thread::sleep(self.limits.poll);
            if reader.dead.load(Ordering::SeqCst) {
                return;
            }
            let used = cpu(reader.pid).unwrap_or(last);
            let working = used.saturating_sub(last) > WORK;
            last = used;
            let oldest = lock(&reader.pending)
                .values()
                .map(|waiting| waiting.since)
                .min();
            let Some(oldest) = oldest else {
                since = Instant::now();
                continue;
            };
            if working {
                since = Instant::now();
            }
            if oldest.elapsed() < self.limits.longest {
                let restarts = self.restarts.load(Ordering::SeqCst).min(4);
                if since.elapsed() < self.limits.idle * 2u32.pow(restarts) {
                    continue;
                }
                if (self.limits.compiling)() {
                    since = Instant::now();
                    continue;
                }
            }
            let idle = since.elapsed();
            let restarts = self.restarts.fetch_add(1, Ordering::SeqCst) + 1;
            reader.killed.store(true, Ordering::SeqCst);
            self.retire(&reader);
            {
                let mut child = lock(&reader.child);
                if !reader.dead.load(Ordering::SeqCst) {
                    let _ = child.kill();
                }
            }
            // Whatever keeps its output open, nobody waits for it any more.
            reader.dead.store(true, Ordering::SeqCst);
            lock(&reader.pending).clear();
            trace::line(format_args!(
                "vision reader restart={restarts} idle_ms={}",
                idle.as_millis()
            ));
            return;
        }
    }

    /// Whether readers cannot be had here: the first ones went without ever answering.
    fn give_up(&self) -> bool {
        !self.answered.load(Ordering::SeqCst) && self.gone.load(Ordering::SeqCst) >= 2
    }

    /// Ask one reader for a request made at `since`: the answer, or nothing when it
    /// went without giving one.
    fn ask(
        &self,
        reader: &Reader,
        head: &[u8],
        png: &[u8],
        id: u64,
        since: Instant,
    ) -> Option<Reply> {
        let (answer, answered) = mpsc::channel();
        lock(&reader.pending).insert(id, Waiting { answer, since });
        if reader.dead.load(Ordering::SeqCst) {
            lock(&reader.pending).remove(&id);
            return None;
        }
        // A reader that cannot be written to is going, which ends the wait below.
        let mut input = lock(&reader.input);
        let _ = input.write_all(head).and_then(|()| input.write_all(png));
        drop(input);
        answered.recv().ok()
    }

    /// What goes before the image's PNG in a request ([`request`]).
    fn head(&self, head: &mut Vec<u8>, id: u64, spelling: &str, image: &pixels::Prepared) {
        head.extend_from_slice(&id.to_le_bytes());
        put_string(head, spelling);
        head.extend_from_slice(&image.width.to_le_bytes());
        head.extend_from_slice(&image.height.to_le_bytes());
        head.extend_from_slice(&(image.png.len() as u64).to_le_bytes());
    }

    fn exchange(self: &Arc<Self>, image: &pixels::Prepared, spelling: &str) -> Exchange {
        if self.unavailable.load(Ordering::SeqCst) {
            return Exchange::Unavailable;
        }
        let began = Instant::now();
        let attempts = if self.failing.load(Ordering::SeqCst) {
            1
        } else {
            self.limits.attempts
        };
        for _ in 0..attempts {
            if began.elapsed() >= self.limits.longest {
                break;
            }
            // Not for good: what stops a start, like descriptors running out, can pass.
            let Ok(reader) = self.reader() else {
                return Exchange::Unavailable;
            };
            let id = self.next.fetch_add(1, Ordering::Relaxed);
            let mut head = Vec::with_capacity(64);
            self.head(&mut head, id, spelling, image);
            if let Some(reply) = self.ask(&reader, &head, &image.png, id, began) {
                return Exchange::Answered(reply);
            }
            if self.give_up() {
                self.unavailable.store(true, Ordering::SeqCst);
                return Exchange::Unavailable;
            }
        }
        self.failing.store(true, Ordering::SeqCst);
        self.restarts.store(0, Ordering::SeqCst);
        Exchange::Stopped
    }
}

unsafe extern "C" {
    fn proc_pid_rusage(
        pid: libc::c_int,
        flavor: libc::c_int,
        buffer: *mut libc::c_void,
    ) -> libc::c_int;
}

/// The CPU a process has used, in the units of `proc_pid_rusage`.
fn cpu(pid: i32) -> Option<u64> {
    #[repr(C)]
    struct Usage {
        uuid: [u8; 16],
        user: u64,
        system: u64,
        rest: [u64; 32],
    }
    let mut usage = std::mem::MaybeUninit::<Usage>::zeroed();
    // SAFETY: The buffer is larger than the version 0 record (flavor 0) it receives.
    if unsafe { proc_pid_rusage(pid, 0, usage.as_mut_ptr().cast()) } != 0 {
        return None;
    }
    // SAFETY: Zeroed, and filled in by the call above.
    let usage = unsafe { usage.assume_init() };
    Some(usage.user.saturating_add(usage.system))
}

/// See [`Limits::compiling`]. The service is shared, so this can also be another
/// program's models, which only makes the wait longer.
fn compiling() -> bool {
    compiling_per(Path::new("/bin/ps"), Duration::from_secs(5))
}

/// What `ps` says, given `wait` to say it, for nothing may hold up [`Client::watch`].
fn compiling_per(ps: &Path, wait: Duration) -> bool {
    let (answer, answered) = mpsc::channel();
    let ps = ps.to_owned();
    let asked = std::thread::Builder::new().spawn(move || {
        // The C locale, for others write a share as 0,4.
        let listing = Command::new(ps)
            .args(["-A", "-o", "pcpu=,comm="])
            .env("LC_ALL", "C")
            .output();
        let _ = answer.send(listing.is_ok_and(|listing| {
            String::from_utf8_lossy(&listing.stdout)
                .lines()
                .any(|line| {
                    let line = line.trim_start();
                    line.contains("ANECompilerService")
                        && line
                            .split_whitespace()
                            .next()
                            .and_then(|n| n.parse::<f32>().ok())
                            .is_some_and(|n| n > 3.0)
                })
        }));
    });
    asked.is_ok() && answered.recv_timeout(wait).unwrap_or(false)
}

/// What a reading through a reader came to.
pub(super) enum Outcome {
    Read(Result<OcrResult>),
    /// No reader can be had here, and nothing was read.
    NoReader,
}

fn outcome(exchange: Exchange, image: &pixels::Prepared, limits: &Limits) -> Outcome {
    match exchange {
        Exchange::Answered(Reply::Lines(scale, language, unread, lines)) => Outcome::Read(
            assemble(lines, image.width, image.height).map(|mut result| {
                result.scale = scale;
                result.language = language;
                result.unread = unread;
                result
            }),
        ),
        Exchange::Answered(Reply::Failed(error)) => Outcome::Read(Err(error)),
        Exchange::Stopped => Outcome::Read(Err(failure(&format!(
            "Vision stopped responding: a text recognition did not finish in up to {} tries. Run the \
             program again, or read with the portable engine (MARKITAI_OCR_BACKEND=paddle)",
            limits.attempts
        )))),
        Exchange::Unavailable => Outcome::NoReader,
    }
}

/// What `read_here` makes of the image, in a reader.
pub(super) fn read(image: &pixels::Prepared, spelling: &str) -> Outcome {
    let Some(client) = client() else {
        return Outcome::NoReader;
    };
    outcome(client.exchange(image, spelling), image, &client.limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS_NOW: Limits = Limits {
        idle: Duration::from_millis(300),
        longest: Duration::from_secs(60),
        attempts: 3,
        poll: Duration::from_millis(25),
        compiling: || false,
    };

    /// In a reader's script: wait for the request, so that its answer has someone waiting.
    const RECEIVED: &str = "dd bs=1 count=1 2>/dev/null >/dev/null";

    /// A client of readers that are this shell script, with a file counting their
    /// starts in the directory returned.
    fn client_of(script: &str, limits: Limits) -> (Arc<Client>, tempfile::TempDir) {
        let directory = tempfile::tempdir().unwrap();
        let starts = directory.path().join("starts");
        let counted = format!("echo x >> '{}'\n{script}", starts.display());
        let arguments = vec![OsString::from("-c"), OsString::from(counted)];
        (
            Arc::new(Client::new("/bin/sh".into(), arguments, limits)),
            directory,
        )
    }

    fn starts(directory: &tempfile::TempDir) -> usize {
        std::fs::read_to_string(directory.path().join("starts"))
            .map_or(0, |text| text.lines().count())
    }

    fn executable(path: &Path, script: &str) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(path, script).unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    fn image() -> pixels::Prepared {
        pixels::Prepared {
            png: vec![0; 16],
            width: 1,
            height: 1,
        }
    }

    fn failed_with(exchange: Exchange) -> String {
        match exchange {
            Exchange::Answered(Reply::Failed(error)) => error.to_string(),
            _ => panic!("not a failure answered"),
        }
    }

    /// The answer to request `id` when it is below 8: a failure saying "later".
    fn failure_answer(id: u8) -> String {
        format!(r"printf '\{id:03o}\000\000\000\000\000\000\000\002\005\000\000\000later'")
    }

    #[test]
    fn a_reading_crosses_the_pipe_to_the_bit() {
        let line = Line {
            text: "第一行 line".into(),
            confidence: f32::from_bits(0x3f7f_ffff),
            bounds: [0.1, f32::MIN_POSITIVE, 1e-30, 1275.5],
            direction: [1.0, -0.0],
            marks: Default::default(),
        };
        let reading = OcrResult {
            scale: 2.5,
            language: "zh-Hans".into(),
            unread: true,
            raw: vec![line.clone(), line],
            ..Default::default()
        };
        let encoded = encode(9, &Ok(reading));
        let (id, reply) = decode(&mut encoded.as_slice()).unwrap();
        assert_eq!(id, 9);
        let Reply::Lines(scale, language, unread, lines) = reply else {
            panic!("not lines")
        };
        assert_eq!(
            (scale, language.as_str(), unread, lines.len()),
            (2.5, "zh-Hans", true, 2)
        );
        assert_eq!(lines[1].text, "第一行 line");
        assert_eq!(lines[1].confidence.to_bits(), 0x3f7f_ffff);
        assert_eq!(
            lines[1].bounds.map(f32::to_bits),
            [0.1, f32::MIN_POSITIVE, 1e-30, 1275.5].map(f32::to_bits)
        );
        assert_eq!(
            lines[1].direction.map(f32::to_bits),
            [1.0f32, -0.0].map(f32::to_bits)
        );
    }

    #[test]
    fn a_failure_crosses_the_pipe_as_the_same_kind() {
        let unsupported = encode(1, &Err(Error::Unsupported("no language".into())));
        let (_, reply) = decode(&mut unsupported.as_slice()).unwrap();
        assert!(
            matches!(reply, Reply::Failed(Error::Unsupported(message)) if message == "no language")
        );
        let setting = encode(3, &Err(Error::Config("ocr.lang is not".into())));
        let (_, reply) = decode(&mut setting.as_slice()).unwrap();
        assert!(
            matches!(reply, Reply::Failed(Error::Config(message)) if message == "ocr.lang is not")
        );
        let other = encode(2, &Err(Error::TextRecognition("not read".into())));
        let (_, reply) = decode(&mut other.as_slice()).unwrap();
        assert!(
            matches!(reply, Reply::Failed(Error::TextRecognition(message)) if message == "not read")
        );
    }

    #[test]
    fn a_cut_off_answer_is_no_answer() {
        let whole = encode(1, &Err(Error::Conversion("not read".into())));
        assert!(decode(&mut &whole[..whole.len() - 1]).is_none());
    }

    #[test]
    fn a_request_is_read_as_it_was_written() {
        let client = Client::new("/bin/sh".into(), Vec::new(), LIMITS_NOW);
        let mut head = Vec::new();
        client.head(&mut head, 7, "zh-Hans", &image());
        head.extend_from_slice(&image().png);
        let (id, spelling, read) = request(&mut head.as_slice()).unwrap();
        assert_eq!(
            (id, spelling.as_str(), read.width, read.height),
            (7, "zh-Hans", 1, 1)
        );
        assert_eq!(read.png, image().png);
    }

    #[test]
    fn a_reader_that_stops_is_killed_and_the_next_one_answers() {
        let directory = tempfile::tempdir().unwrap();
        let script = format!(
            "echo x >> starts\n[ $(wc -l < starts) -gt 1 ] && {}\nexec sleep 100",
            failure_answer(2)
        );
        let script = format!("cd '{}'\n{script}", directory.path().display());
        let arguments = vec![OsString::from("-c"), OsString::from(script)];
        let client = Arc::new(Client::new("/bin/sh".into(), arguments, LIMITS_NOW));
        assert_eq!(failed_with(client.exchange(&image(), "en")), "later");
        assert_eq!(starts(&directory), 2);
        assert_eq!(
            client.restarts.load(Ordering::SeqCst),
            0,
            "an answer forgets the restarts"
        );
    }

    #[test]
    fn readers_that_stop_each_time_are_an_error_in_the_end() {
        let (client, directory) = client_of("exec sleep 100", LIMITS_NOW);
        let began = Instant::now();
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert_eq!(starts(&directory), 3);
        // Each restart in a row waits twice as long: 300, 600 and 1200 ms.
        assert!(began.elapsed() >= Duration::from_millis(2100));
    }

    #[test]
    fn a_reader_using_the_processor_is_left_alone_up_to_the_longest_wait() {
        let limits = Limits {
            idle: Duration::from_secs(1),
            longest: Duration::from_secs(3),
            attempts: 1,
            poll: Duration::from_millis(250),
            compiling: || false,
        };
        let (client, directory) = client_of("while :; do :; done", limits);
        let began = Instant::now();
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert_eq!(starts(&directory), 1);
        assert!(began.elapsed() >= Duration::from_secs(3));
    }

    #[test]
    fn a_reader_waiting_on_compiling_models_is_left_alone() {
        static CALLS: AtomicU32 = AtomicU32::new(0);
        let limits = Limits {
            compiling: || CALLS.fetch_add(1, Ordering::SeqCst) < 3,
            attempts: 2,
            ..LIMITS_NOW
        };
        let (client, directory) = client_of("exec sleep 100", limits);
        let began = Instant::now();
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert_eq!(starts(&directory), 2);
        // Three more waits of the idle time before the first kill.
        assert!(began.elapsed() >= Duration::from_millis(1200));
    }

    /// Every reader stopped: the reading fails as text recognition, the kind an app
    /// tells from a document it cannot convert, and bindings see the same code as ever.
    #[test]
    fn readers_that_all_stop_fail_as_text_recognition() {
        let (client, directory) = client_of("exec sleep 100", LIMITS_NOW);
        let exchange = client.exchange(&image(), "en");
        assert_eq!(starts(&directory), 3);
        let Outcome::Read(Err(error)) = outcome(exchange, &image(), &LIMITS_NOW) else {
            panic!("not a failure");
        };
        assert!(matches!(error, Error::TextRecognition(_)), "{error}");
        assert!(
            error.to_string().contains("Vision stopped responding"),
            "{error}"
        );
        assert_eq!(error.code(), "conversion_error");
    }

    #[test]
    fn readers_that_go_at_once_mean_there_is_no_reader() {
        let (client, directory) = client_of("exit 0", LIMITS_NOW);
        assert!(matches!(
            client.exchange(&image(), "en"),
            Exchange::Unavailable
        ));
        assert_eq!(starts(&directory), 2);
        // Not tried again for the next image.
        assert!(matches!(
            client.exchange(&image(), "en"),
            Exchange::Unavailable
        ));
        assert_eq!(starts(&directory), 2);
    }

    #[test]
    fn a_program_that_cannot_start_means_there_is_no_reader() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing");
        let client = Arc::new(Client::new(missing, Vec::new(), LIMITS_NOW));
        assert!(matches!(
            client.exchange(&image(), "en"),
            Exchange::Unavailable
        ));
    }

    #[test]
    fn readers_that_answered_and_then_go_are_an_error_and_not_no_reader() {
        let script = format!("{}\nexit 0", failure_answer(1));
        let (client, directory) = client_of(&script, LIMITS_NOW);
        assert_eq!(failed_with(client.exchange(&image(), "en")), "later");
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert!(starts(&directory) >= 2);
    }

    #[test]
    fn nothing_waits_for_a_killed_reader_whose_output_stays_open() {
        // A process it started holds its output open after it is killed.
        let (client, directory) = client_of("sleep 100 &\nexec sleep 100", LIMITS_NOW);
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert_eq!(starts(&directory), 3);
    }

    #[test]
    fn once_every_reader_stopped_an_image_gets_one_until_one_answers() {
        let directory = tempfile::tempdir().unwrap();
        let script = format!(
            "cd '{}'\necho x >> starts\n[ -e answer ] && {RECEIVED} && {}\nexec sleep 100",
            directory.path().display(),
            failure_answer(5)
        );
        let arguments = vec![OsString::from("-c"), OsString::from(script)];
        let client = Arc::new(Client::new("/bin/sh".into(), arguments, LIMITS_NOW));
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert_eq!(starts(&directory), 3);

        let began = Instant::now();
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert_eq!(starts(&directory), 4);
        // At the first idle time of 300 ms, and not the 2400 of a fourth in a row.
        assert!(began.elapsed() < Duration::from_millis(2400));

        std::fs::write(directory.path().join("answer"), "").unwrap();
        assert_eq!(failed_with(client.exchange(&image(), "en")), "later");
        assert_eq!(starts(&directory), 5);

        std::fs::remove_file(directory.path().join("answer")).unwrap();
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        // The reader that answered, and two more: an answer brings back every try.
        assert_eq!(starts(&directory), 7);
    }

    #[test]
    fn a_program_that_could_not_start_is_tried_again_for_the_next_image() {
        let directory = tempfile::tempdir().unwrap();
        let program = directory.path().join("reader");
        // The first start of a new file can stop for seconds, using no CPU.
        let limits = Limits {
            idle: Duration::from_secs(30),
            ..LIMITS_NOW
        };
        let client = Arc::new(Client::new(program.clone(), Vec::new(), limits));
        assert!(matches!(
            client.exchange(&image(), "en"),
            Exchange::Unavailable
        ));
        let script = format!(
            "#!/bin/sh\n{RECEIVED}\n{}\nexec sleep 100\n",
            failure_answer(1)
        );
        executable(&program, &script);
        assert_eq!(failed_with(client.exchange(&image(), "en")), "later");
    }

    #[test]
    fn the_longest_wait_of_a_request_covers_every_reader_it_is_tried_on() {
        let limits = Limits {
            idle: Duration::from_secs(60),
            longest: Duration::from_secs(2),
            attempts: 3,
            poll: Duration::from_millis(25),
            compiling: || false,
        };
        let directory = tempfile::tempdir().unwrap();
        // The first goes by itself near the end of the wait; the next never answers.
        let script = format!(
            "cd '{}'\necho x >> starts\n[ $(wc -l < starts) -eq 1 ] && sleep 1.8 && exit 0\n\
             exec sleep 100",
            directory.path().display()
        );
        let arguments = vec![OsString::from("-c"), OsString::from(script)];
        let client = Arc::new(Client::new("/bin/sh".into(), arguments, limits));
        let began = Instant::now();
        assert!(matches!(client.exchange(&image(), "en"), Exchange::Stopped));
        assert_eq!(starts(&directory), 2);
        assert!(began.elapsed() < Duration::from_millis(3700));
    }

    #[test]
    fn a_reader_whose_answers_make_no_sense_does_not_stay() {
        let directory = tempfile::tempdir().unwrap();
        // A failure said to be longer than any string, after which it would wait.
        let nonsense = r"printf '\001\000\000\000\000\000\000\000\002\377\377\377\377'";
        let script = format!(
            "cd '{}'\necho $$ > pid\n{nonsense}\nexec sleep 100",
            directory.path().display()
        );
        let arguments = vec![OsString::from("-c"), OsString::from(script)];
        let client = Arc::new(Client::new("/bin/sh".into(), arguments, LIMITS_NOW));
        client.exchange(&image(), "en");
        let pid: i32 = std::fs::read_to_string(directory.path().join("pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let began = Instant::now();
        // SAFETY: Signal 0 only asks whether the process is there.
        while unsafe { libc::kill(pid, 0) } == 0 {
            assert!(began.elapsed() < Duration::from_secs(20), "still running");
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    #[test]
    fn compiling_is_read_in_the_locale_ps_is_asked_for() {
        let directory = tempfile::tempdir().unwrap();
        let ps = directory.path().join("ps");
        executable(
            &ps,
            "#!/bin/sh\n[ \"$LC_ALL\" = C ] && mark=. || mark=,\n\
             echo \" 12${mark}5 /usr/libexec/ANECompilerService\"\n",
        );
        assert!(compiling_per(&ps, Duration::from_secs(30)));
    }

    #[test]
    fn a_ps_that_does_not_return_does_not_hold_up_the_watch() {
        let directory = tempfile::tempdir().unwrap();
        let ps = directory.path().join("ps");
        executable(&ps, "#!/bin/sh\nexec sleep 100\n");
        let began = Instant::now();
        assert!(!compiling_per(&ps, Duration::from_millis(100)));
        assert!(began.elapsed() < Duration::from_secs(50));
    }

    #[test]
    fn no_reader_is_found_beside_a_test_program() {
        assert_eq!(beside(), None);
    }
}
