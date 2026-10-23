# Native browser installation

The CLI includes its browser control code. Dynamic webpages still require a
Chrome-compatible executable. `markitai doctor` checks a private browser launch (a browser inside a macOS
`.app` bundle is only asked for `--version`, because starting it from a terminal can make macOS
report that the terminal tried to modify an app);
The launch may take a minute, then half a minute more on a second try: a first
launch after installing can be that slow on Windows while antivirus scans the
new executable. A browser that starts in neither try is reported as installed
but slow to start (a warning that blocks no browser workflow), with
`markitai doctor` to run again in a minute; `--fix` installs nothing for it.
A successful launch is remembered for seven days in `MARKITAI_HOME`
(`doctor-chromium-launch.json`), for the same browser executable (path, size and
modification time) and the same markitai, so later doctor runs do not start the
browser again; delete the file to check a launch now.
`markitai doctor --fix` installs the official Chrome for Testing headless shell
when no discovered browser can start. A healthy browser needs no download.
Normal conversions and diagnostics never install software implicitly.

The installer reads Google's Stable manifest and accepts only the exact official
HTTPS archive URL for the selected version and platform. It runs entirely in
Rust, without an npm, Playwright or Python installer. See the upstream
[distribution manifest documentation](https://github.com/GoogleChromeLabs/chrome-for-testing)
and [Chromium headless shell documentation](https://github.com/chromium/chromium/blob/main/headless/README.md).

Files live beneath `MARKITAI_HOME/browsers/native`, or `~/.markitai/browsers/native`
when no isolated home is configured. Discovery gives an explicit
`MARKITAI_BROWSER_EXECUTABLE` precedence, then a completed managed installation,
then system browsers and existing Playwright caches. An explicit executable
override prevents installation: repair or remove that override first.

An OS file lock excludes concurrent installers. Downloads stream to a private
temporary file; the manifest is bounded to 1 MiB, the archive to 512 MiB, and
expanded files to 1.5 GiB and 16,384 entries. Requests use a 15-second connection
timeout and a 600-second overall timeout per response. Redirects are rejected.
Extraction rejects path traversal, unexpected archive roots, duplicate paths,
links and special files. Unix directories are private and executable permissions
are restricted to the current user.

Before publication, the downloaded executable must start in a fresh private
profile and answer a CDP command. A new receipt is atomically published only
after that succeeds. Earlier versions remain available on disk; an interrupted
download or failed launch leaves the previous receipt unchanged. Installation
does not modify a browser's existing user profile or system installation.

The receipt records archive and executable SHA-256 identities. These are local
audit records, not vendor signatures or independently supplied checksums. The
trust source is the official HTTPS distribution. Normal discovery checks the
receipt and regular executable without hashing the entire binary on each fetch.

Platform selection covers macOS arm64/x64, Linux arm64/x64 and Windows x86/x64
when the official Stable manifest contains the corresponding archive. A platform
mapping alone is not a successful installation test. Windows arm64 and other
platforms return an explicit unsupported error. Shared Linux system libraries
may still be required; a launch failure is reported before making the new
browser current. Actual validation results belong in the round's validation
record.

The optional browser's installed size is separate from the single-binary CLI
and must be included when comparing workflows that require dynamic rendering.
