"""The desktop entry of Markitai Mygo on Linux, named and filled as Native's
(desktop/native/linux/data/dev.markitai.native.desktop).

The app runs as dev.markitai.mygo (toolkit_linux.go): that is its window's
class and the name of its data folders. mygo build names the entry after the
slug, which cannot hold a dot (cmd/mygo/config.go:304), and writes no
Comment[zh_CN], Keywords, StartupNotify or StartupWMClass (linuxDesktopEntry,
cmd/mygo/build.go:609). So after mygo build, finish() adds those keys to every
copy of the entry and installs it as dev.markitai.mygo.desktop: in the files
next to the app, the .tar.gz, install.sh, the Debian package and the PKGBUILD.
It also gives the Debian package Native's long description, where mygo writes
"<Name> is a desktop application.", and the PKGBUILD the workspace's license,
which mygo reads from a package.json the app does not have. mygo installs the
icon at 256 and 512 pixels only (linuxIcons, cmd/mygo/deb.go:248); the
packages also get Native's icons at the smaller sizes Native installs, made
from the same desktop/icon.png, which menus and docks show without scaling a
large one down; install.sh's entry shows the archive's original by its path,
as mygo writes it. It fails rather than guess when mygo writes any of them
differently. Its output depends only on its input: the archives it writes
again keep their members' times and the gzip header's."""
from pathlib import Path
import gzip
import hashlib
import io
import re
import struct
import tarfile
import textwrap
import tomllib

APP_ID = "dev.markitai.mygo"
SLUG = "markitai-mygo"
REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "desktop" / "native" / "linux" / "data" / "dev.markitai.native.desktop"
METAINFO = NATIVE.with_name("dev.markitai.native.metainfo.xml")
NATIVE_ICONS = NATIVE.parent / "icons" / "hicolor"
# The sizes Native installs that mygo build does not, which come from Native.
ICON_SIZES = (16, 32, 48, 64, 128)
# Keys the entry takes from Native's, as they are there.
FROM_NATIVE = ("Comment[zh_CN]", "Keywords", "StartupNotify")
INSTALLED = f"usr/share/applications/{SLUG}.desktop"


def native_keys():
    keys = {}
    for line in NATIVE.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            keys[key] = value
    return keys


def long_description():
    """Native's metainfo description, as the lines of a Debian control field."""
    found = re.search(r"<description>\s*<p>(.*?)</p>", METAINFO.read_text(encoding="utf-8"), re.S)
    if not found:
        raise SystemExit(f"{METAINFO.name} has no description paragraph")
    text = " ".join(found.group(1).split()).replace("Markitai turns", "Markitai Mygo turns", 1)
    return "".join(" " + line + "\n" for line in textwrap.wrap(text, 76))


def entry(text, keys):
    """text, the entry mygo wrote, with Native's keys and the window's class."""
    lines = text.rstrip("\n").split("\n")
    comments = [i for i, line in enumerate(lines) if line.startswith("Comment=")]
    if len(comments) != 1 or lines[comments[0]] != "Comment=" + keys["Comment"]:
        raise SystemExit(f"the desktop entry's Comment is not Native's {keys['Comment']!r}: make linux.comment of mygo.json match")
    for key in FROM_NATIVE + ("StartupWMClass",):
        if any(line.startswith(key + "=") for line in lines):
            raise SystemExit(f"mygo build now writes {key} in the desktop entry: drop it from linux_entry.py")
    lines.insert(comments[0] + 1, "Comment[zh_CN]=" + keys["Comment[zh_CN]"])
    lines += ["Keywords=" + keys["Keywords"], "StartupNotify=" + keys["StartupNotify"], "StartupWMClass=" + APP_ID]
    return "\n".join(lines) + "\n"


def icons():
    """Native's icon at each of ICON_SIZES: [(size, png)]."""
    found = []
    for px in ICON_SIZES:
        path = NATIVE_ICONS / f"{px}x{px}" / "apps" / "dev.markitai.native.png"
        if not path.is_file():
            raise SystemExit(f"no {path.relative_to(REPO)}")
        found.append((px, path.read_bytes()))
    return found


def icon_dir(px):
    return f"usr/share/icons/hicolor/{px}x{px}/apps"


def replace_once(text, old, new, where):
    if text.count(old) != 1:
        raise SystemExit(f"{where}: expected {old!r} once, as mygo build wrote it")
    return text.replace(old, new)


def rewrite_tar(data, change, before=lambda member: ()):
    """data, a .tar.gz, with change(member, bytes) -> (member, bytes) applied to
    each regular file, the others as they were, and the members before(member)
    returns, as (member, bytes or None), ahead of it. The gzip header keeps
    data's time, as tarfile would write the time of the run."""
    out = io.BytesIO()
    mtime = struct.unpack("<I", data[4:8])[0]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as src, \
            gzip.GzipFile(fileobj=out, mode="wb", mtime=mtime) as packed, \
            tarfile.open(fileobj=packed, mode="w", format=src.format) as dst:
        for member in src.getmembers():
            for extra, body in before(member):
                dst.addfile(extra, io.BytesIO(body) if body is not None else None)
            body = src.extractfile(member).read() if member.isfile() else None
            if body is not None:
                member, body = change(member, body)
                member.size = len(body)
            dst.addfile(member, io.BytesIO(body) if body is not None else None)
    return out.getvalue()


def like(member, name, body=None):
    """A member named name, a directory without body, with member's owner and
    time: a member of the same archive."""
    new = tarfile.TarInfo(name)
    new.uid, new.gid, new.uname, new.gname, new.mtime = member.uid, member.gid, member.uname, member.gname, member.mtime
    if body is None:
        new.type, new.mode = tarfile.DIRTYPE, 0o755
    else:
        new.mode, new.size = 0o644, len(body)
    return new


def read_ar(data):
    if not data.startswith(b"!<arch>\n"):
        raise SystemExit("the Debian package is not an ar archive")
    members, at = [], 8
    while at < len(data):
        header = data[at:at + 60]
        size = int(header[48:58])
        members.append((header, data[at + 60:at + 60 + size]))
        at += 60 + size + size % 2
    return members


def write_ar(members):
    out = bytearray(b"!<arch>\n")
    for header, body in members:
        out += header[:48] + str(len(body)).encode().ljust(10) + header[58:60] + body
        if len(body) % 2:
            out += b"\n"
    return bytes(out)


def finish(folder, version, keys=None):
    """The Linux build of version in folder, as mygo build left it."""
    keys = keys or native_keys()
    folder = Path(folder)
    archive = folder / f"{SLUG}-{version}-{folder.name}.tar.gz"
    deb = next(folder.glob(f"{SLUG}_*.deb"), None)

    # Next to the app, and in the archive install.sh installs: install.sh
    # reads the entry as <slug>.desktop and writes it where desktop_file says.
    beside = folder / f"{SLUG}.desktop"
    beside.write_text(entry(beside.read_text(encoding="utf-8"), keys), encoding="utf-8")

    def in_archive(member, body):
        if member.name in (f"{SLUG}.desktop", f"./{SLUG}.desktop"):
            body = entry(body.decode("utf-8"), keys).encode("utf-8")
        return member, body

    old_sum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.write_bytes(rewrite_tar(archive.read_bytes(), in_archive))
    new_sum = hashlib.sha256(archive.read_bytes()).hexdigest()

    script = folder / "install.sh"
    script.write_text(replace_once(script.read_text(encoding="utf-8"),
                                   'desktop_file="$data_home/applications/$name.desktop"',
                                   f'desktop_file="$data_home/applications/{APP_ID}.desktop"', "install.sh"), encoding="utf-8")

    # The PKGBUILD writes the entry from its own text and checks the archive
    # by its sum.
    pkgbuild = folder / "PKGBUILD"
    text = pkgbuild.read_text(encoding="utf-8")
    with open(REPO / "Cargo.toml", "rb") as file:
        license = tomllib.load(file)["workspace"]["package"]["license"]
    text = replace_once(text, "license=('LicenseRef-custom')", f"license=('{license}')", "PKGBUILD")
    inline = re.search(r"printf '%s' '(\[Desktop Entry\]\n.*?)' \|", text, re.S)
    if not inline:
        raise SystemExit("PKGBUILD: no desktop entry as mygo build writes it")
    text = text[:inline.start(1)] + entry(inline.group(1), keys) + text[inline.end(1):]
    text = replace_once(text, f'"$pkgdir"/{INSTALLED}', f'"$pkgdir"/usr/share/applications/{APP_ID}.desktop', "PKGBUILD")
    text = replace_once(text, old_sum, new_sum, "PKGBUILD")
    # The smaller icons, as sources next to it, as mygo writes its own.
    small = icons()
    sources, sums, installs = [], [], []
    for px, png in small:
        source = f"{SLUG}-{px}x{px}.png"
        if source in text:
            raise SystemExit(f"mygo build now writes {source}: drop {px} from ICON_SIZES")
        (folder / source).write_bytes(png)
        sources.append(f"'{source}'")
        sums.append(f"'{hashlib.sha256(png).hexdigest()}'")
        installs.append(f'\tinstall -Dm644 {source} "$pkgdir"/{icon_dir(px)}/{SLUG}.png\n')
    first = f"'{SLUG}-256x256.png'"
    text = replace_once(text, first, " ".join(sources) + " " + first, "PKGBUILD")
    sums_at = re.search(r"^sha256sums=\('[0-9a-f]{64}' ", text, re.M)
    if not sums_at:
        raise SystemExit("PKGBUILD: no sha256sums as mygo build writes them")
    text = text[:sums_at.end()] + " ".join(sums) + " " + text[sums_at.end():]
    text = replace_once(text, f'\tinstall -Dm644 {SLUG}-256x256.png', "".join(installs) + f'\tinstall -Dm644 {SLUG}-256x256.png', "PKGBUILD")
    pkgbuild.write_text(text, encoding="utf-8")

    if deb is None:
        return
    members = read_ar(deb.read_bytes())
    names = [header[:16].decode().strip() for header, _ in members]
    if names != ["debian-binary", "control.tar.gz", "data.tar.gz"]:
        raise SystemExit(f"{deb.name}: members {names}, not mygo build's")
    sums = {}

    def in_data(member, body):
        if member.name in (INSTALLED, "./" + INSTALLED):
            body = entry(body.decode("utf-8"), keys).encode("utf-8")
            member.name = member.name.replace(f"{SLUG}.desktop", f"{APP_ID}.desktop")
            sums["entry"] = hashlib.md5(body).hexdigest()
        return member, body

    # The smaller icons go ahead of mygo's 256, each in its directories.
    # tarfile names a directory without its closing slash.
    def icons_before(member):
        name = member.name.removeprefix("./")
        if any((name + "/").startswith(f"usr/share/icons/hicolor/{px}x{px}/") for px, _ in small):
            raise SystemExit(f"mygo build now writes {name} in {deb.name}: drop its size from ICON_SIZES")
        if name != "usr/share/icons/hicolor/256x256":
            return ()
        dot = "./" if member.name.startswith("./") else ""
        added = []
        for px, png in small:
            added += [(like(member, f"{dot}usr/share/icons/hicolor/{px}x{px}"), None),
                      (like(member, f"{dot}{icon_dir(px)}"), None),
                      (like(member, f"{dot}{icon_dir(px)}/{SLUG}.png", png), png)]
            sums[px] = hashlib.md5(png).hexdigest()
        return added

    data = rewrite_tar(members[2][1], in_data, icons_before)
    if "entry" not in sums:
        raise SystemExit(f"{deb.name}: no {INSTALLED}")
    if len(sums) != 1 + len(small):
        raise SystemExit(f"{deb.name}: no usr/share/icons/hicolor/256x256 to put the smaller icons beside")

    def in_control(member, body):
        if member.name in ("control", "./control"):
            text = replace_once(body.decode("utf-8"), " Markitai Mygo is a desktop application.\n", long_description(), "control")
            body = text.encode("utf-8")
        if member.name in ("md5sums", "./md5sums"):
            lines = body.decode("utf-8").splitlines()
            at = [i for i, line in enumerate(lines) if line.endswith("  " + INSTALLED)]
            if len(at) != 1:
                raise SystemExit(f"{deb.name}: md5sums names {INSTALLED} {len(at)} times")
            lines[at[0]] = f"{sums['entry']}  usr/share/applications/{APP_ID}.desktop"
            big = [i for i, line in enumerate(lines) if line.endswith(f"  {icon_dir(256)}/{SLUG}.png")]
            if len(big) != 1:
                raise SystemExit(f"{deb.name}: md5sums names the 256 icon {len(big)} times")
            lines[big[0]:big[0]] = [f"{sums[px]}  {icon_dir(px)}/{SLUG}.png" for px, _ in small]
            body = ("\n".join(lines) + "\n").encode("utf-8")
        return member, body

    control = rewrite_tar(members[1][1], in_control)
    deb.write_bytes(write_ar([members[0], (members[1][0], control), (members[2][0], data)]))
