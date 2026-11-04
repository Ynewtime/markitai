"""desktop/strings.json from the copy table in desktop/DESIGN.md (section 10).

    python3.13 desktop/scripts/strings.py          write it
    python3.13 desktop/scripts/strings.py --check  fail when it differs
"""
from pathlib import Path
import json
import re
import sys

DESKTOP = Path(__file__).resolve().parents[1]
ROW = re.compile(r"^\| `([a-z0-9_.]+)` \|(.*)\|\s*$")


def cells(rest):
    parts = re.split(r"(?<!\\)\|", rest)
    return [part.strip().replace("\\|", "|") for part in parts]


def table(design):
    section = design.split("\n## 10. Copy\n", 1)[1].split("\n## 11.", 1)[0]
    strings = {}
    for line in section.splitlines():
        match = ROW.match(line)
        if not match:
            continue
        key, rest = match.groups()
        values = cells(rest)
        if len(values) != 2 or not all(values):
            raise SystemExit(f"DESIGN.md section 10: {key} needs an en and a zh-Hans text, has {values}")
        if key in strings:
            raise SystemExit(f"DESIGN.md section 10: {key} is listed twice")
        en, zh = values
        if set(re.findall(r"\{(\w+)\}", en)) != set(re.findall(r"\{(\w+)\}", zh)):
            raise SystemExit(f"DESIGN.md section 10: {key} has different placeholders in en and zh-Hans")
        strings[key] = {"en": en, "zh-Hans": zh}
    if not strings:
        raise SystemExit("DESIGN.md section 10 has no copy rows")
    return strings


def main():
    text = json.dumps(table((DESKTOP / "DESIGN.md").read_text(encoding="utf-8")), ensure_ascii=False, indent=1) + "\n"
    target = DESKTOP / "strings.json"
    if "--check" in sys.argv[1:]:
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            raise SystemExit("desktop/strings.json differs from DESIGN.md section 10; run desktop/scripts/strings.py")
        return
    target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
