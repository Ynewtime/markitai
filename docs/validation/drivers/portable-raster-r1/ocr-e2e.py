"""Scanned-PDF OCR end to end: CoreGraphics + Vision against hayro + Vision (macOS).

usage: <reference venv python> ocr-e2e.py [--all-pages] <markitai> <out> <scanned dir|-> [<pdf>|<list.txt>...]

<markitai> is a release CLI built with `--features markitai-core/portable-media`;
each PDF is converted twice by it, with MARKITAI_PDF_RENDERER=coregraphics and
=portable, so only the page renderer differs. <scanned dir> is make-scanned.py's
output (PDFs with ground truth and language); further PDFs (or .txt lists of
them) are converted with the default OCR language and have no ground truth.
--all-pages sets `ocr.per_page_routing: false`, so pages with native text are
rendered and recognized too: rendered text read by OCR, as Office page capture
does with LibreOffice's PDF.

Every run gets its own directory with private MARKITAI_HOME, TMPDIR and
configuration (OCR on, no LLM, screenshots, history or cache), and runs under
sandbox-exec with network access denied and writes limited to the run, the
private HOME and the per-user temporary area. One private HOME (<out>/home)
serves all runs: Vision compiles its recognition model into HOME's caches on
first use (about 25 s), and that compile fails inside the sandbox, so one
unscored warm-up conversion of the first PDF runs first without the sandbox
(same private environment, no model or network option). The recognized text of a page is
its Markdown between page markers, without frontmatter, headings that repeat
the file name, image links and HTML comments. Character error rate is edit
distance over the reference length after collapsing whitespace (removing it for
Chinese): portable against CoreGraphics per document and page, and each against
the ground truth where there is one. Needs rapidfuzz (the reference venv has it).
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

from rapidfuzz.distance import Levenshtein

MODES = ('coregraphics', 'portable')
PAGE = re.compile(r'<!--\s*Page number:\s*(\d+)\s*-->')


def sandbox(run, home):
    q = lambda value: json.dumps(str(value))
    profile = run / 'sandbox.sb'
    profile.write_text('\n'.join([
        '(version 1)', '(allow default)', '(deny network*)',
        '(deny file-write* (require-all (require-not (subpath ' + q(run.resolve()) + '))'
        ' (require-not (subpath ' + q(home.resolve()) + '))'
        ' (require-not (subpath "/private/var/folders")) (require-not (literal "/dev/null"))'
        ' (require-not (literal "/dev/tty"))))']) + '\n')
    return ['/usr/bin/sandbox-exec', '-f', str(profile)]


ALL_PAGES = False


def convert(binary, pdf, run, mode, lang, home, sandboxed=True):
    run.mkdir(parents=True, mode=0o700)
    for name in ('state', 'tmp', 'out'):
        (run / name).mkdir(mode=0o700)
    config = {'llm': {'enabled': False, 'pure': True},
              'ocr': {'enabled': True, **({'lang': lang} if lang else {}),
                      **({'per_page_routing': False} if ALL_PAGES else {})},
              'screenshot': {'enabled': False, 'screenshot_only': False},
              'image': {'alt_enabled': False, 'desc_enabled': False},
              'cache': {'enabled': False, 'global_dir': str(run / 'state/cache')},
              'history': {'record': False}, 'output': {'report': False, 'on_conflict': 'overwrite'}}
    (run / 'config.json').write_text(json.dumps(config))
    local = run / pdf.name
    shutil.copy2(pdf, local)
    env = {'PATH': os.defpath, 'HOME': str(home), 'TMPDIR': str(run / 'tmp'),
           'LANG': 'en_US.UTF-8', 'LC_ALL': 'en_US.UTF-8', 'TZ': 'UTC', 'NO_COLOR': '1', 'TERM': 'dumb',
           'MARKITAI_HOME': str(run / 'state'), 'MARKITAI_CONFIG': str(run / 'config.json'),
           'MARKITAI_LOG_DIR': str(run / 'state/logs'), 'MARKITAI_PDF_RENDERER': mode}
    command = (sandbox(run, home) if sandboxed else []) + [
        str(binary), str(local), '-o', str(run / 'out'), '--ocr', '--no-llm', '--no-screenshot',
        '--no-alt', '--no-desc', '--no-record-history', '--config', str(run / 'config.json')]
    started = time.perf_counter()
    done = subprocess.run(command, env=env, cwd=run, capture_output=True, timeout=900)
    elapsed = time.perf_counter() - started
    (run / 'stdout').write_bytes(done.stdout)
    (run / 'stderr').write_bytes(done.stderr)
    outputs = sorted((run / 'out').glob('*.md'))
    markdown = outputs[0].read_text() if done.returncode == 0 and len(outputs) == 1 else None
    return {'exit': done.returncode, 'seconds': elapsed, 'markdown': markdown}


def page_texts(markdown, stem):
    body = markdown
    if body.startswith('---\n'):
        end = body.find('\n---\n', 4)
        body = body[end + 5:] if end >= 0 else body
    parts = PAGE.split(body)
    pages = [parts[index + 1] for index in range(1, len(parts) - 1, 2)] if len(parts) > 1 else [body]
    cleaned = []
    for text in pages:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('#') and stem.lower() in stripped.lower():
                continue
            if re.fullmatch(r'!\[[^\]]*\]\([^)]*\)', stripped) or re.fullmatch(r'<!--.*-->', stripped):
                continue
            lines.append(line)
        cleaned.append('\n'.join(lines))
    return cleaned


def normal(text, lang):
    text = re.sub(r'\s+', '' if lang == 'zh' else ' ', text).strip()
    return text


def cer(hypothesis, reference):
    return Levenshtein.distance(hypothesis, reference) / max(1, len(reference))


def main(binary, out, scanned, extra):
    binary, out = Path(binary).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    manifest = json.loads((Path(scanned) / 'manifest.json').read_text())['rows'] if scanned else []
    for row in manifest:
        jobs.append((Path(scanned) / row['pdf'], row['lang'], [' '.join(page['truth']) if row['lang'] != 'zh'
                     else ''.join(page['truth']) for page in row['pages']]))
    for item in extra:
        item = Path(item).resolve()
        listed = [Path(line.strip()) for line in item.read_text().splitlines() if line.strip()] \
            if item.suffix == '.txt' else [item]
        jobs.extend((pdf, None, None) for pdf in listed)
    home = out / 'home'
    home.mkdir(mode=0o700)
    warm = convert(binary, jobs[0][0], out / 'warm-up', MODES[0], jobs[0][1], home, sandboxed=False)
    print(json.dumps({'warm_up': str(jobs[0][0]), 'exit': warm['exit'], 'seconds': round(warm['seconds'], 2)}), flush=True)
    results = []
    for number, (pdf, lang, truth) in enumerate(jobs):
        record = {'pdf': str(pdf), 'lang': lang}
        texts = {}
        for mode in MODES:
            done = convert(binary, pdf, out / f'{number:03d}-{pdf.stem}' / mode, mode, lang, home)
            record[mode] = {'exit': done['exit'], 'seconds': round(done['seconds'], 3)}
            if done['markdown'] is not None:
                texts[mode] = [normal(text, lang) for text in page_texts(done['markdown'], pdf.stem)]
        if len(texts) == 2:
            first, second = texts['coregraphics'], texts['portable']
            record['pages'] = [len(first), len(second)]
            record['chars'] = len(''.join(first))
            record['edits_portable_vs_coregraphics'] = Levenshtein.distance(''.join(second), ''.join(first))
            record['cer_portable_vs_coregraphics'] = cer(''.join(second), ''.join(first))
            record['page_cer_portable_vs_coregraphics'] = [cer(b, a) for a, b in zip(first, second)]
            record['identical'] = first == second
            if truth:
                reference = ''.join(normal(page, lang) for page in truth)
                record['truth_chars'] = len(reference)
                for mode in MODES:
                    record[f'edits_{mode}_vs_truth'] = Levenshtein.distance(''.join(texts[mode]), reference)
                    record[f'cer_{mode}_vs_truth'] = cer(''.join(texts[mode]), reference)
        results.append(record)
        print(json.dumps({key: value for key, value in record.items() if not key.startswith('page_cer')}), flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    scored = [row for row in results if 'cer_portable_vs_coregraphics' in row]
    total = lambda key: sum(row[key] for row in scored if key in row)
    truthful = [row for row in scored if 'truth_chars' in row]
    summary = {
        'documents': len(results), 'compared': len(scored),
        'failed': [row['pdf'] for row in results if row not in scored],
        'identical': sum(1 for row in scored if row['identical']),
        'max_cer_portable_vs_coregraphics': max((row['cer_portable_vs_coregraphics'] for row in scored), default=None),
        'mean_cer_portable_vs_coregraphics': total('cer_portable_vs_coregraphics') / max(1, len(scored)),
        'pooled_cer_portable_vs_coregraphics': total('edits_portable_vs_coregraphics') / max(1, total('chars')),
        'pooled_cer_vs_truth': {mode: sum(row[f'edits_{mode}_vs_truth'] for row in truthful)
                                / max(1, sum(row['truth_chars'] for row in truthful)) for mode in MODES},
        'seconds': {mode: round(sum(row[mode]['seconds'] for row in results), 2) for mode in MODES},
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    arguments = sys.argv[1:]
    if arguments[:1] == ['--all-pages']:
        ALL_PAGES = True
        arguments = arguments[1:]
    if len(arguments) < 3:
        raise SystemExit(__doc__)
    main(arguments[0], arguments[1], arguments[2] if arguments[2] != '-' else None, arguments[3:])
