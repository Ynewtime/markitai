"""Render PDFs with CoreGraphics and with hayro and compare every page (macOS).

usage: python3 compare.py [--dpi N] [--pages N] [--step N] [--cargo-config K=V]...
                          [--summarize] <out> <group>=<dir or file>...

--step is the pixel step of the saved pictures (1 = full resolution);
--cargo-config passes `cargo --config K=V`, e.g. a profile opt-level override.

Each <group> names a corpus; directories are searched recursively for *.pdf
and a .txt file lists one PDF path per line.
Inputs are de-duplicated by SHA-256 (first group wins) and listed with their
hashes in <out>/inputs.json. The renderers run in markitai-core's ignored test
`pdf_raster::comparison` (release profile, `portable-media` feature, one test
thread, MARKITAI_HOME inside <out>); it writes documents.jsonl, pages.jsonl and
side-by-side PNGs of dissimilar pages (CoreGraphics | hayro | difference) under
<out>/raw. --summarize only re-reads an existing <out>/raw.

The summary (<out>/summary.json, also printed) gives per group: documents and
pages, open/render failures by cause for each backend, page-count and size
mismatches, similarity (SSIM over 8x8 luma blocks, all and ink-bearing
blocks; ink intersection over union in 3x3 cells; mean absolute channel
difference; hayro's ink share over CoreGraphics', below 1 when hayro draws
lighter) as quantiles, pages under 0.85 ink SSIM or ink IoU, and render
time per page for each backend (open time per document separately: the first
CoreGraphics open includes loading the framework).

Environment: CARGO_TARGET_DIR (default <repo>/.local/w2a/target-compare).
Diagnostic only: CoreGraphics is the reference here, not ground truth.
"""
import hashlib, json, os, statistics, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
BACKENDS = ('coregraphics', 'portable')


def collect(specs):
    rows, seen = [], set()
    for spec in specs:
        group, _, location = spec.partition('=')
        if not location:
            raise SystemExit(f'expected <group>=<path>, got {spec!r}')
        location = Path(location).resolve()
        if location.is_dir():
            paths = sorted(location.rglob('*.pdf'))
        elif location.suffix == '.txt':
            paths = [Path(line.strip()).resolve() for line in location.read_text().splitlines() if line.strip()]
        else:
            paths = [location]
        for path in paths:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append({'group': group, 'path': str(path), 'sha256': digest, 'bytes': len(data)})
    return rows


def run(out, rows, dpi, pages, step, cargo_config):
    (out / 'inputs.txt').write_text(''.join(row['path'] + '\n' for row in rows))
    home = out / 'home'
    home.mkdir(mode=0o700)
    env = dict(os.environ,
               MARKITAI_HOME=str(home / 'markitai'),
               MARKITAI_RASTER_COMPARE_INPUTS=str(out / 'inputs.txt'),
               MARKITAI_RASTER_COMPARE_OUT=str(out / 'raw'),
               MARKITAI_RASTER_COMPARE_DPI=str(dpi),
               MARKITAI_RASTER_COMPARE_PAGES=str(pages),
               MARKITAI_RASTER_COMPARE_PICTURE_STEP=str(step))
    env.setdefault('CARGO_TARGET_DIR', str(REPO / '.local/w2a/target-compare'))
    env.pop('MARKITAI_PDF_RENDERER', None)
    command = ['cargo', *[part for value in cargo_config for part in ('--config', value)],
               'test', '--release', '-p', 'markitai-core', '--features', 'portable-media',
               '--lib', 'pdf_raster::comparison::coregraphics_and_portable_rendering_of_a_corpus',
               '--', '--ignored', '--exact', '--test-threads', '1']
    (out / 'command.json').write_text(json.dumps({'command': command, 'cargo_target_dir': env['CARGO_TARGET_DIR']}) + '\n')
    with open(out / 'cargo.log', 'wb') as log:
        done = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if done.returncode:
        raise SystemExit(f'comparison test failed ({done.returncode}); see {out / "cargo.log"}')


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]
    return {'n': len(values), 'min': values[0], 'p10': pick(0.10), 'median': pick(0.5),
            'mean': statistics.fmean(values), 'max': values[-1]}


def summarize(out):
    inputs = {row['path']: row for row in json.loads((out / 'inputs.json').read_text())}
    documents = [json.loads(line) for line in (out / 'raw/documents.jsonl').read_text().splitlines()]
    pages = [json.loads(line) for line in (out / 'raw/pages.jsonl').read_text().splitlines()]
    by_group = defaultdict(lambda: {'documents': [], 'pages': []})
    for row in documents:
        by_group[inputs[row['input']]['group']]['documents'].append(row)
    for row in pages:
        by_group[inputs[row['input']]['group']]['pages'].append(row)
    summary = {}
    for group, found in sorted(by_group.items()):
        docs, rendered = found['documents'], found['pages']
        entry = {'documents': len(docs), 'pages_compared': len(rendered)}
        for backend in BACKENDS:
            entry[backend] = {
                'open_failures': dict(Counter(row[backend]['cause'] for row in docs if 'error' in row[backend])),
                'render_failures': dict(Counter(row[backend]['cause'] for row in rendered if 'error' in row[backend])),
                'open_ms': quantiles([row[backend]['open_ms'] for row in docs]),
                'render_ms': quantiles([row[backend]['ms'] for row in rendered if 'error' not in row[backend]]),
                'render_ms_total': sum(row[backend]['ms'] for row in rendered if 'error' not in row[backend]),
                'pages_in_documents': sum(row[backend].get('pages', 0) for row in docs),
            }
        entry['page_count_mismatches'] = [row['input'] for row in docs if row.get('page_count_mismatch')]
        entry['size_mismatches'] = [[row['input'], row['page']] for row in rendered if row.get('size_mismatch')]
        for metric in ('ssim', 'ssim_ink', 'ink_iou', 'mean_abs', 'differing'):
            entry[metric] = quantiles([row[metric] for row in rendered if row.get(metric) is not None])
        entry['ink_ratio'] = quantiles([row['ink'][1] / row['ink'][0] for row in rendered
                                        if row.get('ink') and row['ink'][0] > 0])
        entry['dissimilar_pages'] = [
            {key: row.get(key) for key in ('input', 'page', 'ssim_ink', 'ink_iou', 'picture')}
            for row in rendered if row.get('picture')]
        entry['portable_warnings'] = dict(sum((Counter(row.get('portable_warnings', {})) for row in docs), Counter()))
        failures = []
        for row in docs:
            for backend in BACKENDS:
                if 'error' in row[backend]:
                    failures.append({'input': row['input'], 'backend': backend, 'stage': 'open',
                                     'cause': row[backend]['cause'], 'error': row[backend]['error']})
        for row in rendered:
            for backend in BACKENDS:
                if 'error' in row[backend]:
                    failures.append({'input': row['input'], 'page': row['page'], 'backend': backend,
                                     'stage': 'render', 'cause': row[backend]['cause'], 'error': row[backend]['error']})
        entry['failures'] = failures
        summary[group] = entry
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    for group, entry in summary.items():
        print(f'== {group}: {entry["documents"]} documents, {entry["pages_compared"]} pages compared')
        for backend in BACKENDS:
            b = entry[backend]
            render = b['render_ms'] or {}
            print(f'  {backend:12} open failures {b["open_failures"]} render failures {b["render_failures"]}'
                  f' render ms median {render.get("median", 0):.1f} mean {render.get("mean", 0):.1f}'
                  f' total {b["render_ms_total"] / 1e3:.2f} s')
        for metric in ('ssim', 'ssim_ink', 'ink_iou', 'mean_abs', 'ink_ratio'):
            q = entry[metric]
            if q:
                print(f'  {metric:9} min {q["min"]:.4f} p10 {q["p10"]:.4f} median {q["median"]:.4f} mean {q["mean"]:.4f}')
        print(f'  page-count mismatches {len(entry["page_count_mismatches"])}, size mismatches'
              f' {len(entry["size_mismatches"])}, dissimilar pages {len(entry["dissimilar_pages"])},'
              f' hayro warnings {entry["portable_warnings"]}')


def main(argv):
    dpi, pages, step, cargo_config, only_summary = 150, 40, 2, [], False
    while argv and argv[0].startswith('--'):
        flag = argv.pop(0)
        if flag == '--dpi':
            dpi = float(argv.pop(0))
        elif flag == '--pages':
            pages = int(argv.pop(0))
        elif flag == '--step':
            step = int(argv.pop(0))
        elif flag == '--cargo-config':
            cargo_config.append(argv.pop(0))
        elif flag == '--summarize':
            only_summary = True
        else:
            raise SystemExit(__doc__)
    if not argv:
        raise SystemExit(__doc__)
    out = Path(argv.pop(0)).resolve()
    if not only_summary:
        rows = collect(argv)
        out.mkdir(parents=True, exist_ok=False)
        (out / 'inputs.json').write_text(json.dumps(rows, indent=1) + '\n')
        run(out, rows, dpi, pages, step, cargo_config)
    summarize(out)


if __name__ == '__main__':
    main(sys.argv[1:])
