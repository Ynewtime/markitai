#!/usr/bin/env python3
"""Compare four persisted CLI report schemas using private state and loopback HTTP."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import threading

from isolation import install_state_guard


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def guards(port):
    def loopback(event, args):
        if event == 'socket.getaddrinfo':
            return args[0] == '127.0.0.1' and args[1] == port
        if event == 'socket.connect':
            address = args[1]
            return isinstance(address, tuple) and address[:2] == ('127.0.0.1', port)
        return False
    return install_state_guard('Python path/network audit events; not an OS sandbox or native syscall monitor',
                               loopback)


def reference_worker(request_path):
    request = json.loads(request_path.read_text(encoding='utf-8'))
    state = guards(request['port'])
    hard_exit = os._exit
    def audited_exit(code):
        # Keep the reference's own shutdown/flush path, then record the guard
        # before its final os._exit bypasses Python finally/atexit handlers.
        write(request_path.with_name('guard.json'), state)
        hard_exit(99 if state['blocked_state_events'] or state['blocked_network_events'] else code)
    os._exit = audited_exit
    sys.path.insert(0, str(Path(request['reference']) / 'packages/markitai/src'))
    sys.argv = ['markitai', *request['argv']]
    code = 0
    try:
        runpy.run_module('markitai', run_name='__main__')
    except SystemExit as error:
        code = error.code or 0
    finally:
        write(request_path.with_name('guard.json'), state)
    if state['blocked_state_events'] or state['blocked_network_events']:
        raise RuntimeError('Reference attempted protected state or unexpected networking')
    raise SystemExit(code)


def normalize(report, sandbox, origin):
    result = copy.deepcopy(report)
    def path(value):
        prefix = str(sandbox)
        if isinstance(value, str) and (value == prefix or value.startswith(prefix + os.sep)):
            return '<ROOT>' + value[len(prefix):]
        return value
    def clock(obj, keys):
        for key in keys:
            if key not in obj or obj[key] is None: continue
            value = obj[key]
            if not isinstance(value, str): raise ValueError('Timestamp type changed')
            parsed = dt.datetime.fromisoformat(value)
            if parsed.utcoffset() is None: raise ValueError('Timestamp has no UTC offset')
            obj[key] = '<TIME>'
    def duration(obj, keys):
        for key in keys:
            if key not in obj or obj[key] is None: continue
            value = obj[key]
            if not isinstance(value, str) or not re.fullmatch(r'(?:\d+\.\ds|\d{2,}:\d{2}(?::\d{2})?)', value):
                raise ValueError(f'Unexpected report duration: {value!r}')
            obj[key] = '<DURATION>'
    def item(entry):
        clock(entry, ('started_at', 'completed_at'))
        duration(entry, ('duration',))
        if 'output' in entry:
            original = entry['output']
            if original is not None and entry.get('status') == 'completed':
                target = Path(original)
                if not target.is_absolute(): target = sandbox / target
                if not target.is_file(): raise ValueError('Successful report points at a missing output')
            entry['output'] = path(original)
    clock(result, ('generated_at', 'started_at', 'updated_at'))
    if 'log_file' in result: result['log_file'] = path(result['log_file'])
    for key in ('input_dir', 'output_dir'):
        if key in result.get('options', {}): result['options'][key] = path(result['options'][key])
    if 'summary' in result: duration(result['summary'], ('duration', 'processing_time'))
    for entry in result.get('documents', {}).values(): item(entry)
    groups = result.get('url_sources')
    if groups is not None:
        grouped = {}
        for source, group in groups.items():
            urls = {}
            for key, entry in group['urls'].items():
                item(entry)
                new_key = '<HTTP>' + key[len(origin):] if key.startswith(origin) else key
                urls[new_key] = entry
            group['urls'] = urls
            grouped[path(source)] = group
        result['url_sources'] = grouped
    return result


def differing(left, right, path='$'):
    if type(left) is not type(right): return [path]
    if isinstance(left, dict):
        return [p for key in sorted(left.keys() | right.keys()) for p in
                ([path + '.' + key] if key not in left or key not in right else differing(left[key], right[key], path + '.' + key))]
    if isinstance(left, list):
        if len(left) != len(right): return [path]
        return [p for i, (a, b) in enumerate(zip(left, right)) for p in differing(a, b, f'{path}[{i}]')]
    return [] if left == right else [path]


def order_differences(left, right, path='$'):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = [path] if set(left) == set(right) and list(left) != list(right) else []
        for key in left.keys() & right.keys():
            paths.extend(order_differences(left[key], right[key], path + '.' + key))
        return sorted(paths)
    if isinstance(left, list) and isinstance(right, list):
        return [p for index, (a, b) in enumerate(zip(left, right))
                for p in order_differences(a, b, f'{path}[{index}]')]
    return []


class Fixture(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'<html><head><title>Report fixture</title></head><body><main><h1>Report fixture</h1><p>One stable paragraph.</p></main></body></html>'
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_args): pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--require-parity', action='store_true')
    args = parser.parse_args()
    if args.worker: return reference_worker(args.worker)
    if not all((args.reference, args.binary, args.output)): parser.error('reference, binary and output are required')
    reference, binary, output = (p.resolve() for p in (args.reference, args.binary, args.output))
    if output.exists(): parser.error('Use a fresh output directory')
    if output == reference or reference in output.parents: parser.error('Reference checkout is read-only')
    python = reference / '.venv/bin/python'  # Preserve the venv entrypoint symlink.
    if not binary.is_file() or not python.is_file(): parser.error('Build CLI and provide reference venv')
    protected = (Path.home()/'.markitai').resolve()
    if output == protected or protected in output.parents: parser.error('Real Markitai state is read-only')
    output.mkdir(parents=True)
    frozen = output / 'markitai'
    shutil.copy2(binary, frozen)
    before = subprocess.check_output(['git', 'status', '--porcelain'], cwd=reference, text=True)
    report = {'schema': 1, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'native_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'native_source_status': subprocess.check_output(['git', 'status', '--porcelain'], text=True),
              'reference_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=reference, text=True).strip(),
              'reference_status_before': before, 'binary_sha256': sha(frozen), 'script_sha256': sha(__file__),
              'scope': 'Four success-path CLI reports, no models/OCR/history/cache. Loopback HTTP only; guards are Python audit hooks, not an OS sandbox.',
              'cases': [], 'completed': False}
    write(output/'report.json', report)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    try:
        for name in ('single_file', 'directory', 'single_url', 'url_list'):
            saved = {}
            for engine in ('reference', 'native'):
                sandbox = output/name/engine
                sandbox.mkdir(parents=True)
                inputs = sandbox/'input'; inputs.mkdir()
                out = sandbox/'out'
                (inputs/'note.txt').write_text('# Report fixture\n\nStable content.\n')
                (inputs/'links.urls').write_text(f'{origin}/one first\n{origin}/two second\n')
                options = {key: False for key in ('llm', 'ocr', 'screenshot', 'alt', 'desc')}
                config = {'llm': {'enabled': False, 'pure': True}, 'ocr': {'enabled': False},
                          'screenshot': {'enabled': False}, 'image': {'alt_enabled': False, 'desc_enabled': False, 'stdout_fetch_external': False},
                          'history': {'record': False}, 'cache': {'enabled': False, 'global_dir': str(sandbox/'state/cache')},
                          'log': {'dir': None}, 'batch': {'concurrency': 1, 'url_concurrency': 1, 'scan_max_depth': 8},
                          'fetch': {'strategy': 'static', 'no_remote': True, 'remote_consent': 'never'}}
                if name.startswith('single_'): config['output'] = {'report': True}
                cfg_path = sandbox/'config.json'; write(cfg_path, config)
                if name == 'single_file': source = str(inputs/'note.txt'); target = out; identity = source
                elif name == 'directory':
                    (inputs/'links.urls').unlink()
                    source = str(inputs); target = out; identity = source
                    options['scan_max_depth'] = 8
                elif name == 'single_url':
                    source = origin+'/single'; target = out/'chosen.md'; identity = str(out)
                    options = {'llm': False}
                else:
                    source = str(inputs/'links.urls'); target = out; identity = str(out)
                    options = {'llm': False, 'alt': False, 'desc': False}
                input_hashes = {str(path): sha(path) for path in inputs.iterdir()}
                argv = [source, '-o', str(target), '-c', str(cfg_path), '--json', '--quiet']
                env = {k: os.environ[k] for k in ('PATH','SYSTEMROOT','LANG','LC_ALL','TMPDIR') if k in os.environ}
                env.update(MARKITAI_HOME=str(sandbox/'state'), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                           PYTHON_DOTENV_DISABLED='1', LITELLM_LOCAL_MODEL_COST_MAP='True', NO_PROXY='127.0.0.1,localhost')
                command = [str(frozen), *argv]
                request = sandbox/'request.json'
                if engine == 'reference':
                    write(request, {'reference': str(reference), 'port': server.server_port, 'argv': argv})
                    command = [str(python), '-B', str(Path(__file__).resolve()), '--worker', str(request)]
                result = subprocess.run(command, cwd=sandbox, env=env, capture_output=True, timeout=120)
                (sandbox/'stdout').write_bytes(result.stdout); (sandbox/'stderr').write_bytes(result.stderr)
                if result.returncode: raise RuntimeError(f'{name}/{engine} exited {result.returncode}; inspect saved stderr')
                if any(sha(path) != digest for path, digest in input_hashes.items()): raise RuntimeError('Source fixture changed')
                json.loads(result.stdout)  # Exactly one complete JSON value.
                reports = list((out/'.markitai/reports').glob('*.report.json'))
                if len(reports) != 1: raise RuntimeError(f'{name}/{engine}: expected one report, got {len(reports)}')
                identity_data = {'input': str(Path(identity).resolve()), 'output': str(out.resolve()), 'options': options}
                expected_hash = hashlib.md5(json.dumps(identity_data, sort_keys=True).encode(), usedforsecurity=False).hexdigest()[:6]
                if reports[0].name != f'markitai.{expected_hash}.report.json': raise RuntimeError(f'{name}/{engine}: task hash differs')
                raw = json.loads(reports[0].read_text(encoding='utf-8'))
                normalized = normalize(raw, sandbox, origin)
                write(sandbox/'normalized.json', normalized)
                saved[engine] = {'report': str(reports[0]), 'sha256': sha(reports[0]), 'normalized': normalized,
                                 'top_level_order': list(raw), 'stdout_sha256': sha(sandbox/'stdout'),
                                 'input_hashes': input_hashes, 'config_sha256': sha(cfg_path), 'argv': argv}
                if engine == 'reference': saved[engine]['guard'] = json.loads((sandbox/'guard.json').read_text(encoding='utf-8'))
            paths = differing(saved['reference']['normalized'], saved['native']['normalized'])
            ordering = order_differences(saved['reference']['normalized'], saved['native']['normalized'])
            report['cases'].append({'name': name, 'equal': not paths and not ordering, 'different_paths': paths, 'different_order_paths': ordering,
                                    'top_level_order_equal': saved['reference']['top_level_order'] == saved['native']['top_level_order'], 'engines': saved})
            write(output/'report.json', report)
            print(name, 'equal' if not paths and not ordering else {'values': paths, 'ordering': ordering}, flush=True)
    except Exception as error:
        report['failure'] = str(error)
        write(output/'report.json', report)
        raise
    finally:
        server.shutdown(); server.server_close(); thread.join()
    if sha(frozen) != report['binary_sha256']: raise RuntimeError('Frozen binary changed')
    after = subprocess.check_output(['git', 'status', '--porcelain'], cwd=reference, text=True)
    if after != before: raise RuntimeError('Reference checkout changed')
    report['reference_status_after'] = after
    report['completed'] = True
    report['all_equal'] = all(case['equal'] for case in report['cases'])
    report['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write(output/'report.json', report)
    if args.require_parity and not report['all_equal']: raise SystemExit(1)


if __name__ == '__main__':
    main()
