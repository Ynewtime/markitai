#!/usr/bin/env python3
"""Compare authored legacy state round trips with a native unit-test entrypoint."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from audit_reports import differing, guards, order_differences, sha, write


def repository_state(root):
    """Pin the checkout independently of the caller's cwd, including dirty bytes."""
    def git(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=root)
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0')
    return {'revision': git('rev-parse', 'HEAD').decode().strip(),
            'status': git('status', '--porcelain').decode(),
            'diff_sha256': hashlib.sha256(git('diff', '--binary', 'HEAD')).hexdigest(),
            'untracked_sha256': {os.fsdecode(path): sha(root / os.fsdecode(path))
                                 for path in untracked if path}}


def fixtures(root):
    """No input contents or user state are needed to exercise the state codec."""
    result = []
    flags = {key: False for key in ('llm', 'ocr', 'screenshot', 'alt', 'desc')}
    for name in ('minimal', 'detailed', 'named_urls', 'url_list', 'source_presence',
                 'relative_paths', 'missing_options', 'legacy_aliases', 'alias_paths',
                 'replay_valid', 'replay_syntax_unknown', 'replay_semantic_stop', 'replay_nulls'):
        work = root / name
        inputs, output = work / 'input', work / 'out'
        inputs.mkdir(parents=True); output.mkdir()
        (inputs / 'sub').mkdir(); (output / 'sub').mkdir()
        mode = 'url_list' if name in ('url_list', 'source_presence') else 'directory'
        source = inputs / 'links.urls' if mode == 'url_list' else inputs
        options = dict(flags, input_dir=str(inputs), output_dir=str(output))
        state = {'version': '1.0', 'options': options, 'documents': {}, 'urls': {}}
        hash_options = dict(flags)
        if mode == 'directory': hash_options['scan_max_depth'] = 5
        cwd = work
        if name in ('minimal', 'detailed', 'relative_paths', 'missing_options', 'legacy_aliases', 'alias_paths') or name.startswith('replay_'):
            state['documents'] = {
                'sub/笔记.txt': {'status': 'completed', 'output': str(output / 'sub/笔记.txt.md')},
                'failed.txt': {'status': 'failed', 'error': 'Fixture failure', 'target': str(output / 'failed.txt.md')},
                'pending.txt': {'status': 'pending', 'target': str(output / 'pending.txt.md')},
                'active.txt': {'status': 'in_progress', 'target': str(output / 'active.txt.md')},
                'skipped.png': {'status': 'completed'},
            }
        if name == 'detailed':
            state.update(started_at='2026-01-02T03:04:05+00:00', updated_at='2026-01-02T03:04:06+00:00')
            state['documents']['sub/笔记.txt'].update(duration=1.0, images=2, screenshots=0,
                cost_usd=0.125, llm_usage={'fixture': {'requests': 1}}, cache_hit=True,
                started_at='2026-01-02T03:04:05+00:00', completed_at='2026-01-02T03:04:06+00:00')
        if name in ('named_urls', 'url_list', 'source_presence'):
            # Deliberately reverse lexical order. JSON object order is part of this gate.
            state['urls'] = {
                'https://fixture.invalid/z first': {'url': 'https://fixture.invalid/z', 'source_file': str(inputs / 'links.urls'), 'status': 'completed', 'output': str(output / 'first.md')},
                'https://fixture.invalid/a first.md': {'url': 'https://fixture.invalid/a', 'source_file': str(inputs / 'links.urls'), 'status': 'in_progress', 'target': str(output / 'first.v2.md')},
            }
        if name == 'source_presence':
            state['urls'] = {
                'https://fixture.invalid/z': {'status': 'pending'},
                'https://fixture.invalid/a': {'status': 'pending', 'source_file': None},
            }
        if name == 'alias_paths':
            alias = work / 'out-alias'
            try:
                alias.symlink_to(output, target_is_directory=True)
            except OSError as error:
                # Directory junctions exercise the same aliased parent without
                # requiring Windows Developer Mode or the symlink privilege.
                if os.name != 'nt' or error.winerror != 1314:
                    raise
                subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(alias), str(output)],
                               check=True, stdout=subprocess.DEVNULL)
            for entry in state['documents'].values():
                for field in ('output', 'target'):
                    if field in entry: entry[field] = str(alias / Path(entry[field]).relative_to(output))
            output = alias
            options['output_dir'] = str(alias)
        if name == 'relative_paths':
            for entry in state['documents'].values():
                for field in ('output', 'target'):
                    if field in entry: entry[field] = os.path.relpath(entry[field], work)
            state['documents']['pending.txt']['target'] = 'out/sub/../pending.txt.md'
        if name == 'missing_options':
            state.pop('options')
            cwd = inputs
        if name == 'legacy_aliases':
            options['extra_z'] = 1
            options['extra_a'] = 2
            hash_options = {'llm_enabled': False, 'ocr_enabled': False, 'screenshot_enabled': False,
                            'image_alt_enabled': False, 'image_desc_enabled': False,
                            'scan_max_depth': 8, 'glob_patterns': ['*.txt', '!skip*'], 'not_hashed': 'fixture'}
        journal = None
        if name.startswith('replay_'):
            journal = work / 'fixture.jsonl'
            event = lambda key, data, kind='file': json.dumps({'type': kind, 'key': key, 'data': data}, ensure_ascii=False)
            completed = event('pending.txt', {'status': 'completed', 'output': str(output / 'pending.txt.md')})
            tail = event('active.txt', {'status': 'completed', 'output': str(output / 'active.txt.md')})
            if name == 'replay_valid':
                lines = [completed, event('failed.txt', {'error': None}), tail]
            elif name == 'replay_syntax_unknown':
                lines = ['  ', '{broken-json', event('absent.txt', 7), event('x', None, 'other'), completed, tail]
            elif name == 'replay_semantic_stop':
                lines = [completed, event('failed.txt', {'status': 'bad-status'}), tail]
            else:
                state['urls'] = {'https://fixture.invalid/z': {'status': 'failed', 'source_file': str(inputs / 'links.urls'),
                    'error': 'old error', 'target': str(output / 'url.md')}}
                lines = [event('sub/笔记.txt', {'status': 'completed', 'output': None}),
                         event('pending.txt', {'target': None}),
                         event('failed.txt', {'status': 'failed', 'error': None}),
                         event('https://fixture.invalid/z', {'status': 'completed', 'error': None,
                               'output': str(output / 'url.md')}, 'url')]
            journal.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        fixture = work / 'fixture.json'; write(fixture, state)
        result.append({'name': name, 'mode': mode, 'allow_symlinks': name == 'alias_paths', 'input': str(source), 'output': str(output),
                       'cwd': str(cwd), 'fixture': str(fixture), 'journal': str(journal) if journal else None,
                       'hash_options': hash_options})
    return result


def reference_worker(path):
    request = json.loads(path.read_text(encoding='utf-8'))
    state = guards(0)
    sys.path.insert(0, str(Path(request['reference']) / 'packages/markitai/src'))
    try:
        from markitai.batch import BatchProcessor, BatchState
        from markitai.json_order import order_state
        # The identity helper depends only on these three attributes, not configuration or workers.
        identity = object.__new__(BatchProcessor)
        identity.input_path = Path(request['input'])
        identity.output_dir = Path(request['output'])
        identity.task_options = request['hash_options']
        if request.get('journal'):
            identity.state_file = Path(request['fixture'])
            state_value = identity.load_state()
            if state_value is None: raise RuntimeError('Reference could not load authored legacy state')
        else:
            value = json.loads(Path(request['fixture']).read_text(encoding='utf-8'))
            state_value = BatchState.from_dict(value)
        snapshot = order_state(state_value.to_minimal_dict())
        write(request['result'], {'hash': identity._compute_task_hash(), 'snapshot': snapshot})
    finally:
        write(path.with_name('guard.json'), state)
    if state['blocked_state_events'] or state['blocked_network_events']:
        raise RuntimeError('Reference attempted protected state or networking')


def compare(reference, native):
    values = differing(reference, native)
    ordering = order_differences(reference, native)
    return {'equal': not values and not ordering, 'different_paths': values, 'different_order_paths': ordering}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--test-binary', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--require-parity', action='store_true')
    args = parser.parse_args()
    if args.worker: return reference_worker(args.worker)
    if not all((args.reference, args.test_binary, args.output)): parser.error('reference, test-binary and output are required')
    reference, binary, output = (p.resolve() for p in (args.reference, args.test_binary, args.output))
    protected = (Path.home() / '.markitai').resolve()
    if output.exists() or output == reference or reference in output.parents or output == protected or protected in output.parents:
        parser.error('Use a fresh private output directory outside the reference and real user state')
    python = reference / '.venv/bin/python'
    if not python.is_file() or not binary.is_file(): parser.error('Reference venv and native test binary must exist')
    output.mkdir(parents=True)
    frozen = output / 'state-test-binary'; shutil.copy2(binary, frozen)
    native_root = Path(__file__).resolve().parents[1]
    native_before = repository_state(native_root)
    reference_before = repository_state(reference)
    record = {'schema': 1, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'driver_cwd': str(Path.cwd()),
              'native_root': str(native_root), 'native_revision': native_before['revision'],
              'native_source_status': native_before['status'], 'native_source_before': native_before,
              'reference_revision': reference_before['revision'], 'reference_source_before': reference_before,
              'reference_status_before': reference_before['status'], 'binary_sha256': sha(frozen), 'script_sha256': sha(__file__),
              'helper_script_sha256': sha(Path(__file__).with_name('audit_reports.py')),
              'scope': 'Authored legacy snapshot and journal loading, minimal serialization and identity. Native unit-test binary, not production --resume. No conversion, providers, performance or cross-platform gate.',
              'cases': [], 'completed': False}
    write(output / 'report.json', record)
    try:
        for case in fixtures(output / 'cases'):
            before = sha(case['fixture'])
            journal_before = sha(case['journal']) if case['journal'] else None
            engines = {}
            for engine in ('reference', 'native'):
                private = Path(case['fixture']).parent / engine; private.mkdir()
                request = dict(case, reference=str(reference), result=str(private / 'result.json'))
                request_path = private / 'request.json'; write(request_path, request)
                env = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'LANG', 'LC_ALL', 'TMPDIR') if k in os.environ}
                env.update(MARKITAI_HOME=str(private / 'state'), PYTHON_DOTENV_DISABLED='1', PYTHONNOUSERSITE='1',
                           PYTHONDONTWRITEBYTECODE='1', LITELLM_LOCAL_MODEL_COST_MAP='True',
                           MARKITAI_STATE_AUDIT_REQUEST=str(request_path))
                command = [str(python), '-B', str(Path(__file__).resolve()), '--worker', str(request_path)] if engine == 'reference' else [str(frozen), '--ignored', '--exact', 'run_state::audit::legacy_fixture', '--nocapture']
                process = subprocess.run(command, cwd=case['cwd'], env=env, capture_output=True, timeout=60)
                (private / 'stdout').write_bytes(process.stdout); (private / 'stderr').write_bytes(process.stderr)
                if process.returncode: raise RuntimeError(f'{case["name"]}/{engine} exited {process.returncode}; inspect saved stderr')
                saved = Path(request['result'])
                engines[engine] = {'result': json.loads(saved.read_text(encoding='utf-8')), 'result_path': str(saved),
                                   'sha256': sha(saved), 'request_sha256': sha(request_path), 'command': command,
                                   'stdout_sha256': sha(private / 'stdout'), 'stderr_sha256': sha(private / 'stderr')}
                if engine == 'reference': engines[engine]['guard'] = json.loads((private / 'guard.json').read_text(encoding='utf-8'))
            if sha(case['fixture']) != before: raise RuntimeError('Authored state fixture changed')
            if case['journal'] and sha(case['journal']) != journal_before: raise RuntimeError('Authored journal fixture changed')
            verdict = compare(engines['reference']['result'], engines['native']['result'])
            record['cases'].append(dict(name=case['name'], fixture=case['fixture'], fixture_sha256=before,
                                       journal=case['journal'], journal_sha256=journal_before, engines=engines, **verdict))
            write(output / 'report.json', record)
            print(case['name'], 'equal' if verdict['equal'] else verdict, flush=True)
    except Exception as error:
        record['failure'] = str(error); write(output / 'report.json', record)
        raise
    if sha(frozen) != record['binary_sha256']: raise RuntimeError('Frozen test binary changed')
    reference_after = repository_state(reference)
    native_after = repository_state(native_root)
    if reference_after != reference_before: raise RuntimeError('Reference checkout changed')
    if native_after != native_before: raise RuntimeError('Native checkout changed during the audit')
    record.update(completed=True, all_equal=all(c['equal'] for c in record['cases']),
                  reference_status_after=reference_after['status'], reference_source_after=reference_after,
                  native_source_after=native_after,
                  finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
    write(output / 'report.json', record)
    if args.require_parity and not record['all_equal']: raise SystemExit(1)


if __name__ == '__main__':
    main()
