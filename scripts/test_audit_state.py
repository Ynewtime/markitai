import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from audit_state import compare, fixtures, repository_state


class StateAuditTests(unittest.TestCase):
    def test_provenance_uses_explicit_checkout_and_detects_same_status_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)
            git('init', '-q')
            tracked = root / 'tracked.txt'; tracked.write_text('initial')
            git('add', 'tracked.txt')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
            self.assertEqual(repository_state(root)['revision'], git('rev-parse', 'HEAD').decode().strip())
            tracked.write_text('first edit')
            before = repository_state(root)
            tracked.write_text('second edit')
            after = repository_state(root)
            self.assertEqual(before['status'], after['status'])
            self.assertNotEqual(before['diff_sha256'], after['diff_sha256'])
            untracked = root / 'new.rs'; untracked.write_text('first')
            before = repository_state(root)
            untracked.write_text('second')
            after = repository_state(root)
            self.assertEqual(before['status'], after['status'])
            self.assertNotEqual(before['untracked_sha256'], after['untracked_sha256'])

    def test_comparison_preserves_types_omissions_null_and_url_order(self):
        reference = {'snapshot': {'urls': {'z': {'source_file': None}, 'a': {'requests': 0}}}}
        self.assertTrue(compare(reference, reference)['equal'])
        reordered = {'snapshot': {'urls': {'a': {'requests': 0}, 'z': {'source_file': None}}}}
        result = compare(reference, reordered)
        self.assertFalse(result['equal'])
        self.assertEqual(result['different_paths'], [])
        self.assertEqual(result['different_order_paths'], ['$.snapshot.urls'])
        for changed in ({'z': {}, 'a': {'requests': 0}},
                        {'z': {'source_file': ''}, 'a': {'requests': 0}},
                        {'z': {'source_file': None}, 'a': {'requests': False}},
                        {'z': {'source_file': None}, 'a': {'requests': 0.0}}):
            self.assertFalse(compare(reference, {'snapshot': {'urls': changed}})['equal'])

    def test_fixtures_exercise_order_presence_and_both_state_hash_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            cases = fixtures(Path(temporary) / 'cases')
            self.assertEqual(len(cases), 13)
            states = {case['name']: json.loads(Path(case['fixture']).read_text(encoding='utf-8')) for case in cases}
            self.assertTrue(next(iter(states['named_urls']['urls'])).endswith('/z first'))
            presence = list(states['source_presence']['urls'].values())
            self.assertNotIn('source_file', presence[0])
            self.assertIsNone(presence[1]['source_file'])
            self.assertNotIn('options', states['missing_options'])
            alias_output = Path(states['alias_paths']['options']['output_dir'])
            self.assertTrue(alias_output.samefile(alias_output.parent / 'out'))
            self.assertTrue(any(case['mode'] == 'url_list' and case['input'].endswith('links.urls') for case in cases))
            self.assertFalse(Path(states['relative_paths']['documents']['pending.txt']['target']).is_absolute())
            replay = {case['name']: Path(case['journal']).read_text(encoding='utf-8') for case in cases if case['journal']}
            self.assertEqual(len(replay), 4)
            self.assertIn('{broken-json', replay['replay_syntax_unknown'])
            self.assertIn('"data": 7', replay['replay_syntax_unknown'])
            self.assertIn('bad-status', replay['replay_semantic_stop'])
            self.assertIn('"output": null', replay['replay_nulls'])


if __name__ == '__main__':
    unittest.main()
