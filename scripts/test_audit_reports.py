import copy
import os
import tempfile
import unittest
from pathlib import Path

from audit_reports import differing, normalize, order_differences


class ReportEvidenceTests(unittest.TestCase):
    def test_only_schema_clock_fields_and_exact_root_prefix_are_normalized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'out.md'
            output.write_text('body')
            source = {
                'generated_at': '2026-09-28T12:00:00+08:00',
                'summary': {'duration': '0.1s', 'processing_time': '01:02'},
                'options': {'input_dir': str(root/'input'), 'extra': str(root/'keep')},
                'documents': {'item': {'status': 'completed', 'output': str(output),
                                      'duration': '0.1s', 'llm_usage': {'duration': 42}}},
                'extra': {'generated_at': 'not a clock', 'root': str(root)+'-other'},
            }
            original = copy.deepcopy(source)
            result = normalize(source, root, 'http://127.0.0.1:1')
            self.assertEqual(source, original)
            self.assertEqual(result['generated_at'], '<TIME>')
            self.assertEqual(result['documents']['item']['output'], '<ROOT>' + os.sep + 'out.md')
            self.assertEqual(result['documents']['item']['llm_usage'], {'duration': 42})
            self.assertEqual(result['extra'], source['extra'])
            self.assertEqual(result['options']['extra'], str(root/'keep'))
            relative = {'documents': {'a': {'status': 'completed', 'output': 'out.md'}}}
            self.assertEqual(normalize(relative, root, ''), relative)

    def test_unexpected_fields_json_types_and_null_are_not_hidden(self):
        self.assertEqual(differing({'x': True}, {'x': 1}), ['$.x'])
        self.assertEqual(differing({'x': None}, {}), ['$.x'])
        self.assertEqual(differing({'x': []}, {'x': {}}), ['$.x'])
        self.assertEqual(differing({'usage': {'tokens': 2}}, {'usage': {'tokens': 3}}), ['$.usage.tokens'])

    def test_nested_order_is_distinct_from_semantic_equality(self):
        left = {'group': {'first': 1, 'second': 2}}
        right = {'group': {'second': 2, 'first': 1}}
        self.assertEqual(differing(left, right), [])
        self.assertEqual(order_differences(left, right), ['$.group'])
        self.assertEqual(order_differences({'x': 1}, {'y': 1}), [])

    def test_missing_output_and_invalid_clock_or_duration_reject_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for value in (
                {'documents': {'a': {'status': 'completed', 'output': str(root/'missing')}}},
                {'generated_at': 123}, {'generated_at': 'yesterday'}, {'generated_at': '2026-09-28T12:00:00'},
                {'summary': {'duration': 1.25}}, {'summary': {'duration': 'quick'}},
            ):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    normalize(value, root, 'http://127.0.0.1:1')
            self.assertEqual(normalize({'summary': {'duration': None}}, root, ''),
                             {'summary': {'duration': None}})

    def test_url_custom_name_and_unknown_source_shape_survive(self):
        origin = 'http://127.0.0.1:321'
        value = {'url_sources': {'unknown.urls': {'total': 1, 'urls': {
            origin+'/path?x=12#part name.md': {'status': 'failed', 'error': 'failure 321', 'usage': 12}
        }}}}
        actual = normalize(value, Path('/isolated/root'), origin)
        item = actual['url_sources']['unknown.urls']['urls']['<HTTP>/path?x=12#part name.md']
        self.assertEqual(item, {'status': 'failed', 'error': 'failure 321', 'usage': 12})
        self.assertEqual(actual['url_sources']['unknown.urls']['total'], 1)


if __name__ == '__main__':
    unittest.main()
