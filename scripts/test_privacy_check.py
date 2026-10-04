"""Privacy regression checks; all sample identities and secrets are invented."""
from pathlib import Path
import unittest

from privacy_check import check_repository, findings


class PrivacyCheckTests(unittest.TestCase):
    def test_personal_paths_are_rejected_without_echoing_values(self):
        for prefix in (b"/" + b"Users/", b"/" + b"home/", b"C:\\" + b"Users\\"):
            value = prefix + b"invented-private-person/project"
            result = findings(b"first line\n" + value)
            self.assertEqual(result, [("personal home path", 2)])
            self.assertNotIn("invented-private-person", repr(result))

    def test_synthetic_paths_and_portable_locations_are_allowed(self):
        for value in (b"/Users/example/work", b"/home/tester/work", b"C:\\Users\\User\\work",
                      b"${HOME}/work", b"${CARGO_HOME}/registry", b"crates/core/src/lib.rs"):
            self.assertEqual(findings(value), [])

    def test_mac_temporary_identity_is_rejected(self):
        self.assertEqual(findings(b"/var/" + b"folders/ab/0123456789/T/log"),
                         [("personal temporary path", 1)])

    def test_secret_shapes_are_rejected_without_echoing(self):
        for value in (b"ghp_" + b"x" * 36, b"-----BEGIN PRIVATE" + b" KEY-----"):
            self.assertEqual(findings(value), [("secret-shaped value", 1)])

    def test_tracked_files_do_not_contain_private_machine_paths(self):
        self.assertEqual(check_repository(Path(__file__).resolve().parents[1]), [])
