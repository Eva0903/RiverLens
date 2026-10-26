from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class PublicReleaseTests(unittest.TestCase):
    def test_publish_audit_rejects_observations_even_when_renamed(self):
        audit = module("release_audit", ROOT / "scripts/audit_public_release.py")
        self.assertTrue(audit.inspect("data/raw/observation.json", b"{}"))
        self.assertTrue(audit.inspect("results/predictions.md", b"rows"))
        self.assertTrue(audit.inspect("table.xlsx", b"PK"))
        self.assertFalse(audit.inspect("data/raw/README.md", b"No data distributed."))

    def test_publish_audit_detects_credentials_without_showing_values(self):
        audit = module("release_audit", ROOT / "scripts/audit_public_release.py")
        fake = ("gh" + "p_" + "a" * 36).encode()
        reasons = audit.inspect("config.json", fake)
        self.assertTrue(reasons)
        self.assertNotIn(fake.decode(), " ".join(reasons))

    def test_current_entry_points_and_modes_are_selected(self):
        launcher = module("study_launcher", ROOT / "run_study.py")
        for stage in launcher.STAGES:
            for mode in ["audit", "run"]:
                cmd = launcher.command(stage, mode)
                self.assertTrue(Path(cmd[1]).is_file())
                self.assertTrue((ROOT / cmd[3]).is_file())
        self.assertIn("06_interpretation_precision_fix.py", launcher.command("06", "run")[1])
        self.assertNotIn("--replay-missing", launcher.command("06", "run"))
        self.assertIn("--replay-missing", launcher.command("06", "recover"))
        with self.assertRaises(ValueError):
            launcher.command("07", "recover")

    def test_synthetic_generator_refuses_to_overwrite_existing_data(self):
        from scripts.make_synthetic_data import generate
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "station_A_Anshun.xlsx"
            p.write_bytes(b"do not overwrite")
            with self.assertRaises(FileExistsError):
                generate(folder)
            self.assertEqual(p.read_bytes(), b"do not overwrite")
