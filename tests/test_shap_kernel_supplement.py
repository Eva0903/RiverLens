from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jinjiang.shap_kernel_supplement import CoalitionGame, explain_origin, player_groups, validate_spec

HAS_SHAP = importlib.util.find_spec("shap") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class KernelTests(unittest.TestCase):
    def test_import_succeeds_with_old_07_explicitly_blocked(self):
        import subprocess
        import sys
        command = ("import sys; sys.modules['jinjiang.shap_supplement'] = None; "
                   "sys.modules['jinjiang.shap_cuda_forward_fix'] = None; "
                   "from jinjiang.shap_kernel_supplement import run; "
                   "from jinjiang.shap_kernel_common import ReadOnlyCaseSource; "
                   "print('standalone import passed')")
        result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('standalone import passed', result.stdout)

    def test_group_coverage_and_related_encodings_joint(self):
        columns = [f"A_X{i}" for i in range(26)] + ["W_PRS_MEAN", "W_RHU_MEAN", "W_TEM_MEAN"] + [f"W_RAIN{i}" for i in range(7)] + ["W_WIND_SPEED_MAX", "W_WIND_DIR_SIN", "W_WIND_DIR_COS"]
        groups = player_groups(columns)
        self.assertEqual(len(groups), 31)
        self.assertEqual(sorted(i for g in groups for i in g["columns"]), list(range(39)))
        self.assertEqual(len(groups[-2]["columns"]), 7)
        self.assertEqual(len(groups[-1]["columns"]), 3)

    def test_mask_preserves_intact_trajectories_and_endpoint_cache(self):
        x = np.arange(24).reshape(4, 6).astype(np.float32)
        bg = np.stack([x + 100, x + 200])
        groups = [dict(name="a", columns=[0, 1]), dict(name="b", columns=[2, 3]), dict(name="c", columns=[4, 5])]
        seen = []
        def predict(values):
            seen.append(values.copy())
            return values.sum(axis=(1, 2))[:, None]
        game = CoalitionGame(predict, x, bg, groups, [x.sum()], [bg.sum(axis=(1, 2)).mean()])
        values = game(np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1]]))
        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0][:, :, [0, 1, 4, 5]], np.broadcast_to(x[:, [0, 1, 4, 5]], (2, 4, 4)))
        np.testing.assert_array_equal(seen[0][:, :, 2:4], bg[:, :, 2:4])
        game(np.array([[1, 0, 1]]))
        self.assertEqual(len(seen), 1)
        with self.assertRaises(ValueError):
            game(np.array([[.5, 0, 1]]))

    @unittest.skipUnless(HAS_SHAP, "requires installed shap")
    def test_known_affine_shap_all_outputs(self):
        x = np.array([[2., 3., 4.], [4., 5., 6.]])
        bg = np.stack([np.zeros_like(x), np.ones_like(x)])
        weights = np.array([1., -2., 3.])
        groups = [dict(name=str(i), columns=[i]) for i in range(3)]
        def predict(a):
            v = (a * weights).sum(axis=(1, 2))
            return np.stack([v, 2 * v + 10], axis=1)
        game = CoalitionGame(predict, x, bg, groups, predict(x[None])[0], predict(bg).mean(axis=0))
        result = explain_origin(game, groups, 128, 2026)
        expected = ((x - bg.mean(axis=0)) * weights).sum(axis=0)
        np.testing.assert_allclose(result["phi"][:, 0], expected, atol=1e-8)
        np.testing.assert_allclose(result["phi"][:, 1], 2 * expected, atol=1e-8)
        self.assertEqual(int(result["design_rank"]), 3)

    @unittest.skipUnless(HAS_SHAP, "requires installed shap")
    def test_interaction_matches_exhaustive_group_shapley(self):
        x = np.array([[2., 3., 4.]])
        bg = np.array([[[0., 0., 0.]], [[1., 1., 1.]]])
        groups = [dict(name=str(i), columns=[i]) for i in range(3)]
        def predict(a):
            return (a[:, 0, 0] * a[:, 0, 1] + a[:, 0, 2] ** 2)[:, None]
        game = CoalitionGame(predict, x, bg, groups, predict(x[None])[0], predict(bg).mean(axis=0))
        result = explain_origin(game, groups, 128, 2026)
        exact = np.zeros(3)
        for j in range(3):
            others = [i for i in range(3) if i != j]
            for size in range(3):
                for subset in itertools.combinations(others, size):
                    mask = np.zeros(3, dtype=int); mask[list(subset)] = 1
                    before = game(mask[None])[0, 0]
                    mask[j] = 1
                    after = game(mask[None])[0, 0]
                    exact[j] += math.factorial(size) * math.factorial(2 - size) / math.factorial(3) * (after - before)
        np.testing.assert_allclose(result["phi"][:, 0], exact, atol=1e-8)

    @unittest.skipUnless(HAS_SHAP, "requires installed shap")
    def test_no_default_top10_feature_selection(self):
        columns = [f"A_X{i}" for i in range(26)] + ["W_PRS_MEAN", "W_RHU_MEAN", "W_TEM_MEAN"] + [f"W_RAIN{i}" for i in range(7)] + ["W_WIND_SPEED_MAX", "W_WIND_DIR_SIN", "W_WIND_DIR_COS"]
        groups = player_groups(columns)
        x, bg = np.ones((2, 39)), np.zeros((2, 2, 39))
        def predict(a):
            return a.sum(axis=(1, 2))[:, None]
        game = CoalitionGame(predict, x, bg, groups, [78.], [0.])
        result = explain_origin(game, groups, 128, 2026)
        np.testing.assert_allclose(result["phi"][:, 0], [2 * len(g["columns"]) for g in groups], atol=1e-7)
        self.assertEqual((np.abs(result["phi"][:, 0]) > 1e-6).sum(), 31)

    def test_protocol_changes_rejected(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root / "configs/shap_kernel.json").read_text())
        validate_spec(c)
        with self.assertRaises(ValueError):
            validate_spec(dict(c, model_seed=99))
        with self.assertRaises(ValueError):
            validate_spec(dict(c, kernel_nsamples=2))

    @unittest.skipUnless(HAS_TORCH and HAS_SHAP, "requires installed torch and shap")
    def test_driver_resume_without_any_backward_or_weight_change(self):
        import torch
        import pandas as pd
        from types import SimpleNamespace
        from unittest.mock import patch
        import jinjiang.shap_kernel_supplement as core
        from jinjiang.interpretation_precision_fix import load_predictor_fixed, prediction_diagnostics
        torch.set_num_threads(1); torch.manual_seed(7)

        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.recurrent = torch.nn.LSTM(39, 3, 2, dropout=.1, batch_first=True)
                self.head = torch.nn.Linear(3, 6)

        c = json.loads((Path(__file__).resolve().parents[1] / "configs/shap_kernel.json").read_text())
        c["device"] = "cpu"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); checkpoint = root / "original_checkpoint"; checkpoint.mkdir()
            model = Small().eval(); torch.save(model.state_dict(), checkpoint / "model.pt")
            np.savez(checkpoint / "scalers.npz", x_mean=np.zeros(39), x_scale=np.ones(39), y_mean=np.ones(6), y_scale=np.ones(6))
            columns = [f"A_X{i}" for i in range(26)] + ["W_PRS_MEAN", "W_RHU_MEAN", "W_TEM_MEAN"] + [f"W_RAIN{i}" for i in range(7)] + ["W_WIND_SPEED_MAX", "W_WIND_DIR_SIN", "W_WIND_DIR_COS"]
            meta = dict(input_columns=columns, training=dict(hidden_size=3, num_layers=2, dropout=.1, batch_size=32),
                        case=dict(parameter_count=sum(p.numel() for p in model.parameters()), source_task_id="synthetic_fixture_only"))
            (checkpoint / "metadata.json").write_text(json.dumps(meta))
            x = np.zeros((8, 3, 39), dtype=np.float32); x[6:] = .01
            split = dict(train=np.arange(6), validation=np.array([], dtype=int), test=np.array([6, 7]))
            seq = SimpleNamespace(x=x, feature_columns=columns, issue_times=pd.date_range("2020-01-01", periods=8, freq="4h"))
            expected = load_predictor_fixed(checkpoint, "cpu")(x[split["test"]])
            old_hashes = {p.name: core.digest(p) for p in checkpoint.iterdir()}

            class Source:
                def __init__(self, *args):
                    self.fingerprint = {"synthetic_fixture_only": True}
                    self._predictor_loader = load_predictor_fixed
                    self._prediction_diagnostics = prediction_diagnostics
                def case(self, target):
                    return seq, split, checkpoint, meta, expected

            with patch.object(core, "ReadOnlyCaseSource", Source), patch.object(core, "figures"), \
                 patch.object(torch.autograd, "grad", side_effect=AssertionError("No gradient calls allowed")):
                output = core.run(c, root, "run", max_new_origins=1)
                shard = next((output / "tasks").glob("*/origin_0000.npz")); before = shard.stat().st_mtime_ns
                self.assertEqual(core.read_json(output / "run_status.json")["newly_explained_origins"], 1)
                core.run(c, root, "run", max_new_origins=1)
                status = core.read_json(output / "run_status.json")
                self.assertEqual(len(status["completed_background_tasks"]), 1)
                self.assertEqual(status["gradient_calls"], 0)
                self.assertEqual(before, shard.stat().st_mtime_ns)
                self.assertEqual({p.name: core.digest(p) for p in checkpoint.iterdir()}, old_hashes)


if __name__ == "__main__":
    unittest.main()
