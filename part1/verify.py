'Behavioral contract tests without full training or test-set tuning.'
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import common
TEST_TMP = common.OUTPUT_DIR / "verification_tmp"
TEST_TMP.mkdir(parents=True, exist_ok=True)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


template = load_module("verify_template", "01_template_baseline.py")
evaluation = load_module("verify_evaluation", "03_evaluate.py")


class ContractChecks(unittest.TestCase):
    def test_splits_all_categories(self):
        for category in common.REQUIRED_CATEGORIES:
            train = common.load_visa_split(category, "train")
            fit, validation = common.split_normal_train(train)
            test = common.load_visa_split(category, "test")
            self.assertFalse(set(fit.image) & set(validation.image))
            self.assertEqual(set(fit.image) | set(validation.image), set(train.image))
            self.assertFalse(set(train.image) & set(test.image))
            self.assertTrue(fit.label.eq("normal").all())

    def test_mask_low_positive_values(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            path = Path(tmp) / "mask.png"
            Image.fromarray(np.array([[0, 1], [6, 255]], dtype=np.uint8)).save(path)
            self.assertEqual(common.load_mask(path, (2, 2)).tolist(), [[0, 1], [1, 1]])

    def test_translation_and_border_fallback(self):
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[100:110, 170:180] = 255
        image[90:100, 0:6] = 255
        reference = np.zeros_like(image)
        warp = np.array([[1, 0, 20], [0, 1, 0]], dtype=np.float32)
        aligned = cv2.warpAffine(image, warp, (256, 256), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        with patch.object(template, "register_to_template", return_value=(aligned, warp, True)):
            heatmap, ok, coverage = template.make_difference_map(image, reference)
        self.assertTrue(ok)
        self.assertGreater(float(heatmap[102:108, 172:178].mean()), 0.8)
        self.assertLess(float(heatmap[100:110, 149:160].max()), 0.01)
        self.assertGreater(float(heatmap[92:98, 0:4].mean()), 0.5)
        self.assertLess(coverage, 1.0)

    def test_rotation_coordinates(self):
        heatmap = np.zeros((128, 128), dtype=np.float32)
        heatmap[45:55, 75:85] = 1
        warp = cv2.getRotationMatrix2D((64, 64), 12, 1)
        returned, _ = template.restore_map(heatmap, warp, np.zeros_like(heatmap))
        expected = warp @ np.array([79.5, 49.5, 1])
        ys, xs = np.where(returned > 0.5)
        self.assertLess(abs(xs.mean() - expected[0]), 1)
        self.assertLess(abs(ys.mean() - expected[1]), 1)

    def test_failed_registration_retains_sample(self):
        image = np.full((32, 32, 3), 127, dtype=np.uint8)
        with patch.object(cv2, "findTransformECC", side_effect=cv2.error("forced failure")):
            heatmap, ok, coverage = template.make_difference_map(image, np.zeros_like(image))
        self.assertFalse(ok)
        self.assertEqual(coverage, 1)
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreater(float(heatmap.mean()), 0.4)

    def test_threshold_normal_only_and_ties(self):
        validation = pd.DataFrame({"label": ["normal"] * 90, "score": np.ones(90)})
        threshold = evaluation.calibrate_threshold(validation, 0.05)
        self.assertEqual(float((validation.score > threshold).mean()), 0)
        validation.loc[0, "label"] = "anomaly"
        with self.assertRaises(ValueError):
            evaluation.calibrate_threshold(validation, 0.05)

    def test_aupro_perfect_and_constant(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            tmp = Path(tmp)
            mask = np.zeros((448, 448), dtype=np.uint8)
            mask[100:120, 100:120] = 1
            Image.fromarray(mask).save(tmp / "mask.png")
            paths = [tmp / "good.npy", tmp / "bad.npy"]
            np.save(paths[0], np.zeros_like(mask, dtype=np.float32))
            np.save(paths[1], mask.astype(np.float32))
            rows = pd.DataFrame({"mask_path": [None, str(tmp / "mask.png")], "map_path": list(map(str, paths))})
            perfect = evaluation.evaluate_aupro(rows, "synthetic")
            self.assertAlmostEqual(perfect["test_aupro"], 1, places=7)
            np.save(paths[1], np.zeros_like(mask, dtype=np.float32))
            constant = evaluation.evaluate_aupro(rows, "synthetic")["test_aupro"]
            self.assertAlmostEqual(constant, 0.15, places=7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
