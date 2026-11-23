"""Check how an observed re-anchor affects both basal terminal regions."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import hashlib
import sys
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'echoforge' / 'chrono_tracker.py'
spec = spec_from_file_location('placement_component_under_test', SCRIPT)
component = module_from_spec(spec)
sys.modules[spec.name] = component
spec.loader.exec_module(component)


def mesh(left=0., right=0.):
    points = np.empty((5, 36, 2), dtype=np.float64)
    points[..., 0] = np.arange(36)
    points[..., 1] = np.arange(5)[:, None]
    points[:, 0, 1] += left
    points[:, -1, 1] += right
    return points.reshape(180, 2)


def fingerprint(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


class BasalSourceSelectionTests(unittest.TestCase):
    def initialize(self, baseline, candidate, proposed_frame=2):
        meta = dict(initialization_reference_position=0, reference_selection={})
        chosen_meta = dict(initialization_reference_position=proposed_frame)
        selector = SimpleNamespace(admitted_indices=lambda *_: [0, 2],
            select=lambda *_, **__: ((candidate, chosen_meta), dict(selected_reference=proposed_frame)))
        gray = np.zeros((3, 8, 8), dtype=np.uint8)
        fan = np.ones((8, 8), dtype=bool)
        engine = SimpleNamespace(auto_initial_queries_for_cine=lambda *_, **__: (baseline, meta),
            _gray_frames=lambda frames, _: frames, source_fan_support=lambda _: fan,
            _material_array_pin=fingerprint, _conditional_array_hash_v55=fingerprint)
        with patch.object(component, '_repair_module', side_effect=[selector, SimpleNamespace()]):
            component._install_source_shape_selection(engine)
        return engine.auto_initial_queries_for_cine(gray, return_metadata=True)

    def test_one_worse_basal_end_retains_original(self):
        baseline = mesh(left=1., right=1.)
        for side, candidate in (('left', mesh(left=2.)), ('right', mesh(right=2.))):
            with self.subTest(side=side):
                q, metadata = self.initialize(baseline, candidate)
                np.testing.assert_array_equal(q, baseline)
                report = metadata['pretracking_source_shape_selection']
                self.assertEqual(report['effective_reference'], 0)
                self.assertFalse(report['selected_reference_changed'])
                self.assertEqual(report['basal_endcap_check']['per_side_passed'],
                                 [side != 'left', side != 'right'])

    def test_both_ends_no_worse_allow_new_reference(self):
        baseline, candidate = mesh(left=1., right=1.), mesh(right=1.)
        q, metadata = self.initialize(baseline, candidate)
        np.testing.assert_array_equal(q, candidate)
        self.assertEqual(metadata['initialization_reference_position'], 2)
        self.assertTrue(metadata['pretracking_source_shape_selection']['basal_endcap_check']['passed'])

    def test_invalid_proposal_keeps_original(self):
        baseline, candidate = mesh(), mesh()
        candidate[0] = candidate[1]
        q, metadata = self.initialize(baseline, candidate)
        np.testing.assert_array_equal(q, baseline)
        report = metadata['pretracking_source_shape_selection']
        self.assertEqual(report['basal_endcap_check']['reason'], 'invalid_basal_intervals')

    def test_same_frame_keeps_exact_initial_points(self):
        baseline, candidate = mesh(left=1.), mesh()
        q, metadata = self.initialize(baseline, candidate, proposed_frame=0)
        np.testing.assert_array_equal(q, baseline)
        self.assertFalse(metadata['pretracking_source_shape_selection']['selected_reference_changed'])


if __name__ == '__main__':
    unittest.main()
