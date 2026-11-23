import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from echoforge.ef_pipeline import (
    echolvsnet_volume_pipeline,
    mclv_pipeline,
    phase_mclv_pipeline,
    pipeline,
    volume_phase_pipeline,
)


class PhaseModel:
    def predict(self, windows, verbose=0):
        return np.zeros(windows.shape[:2])


def fake_segment(frame, _model):
    # Keep every native frame distinguishable in the volume results.
    return np.full_like(frame, 50 - frame[0, 0])


def fake_segment_batch(frames, model):
    return [fake_segment(frame, model) for frame in frames]


class EfProgressTests(unittest.TestCase):
    def test_all_pipelines_report_real_stage_work_without_changing_results(self):
        frames = [np.full((8, 8), index, dtype=np.uint8) for index in range(48)]
        pairs = [(3, 9), (20, 29)]
        phase_steps = [0.0, 16 / 19, 1.0]
        paired_steps = [index / 4 for index in range(5)]
        volume_steps = [0.0, 1.0]
        cases = (
            (pipeline, pipeline.compute_ef, "segment_frame", fake_segment,
             [("phase", phase_steps), ("segmentation", paired_steps)]),
            (phase_mclv_pipeline, phase_mclv_pipeline.compute_phase_mclv_ef,
             "segment_mclv_frames", fake_segment_batch,
             [("phase", phase_steps), ("segmentation", paired_steps)]),
            (mclv_pipeline, mclv_pipeline.compute_mclv_ef, "segment_frames", fake_segment_batch,
             [("segmentation", [count / 48 for count in range(0, 49, 4)])]),
            (echolvsnet_volume_pipeline, echolvsnet_volume_pipeline.compute_echolvsnet_volume_ef,
             "segment_frame", fake_segment,
             [("segmentation", [count / 48 for count in range(0, 49, 4)])]),
            (volume_phase_pipeline, volume_phase_pipeline.compute_volume_phase_ef,
             "segment_mclv_frames", fake_segment_batch,
             [("segmentation", [0.0, 1 / 3, 2 / 3, 1.0]), ("phase", phase_steps)]),
        )
        for module, compute, segment_name, segment, stages in cases:
            with self.subTest(method=compute.__name__), ExitStack() as stack:
                stack.enter_context(patch.object(module, "read_input", return_value=(frames, None, 30)))
                stack.enter_context(patch.object(module, segment_name, side_effect=segment))
                stack.enter_context(patch.object(module, "mask_volume",
                                                 side_effect=lambda mask, _: (float(mask[0, 0]), None)))
                models = {"seg_model": object()}
                if any(stage == "phase" for stage, _ in stages):
                    models["pd_model"] = PhaseModel()
                    stack.enter_context(patch.object(pipeline, "PD_SIZE", 4))
                    stack.enter_context(patch.object(module, "detect_beats",
                                                     return_value=([3, 20], [9, 29])))
                if hasattr(module, "select_beats"):
                    stack.enter_context(patch.object(module, "select_beats",
                                                     side_effect=lambda curve: (pairs, curve, np.zeros(len(curve), bool))))
                observed = []
                expected = [(stage, fraction) for stage, fractions in (*stages, ("volumes", volume_steps))
                            for fraction in fractions]
                baseline = compute("sample.mp4", **models)
                reported = compute("sample.mp4", **models,
                                   progress=lambda stage, fraction: observed.append((stage, fraction)))
                self.assertEqual(observed, expected)
                self.assertEqual(reported, baseline)


if __name__ == "__main__":
    unittest.main()
