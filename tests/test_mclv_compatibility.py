import json
import os
import tempfile
import unittest

import h5py
import numpy as np
import tensorflow as tf

from echoforge.segmentation.models.mclv.mclv import _load_mclv


class MclvCompatibilityTests(unittest.TestCase):
    def test_keras3_graph_loads_original_weights(self):
        inputs = tf.keras.Input(shape=(2,), name="input")
        first = tf.keras.layers.Dense(2, name="first")(inputs)
        merged = tf.keras.layers.Concatenate()([first, inputs])
        model = tf.keras.Model(inputs, merged)
        sample = np.array([[1.0, 2.0]], dtype=np.float32)
        expected = model(sample).numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.h5")
            model.save(path, include_optimizer=False)
            with h5py.File(path, "r+") as saved:
                config = json.loads(saved.attrs["model_config"])
                for layer in config["config"]["layers"]:
                    if layer["class_name"] == "InputLayer":
                        layer["config"]["batch_shape"] = layer["config"].pop("batch_input_shape")
                    layer["inbound_nodes"] = [
                        {
                            "args": [[
                                {"class_name": "__keras_tensor__", "config": {"keras_history": node[:3]}}
                                for node in group
                            ]],
                            "kwargs": {},
                        }
                        for group in layer["inbound_nodes"]
                    ]
                saved.attrs["model_config"] = json.dumps(config)
                saved.attrs["keras_version"] = "3.0.5"
            loaded = _load_mclv(path)
            np.testing.assert_allclose(loaded(sample).numpy(), expected)

    def test_rejects_unsupported_node_arguments(self):
        inputs = tf.keras.Input(shape=(2,))
        model = tf.keras.Model(inputs, tf.keras.layers.Dense(1)(inputs))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.h5")
            model.save(path, include_optimizer=False)
            with h5py.File(path, "r+") as saved:
                config = json.loads(saved.attrs["model_config"])
                input_config = config["config"]["layers"][0]["config"]
                input_config["batch_shape"] = input_config.pop("batch_input_shape")
                config["config"]["layers"][1]["inbound_nodes"] = [
                    {"args": [], "kwargs": {"training": True}}
                ]
                saved.attrs["model_config"] = json.dumps(config)
                saved.attrs["keras_version"] = "3.0.5"
            with self.assertRaisesRegex(ValueError, "node arguments"):
                _load_mclv(path)


if __name__ == "__main__":
    unittest.main()
