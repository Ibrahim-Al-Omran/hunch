"""Convert the MediaPipe pose .task bundles into ONNX models (dev-time only).

Run with the separate conversion venv (TensorFlow + tf2onnx), not the
prototype venv:

    .venv-convert\\Scripts\\python tools\\convert_models.py

The pose detector stores some weights sparsely (DENSIFY ops), which tf2onnx
cannot handle, so those constants are densified with the TFLite interpreter
and baked back into the flatbuffer before conversion.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils

MODELS = Path(__file__).resolve().parents[1] / "models"


def densify(src: Path, dst: Path) -> int:
    model = flatbuffer_utils.read_model(str(src))
    interp = tf.lite.Interpreter(model_path=str(src), experimental_preserve_all_tensors=True)
    interp.allocate_tensors()
    for d in interp.get_input_details():
        interp.set_tensor(d["index"], np.zeros(d["shape"], d["dtype"]))
    interp.invoke()

    densify_code = schema.BuiltinOperator.DENSIFY
    sub = model.subgraphs[0]
    keep, removed = [], 0
    for op in sub.operators:
        code = model.operatorCodes[op.opcodeIndex]
        builtin = max(code.builtinCode, code.deprecatedBuiltinCode)
        if builtin != densify_code:
            keep.append(op)
            continue
        out_idx = int(op.outputs[0])
        dense = interp.get_tensor(out_idx)
        buf = schema.BufferT()
        buf.data = np.frombuffer(dense.tobytes(), dtype=np.uint8)
        model.buffers.append(buf)
        sub.tensors[out_idx].buffer = len(model.buffers) - 1
        removed += 1
    sub.operators = keep
    _drop_unused_tensors(sub)
    flatbuffer_utils.write_model(model, str(dst))
    return removed


def _drop_unused_tensors(sub) -> None:
    # tf2onnx crashes when it inspects the orphaned sparse tensors.
    used = set(int(i) for i in sub.inputs) | set(int(i) for i in sub.outputs)
    for op in sub.operators:
        used.update(int(i) for i in op.inputs if i >= 0)
        used.update(int(i) for i in op.outputs if i >= 0)
    order = sorted(used)
    remap = {old: new for new, old in enumerate(order)}
    sub.tensors = [sub.tensors[i] for i in order]
    sub.inputs = [remap[int(i)] for i in sub.inputs]
    sub.outputs = [remap[int(i)] for i in sub.outputs]
    for op in sub.operators:
        op.inputs = [remap[int(i)] if i >= 0 else -1 for i in op.inputs]
        op.outputs = [remap[int(i)] for i in op.outputs]


def main() -> None:
    for variant in ("lite", "full"):
        task = MODELS / f"pose_landmarker_{variant}.task"
        out_dir = MODELS / variant
        zipfile.ZipFile(task).extractall(out_dir)
        det = out_dir / "pose_detector.tflite"
        det_dense = out_dir / "pose_detector_dense.tflite"
        n = densify(det, det_dense)
        print(f"{variant}: densified {n} sparse constants")
        for src, name in ((det_dense, "pose_detector"), (out_dir / "pose_landmarks_detector.tflite", "pose_landmarks_detector")):
            subprocess.run(
                [sys.executable, "-m", "tf2onnx.convert", "--tflite", str(src),
                 "--output", str(out_dir / f"{name}.onnx"), "--opset", "17"],
                check=True,
            )


if __name__ == "__main__":
    main()
