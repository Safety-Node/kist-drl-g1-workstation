"""
Unit tests for PlannerInputBuilder.

Run:
  uv run src/pipeline/gearsonic/planner/test/test_input_builder.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../.."))

import numpy as np
import onnxruntime as ort

from src.pipeline.gearsonic.planner.input_builder import PlannerInputBuilder

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

_ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "../../models/onnx/planner_sonic.onnx"
)

# Expected ONNX input shapes
_EXPECTED_SHAPES = {
    "context_mujoco_qpos":       (1, 4, 36),
    "target_vel":                (1,),
    "mode":                      (1,),
    "movement_direction":        (1, 3),
    "facing_direction":          (1, 3),
    "random_seed":               (1,),
    "has_specific_target":       (1, 1),
    "specific_target_positions": (1, 4, 3),
    "specific_target_headings":  (1, 4),
    "allowed_pred_num_tokens":   (1, 11),
    "height":                    (1,),
}

_EXPECTED_DTYPES = {
    "context_mujoco_qpos":       np.float32,
    "target_vel":                np.float32,
    "mode":                      np.int64,
    "movement_direction":        np.float32,
    "facing_direction":          np.float32,
    "random_seed":               np.int64,
    "has_specific_target":       np.int64,
    "specific_target_positions": np.float32,
    "specific_target_headings":  np.float32,
    "allowed_pred_num_tokens":   np.int64,
    "height":                    np.float32,
}


def _result(ok: bool, label: str, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {label}{suffix}")
    return ok


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_build_shapes() -> bool:
    print("[1] build() output shapes and dtypes")
    builder = PlannerInputBuilder()
    builder.initialize(np.zeros(29, dtype=np.float32))
    inputs = builder.build()

    ok = True
    for key, expected_shape in _EXPECTED_SHAPES.items():
        arr = inputs[key]
        shape_ok = arr.shape == expected_shape
        dtype_ok = arr.dtype == _EXPECTED_DTYPES[key]
        ok &= _result(
            shape_ok and dtype_ok,
            key,
            f"shape={arr.shape} dtype={arr.dtype}"
        )
    return ok


def test_initialize() -> bool:
    print("[2] initialize() — height, identity quat, joint remap")
    builder = PlannerInputBuilder()

    joint_pos = np.arange(29, dtype=np.float32) * 0.1
    builder.initialize(joint_pos)
    inputs = builder.build()
    ctx = inputs["context_mujoco_qpos"][0]  # (4, 36)

    results = []

    # all 4 frames identical
    results.append(_result(
        np.allclose(ctx[0], ctx[1]) and np.allclose(ctx[0], ctx[2]) and np.allclose(ctx[0], ctx[3]),
        "4 frames identical after initialize"
    ))

    frame = ctx[0]

    # position normalized to origin
    results.append(_result(
        frame[0] == 0.0 and frame[1] == 0.0,
        "xy = 0"
    ))
    results.append(_result(
        frame[2] > 0.0,
        f"height > 0  (got {frame[2]:.3f})"
    ))

    # identity quaternion [qw, qx, qy, qz]
    results.append(_result(
        np.allclose(frame[3:7], [1.0, 0.0, 0.0, 0.0]),
        f"identity quat  (got {frame[3:7]})"
    ))

    # joint remap: mujoco_to_isaaclab[mujoco_i] = isaac_i
    from src.pipeline.gearsonic.planner.input_builder import _load_config
    mapping = _load_config()["mujoco_to_isaaclab"]
    remap_ok = all(
        np.isclose(frame[7 + isaac_i], joint_pos[mujoco_i])
        for mujoco_i, isaac_i in enumerate(mapping)
    )
    results.append(_result(remap_ok, "mujoco→isaaclab joint remap"))

    return all(results)


def test_update_command() -> bool:
    print("[3] update_command() — values stored in build()")
    builder = PlannerInputBuilder()
    builder.initialize(np.zeros(29, dtype=np.float32))

    move = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    face = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    builder.update_command(
        mode=2, target_vel=-1.0,
        movement_direction=move, facing_direction=face, random_seed=42,
    )
    inputs = builder.build()

    results = []
    results.append(_result(inputs["mode"][0] == 2,           f"mode=2 (got {inputs['mode'][0]})"))
    results.append(_result(inputs["target_vel"][0] == -1.0,  f"target_vel=-1.0"))
    results.append(_result(np.allclose(inputs["movement_direction"][0], move), "movement_direction"))
    results.append(_result(np.allclose(inputs["facing_direction"][0], face),   "facing_direction"))
    results.append(_result(inputs["random_seed"][0] == 42,   f"random_seed=42"))
    return all(results)


def test_set_trajectory() -> bool:
    print("[4] set_trajectory() — context updates on next build()")
    builder = PlannerInputBuilder()
    builder.initialize(np.zeros(29, dtype=np.float32))

    ctx_before = builder.build()["context_mujoco_qpos"].copy()

    # synthetic trajectory: linearly increasing position
    T = 24
    traj = np.zeros((T, 36), dtype=np.float32)
    for i in range(T):
        traj[i, 0] = i * 0.1   # x grows
        traj[i, 3] = 1.0       # qw = 1 (identity)

    builder.set_trajectory(traj)
    time.sleep(0.05)            # let gen_frame > 0
    ctx_after = builder.build()["context_mujoco_qpos"].copy()

    changed = not np.allclose(ctx_before, ctx_after)
    return _result(changed, "context changed after set_trajectory + build()")


def test_onnx_e2e() -> bool:
    print("[5] ONNX end-to-end — build() → sess.run() succeeds")
    try:
        sess = ort.InferenceSession(_ONNX_PATH)
    except Exception as e:
        return _result(False, f"ONNX session load failed: {e}")

    builder = PlannerInputBuilder()
    builder.initialize(np.zeros(29, dtype=np.float32))
    builder.update_command(
        mode=1, target_vel=-1.0,
        movement_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        facing_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        random_seed=0,
    )

    try:
        inputs = builder.build()
        mujoco_qpos, num_pred_frames = sess.run(None, inputs)
        n = int(num_pred_frames[0])
        traj = mujoco_qpos[0, :n]
    except Exception as e:
        return _result(False, f"sess.run() failed: {e}")

    results = []
    results.append(_result(n > 0,               f"num_pred_frames={n} > 0"))
    results.append(_result(traj.shape[1] == 36, f"trajectory shape={traj.shape}"))
    results.append(_result(traj.dtype == np.float32, f"dtype={traj.dtype}"))

    # second inference with set_trajectory
    builder.set_trajectory(traj)
    try:
        inputs2 = builder.build()
        mujoco_qpos2, num_pred_frames2 = sess.run(None, inputs2)
        n2 = int(num_pred_frames2[0])
    except Exception as e:
        return _result(False, f"second sess.run() failed: {e}")

    results.append(_result(n2 > 0, f"2nd inference num_pred_frames={n2} > 0"))
    return all(results)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    tests = [
        ("build() shapes",    test_build_shapes),
        ("initialize()",      test_initialize),
        ("update_command()",  test_update_command),
        ("set_trajectory()",  test_set_trajectory),
        ("ONNX end-to-end",   test_onnx_e2e),
    ]

    results = {}
    for name, fn in tests:
        results[name] = fn()

    print("\n--- Results ---")
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {name}")

    passed = sum(results.values())
    print(f"\n{passed}/{len(results)} passed")


if __name__ == "__main__":
    main()
