# gearsonic

GearSonic wholebody control pipeline — converts VR input into encoder input (1762-dim).

## Architecture

![architecture](docs/gearsonic_architecture.png)

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | array operations |
| `scipy` | rotation transforms (`Rotation`) |
| `onnxruntime` | ONNX encoder / decoder inference |

## Encoder Modes

| mode_id | name | upper body input | lower body input |
|---|---|---|---|
| 0 | `g1` | motion capture reference | full robot state |
| 1 | `upperbody` | VR 3-point (wrist×2 + neck) | G1 lower body measured |
| 2 | `wholebody` | SMPL full body tracking | SMPL full body tracking |

## Encoder Input Layout

See [docs/encoder_input_config.md](docs/encoder_input_config.md)

## Files

| file | role |
|---|---|
| `three_point_pose.py` | SMPL (24,7) → VR 3-point (3,7) transform + calibration |
| `obs_builder_base.py` | common interface for encoder input assembly |
| `obs_upperbody.py` | mode 1: VR 3-point + G1 lower body → 1762-dim |
| `obs_wholebody.py` | mode 2: SMPL full body tracking → 1762-dim |

## Reference

- Encoder/Decoder ONNX: `src/policy/model_encoder.onnx`, `src/policy/model_decoder.onnx`
- Observation config: `gear_sonic_deploy/policy/release/observation_config.yaml`
- GearSonic source: `gear_sonic/scripts/pico_manager_thread_server.py`
