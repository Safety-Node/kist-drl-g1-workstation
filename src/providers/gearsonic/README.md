# gearsonic

GearSonic wholebody control pipeline — converts VR input into encoder input (1762-dim).

## Architecture

![architecture](docs/gearsonic_architecture.png)

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | array operations |
| `scipy` | rotation transforms |
| `onnxruntime` | ONNX encoder / decoder inference |

## Encoder Modes

| mode_id | name | upper body input | lower body input |
|---|---|---|---|
| 0 | `g1` | motion capture reference | full robot state |
| 1 | `upperbody` | VR 3-point (wrist×2 + neck) | G1 lower body measured |
| 2 | `wholebody` | SMPL full body tracking | SMPL full body tracking |

## Encoder Input Layout

See [docs/dev.md](docs/dev.md)

