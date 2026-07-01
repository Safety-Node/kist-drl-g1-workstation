# gearsonic

GearSonic wholebody control 파이프라인 — VR 입력을 encoder input (1762-dim)으로 변환한다.

## Architecture

![architecture](docs/gearsonic_architecture.png)

## Encoder Modes

| mode_id | 이름 | 상체 입력 | 하체 입력 |
|---|---|---|---|
| 0 | `g1` | 모션 캡처 레퍼런스 | 로봇 전신 실측 |
| 1 | `upperbody` | VR 3-point (wrist×2 + neck) | G1 하체 실측 |
| 2 | `wholebody` | SMPL 전신 트래킹 | SMPL 전신 트래킹 |

## Encoder Input Layout

상세 레이아웃: [docs/encoder_input_config.md](docs/encoder_input_config.md)

## Files

| 파일 | 역할 |
|---|---|
| `three_point_pose.py` | SMPL (24,7) → VR 3-point (3,7) 변환 + 캘리브레이션 |
| `obs_builder_base.py` | encoder input 조립 공통 인터페이스 |
| `obs_upperbody.py` | mode 1: VR 3-point + G1 하체 → 1762-dim |
| `obs_wholebody.py` | mode 2: SMPL 전신 트래킹 → 1762-dim |

## Reference

- Encoder/Decoder ONNX: `src/policy/model_encoder.onnx`, `src/policy/model_decoder.onnx`
- Observation config: `gear_sonic_deploy/policy/release/observation_config.yaml`
- GearSonic 원본: `gear_sonic/scripts/pico_manager_thread_server.py`
