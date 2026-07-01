# gearsonic

GearSonic wholebody control 파이프라인 — VR 입력을 encoder input (1762-dim)으로 변환한다.

## Pipeline

```
PicoVRReader  ──────────────────────────────────┐
                                                 ├──▶ obs_*.py ──▶ encoder.onnx ──▶ decoder.onnx ──▶ q_target [29]
UnitreeG1Provider ──▶ G1ObsProvider ────────────┘
```

## Encoder Modes

| mode_id | 이름 | 상체 입력 | 하체 입력 |
|---|---|---|---|
| 0 | `g1` | 모션 캡처 레퍼런스 | 로봇 전신 실측 |
| 1 | `upperbody` | VR 3-point (wrist×2 + neck) | G1 하체 실측 |
| 2 | `wholebody` | SMPL 전신 트래킹 | SMPL 전신 트래킹 |

## Encoder Input Layout (1762-dim)

| 범위 | 필드 | 크기 | 활성 모드 |
|---|---|---|---|
| `[0:4]` | `encoder_mode_4` | 4 | 전체 |
| `[4:294]` | `motion_joint_positions_10frame_step5` | 290 | g1 |
| `[294:584]` | `motion_joint_velocities_10frame_step5` | 290 | g1 |
| `[584:594]` | `motion_root_z_position_10frame_step5` | 10 | g1 |
| `[594:595]` | `motion_root_z_position` | 1 | g1 |
| `[595:601]` | `motion_anchor_orientation` | 6 | upperbody |
| `[601:661]` | `motion_anchor_orientation_10frame_step5` | 60 | g1 |
| `[661:781]` | `motion_joint_positions_lowerbody_10frame_step5` | 120 | upperbody |
| `[781:901]` | `motion_joint_velocities_lowerbody_10frame_step5` | 120 | upperbody |
| `[901:910]` | `vr_3point_local_target` | 9 | upperbody |
| `[910:922]` | `vr_3point_local_orn_target` | 12 | upperbody |
| `[922:1642]` | `smpl_joints_10frame_step1` | 720 | wholebody |
| `[1642:1702]` | `smpl_anchor_orientation_10frame_step1` | 60 | wholebody |
| `[1702:1762]` | `motion_joint_positions_wrists_10frame_step1` | 60 | wholebody |

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
