# Encoder Input Configuration

Encoder input 1762-dim 벡터 상세 레이아웃.

Reference: `gear_sonic_deploy/policy/release/observation_config.yaml`

## Fields

| 범위 | 필드 | 크기 | 활성 모드 | 설명 |
|---|---|---|---|---|
| `[0:4]` | `encoder_mode_4` | 4 | 전체 | 모드 one-hot. `[0,1,0,0]`=upperbody, `[0,0,1,0]`=wholebody |
| `[4:294]` | `motion_joint_positions_10frame_step5` | 290 | g1 | 전신 29관절 위치 히스토리 (10프레임, step=5) |
| `[294:584]` | `motion_joint_velocities_10frame_step5` | 290 | g1 | 전신 29관절 속도 히스토리 (10프레임, step=5) |
| `[584:594]` | `motion_root_z_position_10frame_step5` | 10 | g1 | root 높이(Z) 히스토리 (10프레임, step=5) |
| `[594:595]` | `motion_root_z_position` | 1 | g1 | 현재 root 높이 |
| `[595:601]` | `motion_anchor_orientation` | 6 | upperbody | 현재 torso 방향 (6D rotation, G1 IMU) |
| `[601:661]` | `motion_anchor_orientation_10frame_step5` | 60 | g1 | torso 방향 히스토리 (10프레임, step=5) |
| `[661:781]` | `motion_joint_positions_lowerbody_10frame_step5` | 120 | upperbody | G1 하체 12관절 위치 히스토리 (10프레임, step=5) |
| `[781:901]` | `motion_joint_velocities_lowerbody_10frame_step5` | 120 | upperbody | G1 하체 12관절 속도 히스토리 (10프레임, step=5) |
| `[901:910]` | `vr_3point_local_target` | 9 | upperbody | L-Wrist/R-Wrist/Neck 위치 (root 기준, 3×3) |
| `[910:922]` | `vr_3point_local_orn_target` | 12 | upperbody | L-Wrist/R-Wrist/Neck 방향 (scalar-first quat, 3×4) |
| `[922:1642]` | `smpl_joints_10frame_step1` | 720 | wholebody | SMPL 전신 24관절 위치 히스토리 (10프레임, step=1) |
| `[1642:1702]` | `smpl_anchor_orientation_10frame_step1` | 60 | wholebody | SMPL root 방향 히스토리 (10프레임, step=1) |
| `[1702:1762]` | `motion_joint_positions_wrists_10frame_step1` | 60 | wholebody | 손목 위치 히스토리 (10프레임×2손목×3, step=1) |

## Mode One-hot Encoding

| 인덱스 | mode_id | 이름 | 벡터 |
|---|---|---|---|
| 0 | 0 | g1 | `[1,0,0,0]` |
| 1 | 1 | upperbody | `[0,1,0,0]` |
| 2 | 2 | wholebody | `[0,0,1,0]` |

## ONNX Models

| 모델 | 입력 | 출력 |
|---|---|---|
| `model_encoder.onnx` | `obs_dict [1, 1762]` | `encoded_tokens [1, 64]` |
| `model_decoder.onnx` | `obs_dict [1, 994]` | `action [1, 29]` |
