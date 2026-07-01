# Encoder Input Configuration

Detailed layout of the encoder input 1762-dim vector.

Reference: `gear_sonic_deploy/policy/release/observation_config.yaml`

## Fields

| range | field | size | active mode | description |
|---|---|---|---|---|
| `[0:4]` | `encoder_mode_4` | 4 | all | mode one-hot. `[0,1,0,0]`=upperbody, `[0,0,1,0]`=wholebody |
| `[4:294]` | `motion_joint_positions_10frame_step5` | 290 | g1 | full body 29-joint position history (10 frames, step=5) |
| `[294:584]` | `motion_joint_velocities_10frame_step5` | 290 | g1 | full body 29-joint velocity history (10 frames, step=5) |
| `[584:594]` | `motion_root_z_position_10frame_step5` | 10 | g1 | root height (Z) history (10 frames, step=5) |
| `[594:595]` | `motion_root_z_position` | 1 | g1 | current root height |
| `[595:601]` | `motion_anchor_orientation` | 6 | upperbody | current torso orientation (6D rotation, G1 IMU) |
| `[601:661]` | `motion_anchor_orientation_10frame_step5` | 60 | g1 | torso orientation history (10 frames, step=5) |
| `[661:781]` | `motion_joint_positions_lowerbody_10frame_step5` | 120 | upperbody | G1 lower body 12-joint position history (10 frames, step=5) |
| `[781:901]` | `motion_joint_velocities_lowerbody_10frame_step5` | 120 | upperbody | G1 lower body 12-joint velocity history (10 frames, step=5) |
| `[901:910]` | `vr_3point_local_target` | 9 | upperbody | L-Wrist/R-Wrist/Neck position relative to root (3×3) |
| `[910:922]` | `vr_3point_local_orn_target` | 12 | upperbody | L-Wrist/R-Wrist/Neck orientation, scalar-first quat (3×4) |
| `[922:1642]` | `smpl_joints_10frame_step1` | 720 | wholebody | SMPL full body 24-joint position history (10 frames, step=1) |
| `[1642:1702]` | `smpl_anchor_orientation_10frame_step1` | 60 | wholebody | SMPL root orientation history (10 frames, step=1) |
| `[1702:1762]` | `motion_joint_positions_wrists_10frame_step1` | 60 | wholebody | wrist position history (10 frames × 2 wrists × 3, step=1) |

## Mode One-hot Encoding

| index | mode_id | name | vector |
|---|---|---|---|
| 0 | 0 | g1 | `[1,0,0,0]` |
| 1 | 1 | upperbody | `[0,1,0,0]` |
| 2 | 2 | wholebody | `[0,0,1,0]` |

## ONNX Models

| model | input | output |
|---|---|---|
| `model_encoder.onnx` | `obs_dict [1, 1762]` | `encoded_tokens [1, 64]` |
| `model_decoder.onnx` | `obs_dict [1, 994]` | `action [1, 29]` |
