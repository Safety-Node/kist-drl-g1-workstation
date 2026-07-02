# GearSonic — Developer Notes

## Planner Model

| model | description |
|---|---|
| `planner_sonic.onnx` | locomotion trajectory planner |

**Input:**

| range | field | shape | description |
|---|---|---|---|
| `[1]` | `context_mujoco_qpos` | `[1,4,36]` | robot state history — 4 frames × 36 (7 floating base + 29 joints in isaaclab order), float32 |
| `[2]` | `target_vel` | `[1]` | target speed m/s (−1 = default), float32 |
| `[3]` | `mode` | `[1]` | locomotion mode (int64): 0=IDLE, 1=SLOW_WALK, 2=WALK, 3=RUN, 4=IDLE_SQUAT, … |
| `[4]` | `movement_direction` | `[1,3]` | direction to move (unit vector), float32 |
| `[5]` | `facing_direction` | `[1,3]` | direction to face (unit vector), float32 |
| `[6]` | `random_seed` | `[1]` | random seed for stochastic generation, int64 |
| `[7]` | `has_specific_target` | `[1,1]` | specific waypoint target (int64): 0=no, 1=yes |
| `[8]` | `specific_target_positions` | `[1,4,3]` | 4 waypoint positions xyz, float32 |
| `[9]` | `specific_target_headings` | `[1,4]` | 4 waypoint heading angles, float32 |
| `[10]` | `allowed_pred_num_tokens` | `[1,11]` | prediction token mask, int64 (default: [1,1,1,1,1,1,0,0,0,0,0]) |
| `[11]` | `height` | `[1]` | target body height in meters (−1 = use model default), float32 |

**Output:**

| range | field | shape | description |
|---|---|---|---|
| `[1]` | `mujoco_qpos` | `[1,64,36]` | future trajectory — 64 frames × 36 (7 floating base + 29 joints in mujoco order), float32 |
| `[2]` | `num_pred_frames` | `[1]` | number of valid predicted frames, int32 |

## Encoder Modes

| mode_id | name | one-hot | description |
|---|---|---|---|
| 0 | `g1` | `[1,0,0,0]` | motion capture imitation — follows pre-recorded reference motion clips |
| 1 | `upperbody` | `[0,1,0,0]` | VR upper body teleop — VR 3-point (wrists + neck) controls arms, G1 lower body sensors as context |
| 2 | `wholebody` | `[0,0,1,0]` | VR full body teleop — SMPL 24-joint tracking drives the full 29-DOF output |
| 3 | — | `[0,0,0,1]` | reserved / unused |

## Encoder Input Layout (1762-dim)

| range | field | size | active mode | description |
|---|---|---|---|---|
| `[0:4]` | `encoder_mode_4` | 4 | all | mode one-hot |
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

## Decoder Input Layout (994-dim)

| range | field | size | description |
|---|---|---|---|
| `[0:64]` | `token_state` | 64 | encoder output |
| `[64:94]` | `his_base_angular_velocity_10f_s1` | 30 | base angular velocity history (10 frames × 3) |
| `[94:384]` | `his_body_joint_positions_10f_s1` | 290 | full body joint position history (10 frames × 29) |
| `[384:674]` | `his_body_joint_velocities_10f_s1` | 290 | full body joint velocity history (10 frames × 29) |
| `[674:964]` | `his_last_actions_10f_s1` | 290 | action history (10 frames × 29) |
| `[964:994]` | `his_gravity_dir_10f_s1` | 30 | gravity direction history (10 frames × 3) |

## ONNX Models

| model | input | output |
|---|---|---|
| `model_encoder.onnx` | `obs_dict [1, 1762] float32` | `encoded_tokens [1, 64] float32` |
| `model_decoder.onnx` | `obs_dict [1, 994] float32` | `action [1, 29] float32` |
