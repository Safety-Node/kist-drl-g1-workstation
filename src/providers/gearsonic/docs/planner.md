# GearSonic — Planner

## Architecture

![Planner Architecture](planner_architecture.png)

## Model

| model | description |
|---|---|
| `planner_sonic.onnx` | locomotion trajectory planner |

## Locomotion Modes

| mode | name |
|---|---|
| 0 | `IDLE` |
| 1 | `SLOW_WALK` |
| 2 | `WALK` |
| 3 | `RUN` |
| 4 | `IDLE_SQUAT` |
| 5 | `IDLE_KNEEL_TWO_LEGS` |
| 6 | `IDLE_KNEEL` |
| 7 | `IDLE_LYING` |
| 8 | `IDLE_CRAWLING` |
| 9 | `IDLE_BOXING` |
| 10 | `WALK_BOXING` |
| 11 | `LEFT_PUNCH` |
| 12 | `RIGHT_PUNCH` |
| 13 | `RANDOM_PUNCH` |
| 14 | `ELBOW_CRAWLING` |
| 15 | `LEFT_HOOK` |
| 16 | `RIGHT_HOOK` |
| 17 | `FORWARD_JUMP` |
| 18 | `STEALTH_WALK` |
| 19 | `INJURED_WALK` |

## Input

| range | field | shape | description |
|---|---|---|---|
| `[1]` | `context_mujoco_qpos` | `[1,4,36]` | robot state history — 4 frames × 36 (7 floating base + 29 joints in isaaclab order), float32 |
| `[2]` | `target_vel` | `[1]` | target speed m/s (−1 = default), float32 |
| `[3]` | `mode` | `[1]` | locomotion mode (int64): see Locomotion Modes |
| `[4]` | `movement_direction` | `[1,3]` | direction to move (unit vector), float32 |
| `[5]` | `facing_direction` | `[1,3]` | direction to face (unit vector), float32 |
| `[6]` | `random_seed` | `[1]` | random seed for stochastic generation, int64 |
| `[7]` | `has_specific_target` | `[1,1]` | specific waypoint target (int64): 0=no, 1=yes |
| `[8]` | `specific_target_positions` | `[1,4,3]` | 4 waypoint positions xyz, float32 |
| `[9]` | `specific_target_headings` | `[1,4]` | 4 waypoint heading angles, float32 |
| `[10]` | `allowed_pred_num_tokens` | `[1,11]` | prediction token mask, int64 (default: [1,1,1,1,1,1,0,0,0,0,0]) |
| `[11]` | `height` | `[1]` | target body height in meters (−1 = use model default), float32 |

## Output

| range | field | shape | description |
|---|---|---|---|
| `[1]` | `mujoco_qpos` | `[1,64,36]` | future trajectory — 64 frames × 36 (7 floating base + 29 joints in isaaclab order), float32 |
| `[2]` | `num_pred_frames` | `[1]` | number of valid predicted frames, int32 |
