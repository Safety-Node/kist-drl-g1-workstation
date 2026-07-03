from dataclasses import dataclass

import numpy as np


@dataclass
class PlannerCommand:
    mode: int
    target_vel: float
    movement_direction: np.ndarray  # (3,) float32 unit vector
    facing_direction: np.ndarray    # (3,) float32 unit vector
    random_seed: int
