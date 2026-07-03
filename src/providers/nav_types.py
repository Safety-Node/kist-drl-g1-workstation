from dataclasses import dataclass


@dataclass(frozen=True)
class NavVelCmd:
    vx:   float = 0.0
    vy:   float = 0.0
    vyaw: float = 0.0
