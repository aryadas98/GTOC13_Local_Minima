from dataclasses import dataclass
from typing import Optional
from .Vector import Vector3

@dataclass
class SolutionRow:
    description: str
    body_id: int
    flag: int
    epoch: float
    position: Vector3
    velocity: Vector3
    control: Optional[Vector3] = None  # None or [∞,∞,∞] for flyby