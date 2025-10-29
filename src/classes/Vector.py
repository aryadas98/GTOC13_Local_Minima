import math
from dataclasses import dataclass

@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def distance_to(self, other: "Vector3") -> float:
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2 +
                         (self.z - other.z) ** 2)

    def norm(self) -> float:
        """Euclidean norm of the vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def is_close_to_zero(self, eps: float = 1e-9) -> bool:
        """Check if vector magnitude is effectively zero within tolerance."""
        return self.norm() <= eps
