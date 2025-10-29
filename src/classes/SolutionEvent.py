from enum import Enum
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence
from .SolutionRow import SolutionRow
from .Vector import Vector3

class EventType(Enum):
    FLYBY = "Flyby"
    CONIC = "Conic"
    PROPAGATED = "Propagated"


@dataclass
class SolutionEvent:
    type: EventType
    rows: List[SolutionRow] = field(default_factory=list)

    def validate(self):
        raise NotImplementedError("Validation not implemented for base class")

def _as_vector3(value: Vector3 | Sequence[float]) -> Vector3:
    if isinstance(value, Vector3):
        return value
    if isinstance(value, Sequence) and len(value) == 3:
        return Vector3(float(value[0]), float(value[1]), float(value[2]))
    raise ValueError("Vector value must be a Vector3 or a sequence of three numeric values.")


@dataclass(init=False)
class FlybyEvent(SolutionEvent):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            rows = args[0]
        elif len(args) == 8:
            k, flag, epoch, position, v_minus, v_plus, u_minus, u_plus = args
            pos_vec = _as_vector3(position)
            vel_in = _as_vector3(v_minus)
            vel_out = _as_vector3(v_plus)
            control_in = None if u_minus is None else _as_vector3(u_minus)
            control_out = None if u_plus is None else _as_vector3(u_plus)

            rows = [
                SolutionRow(
                    description="Flyby incoming",
                    body_id=int(k),
                    flag=int(flag),
                    epoch=float(epoch),
                    position=pos_vec,
                    velocity=vel_in,
                    control=control_in,
                ),
                SolutionRow(
                    description="Flyby outgoing",
                    body_id=int(k),
                    flag=int(flag),
                    epoch=float(epoch),
                    position=Vector3(pos_vec.x, pos_vec.y, pos_vec.z),
                    velocity=vel_out,
                    control=control_out,
                ),
            ]
        else:
            raise TypeError("FlybyEvent expects either a list of rows or the parameters (k, f, epoch, position, v_minus, v_plus, u_minus, u_plus).")

        super().__init__(type=EventType.FLYBY, rows=rows)

    def validate(self, eps_time=1e-6, eps_pos=1e-6):
        assert len(self.rows) == 2, "Flyby must have exactly 2 rows."
        r1, r2 = self.rows

        assert r1.body_id > 0 and r2.body_id > 0, "Body ID must be >0 for flyby."
        assert r1.body_id == r2.body_id, "Body ID mismatch in flyby rows."
        assert r1.flag == r2.flag, "Flags must match in flyby."
        assert abs(r2.epoch - r1.epoch) < eps_time, "Epochs must be identical."
        assert r1.position.distance_to(r2.position) < eps_pos, "Positions must match."


@dataclass(init=False)
class ConicEvent(SolutionEvent):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            rows = args[0]
        elif len(args) == 6:
            t_start, t_end, r_start, r_end, v_start, v_end = args
            rows = [
                SolutionRow(
                    description="Conic start",
                    body_id=0,
                    flag=0,
                    epoch=float(t_start),
                    position=_as_vector3(r_start),
                    velocity=_as_vector3(v_start),
                    control=None,
                ),
                SolutionRow(
                    description="Conic end",
                    body_id=0,
                    flag=0,
                    epoch=float(t_end),
                    position=_as_vector3(r_end),
                    velocity=_as_vector3(v_end),
                    control=None,
                ),
            ]
        else:
            raise TypeError("ConicEvent expects either a list of rows or the parameters (t_start, t_end, r_start, r_end, v_start, v_end).")

        super().__init__(type=EventType.CONIC, rows=rows)

    def validate(self, eps_time=1e-6):
        assert len(self.rows) == 2, "Conic arc must have exactly 2 rows."
        start, end = self.rows

        assert start.body_id == 0 and end.body_id == 0, "Body ID must be 0 (heliocentric)."
        assert start.flag == 0 and end.flag == 0, "Flag must be 0 for conic propagation."
        assert self._control_is_zero(start.control) and self._control_is_zero(end.control), \
            "Control must be zero for conic arcs."
        assert end.epoch > start.epoch + eps_time, "Epoch must strictly increase."

    @staticmethod
    def _control_is_zero(control):
        if control is None:
            return True
        if isinstance(control, Vector3):
            return control.is_close_to_zero()
        return False


@dataclass(init=False)
class PropagatedEvent(SolutionEvent):
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            rows = args[0]
        elif len(args) == 4:
            epochs, positions, velocities, controls = args
            rows = PropagatedEvent._rows_from_sequences(epochs, positions, velocities, controls)
        else:
            raise TypeError(
                "PropagatedEvent expects either a list of rows or the parameters (epochs, positions, velocities, controls)."
            )

        super().__init__(type=EventType.PROPAGATED, rows=rows)

    def validate(self, eps_time=1e-6, eps_pos=1e-6, eps_vel=1e-6):
        assert len(self.rows) >= 2, "Propagated arc must have at least 2 rows."
        for i in range(len(self.rows) - 1):
            r1, r2 = self.rows[i], self.rows[i + 1]
            assert r1.body_id == 0 and r2.body_id == 0, "Body ID must be 0 for propagated."
            assert r1.flag == 1 and r2.flag == 1, "Flag must be 1 for propagated arcs."
            assert r2.epoch >= r1.epoch, "Time must be non-decreasing."
            assert r2.epoch - r1.epoch >= 60 or abs(r2.epoch - r1.epoch) < eps_time, \
                "Minimum 60-second timestep unless control discontinuity."

            # If same epoch (control discontinuity)
            if abs(r2.epoch - r1.epoch) < eps_time:
                assert r1.position.distance_to(r2.position) < eps_pos, "Position mismatch at discontinuity."
                assert r1.velocity.distance_to(r2.velocity) < eps_vel, "Velocity mismatch at discontinuity."

    @staticmethod
    def _rows_from_sequences(
        epochs: Iterable[float],
        positions: Iterable[Vector3 | Sequence[float]],
        velocities: Iterable[Vector3 | Sequence[float]],
        controls: Optional[Iterable[Optional[Vector3 | Sequence[float]]]] = None,
    ) -> List[SolutionRow]:
        epoch_list = [float(e) for e in epochs]
        pos_list = [_as_vector3(p) for p in positions]
        vel_list = [_as_vector3(v) for v in velocities]

        length = len(epoch_list)
        if length < 2:
            raise ValueError("Propagated arc requires at least 2 rows.")
        if not (len(pos_list) == len(vel_list) == length):
            raise ValueError("Propagated arc epoch, position, and velocity sequences must share the same length.")

        if controls is None:
            ctrl_list = [None] * length
        else:
            ctrl_list = [None if c is None else _as_vector3(c) for c in controls]
            if len(ctrl_list) != length:
                raise ValueError("Propagated arc control sequence must match the length of epochs.")

        rows: List[SolutionRow] = []
        for idx in range(length):
            rows.append(
                SolutionRow(
                    description="Propagated",
                    body_id=0,
                    flag=1,
                    epoch=epoch_list[idx],
                    position=pos_list[idx],
                    velocity=vel_list[idx],
                    control=ctrl_list[idx],
                )
            )

        return rows



