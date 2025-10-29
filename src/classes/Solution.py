from typing import List, Optional, Sequence, Union

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp

from .SolutionEvent import (
    SolutionEvent,
    FlybyEvent,
    ConicEvent,
    PropagatedEvent,
)
from .SolutionBuilder import SolutionBuilder
from .SolutionRow import SolutionRow
from .Vector import Vector3
from common.constants import ALTAIRA_MU


class Solution:
    def __init__(self):
        self.events: List[SolutionEvent] = []

    def add_event(self, event: SolutionEvent, eps_time: float = 1e-6,
                  eps_pos: float = 1e-6, eps_vel: float = 1e-6):
        if isinstance(event, FlybyEvent):
            event.validate(eps_time=eps_time, eps_pos=eps_pos)
        elif isinstance(event, ConicEvent):
            event.validate(eps_time=eps_time)
        elif isinstance(event, PropagatedEvent):
            event.validate(eps_time=eps_time, eps_pos=eps_pos, eps_vel=eps_vel)
        else:
            event.validate()
        self.events.append(event)

    @staticmethod
    def row_from_series(row: pd.Series, eps_control: float = 1e-9) -> SolutionRow:
        control_vec = Vector3(row["ux"], row["uy"], row["uz"])
        control = None if control_vec.is_close_to_zero(eps_control) else control_vec

        return SolutionRow(
            description="",
            body_id=int(row["#body_id"]),
            flag=int(row["flag"]),
            epoch=float(row["epoch"]),
            position=Vector3(row["rx"], row["ry"], row["rz"]),
            velocity=Vector3(row["vx"], row["vy"], row["vz"]),
            control=control
        )

    @classmethod
    def from_csv(cls, path: str, eps_time: float = 1e-6, eps_pos: float = 1e-6,
                 eps_vel: float = 1e-6, eps_control: float = 1e-9):
        solution = cls()

        df = pd.read_csv(path)
        rows = [cls.row_from_series(df.iloc[i], eps_control=eps_control)
                for i in range(len(df))]

        events: List[SolutionEvent] = []
        i = 0
        while i < len(rows):
            if cls._is_flyby(rows, i, eps_time, eps_pos):
                events.append(FlybyEvent([rows[i], rows[i + 1]]))
                i += 2
                continue

            if cls._is_conic(rows, i, eps_time, eps_control):
                events.append(ConicEvent([rows[i], rows[i + 1]]))
                i += 2
                continue

            if cls._is_propagated_start(rows[i]):
                bundle, count = cls._collect_propagated(rows, i)
                events.append(PropagatedEvent(bundle))
                i += count
                continue

            fallback_events, consumed = cls._coerce_remaining_into_flybys(rows, i)
            events.extend(fallback_events)
            i += consumed
            continue

        for event in events:
            solution.add_event(event, eps_time=eps_time, eps_pos=eps_pos, eps_vel=eps_vel)

        solution._validate_transitions(eps_time=eps_time)
        return solution

    @classmethod
    def from_events(cls, event_specs: Sequence[Union[SolutionEvent, dict]], eps_time: float = 1e-6,
                    eps_pos: float = 1e-6, eps_vel: float = 1e-6) -> "Solution":
        return SolutionBuilder.build(
            event_specs,
            eps_time=eps_time,
            eps_pos=eps_pos,
            eps_vel=eps_vel,
        )

    def to_csv(self, path: str):
        """Serialize solution rows to CSV in the same layout accepted by from_csv."""
        data = []
        for event in self.events:
            for row in event.rows:
                control = row.control if row.control is not None else Vector3(0.0, 0.0, 0.0)
                data.append({
                    "#body_id": row.body_id,
                    "flag": row.flag,
                    "epoch": row.epoch,
                    "rx": row.position.x,
                    "ry": row.position.y,
                    "rz": row.position.z,
                    "vx": row.velocity.x,
                    "vy": row.velocity.y,
                    "vz": row.velocity.z,
                    "ux": control.x,
                    "uy": control.y,
                    "uz": control.z,
                })

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def trajectory_samples(self, num_conic_points: int = 200) -> List[dict]:
        """Sample each event's trajectory for plotting purposes."""
        samples: List[dict] = []
        for event in self.events:
            if isinstance(event, FlybyEvent):
                row = event.rows[0]
                samples.append({
                    "type": event.type.value,
                    "epochs": [row.epoch],
                    "positions": [Vector3(row.position.x, row.position.y, row.position.z)],
                })
            elif isinstance(event, PropagatedEvent):
                epochs = [r.epoch for r in event.rows]
                positions = [Vector3(r.position.x, r.position.y, r.position.z) for r in event.rows]
                samples.append({
                    "type": event.type.value,
                    "epochs": epochs,
                    "positions": positions,
                })
            elif isinstance(event, ConicEvent):
                start, end = event.rows
                epochs, positions = self._integrate_conic_segment(start, end, num_conic_points)
                samples.append({
                    "type": event.type.value,
                    "epochs": epochs,
                    "positions": positions,
                })
        return samples

    @staticmethod
    def _is_flyby(rows: List[SolutionRow], index: int, eps_time: float, eps_pos: float) -> bool:
        if index + 1 >= len(rows):
            return False

        current, nxt = rows[index], rows[index + 1]
        if current.body_id <= 0 or nxt.body_id <= 0:
            return False
        if current.body_id != nxt.body_id or current.flag != nxt.flag:
            return False
        if abs(nxt.epoch - current.epoch) >= eps_time:
            return False
        if current.position.distance_to(nxt.position) >= eps_pos:
            return False
        return True

    @staticmethod
    def _is_zero_control(control: Optional[Vector3], eps_control: float) -> bool:
        if control is None:
            return True
        return control.is_close_to_zero(eps_control)

    @classmethod
    def _is_conic(cls, rows: List[SolutionRow], index: int, eps_time: float,
                  eps_control: float) -> bool:
        if index + 1 >= len(rows):
            return False

        start, end = rows[index], rows[index + 1]
        if start.body_id != 0 or end.body_id != 0:
            return False
        if start.flag != 0 or end.flag != 0:
            return False
        if not cls._is_zero_control(start.control, eps_control):
            return False
        if not cls._is_zero_control(end.control, eps_control):
            return False
        if end.epoch <= start.epoch + eps_time:
            return False
        return True

    @staticmethod
    def _is_propagated_start(row: SolutionRow) -> bool:
        return row.body_id == 0 and row.flag == 1

    @classmethod
    def _collect_propagated(cls, rows: List[SolutionRow], index: int) -> tuple[List[SolutionRow], int]:
        collected = [rows[index]]
        j = index + 1
        while j < len(rows) and cls._is_propagated_start(rows[j]):
            collected.append(rows[j])
            j += 1

        if len(collected) < 2:
            raise ValueError("Propagated arcs require at least 2 consecutive rows.")

        return collected, len(collected)

    @staticmethod
    def _epochs_match(t1: float, t2: float, eps_time: float) -> bool:
        return abs(t1 - t2) <= eps_time

    def _validate_transitions(self, eps_time: float):
        if not self.events:
            return

        for idx, event in enumerate(self.events):
            start_row = event.rows[0]
            end_row = event.rows[-1]

            if idx > 0:
                prev_end = self.events[idx - 1].rows[-1]
                assert self._epochs_match(prev_end.epoch, start_row.epoch, eps_time), \
                    "Arc transitions must share the boundary epoch."

            if idx < len(self.events) - 1:
                next_start = self.events[idx + 1].rows[0]
                assert self._epochs_match(next_start.epoch, end_row.epoch, eps_time), \
                    "Arc transitions must share the boundary epoch."

            if isinstance(event, FlybyEvent):
                assert idx > 0, "Flyby arcs must be preceded by a heliocentric arc."
                prev_end = self.events[idx - 1].rows[-1]
                assert prev_end.body_id == 0, "Flyby must be preceded by heliocentric state."
                assert self._epochs_match(prev_end.epoch, start_row.epoch, eps_time), \
                    "Flyby entry epoch must match preceding arc end."

                if idx < len(self.events) - 1:
                    next_start = self.events[idx + 1].rows[0]
                    assert next_start.body_id == 0, "Flyby must be followed by heliocentric state."
                    assert self._epochs_match(next_start.epoch, start_row.epoch, eps_time), \
                        "Flyby exit epoch must match following arc start."

    @classmethod
    def _coerce_remaining_into_flybys(
        cls,
        rows: List[SolutionRow],
        start: int,
    ) -> tuple[List[SolutionEvent], int]:
        remaining = rows[start:]
        if not remaining:
            return [], 0

        consumed = len(remaining)
        working_block: List[SolutionRow] = [cls._clone_row(r) for r in remaining]

        if len(working_block) % 2 == 1:
            working_block.append(cls._clone_row(working_block[-1]))

        flyby_events: List[SolutionEvent] = []
        for idx in range(0, len(working_block), 2):
            incoming = working_block[idx]
            outgoing = working_block[idx + 1]
            pair = cls._normalize_flyby_pair(incoming, outgoing)
            flyby_events.append(FlybyEvent(pair))

        return flyby_events, consumed

    @staticmethod
    def _clone_row(row: SolutionRow) -> SolutionRow:
        control_clone = None
        if row.control is not None:
            control_clone = Vector3(float(row.control.x), float(row.control.y), float(row.control.z))

        return SolutionRow(
            description=row.description,
            body_id=row.body_id,
            flag=row.flag,
            epoch=row.epoch,
            position=Vector3(float(row.position.x), float(row.position.y), float(row.position.z)),
            velocity=Vector3(float(row.velocity.x), float(row.velocity.y), float(row.velocity.z)),
            control=control_clone,
        )

    @classmethod
    def _normalize_flyby_pair(cls, incoming: SolutionRow, outgoing: SolutionRow) -> List[SolutionRow]:
        incoming_clone = cls._clone_row(incoming)

        outgoing_velocity = Vector3(float(outgoing.velocity.x), float(outgoing.velocity.y), float(outgoing.velocity.z))
        outgoing_control = None
        if outgoing.control is not None:
            outgoing_control = Vector3(float(outgoing.control.x), float(outgoing.control.y), float(outgoing.control.z))

        outgoing_row = SolutionRow(
            description=outgoing.description,
            body_id=incoming_clone.body_id,
            flag=incoming_clone.flag,
            epoch=incoming_clone.epoch,
            position=Vector3(
                float(incoming_clone.position.x),
                float(incoming_clone.position.y),
                float(incoming_clone.position.z),
            ),
            velocity=outgoing_velocity,
            control=outgoing_control,
        )

        return [incoming_clone, outgoing_row]

    @staticmethod
    def _integrate_conic_segment(start: SolutionRow, end: SolutionRow, num_samples: int) -> tuple[List[float], List[Vector3]]:
        duration = end.epoch - start.epoch
        if duration <= 0:
            return [start.epoch, end.epoch], [
                Vector3(start.position.x, start.position.y, start.position.z),
                Vector3(end.position.x, end.position.y, end.position.z),
            ]

        y0 = np.array([
            start.position.x,
            start.position.y,
            start.position.z,
            start.velocity.x,
            start.velocity.y,
            start.velocity.z,
        ], dtype=float)

        def dynamics(_, state):
            x, y, z, vx, vy, vz = state
            r_sq = x * x + y * y + z * z
            r = np.sqrt(r_sq)
            if r == 0.0:
                raise RuntimeError("Conic integration encountered zero radius.")
            inv_r3 = 1.0 / (r * r_sq)
            ax = -ALTAIRA_MU * x * inv_r3
            ay = -ALTAIRA_MU * y * inv_r3
            az = -ALTAIRA_MU * z * inv_r3
            return np.array([vx, vy, vz, ax, ay, az], dtype=float)

        t_eval = np.linspace(0.0, duration, max(2, num_samples))
        solution = solve_ivp(
            dynamics,
            (0.0, duration),
            y0,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )

        if not solution.success:
            raise RuntimeError(f"Conic arc integration failed: {solution.message}")

        epochs = [start.epoch + float(t) for t in solution.t]
        positions = [
            Vector3(float(solution.y[0, i]), float(solution.y[1, i]), float(solution.y[2, i]))
            for i in range(solution.y.shape[1])
        ]

        return epochs, positions

if __name__ == "__main__":
    try:
        solution = Solution.from_csv(
            "/home/shin0bi/dev/IITInternship/GTOC13_Local_Minima/src/dev/rogue1_intercept.csv"
        )
    except ValueError as exc:
        print(f"Failed to build solution: {exc}")
    else:
        for event in solution.events:
            print(event.type, "Rows:", len(event.rows))




