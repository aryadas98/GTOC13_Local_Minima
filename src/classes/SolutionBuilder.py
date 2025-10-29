from typing import Sequence, Union, TYPE_CHECKING

from .SolutionEvent import (
    SolutionEvent,
    FlybyEvent,
    ConicEvent,
    PropagatedEvent,
)

if TYPE_CHECKING:
    from .Solution import Solution


class SolutionBuilder:
    """Utility to assemble a Solution object from declarative event specs."""

    @classmethod
    def build(
        cls,
        event_specs: Sequence[Union[SolutionEvent, dict]],
        eps_time: float = 1e-6,
        eps_pos: float = 1e-6,
        eps_vel: float = 1e-6,
    ) -> "Solution":
        from .Solution import Solution  # Local import to avoid circular dependency

        solution = Solution()
        for spec in event_specs:
            if isinstance(spec, SolutionEvent):
                solution.add_event(spec, eps_time=eps_time, eps_pos=eps_pos, eps_vel=eps_vel)
                continue

            if not isinstance(spec, dict):
                raise ValueError("Each element must be a SolutionEvent or a specification dictionary.")

            event_type = spec.get("type")
            if not isinstance(event_type, str):
                raise ValueError("Event specification requires a 'type' string field.")

            event_type_lower = event_type.lower()
            if event_type_lower == "flyby":
                event = cls._build_flyby(spec)
                solution.add_event(event, eps_time=eps_time, eps_pos=eps_pos)
            elif event_type_lower == "conic":
                event = cls._build_conic(spec)
                solution.add_event(event, eps_time=eps_time)
            elif event_type_lower in {"propagated", "propagated_arc", "propagatedarc"}:
                event = cls._build_propagated(spec)
                solution.add_event(event, eps_time=eps_time, eps_pos=eps_pos, eps_vel=eps_vel)
            else:
                raise ValueError(f"Unsupported event type '{event_type}'.")

        solution._validate_transitions(eps_time=eps_time)
        return solution

    @classmethod
    def _build_flyby(cls, spec: dict) -> FlybyEvent:
        return FlybyEvent(
            spec["k"],
            spec["flag"],
            spec["epoch"],
            spec["position"],
            spec["velocity_incoming"],
            spec["velocity_outgoing"],
            spec.get("control_incoming"),
            spec.get("control_outgoing"),
        )

    @classmethod
    def _build_conic(cls, spec: dict) -> ConicEvent:
        return ConicEvent(
            spec["epoch_incoming"],
            spec["epoch_outgoing"],
            spec["position_incoming"],
            spec["position_outgoing"],
            spec["velocity_incoming"],
            spec["velocity_outgoing"],
        )

    @classmethod
    def _build_propagated(cls, spec: dict) -> PropagatedEvent:
        epochs = spec["epoch"]
        positions = spec["position"]
        velocities = spec["velocity"]
        controls = spec.get("control")

        return PropagatedEvent(epochs, positions, velocities, controls)
