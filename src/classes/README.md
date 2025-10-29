# Classes Overview

This package gathers the building blocks used to load, validate, manipulate, and analyse trajectory solutions for the Altaira system challenge. The notes below document the key functions provided by each module so you can quickly identify where to extend the pipeline or hook in custom logic.

## Solution.py

- `Solution.add_event(event, eps_time=1e-6, eps_pos=1e-6, eps_vel=1e-6)`: Dispatches to the event-specific `validate` routine before appending to the solution timeline.
- `Solution.row_from_series(row, eps_control=1e-9)`: Converts a pandas row into a `SolutionRow`, coercing near-zero control vectors to `None`.
- `Solution.from_csv(path, eps_time=1e-6, eps_pos=1e-6, eps_vel=1e-6, eps_control=1e-9)`: Parses the canonical CSV format, segments rows into flyby/conic/propagated events, validates each arc, and returns a populated `Solution`.
- `Solution.from_events(event_specs, eps_time=1e-6, eps_pos=1e-6, eps_vel=1e-6)`: Creates a solution from a mix of specification dictionaries and `SolutionEvent` objects via `SolutionBuilder`.
- `Solution.to_csv(path)`: Serialises the current event list back to CSV, writing zero-vector controls wherever thrust is absent.
- `Solution.trajectory_samples(num_conic_points=200)`: Emits epoch/position samples suitable for plotting; conic arcs are re-integrated through `_integrate_conic_segment` for smooth traces.
- `_is_flyby`, `_is_conic`, `_is_propagated_start`, `_collect_propagated`, `_coerce_remaining_into_flybys`: Helper routines that classify and normalise raw rows during CSV import.
- `_validate_transitions(eps_time)`: Ensures consecutive events meet at the same epoch and that flybys sit between heliocentric arcs.
- `_integrate_conic_segment(start, end, num_samples)`: Numerically solves the two-body problem between conic endpoints using `scipy.integrate.solve_ivp` to support trajectory overlays.

## SolutionEvent.py

- `SolutionEvent`: Abstract base storing the event `type` and associated `rows`; subclasses override `validate`.
- `FlybyEvent.__init__`: Accepts either a list of `SolutionRow` instances or `(k, flag, epoch, position, v_minus, v_plus, u_minus, u_plus)` data and builds the canonical incoming/outgoing pair.
- `FlybyEvent.validate(eps_time=1e-6, eps_pos=1e-6)`: Confirms body/flag agreement, identical epochs, and co-located positions for the two flyby rows.
- `ConicEvent.__init__`: Builds heliocentric start/end rows from a list or `(t_start, t_end, r_start, r_end, v_start, v_end)` tuple.
- `ConicEvent.validate(eps_time=1e-6)`: Checks for heliocentric markers, zero control vectors, and strictly increasing epoch across the segment.
- `PropagatedEvent.__init__`: Accepts a row list or sequences of `(epochs, positions, velocities, controls)` and funnels them through `_rows_from_sequences`.
- `PropagatedEvent.validate(eps_time=1e-6, eps_pos=1e-6, eps_vel=1e-6)`: Verifies heliocentric propagated arcs, minimum step size (unless handling control discontinuities), and state continuity when epochs repeat.
- `PropagatedEvent._rows_from_sequences(epochs, positions, velocities, controls=None)`: Vectorises input sequences into `SolutionRow` objects while enforcing matching lengths and converting controls to `Vector3` instances.
- `_as_vector3(value)`: Module helper that casts 3-tuples into `Vector3` objects, ensuring constructors accept both raw sequences and `Vector3` values.

## SolutionRow.py

- `SolutionRow`: Dataclass representing a single state/control sample (metadata, epoch, position, velocity, and optional control vector) consumed by every event and CSV transformer.

## SolutionBuilder.py

- `SolutionBuilder.build(event_specs, eps_time=1e-6, eps_pos=1e-6, eps_vel=1e-6)`: Iterates event specifications, instantiates the correct subclass, appends validated events to a `Solution`, and enforces boundary continuity.
- `SolutionBuilder._build_flyby(spec)`: Constructs a `FlybyEvent` from dictionary keys (`k`, `flag`, `epoch`, `position`, `velocity_incoming`, `velocity_outgoing`, optional `control_incoming`, optional `control_outgoing`).
- `SolutionBuilder._build_conic(spec)`: Translates dictionary data into a `ConicEvent` spanning `epoch_incoming` to `epoch_outgoing` with matching state vectors.
- `SolutionBuilder._build_propagated(spec)`: Generates a `PropagatedEvent` from sequences of epochs, state vectors, and optional controls.

## Vector.py

- `Vector3.distance_to(other)`: Returns the Euclidean separation between vectors; used in validation tolerances.
- `Vector3.norm()`: Computes vector magnitude.
- `Vector3.is_close_to_zero(eps=1e-9)`: Tests whether the magnitude falls below a tolerance, signalling an absent thrust command.

## test_solution.py

- `test_solution.py`: Pytest module exercising event classification, builder dictionary ingestion, CSV round-tripping, propagated control discontinuities, and trajectory sampling to document expected behaviours.
