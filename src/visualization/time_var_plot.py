from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from orbital_mechanics.solar_system import SolarSystem
from common.constants import ALTAIRA_AU as AU, YEAR

PLOT = True

# Initialize your model
solar_system = SolarSystem()

# calculate the orbits
orbits = solar_system.get_orbit_points(solar_system.planets_idx,
                                           num_points=50)  # orbit points

# Create static base figure with orbits once
base_fig = go.Figure()

# draw the sun
base_fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        text='Altaira',
        name='ALTAIRA',
        marker=dict(size=6, color='orange', opacity=1)
    ))

# draw the orbits (only once)
for i in range(len(orbits)):
    row = orbits.iloc[i]
    pts = row['orbit'] / AU

    base_fig.add_trace(go.Scatter3d(
        x=pts[0], y=pts[1], z=pts[2],
        mode='lines',
        marker=dict(size=1, color='darkslategray'),
        name=' Planet Orbits',                      # all will share one legend entry
        legendgroup='orbits',               # this is the key line
        showlegend=(i == 0)                 # show only once in the legend
        ))


def _load_default_solution_samples():
    if not PLOT:
        return []

    solution_path = Path(__file__).resolve().parents[1] / "dev" / "capture_test.csv"
    if not solution_path.exists():
        return []

    try:
        from classes.Solution import Solution
        solution = Solution.from_csv(str(solution_path))
        return solution.trajectory_samples()
    except Exception:
        return []


SOLUTION_SAMPLES = _load_default_solution_samples()


app = dash.Dash(__name__)

# Layout
controls_children = [
    html.H4("Controls"),
    html.Label("Time (years):"),
    dcc.Slider(
        id='time-slider',
        min=0,
        max=200,
        step=1,
        value=0,
        marks={i: str(i) for i in [0, 25, 50, 75, 100, 125, 150, 175, 200]},
    ),
    html.Label("Body Types:"),
    dcc.Checklist(
        id='body-types',
        options=[
            {'label': 'Planets', 'value': 'planet'},
            {'label': 'Asteroids', 'value': 'asteroid'},
            {'label': 'Comets', 'value': 'comet'},
        ],
        value=['planet', 'asteroid', 'comet'],
    ),
]

if PLOT:
    controls_children.extend([
        html.Label("Overlays:"),
        dcc.Checklist(
            id='overlays',
            options=[
                {'label': 'Solution trajectory', 'value': 'solution'},
            ],
            value=[],
        ),
    ])


app.layout = html.Div([
    html.Div([
        html.H2("Altaira System Visualizer (GTOC13)"),
        dcc.Graph(id='orbit-plot', style={'height': '90vh', 'width': '100%'}),
        dcc.Store(id='camera-store'),
    ], style={'flex': '3', 'padding': '10px'}),

    html.Div(controls_children, style={'flex': '1', 'padding': '10px'})
],
style={
    'display': 'flex',
    'flexDirection': 'row',
    'height': '100vh'
})

# 1️⃣ Callback to store camera state when user moves it
@app.callback(
    Output('camera-store', 'data'),
    Input('orbit-plot', 'relayoutData'),
    State('camera-store', 'data'),
    prevent_initial_call=True
)
def save_camera_state(relayout_data, current_state):
    if relayout_data and 'scene.camera' in relayout_data:
        return relayout_data['scene.camera']
    return current_state

def _render_plot(time_value, selected_types, overlays, camera_state):
    time_value *= YEAR
    df = solar_system.get_state_at_t(time_value)
    df[['rx', 'ry', 'rz']] /= AU

    fig = base_fig.to_dict()        # copy as dict
    fig = go.Figure(fig)            # recreate figure

    for btype, color, size in [('planet', 'yellow', 6),
                               ('asteroid', 'gray', 2),
                               ('comet', 'lightblue', 3)]:
        if btype in selected_types:
            data = df[df['type'] == btype]
            fig.add_trace(go.Scatter3d(
                x=data['rx'], y=data['ry'], z=data['rz'],
                mode='markers' if btype != 'planet' else 'markers+text',
                text=data['name'],
                name=btype.capitalize() + 's',
                marker=dict(size=size, color=color, opacity=1)
            ))

    overlays = overlays or []
    if 'solution' in overlays and SOLUTION_SAMPLES:
        for trace in _build_solution_traces(SOLUTION_SAMPLES, time_value):
            fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X [AU]', range=[-200, 200], autorange=False),
            yaxis=dict(title='Y [AU]', range=[-200, 200], autorange=False),
            zaxis=dict(title='Z [AU]', range=[-100, 100], autorange=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=100/200)
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    # Restore previous camera view if available
    if camera_state:
        fig.update_layout(scene_camera=camera_state)

    return fig


if PLOT:
    @app.callback(
        Output('orbit-plot', 'figure'),
        Input('time-slider', 'value'),
        Input('body-types', 'value'),
        Input('overlays', 'value'),
        State('camera-store', 'data'),
    )
    def update_plot(time_value, selected_types, overlays, camera_state):
        return _render_plot(time_value, selected_types, overlays, camera_state)
else:
    @app.callback(
        Output('orbit-plot', 'figure'),
        Input('time-slider', 'value'),
        Input('body-types', 'value'),
        State('camera-store', 'data'),
    )
    def update_plot_no_overlay(time_value, selected_types, camera_state):
        return _render_plot(time_value, selected_types, [], camera_state)


def _build_solution_traces(samples, time_limit_seconds):
    traces = []
    legend_tracker = {}
    limit = float(time_limit_seconds)

    color_map = {
        'Flyby': 'red',
        'Conic': 'royalblue',
        'Propagated': 'limegreen',
    }

    for segment in samples:
        seg_type = segment['type']
        epochs = segment['epochs']
        positions = segment['positions']

        indices = [idx for idx, epoch in enumerate(epochs) if epoch <= limit + 1e-6]
        if not indices:
            continue

        xs = [positions[idx].x / AU for idx in indices]
        ys = [positions[idx].y / AU for idx in indices]
        zs = [positions[idx].z / AU for idx in indices]

        color = color_map.get(seg_type, 'white')
        legend_group = f'solution-{seg_type.lower()}'
        showlegend = not legend_tracker.get(seg_type, False)
        legend_tracker[seg_type] = True

        if seg_type == 'Flyby' or len(indices) == 1:
            mode = 'markers'
            trace = go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode=mode,
                name=f'Solution {seg_type}',
                legendgroup=legend_group,
                showlegend=showlegend,
                marker=dict(size=6, color=color, symbol='diamond'),
            )
        else:
            trace = go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='lines+markers',
                name=f'Solution {seg_type}',
                legendgroup=legend_group,
                showlegend=showlegend,
                marker=dict(size=3, color=color),
                line=dict(color=color, width=3),
            )

        traces.append(trace)

    return traces


if __name__ == '__main__':
    app.run(debug=True)
