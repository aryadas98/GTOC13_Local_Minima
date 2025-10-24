import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from orbital_mechanics.solar_system import SolarSystem
from orbital_mechanics.constants import AU, YEAR

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


app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.Div([
        html.H2("Altaira System Visualizer (GTOC13)"),
        dcc.Graph(id='orbit-plot', style={'height': '90vh', 'width': '100%'}),
        dcc.Store(id='camera-store'),
    ], style={'flex': '3', 'padding': '10px'}),

    html.Div([
        html.H4("Controls"),
        html.Label("Time (years):"),
        dcc.Slider(id='time-slider', min=0, max=200, step=1, value=0,
                   marks={i:str(i) for i in [0,25,50,75,100,125,150,175,200]}),

        html.Label("Body Types:"),
        dcc.Checklist(
            id='body-types',
            options=[
                {'label': 'Planets', 'value': 'planet'},
                {'label': 'Asteroids', 'value': 'asteroid'},
                {'label': 'Comets', 'value': 'comet'},
            ],
            value=['planet', 'asteroid', 'comet']
        ),
    ], style={'flex': '1', 'padding': '10px'})
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

# Callback
@app.callback(
    Output('orbit-plot', 'figure'),
    Input('time-slider', 'value'),
    Input('body-types', 'value'),
    State('camera-store', 'data'),
)
def update_plot(time_value, selected_types, camera_state):
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


if __name__ == '__main__':
    app.run(debug=True)
