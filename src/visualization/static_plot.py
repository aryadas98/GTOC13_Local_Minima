import plotly.graph_objects as go

from orbital_mechanics.solar_system import SolarSystem
from common.constants import ALTAIRA_AU as AU

ss = SolarSystem()
df = ss.get_state_at_t(0)  # example snapshot
orbits = ss.get_orbit_points(ss.planets_idx, num_points=50)  # orbit points
df[['rx', 'ry', 'rz']] /= AU

fig = go.Figure()

# draw the sun
fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        text='Altaira',
        name='ALTAIRA',
        marker=dict(size=6, color='orange', opacity=1)
    ))

# Add all planets
planets = df.iloc[ss.planets_idx]
fig.add_trace(go.Scatter3d(
    x=planets['rx'], y=planets['ry'], z=planets['rz'],
    mode='markers+text',
    text=planets['name'],
    name='Planets',
    marker=dict(size=6, color='yellow')
))

# Add asteroids
asteroids = df.iloc[ss.asteroids_idx]
fig.add_trace(go.Scatter3d(
    x=asteroids['rx'], y=asteroids['ry'], z=asteroids['rz'],
    mode='markers',
    name='Asteroids',
    marker=dict(size=2, color='gray')
))

# Add comets
comets = df.iloc[ss.comets_idx]
fig.add_trace(go.Scatter3d(
    x=comets['rx'], y=comets['ry'], z=comets['rz'],
    mode='markers',
    name='Comets',
    marker=dict(size=3, color='lightblue')
))

# draw the orbits
for i in range(len(orbits)):
    row = orbits.iloc[i]
    pts = row['orbit']
    pts /= AU

    fig.add_trace(go.Scatter3d(
        x=pts[0], y=pts[1], z=pts[2],
        mode='lines',
        marker=dict(size=1, color='darkslategray'),
        name=' Planet Orbits',                      # all will share one legend entry
        legendgroup='orbits',               # this is the key line
        showlegend=(i == 0)                 # show only once in the legend
        ))

fig.update_layout(
    scene=dict(
        xaxis_title='X [AU]',
        yaxis_title='Y [AU]',
        zaxis_title='Z [AU]',
        aspectmode='data'
    ),
    legend=dict(x=0, y=1)
)

fig.show()