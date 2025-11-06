import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from src.application.interfaces import MeshPlotter
from src.domain import Mesh2D

#pio.renderers.default = "browser"

pio.renderers.default = "browser"

PLOT_TABBING = 1 # extra plot range to avoid cutting of surfaces touching the extreams 


class GroundGridPlot(MeshPlotter):
    def __init__(self):
        super().__init__()

        self.X = None
        self.Y = None
        self.Z = None

    def plot(self, mesh: Mesh2D) -> None:
        grid = mesh.to_ground_grid()
        self.X = grid.X
        self.Y = grid.Y
        self.Z = grid.point_values["Z"]

        self.plot_plotly()

    def plot_plotly(self, title="Mesh2D Surface"):
        """
        Interactive Plotly surface with:
         - aspectmode='cube' (equal scales)
         - white faces, no colorbar
         - black grid-lines at each mesh step
        """
        # figure out your grid spacing
        # assumes X.shape == (M,N), X[i,j] = x_vals[i],  Y[i,j] = y_vals[j]
        x_vals = self.X[:, 0]
        y_vals = self.Y[0, :]
        dx = np.diff(x_vals).mean() if x_vals.size > 1 else 1.0
        dy = np.diff(y_vals).mean() if y_vals.size > 1 else 1.0

        surf = go.Surface(
            x=self.X,
            y=self.Y,
            z=self.Z,
            colorscale=[[0, "blue"], [1, "blue"]],
            showscale=False,
            contours={
                "x": dict(
                    show=True,
                    start=x_vals.min(),
                    end=x_vals.max(),
                    size=dx,
                    color="black",
                    width=1
                ),
                "y": dict(
                    show=True,
                    start=y_vals.min(),
                    end=y_vals.max(),
                    size=dy,
                    color="black",
                    width=1
                ),
                # you can enable z‐contours too if you want horizontal lines:
                # "z": dict(show=True, color="black", width=1, start=self.Z.min(), end=self.Z.max(), size=np.diff(np.unique(self.Z)).mean())
            }
        )

        # the range has to be set the same for all axis, lets figure out the delta and min values
        minX = self.X.min()
        minY = self.Y.min()
        minZ = self.Z.min()

        maxX = self.X.max()
        maxY = self.Y.max()
        maxZ = self.Z.max()

        maxDelta = max(maxX-minX, maxY-minY, maxZ-minZ)
        
        fig = go.Figure(data=[surf])

        camera = dict(
            up=dict(x=0, y=0, z=1), 
            center=dict(x=0, y=0, z=-0.5), # corner, ground level
            eye=dict(x=1.0, y=1.0, z=0.5) # oposite corner, mid point in Z
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X", range=[minX-PLOT_TABBING, minX+maxDelta+PLOT_TABBING]),
                yaxis=dict(title="Y", range=[minY-PLOT_TABBING, minY+maxDelta+PLOT_TABBING]),
                zaxis=dict(title="Z", range=[minZ-PLOT_TABBING, minZ+maxDelta+PLOT_TABBING]),
                aspectmode="manual",  # manual mode lets you set the ratio
                aspectratio=dict(x=1, y=1, z=1),
            ),
            scene_camera = camera,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        fig.show()

    def plot_wireframe(self, title="Wireframe Only"):
        """
        Draws only the grid lines of a structured mesh:
        - one 3D line per row
        - one 3D line per column
        - equal aspect scaling (cube)
        """
        n_i, n_j = self.X.shape
        traces = []

        # horizontal lines (rows)
        for i in range(n_i):
            traces.append(go.Scatter3d(
                x=self.X[i, :],
                y=self.Y[i, :],
                z=self.Z[i, :],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

        # vertical lines (columns)
        for j in range(n_j):
            traces.append(go.Scatter3d(
                x=self.X[:, j],
                y=self.Y[:, j],
                z=self.Z[:, j],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

        fig = go.Figure(data=traces)

        # the range has to be set the same for all axis, lets figure out the delta and min values
        minX = self.X.min()
        minY = self.Y.min()
        minZ = self.Z.min()

        maxX = self.X.max()
        maxY = self.Y.max()
        maxZ = self.Z.max()

        maxDelta = max(maxX-minX, maxY-minY, maxZ-minZ)

        camera = dict(
            up=dict(x=0, y=0, z=1), 
            center=dict(x=0, y=0, z=-0.5), # corner, ground level
            eye=dict(x=1.0, y=1.0, z=0.5) # oposite corner, mid point in Z
        )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title="X", range=[minX-PLOT_TABBING, minX+maxDelta+PLOT_TABBING]),
                yaxis=dict(title="Y", range=[minY-PLOT_TABBING, minY+maxDelta+PLOT_TABBING]),
                zaxis=dict(title="Z", range=[minZ-PLOT_TABBING, minZ+maxDelta+PLOT_TABBING]),
                aspectmode="manual",  # manual mode lets you set the ratio
                aspectratio=dict(x=1, y=1, z=1),
            ),
            scene_camera = camera,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        fig.show()