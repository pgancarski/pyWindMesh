import numpy as np
import plotly.graph_objects as go
from config import Config

import plotly.io as pio
pio.renderers.default = "browser"

PLOT_TABBING = 1 # extra plot range to avoid cutting of surfaces touching the extreams

# helper to compute angle between a,b: arccos((a·b)/(|a||b|))
def angle(a, b):
    dot = np.sum(a*b, axis=-1)
    na  = np.linalg.norm(a, axis=-1)
    nb  = np.linalg.norm(b, axis=-1)
    # clamp to [-1,1] to avoid NaNs
    cosang = np.clip(dot/(na*nb), -1.0, 1.0)
    return np.arccos(cosang)

class Mesh2D:
    """
    Constructs a regular 2D mesh based on a configuration dictionary.

    Attributes:
        X (np.ndarray): 2D array of X-coordinates.
        Y (np.ndarray): 2D array of Y-coordinates.
        Z (np.ndarray): 2D array of Z-coordinates (initialized to zeros).
    """
    def __init__(self, config: Config):
        # Unpack configuration
        xt = config.farm_xt
        xf = config.farm_xf
        yt = config.farm_yt
        yf = config.farm_yf
        dx = config.farm_cellsize_x
        dy = config.farm_cellsize_y

        # Determine number of points along each axis
        nx = int(round((xf - xt) / dx)) + 1
        ny = int(round((yf - yt) / dy)) + 1

        # Generate linspace for each axis
        x_vals = np.linspace(xt, xf, nx)
        y_vals = np.linspace(yt, yf, ny)

        # Create coordinate grids
        self.X, self.Y = np.meshgrid(x_vals, y_vals, indexing='ij')

        # Initialize Z to zero
        self.Z = np.zeros_like(self.X)

    def get_point(self, ix: int, iy: int) -> tuple[float, float, float]:
        """
        Retrieve the point at mesh indices (ix, iy).

        Args:
            ix (int): Index along the X-direction (0 <= ix < nx).
            iy (int): Index along the Y-direction (0 <= iy < ny).

        Returns:
            Point: The Point object at the specified indices.
        """
        x = float(self.X[ix, iy])
        y = float(self.Y[ix, iy])
        z = float(self.Z[ix, iy])
        return x, y, z

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the mesh as (nx, ny).
        """
        return self.X.shape
    
    def check_mesh_quality(self):
        max_skewness, mean_skewness = self.mesh_skewness_stats()

        print("The maximum skewness angle [rad]: ",max_skewness)
        print("The mean skewness angle [rad]: ",mean_skewness)
    
    def mesh_skewness_stats(self):
        """
        Given X, Y, Z of shape (n, m), interpreted as a regular quad‐mesh,
        compute for each quad (cell) its skewness = max_i(|angle_i - 90°|)/90°,
        where angle_i are the four interior angles of the quad.
        Returns (max_skewness, avg_skewness), both in [0, 1].
        """
        # pack into points array of shape (n, m, 3)
        P = np.stack((self.X, self.Y, self.Z), axis=-1)

        # grab the four corners of each quad:
        P00 = P[:-1, :-1]   # lower‐left
        P10 = P[1:,  :-1]   # upper‐left
        P11 = P[1:,  1:]    # upper‐right
        P01 = P[:-1, 1:]    # lower‐right

        # for each corner, form the two edges meeting there:
        # at P00: edges to P10 and to P01
        e00a = P10 - P00
        e00b = P01 - P00

        # at P10: edges to P11 and to P00
        e10a = P11 - P10
        e10b = P00 - P10

        # at P11: edges to P01 and to P10
        e11a = P01 - P11
        e11b = P10 - P11

        # at P01: edges to P00 and to P11
        e01a = P00 - P01
        e01b = P11 - P01

        # compute the four angle arrays (shape (n-1, m-1))
        θ00 = angle(e00a, e00b)
        θ10 = angle(e10a, e10b)
        θ11 = angle(e11a, e11b)
        θ01 = angle(e01a, e01b)

        # deviation from 90° = |θ - π/2|
        dev00 = np.abs(θ00 - np.pi/2)
        dev10 = np.abs(θ10 - np.pi/2)
        dev11 = np.abs(θ11 - np.pi/2)
        dev01 = np.abs(θ01 - np.pi/2)

        # per‐face skewness = (max deviation)/(π/2)
        max_dev = np.maximum.reduce([dev00, dev10, dev11, dev01])
        skewness = max_dev / (np.pi/2)

        return skewness.max(), skewness.mean()
        
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

        center = 50

        surf = go.Surface(
            x=self.X[center-10 : center+10, :],
            y=self.Y[center-10 : center+10, :],
            z=self.Z[center-10 : center+10, :],
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