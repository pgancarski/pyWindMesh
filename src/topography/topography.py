from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import plotly.io as pio

from config import Config
pio.renderers.default = "notebook"

@dataclass(slots=True, kw_only=True)
class Topography(ABC):
    config:Config
    xmin:float = 0
    xmax:float = 0
    ymin:float = 0
    ymax:float = 0

    def __post_init__(self) -> None:
        self.xmin = self.config.farm_xt
        self.xmax = self.config.farm_xf
        self.ymin = self.config.farm_yt
        self.ymax = self.config.farm_yf

        # Validate domain range
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Domain maxima must be greater than minima.")
            

    # --------------------------------------------------------------------- #
    # the one piece of behaviour sub-classes must supply
    # --------------------------------------------------------------------- #
    @abstractmethod
    def get_domain_range(self) -> tuple[float, float, float, float]:
        """Return default (xmin, xmax, ymin, ymax) if not provided by user."""
        raise NotImplementedError

    @abstractmethod
    def sample_z(self, x: float, y: float) -> float:
        """Return the scalar value at a single point (x, y)."""
        raise NotImplementedError

    @abstractmethod
    def array_sample_Z(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def Lx(self) -> float:          # domain length in x
        return self.xmax - self.xmin

    @property
    def Ly(self) -> float:          # domain length in y
        return self.ymax - self.ymin
    
    # --------------------------------------------------------------------- #
    # convenience helpers that rely *only* on `sample_z`
    # --------------------------------------------------------------------- #
    def sample_grid(
        self,
        resolution: int = 100,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Evaluate the field on a regular grid (Δx = Δy).

        `resolution` is the maximum number of points in one direction.
        """
        Δ = max(self.Lx, self.Ly) / (resolution - 1)
        xs = np.arange(self.xmin, self.xmax + 0.5 * Δ, Δ)
        ys = np.arange(self.ymin, self.ymax + 0.5 * Δ, Δ)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Z = self.array_sample_Z(X, Y)
        return X, Y, Z

    def visualize(
        self,
        resolution: int = 100,
        show: bool = True,
    ):
        """Quick visualisation of *any* ScalarField2D."""
        _, _, Z = self.sample_grid(resolution)

        fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.imshow(
            Z,
            extent=(self.xmin, self.xmax, self.ymin, self.ymax),
            origin="lower",
            cmap="viridis",
            interpolation="bilinear",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Elevation (z)")

        if show:
            plt.show()
        else:
            return fig, ax, im