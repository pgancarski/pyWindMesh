from abc import ABC, abstractmethod

from src.domain import Mesh2D

class MeshPlotter(ABC):
    @abstractmethod
    def plot(self, mesh: Mesh2D, field_name: str = "Z") -> None:
        ...
