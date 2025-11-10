from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RegionStatistics:
    name: str
    value: int
    voxel_count: int
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    median_intensity: float
    centroid: tuple[float, float, float]
    bounding_box: tuple[tuple[float, float, float], tuple[float, float, float]]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
