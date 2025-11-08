from typing import Any

import numpy as np

type Serializable = int | float | list[Serializable] | dict[str, Serializable]


def into_serializable(value: Any) -> Serializable:
    """
    Convert numpy data values into serializable Python values.
    """

    if isinstance(value, (np.integer, np.int64, np.int32)):  # type: ignore
        return int(value)  # type: ignore
    elif isinstance(value, (np.floating, np.float64, np.float32)):  # type: ignore
        return float(value)  # type: ignore
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {
            key: into_serializable(value)
            for key, value  # type: ignore
            in value.items()
        }
    elif isinstance(value, list):
        return [
            into_serializable(item)
            for item  # type: ignore
            in value
        ]
    else:
        return value
