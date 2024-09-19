from typing import List

import numpy as np
from scipy.ndimage import map_coordinates


def generate_concentric_sample_pts(
    peak_radii: np.ndarray[np.int64],
    center: List[float],
    num_pts: int = 200,
) -> np.ndarray[np.float64]:
    # X,Y labelling seems backwards
    cx: float = center[0]
    cy: float = center[1]
    # Reshape linear radii (peak indices) for broadcasting
    radii: np.ndarray[np.int64] = np.array([peak_radii]).reshape(-1, 1)
    theta: np.ndarray[np.float64] = np.linspace(0.0, 2 * np.pi, 200)

    coords_x: np.ndarray[np.float64] = radii * np.cos(theta) + cx
    coords_y: np.ndarray[np.float64] = radii * np.sin(theta) + cy

    # Reshape for optimization routines
    coords: np.ndarray[np.float64] = np.zeros((2, num_pts * len(peak_radii)))
    coords[1] = coords_x.reshape(-1)
    coords[0] = coords_y.reshape(-1)

    return coords


def geometry_optimize_residual(
    params: List[float], powder: np.ndarray[np.float64]
) -> np.ndarray[np.float64]:
    # Unpack the parameters
    center_guess: List[float] = params[:2]
    # Indices are in radii units since they are for a radial profile
    indices: List[float] = params[2:]
    coords: np.ndarray[np.float64] = generate_concentric_sample_pts(
        indices, center_guess
    )

    # Use residual for fitting - difference between intensity in ring
    # and powder max
    pixel_values: np.ndarray[np.float64] = map_coordinates(powder, coords)
    pixel_values -= np.max(powder)
    return pixel_values
