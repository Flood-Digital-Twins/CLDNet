import numpy as np
from typing import Union

# Vectorized minmod limited function
def minmod(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    a = np.asarray(a)
    b = np.asarray(b)
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def limited_gradient(z: np.ndarray, cell_size: float) -> np.ndarray:
    """
    -----------------------------------------------
    Grid Indexing Convention:
    z[i, j] corresponds to the value at:
        - Row index i (vertical direction, increases downwards → x-axis)
        - Column index j (horizontal direction, increases to the right → y-axis)

    Coordinate Axes:
        - x: down (increasing row index)
        - y: right (increasing column index)

    Grid layout (schematic):

        y →
        ┌───────────────────────────────┐
        x ↓  [0,0]  [0,1]  [0,2] ... [0,n]
            [1,0]  [1,1]  [1,2] ... [1,n]
            [2,0]  [2,1]  [2,2] ... [2,n]
            ...    ...    ...     ...
            [m,0]  [m,1]  [m,2] ... [m,n]
    -----------------------------------------------

        Directional Reference:
                ↑
                North
        ← West         East →
                South
                ↓
    """

    padded = np.pad(z, pad_width=1, mode='edge')

    # One-sided slopes
    grad_x_up   = (padded[2:, 1:-1] - z) / cell_size     # south
    grad_x_down = (z - padded[:-2, 1:-1]) / cell_size    # north
    grad_y_up   = (padded[1:-1, 2:] - z) / cell_size     # east
    grad_y_down = (z - padded[1:-1, :-2]) / cell_size    # west

    # Minmod limiter
    grad_x = minmod(grad_x_up, grad_x_down)
    grad_y = minmod(grad_y_up, grad_y_down)

    # Stack gradients along last axis
    gradient = np.stack((grad_x, grad_y), axis=-1)
    return gradient