import numpy as np
import matplotlib.pyplot as plt

def hexagon_grid(radius: int) -> np.ndarray:
    result = np.array([[0., 0.]])

    directions = np.array([[-0.5, np.sqrt(3.) / 2.], [-1., 0.], [-0.5, -np.sqrt(3.) / 2.], [0.5, -np.sqrt(3.) / 2.], [1., 0.], [0.5, np.sqrt(3.) / 2.]])

    for r in range(radius):
        point = np.array([r + 1., 0.])
        result = np.vstack((result, point))
        for i, d in enumerate(directions):
            for _ in range(r + 1 if i < 5 else r):
                point += d
                result = np.vstack((result, point))

    return result