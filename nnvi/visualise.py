from typing import Optional

import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_3d(
    title: str,
    X: list[NDArray],
    Y: list[NDArray],
    Z: list[NDArray],
    ax: Optional[plt.Axes] = None,
    use_trisurf = False,
) -> None:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

    ax.set_title(title)
    for i in range(0, len(X)):
        if use_trisurf:
            ax.plot_trisurf(X[i], Y[i], Z[i].flatten())
        else:
            ax.scatter(X[i], Y[i], Z[i])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_2d(
    title: str,
    X: list[NDArray],
    Z: list[NDArray],
    ax: Optional[plt.Axes] = None,
) -> None:
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    ax.set_title(title)
    for i in range(0, len(X)):
        ax.scatter(X[i], Z[i], s=0.3)
