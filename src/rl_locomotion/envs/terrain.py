"""Read and inspect the terrain height fields of Playground environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ------------------------------------------------------------------ readers and plotters
@dataclass(frozen=True)
class HeightMap:
    """One terrain height field, in metres.
    This is a frozen dataclass holding one terrain, already in metres, plus helpers:"""

    heights: np.ndarray  # (nrow, ncol), metres above the base
    radius_x: float 
    radius_y: float  
    elevation: float  # metres corresponding to a normalised value of 1.0
    base: float  

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) in metres — ready for `imshow`."""
        return (-self.radius_x, self.radius_x, -self.radius_y, self.radius_y)

    @property
    def cell_size(self) -> tuple[float, float]:
        """Spacing between adjacent grid points, in metres."""
        nrow, ncol = self.shape
        return (2 * self.radius_x / ncol, 2 * self.radius_y / nrow)

    @property
    def peak_to_peak(self) -> float:
        """"Gap between the highest and lowest points, in metres."""
        return float(np.ptp(self.heights))

    @property
    def roughness(self) -> float:
        """Standard deviation of height, in metres.
        """
        return float(self.heights.std())

    def summary(self) -> dict[str, Any]:
        cx, cy = self.cell_size
        return {
            "grid": f"{self.shape[0]}x{self.shape[1]}",
            "area_m": f"{2 * self.radius_x:g}x{2 * self.radius_y:g}",
            "cell_size_m": round(cx, 4),
            "peak_to_peak_m": round(self.peak_to_peak, 4),
            "roughness_std_m": round(self.roughness, 4),
        }

    def sample(self, x: float, y: float) -> float:
        """Height in metres at a world (x, y), by nearest grid point."""
        nrow, ncol = self.shape
        col = int(np.clip((x + self.radius_x) / (2 * self.radius_x) * (ncol - 1), 0, ncol - 1))
        row = int(np.clip((y + self.radius_y) / (2 * self.radius_y) * (nrow - 1), 0, nrow - 1))
        return float(self.heights[row, col])


def height_map(env: Any, index: int = 0) -> HeightMap | None:
    """The terrain height field of an environment, or None if it has no terrain.
    `env` may be an `MjxEnv` or a raw `MjModel`.
    """
    model = getattr(env, "mj_model", env)
    if model.nhfield <= index:
        return None

    nrow = int(model.hfield_nrow[index])
    ncol = int(model.hfield_ncol[index])
    radius_x, radius_y, elevation, base = (float(v) for v in model.hfield_size[index])

    start = int(model.hfield_adr[index])
    raw = model.hfield_data[start : start + nrow * ncol].reshape(nrow, ncol)

    return HeightMap(
        heights=raw * elevation,  # normalised [0, 1] -> metres
        radius_x=radius_x,
        radius_y=radius_y,
        elevation=elevation,
        base=base,
    )


def plot_height_map(
    hmap: HeightMap,
    title: str = "",
    cmap: str = "terrain",
    figsize: tuple[float, float] = (11.0, 4.5),
) -> Any:
    """Top-down view alongside a cross-section through the middle.
    """
    import matplotlib.pyplot as plt

    fig, (ax_map, ax_cut) = plt.subplots(1, 2, figsize=figsize)

    image = ax_map.imshow(
        hmap.heights, extent=hmap.extent, origin="lower", cmap=cmap
    )
    ax_map.set(xlabel="x (m)", ylabel="y (m)", title=title or "height map")
    fig.colorbar(image, ax=ax_map, label="height (m)")

    nrow, ncol = hmap.shape
    xs = np.linspace(-hmap.radius_x, hmap.radius_x, ncol)
    ax_cut.plot(xs, hmap.heights[nrow // 2], lw=0.8)
    ax_cut.set(xlabel="x (m)", ylabel="height (m)", title="cross-section at y = 0")
    ax_cut.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def terrain_table(env_names: list[str] | None = None) -> Any:
    """Compare the terrain of every environment that has one.
    Loads every locomotion environment from your registry, keeps the 
    ones with terrain, and returns a pandas table comparing their summaries.
    """
    import pandas as pd
    from mujoco_playground import registry as pg_registry

    if env_names is None:
        from rl_locomotion.envs.registry import list_envs

        env_names = [i.name for i in list_envs("locomotion")]

    rows = []
    for name in env_names:
        hmap = height_map(pg_registry.load(name))
        if hmap is None:
            continue
        rows.append({"env": name, **hmap.summary()})

    return pd.DataFrame(rows).set_index("env") if rows else pd.DataFrame()


# ------------------------------------------------------------------ generators

def make_bowl(
    nrow: int = 256,
    ncol: int = 256,
    radius_m: float = 10.0,
    slope_deg: float = 10.0,
    flat_radius_m: float = 1.0,
) -> tuple[np.ndarray, float]:
    """A bowl: flat disc (radius 1m), then a constant slope in every direction.

        h(r) = tan(slope) * max(0, r - flat_radius)
    """
    if slope_deg < 0:
        raise ValueError("slope_deg must be >= 0")
    ys, xs = np.meshgrid(
        np.linspace(-radius_m, radius_m, nrow), np.linspace(-radius_m, radius_m, ncol), indexing="ij"
    )
    r = np.hypot(xs, ys)
    heights = np.tan(np.radians(slope_deg)) * np.clip(r - flat_radius_m, 0.0, None)
    elevation = float(heights.max())
    normalised = heights / elevation if elevation > 0 else np.zeros_like(heights)
    return normalised.astype(np.float32), elevation


def apply_heightfield(model: Any, heights01: np.ndarray, elevation: float, index: int = 0) -> None:
    """Write a normalised height grid and its elevation into `model` (mujoco.MjModel) in place."""
    nrow, ncol = int(model.hfield_nrow[index]), int(model.hfield_ncol[index])
    if heights01.shape != (nrow, ncol):
        raise ValueError(f"heights must be {(nrow, ncol)}, got {heights01.shape}")
    start = int(model.hfield_adr[index])
    model.hfield_data[start : start + nrow * ncol] = heights01.ravel()
    model.hfield_size[index, 2] = elevation


def combine_heightfields(*fields: tuple[np.ndarray, float]) -> tuple[np.ndarray, float]:
    """Sum heightfields given as (normalised grid, elevation) pairs; returns the same form.

    Used for rough + bowl: Playground's rocky relief (normalised grid x 0.05 m)
    added on top of the bowl slope. Grids must share a shape.
    """
    total = None
    for grid, elevation in fields:
        metres = np.asarray(grid, dtype=np.float64) * float(elevation)
        total = metres if total is None else total + metres
    assert total is not None
    elevation = float(total.max())
    normalised = total / elevation if elevation > 0 else np.zeros_like(total)
    return normalised.astype(np.float32), elevation


def build_terrain(
    base: tuple[np.ndarray, float] | None,
    shape: str,
    amplitude: float,
    slope_deg: float,
    nrow: int,
    ncol: int,
    radius_m: float,
) -> tuple[np.ndarray, float]:
    """(normalised grid, elevation in m) for a terrain shape.

    base: the scene's own rocky field as (normalised grid, elevation); needed for
    "playground" and "rough_bowl". "playground" = rocky field scaled to
    `amplitude`; "bowl" = smooth bowl at `slope_deg`; "rough_bowl" = bowl plus
    rocky relief of `amplitude`.
    """
    if shape == "bowl":
        return make_bowl(nrow, ncol, radius_m, slope_deg=slope_deg)
    if base is None:
        raise ValueError(f"terrain shape {shape!r} needs the scene's heightfield")
    rocky_grid, _ = base
    if shape == "playground":
        return rocky_grid.astype(np.float32), float(amplitude)
    if shape == "rough_bowl":
        bowl = make_bowl(nrow, ncol, radius_m, slope_deg=slope_deg)
        return combine_heightfields(bowl, (rocky_grid, float(amplitude)))
    raise ValueError(f"unknown terrain shape {shape!r}")
