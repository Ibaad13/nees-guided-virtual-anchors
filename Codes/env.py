"""Hall environment: dims, anchors, wall segments (NLOS blockers)."""
import numpy as np

HALL_W, HALL_H = 20.0, 15.0

# 6 real anchors: 4 corners + 2 mid-walls
ANCHORS = np.array([
    [0.0, 0.0],
    [HALL_W, 0.0],
    [HALL_W, HALL_H],
    [0.0, HALL_H],
    [HALL_W / 2, 0.0],
    [HALL_W / 2, HALL_H],
], dtype=float)

# Rectangular wall blockers as list of 4 line segments each: [(x1,y1),(x2,y2)]
def _rect_segments(cx, cy, w, h):
    x0, x1 = cx - w / 2, cx + w / 2
    y0, y1 = cy - h / 2, cy + h / 2
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]

WALLS = (
    _rect_segments(6.0, 4.0, 2.5, 1.5)
    + _rect_segments(12.0, 10.0, 3.0, 1.0)
    + _rect_segments(15.0, 4.0, 1.5, 2.5)
)


def get_env():
    return HALL_W, HALL_H, ANCHORS, WALLS


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, HALL_W)
    ax.set_ylim(0, HALL_H)
    ax.scatter(ANCHORS[:, 0], ANCHORS[:, 1], c="green", marker="^", s=80, label="Anchors")
    for seg in WALLS:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)
    ax.set_aspect("equal")
    ax.legend()
    plt.savefig("fig_env_only.png", dpi=150)
