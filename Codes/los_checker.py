"""LOS check: does segment(anchor,tag) intersect any wall segment?"""
import numpy as np


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1, p2, p3, p4):
    return (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)) and (_ccw(p1, p2, p3) != _ccw(p1, p2, p4))


def is_los(anchor_pos, tag_pos, walls):
    """Return True if LOS (no wall intersects anchor-tag segment)."""
    a = tuple(anchor_pos)
    t = tuple(tag_pos)
    for (w1, w2) in walls:
        if _segments_intersect(a, t, w1, w2):
            return False
    return True


def los_mask(anchors, tag_pos, walls):
    """Vectorized-ish over anchors (6,). Returns bool array."""
    return np.array([is_los(a, tag_pos, walls) for a in anchors], dtype=bool)
