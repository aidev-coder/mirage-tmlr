"""Deterministic label placement.

Random jitter dodges collisions by luck and cannot report failure. This tries a fixed
ladder of offsets per label, measures the rendered bounding box each time, and keeps the
first placement that overlaps nothing already placed. If every candidate collides it takes
the least-bad one and returns it in the report, so a caller can see that a label is still
sitting on something instead of assuming the figure is clean.
"""

from __future__ import annotations

# (dx, dy) in points, tried in order: right, left, above, below, then diagonals further out
CANDIDATES = [
    (6, 0), (-6, 0), (0, 7), (0, -8),
    (7, 6), (-7, 6), (7, -7), (-7, -7),
    (12, 0), (-12, 0), (0, 13), (0, -14),
    (14, 10), (-14, 10), (14, -11), (-14, -11),
]


def _overlaps(a, b, pad: float = 1.0) -> bool:
    return not (a.x1 + pad < b.x0 or b.x1 + pad < a.x0
                or a.y1 + pad < b.y0 or b.y1 + pad < a.y0)


def place_labels(ax, points, texts, fontsize=6.0, color="#5A6068", avoid=(),
                 marker_pad=4.0):
    """points: [(x, y)]; texts: [str]. Returns (artists, unresolved_labels)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    taken = list(avoid)
    # the markers themselves are obstacles
    from matplotlib.transforms import Bbox
    for (px, py) in points:
        dx, dy = ax.transData.transform((px, py))
        taken.append(Bbox.from_extents(dx - marker_pad, dy - marker_pad,
                                       dx + marker_pad, dy + marker_pad))
    artists, unresolved = [], []

    # Place the most constrained first: points closest to their neighbours.
    order = sorted(range(len(points)),
                   key=lambda i: min((abs(points[i][0] - points[j][0])
                                      + abs(points[i][1] - points[j][1]))
                                     for j in range(len(points)) if j != i))

    for i in order:
        x, y = points[i]
        best, best_cost, best_box = None, None, None
        for (dx, dy) in CANDIDATES:
            t = ax.annotate(texts[i], (x, y), textcoords="offset points", xytext=(dx, dy),
                            fontsize=fontsize, color=color, annotation_clip=False,
                            ha="left" if dx >= 0 else "right",
                            va="center" if dy == 0 else ("bottom" if dy > 0 else "top"))
            fig.canvas.draw()
            box = t.get_window_extent(renderer=renderer)
            cost = sum(1 for b in taken if _overlaps(box, b))
            if cost == 0:
                if best is not None:
                    best.remove()      # discard the earlier best-effort placement
                    best = None
                artists.append(t)
                taken.append(box)
                break
            if best_cost is None or cost < best_cost:
                if best is not None:
                    best.remove()
                best, best_cost, best_box = t, cost, box
            else:
                t.remove()
        else:
            artists.append(best)
            taken.append(best_box)
            unresolved.append(texts[i])

    return artists, unresolved
