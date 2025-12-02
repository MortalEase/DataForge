from dataclasses import dataclass
from typing import List, Sequence, Tuple

from shapely.geometry import Polygon, box


@dataclass(frozen=True)
class ClipResult:
    quad: List[Tuple[float, float]]
    area_ratio: float


def compute_quad_area(quad: Sequence[Tuple[float, float]]) -> float:
    """计算四点多边形的面积."""
    poly = Polygon(quad)
    return float(poly.area) if poly.is_valid else 0.0


def quad_to_rotated_bbox(quad: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float, float]:
    """根据四点多边形返回旋转框的 (cx, cy, w, h, angle)."""
    x1, y1 = quad[0]
    x2, y2 = quad[1]
    x3, y3 = quad[2]
    x4, y4 = quad[3]

    cx = (x1 + x2 + x3 + x4) / 4.0
    cy = (y1 + y2 + y3 + y4) / 4.0

    w = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    h = ((x3 - x2) ** 2 + (y3 - y2) ** 2) ** 0.5

    # 角度采用第一条边的方向
    from math import atan2, degrees

    angle = degrees(atan2(y2 - y1, x2 - x1))
    return cx, cy, w, h, angle


def quad_to_hbb(quad: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """将四点多边形转换为 axis-aligned HBB."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax


def clip_quad_to_tile(
    quad: Sequence[Tuple[float, float]],
    tile_rect: Tuple[int, int, int, int],
    original_area: float,
    min_pixels: int = 16,
) -> List[ClipResult]:
    """将 OBB 与 tile 矩形相交，返回重建的多边形与面积比例."""
    if original_area <= 0:
        return []

    tx, ty, tw, th = tile_rect
    tile_poly = box(tx, ty, tx + tw, ty + th)
    quad_poly = Polygon(quad)
    if not quad_poly.is_valid:
        quad_poly = quad_poly.buffer(0)
    inter = quad_poly.intersection(tile_poly)
    if inter.is_empty or inter.area < min_pixels:
        return []

    mrr = inter.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:-1]
    if len(coords) != 4:
        coords = coords[:4]
    if len(coords) < 4:
        return []

    area_ratio = float(inter.area) / original_area
    return [ClipResult(quad=[(float(pt[0]), float(pt[1])) for pt in coords], area_ratio=area_ratio)]
