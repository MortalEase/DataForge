#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DOTA-v1.0 -> YOLO 格式转换与切分辅助脚本

核心: 解析 DOTA-v1.0 的四点有向框, 输出标准 YOLO 或 axis-aligned YOLO 标签
扩展: 提供 --hbb 选项导出 HBB, --tile 选项执行 tile 切分并按面积比标记截断目标
默认: 未启用 --tile 时只转换原始分割, 否则生成 tile_size/overlap 小图与标签
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
from tqdm import tqdm

try:
    from utils.logging_utils import tee_stdout_stderr, log_error, log_info, log_warn
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from utils.logging_utils import tee_stdout_stderr, log_error, log_info, log_warn

from utils.obb_utils import clip_quad_to_tile, compute_quad_area, quad_to_hbb, quad_to_rotated_bbox

_LOG_FILE = tee_stdout_stderr('logs')

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
MIN_AREA_PIXELS_DEFAULT = 16


@dataclass
class QuadAnnotation:
    quad: List[tuple[float, float]]
    class_name: str
    difficulty: int
    area: float


def normalize_bbox(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    if img_w == 0 or img_h == 0:
        return 0.0, 0.0, 0.0, 0.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def parse_dota_label(label_path: Path) -> List[QuadAnnotation]:
    annotations: List[QuadAnnotation] = []
    try:
        with label_path.open('r', encoding='utf-8') as fp:
            for line_idx, line in enumerate(fp):
                line = line.strip()
                if not line or line.startswith('imagesource') or line.startswith('gsd') or 'acquisition dates' in line:
                    continue
                parts = line.split()
                if len(parts) < 9:
                    log_warn(f"{label_path} 第 {line_idx + 1} 行格式异常: {line}")
                    continue
                try:
                    pts = [(float(parts[i]), float(parts[i + 1])) for i in range(0, 8, 2)]
                except ValueError as exc:
                    log_warn(f"{label_path} 第 {line_idx + 1} 行解析失败: {line} ({exc})")
                    continue
                class_name = parts[8]
                difficulty = int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0
                area = compute_quad_area(pts)
                if area <= 0:
                    log_warn(f"{label_path} 第 {line_idx + 1} 行面积为 0, 跳过")
                    continue
                annotations.append(QuadAnnotation(quad=pts, class_name=class_name, difficulty=difficulty, area=area))
    except Exception as exc:
        log_error(f"读取标签 {label_path} 失败: {exc}")
    return annotations


def convert_to_yolo(quad: List[tuple[float, float]], img_w: int, img_h: int, use_hbb: bool) -> tuple[float, float, float, float, float]:
    if use_hbb:
        xmin, ymin, xmax, ymax = quad_to_hbb(quad)
        cx = xmin + (xmax - xmin) / 2.0
        cy = ymin + (ymax - ymin) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        angle = 0.0
    else:
        cx, cy, w, h, angle = quad_to_rotated_bbox(quad)
    return (*normalize_bbox(cx, cy, w, h, img_w, img_h), angle)


def write_label_file(label_path: Path, entries: Iterable[tuple[int, float, float, float, float, float, int]]) -> None:
    with label_path.open('w', encoding='utf-8') as out:
        for class_id, cx, cy, w, h, angle, difficulty in entries:
            out.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.2f} {difficulty}\n")


def discover_image_path(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_class_mapping(root_dir: Path) -> dict[str, int]:
    classes: set[str] = set()
    for label_file in root_dir.rglob('*.txt'):
        annotations = parse_dota_label(label_file)
        for ann in annotations:
            classes.add(ann.class_name)
    return {name: idx for idx, name in enumerate(sorted(classes))}


def analyze_image_sizes(split_dir: Path) -> None:
    images_dir = split_dir / 'images'
    if not images_dir.exists():
        log_warn(f"缺少 images 目录: {images_dir}")
        return
    widths: list[int] = []
    heights: list[int] = []
    for img_path in tqdm(list(images_dir.iterdir()), desc=f"统计 {split_dir.name}", leave=False):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
    if not widths:
        log_warn(f"{split_dir.name} 无图像可统计")
        return
    avg_w = sum(widths) / len(widths)
    avg_h = sum(heights) / len(heights)
    max_w = max(widths)
    max_h = max(heights)
    log_info(
        f"[{split_dir.name}] 图像尺寸: min=({min(widths)},{min(heights)}), max=({max_w},{max_h}), avg=({avg_w:.1f},{avg_h:.1f}), count={len(widths)}"
    )
    if max(max_w, max_h) > 4000 or max(avg_w, avg_h) > 2000:
        log_warn(f"[{split_dir.name}] 建议 tile 切分以降低内存消耗。")
    else:
        log_info(f"[{split_dir.name}] 尺寸适中, 可直接训练或通过 resize 处理。")


def process_split(
    split_dir: Path,
    output_dir: Path,
    class_mapping: dict[str, int],
    args: argparse.Namespace,
) -> tuple[int, int]:
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labelTxt'
    processed = 0
    skipped = 0

    # 兼容两种标签目录结构：labelTxt/ 子目录或直接在 split_dir 下的 *.txt 文件
    if not labels_dir.exists():
        labels_dir = split_dir

    if not images_dir.exists():
        log_warn(f"未找到图像目录: {images_dir}")
        return processed, skipped

    tile_root = output_dir / 'tiles' / split_dir.name if args.tile else None

    # 仅收集 split_dir 或 labelTxt/ 下的 *.txt（过滤掉 classes.txt）
    label_files = sorted([f for f in labels_dir.glob('*.txt') if f.name != 'classes.txt' and f.is_file()])
    if not label_files:
        log_warn(f"未在 {labels_dir} 下找到标签文件")
        return processed, skipped

    for label_file in tqdm(label_files, desc=f"{split_dir.name}"):
        stem = label_file.stem
        img_path = discover_image_path(images_dir, stem)
        if img_path is None:
            log_warn(f"未找到图像: {stem}")
            skipped += 1
            continue

        annotations = parse_dota_label(label_file)
        if not annotations:
            skipped += 1
            if args.skip_no_label:
                continue
            log_warn(f"{label_file.name} 没有有效标注")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            log_warn(f"无法读取图像: {img_path}")
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]

        if args.tile:
            tiles_created = split_into_tiles(
                image_path=img_path,
                image_array=img,
                annotations=annotations,
                class_mapping=class_mapping,
                tile_root=tile_root,
                args=args,
            )
            if tiles_created == 0:
                log_warn(f"{label_file.name} 未生成任何 tile")
                skipped += 1
                continue
            processed += 1
            continue

        dest_images = output_dir / split_dir.name / 'images'
        dest_labels = output_dir / split_dir.name / 'labels'
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        from shutil import copy2

        try:
            copy2(img_path, dest_images / img_path.name)
        except Exception as exc:
            log_warn(f"复制图像失败: {img_path} ({exc})")
            skipped += 1
            continue

        label_entries: list[tuple[int, float, float, float, float, float, int]] = []
        for ann in annotations:
            class_id = class_mapping.get(ann.class_name, 0)
            cx, cy, w, h, angle = convert_to_yolo(ann.quad, img_w, img_h, args.hbb)
            label_entries.append((class_id, cx, cy, w, h, angle, ann.difficulty))

        write_label_file(dest_labels / f"{stem}.txt", label_entries)
        processed += 1

    return processed, skipped


def split_into_tiles(
    image_path: Path,
    image_array,
    annotations: List[QuadAnnotation],
    class_mapping: dict[str, int],
    tile_root: Path,
    args: argparse.Namespace,
) -> int:
    tile_images_dir = tile_root / 'images'
    tile_labels_dir = tile_root / 'labels'
    tile_images_dir.mkdir(parents=True, exist_ok=True)
    tile_labels_dir.mkdir(parents=True, exist_ok=True)

    height, width = image_array.shape[:2]
    step = args.tile_size - args.tile_overlap
    if step <= 0:
        log_warn("tile_size 必须大于 overlap")
        return 0

    tile_count = 0
    extension = image_path.suffix
    stem = image_path.stem

    for y in tqdm(range(0, height, step), desc=f"{image_path.stem} tiles", leave=False):
        for x in range(0, width, step):
            tile_w = min(args.tile_size, width - x)
            tile_h = min(args.tile_size, height - y)
            if tile_w <= 0 or tile_h <= 0:
                continue
            clipped: list[tuple[List[tuple[float, float]], str, int]] = []
            for ann in annotations:
                clips = clip_quad_to_tile(ann.quad, (x, y, tile_w, tile_h), ann.area, min_pixels=args.min_area_pixels)
                for clip in clips:
                    quad_local = [(pt_x - x, pt_y - y) for pt_x, pt_y in clip.quad]
                    difficulty = ann.difficulty
                    if clip.area_ratio < args.min_area_ratio:
                        difficulty = max(difficulty, 1)
                    clipped.append((quad_local, ann.class_name, difficulty))
            if not clipped:
                continue
            tile_img = image_array[y : y + tile_h, x : x + tile_w]
            tile_name = f"{stem}_x{x:04d}_y{y:04d}{extension}"
            cv2.imwrite(str(tile_images_dir / tile_name), tile_img)

            label_entries: list[tuple[int, float, float, float, float, float, int]] = []
            for quad_local, class_name, difficulty in clipped:
                class_id = class_mapping.get(class_name, 0)
                cx, cy, w, h, angle = convert_to_yolo(quad_local, tile_w, tile_h, args.hbb)
                label_entries.append((class_id, cx, cy, w, h, angle, difficulty))

            write_label_file(tile_labels_dir / f"{stem}_x{x:04d}_y{y:04d}.txt", label_entries)
            tile_count += 1

    return tile_count


def extract_zip_if_needed(input_dir: Path) -> Path:
    zips = list(input_dir.glob('*.zip'))
    if not zips:
        return input_dir
    for zp in zips:
        try:
            with zipfile.ZipFile(zp, 'r') as archive:
                archive.extractall(input_dir)
            log_info(f"已解压: {zp.name}")
        except Exception as exc:
            log_warn(f"解压 {zp.name} 失败: {exc}")
    return input_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description='将 DOTA-v1.0 转为 YOLO 格式并可选 tile 切分',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('-i', '--input', required=True, help='DOTA 数据集路径, 包含 train/val/test')
    parser.add_argument('-o', '--output', default=None, help='输出目录 (默认: input 同级的 dota_yolo_format1)')
    parser.add_argument('--skip-no-label', action='store_true', help='跳过无有效标签的图像')
    parser.add_argument('--hbb', action='store_true', help='以 axis-aligned bbox 导出标签 (angle=0)')
    parser.add_argument('--analyze-only', action='store_true', help='仅统计尺寸分布, 不执行转换')
    parser.add_argument('--tile', action='store_true', help='对图像按 tile_size/overlap 切分并写入小图')
    parser.add_argument('--tile-size', type=int, default=1024, help='tile 边长 (默认 1024)')
    parser.add_argument('--tile-overlap', type=int, default=200, help='tile 重叠 (默认 200)')
    parser.add_argument('--min-area-ratio', type=float, default=0.3, help='裁剪后占原始面积的比率, 低于则标记难例')
    parser.add_argument('--min-area-pixels', type=int, default=MIN_AREA_PIXELS_DEFAULT, help='裁剪后最小面积阈值 (像素)')
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        log_error(f"输入目录不存在: {input_dir}")
        return

    data_dir = extract_zip_if_needed(input_dir)
    splits = [data_dir / name for name in ['train', 'val', 'test'] if (data_dir / name).exists()]
    if not splits:
        log_error(f"在 {data_dir} 下未检测到 train/val/test 子目录")
        return

    # 仅在 --analyze-only 模式下做完整尺寸统计
    if args.analyze_only:
        for split in splits:
            analyze_image_sizes(split)
        log_info('图像尺寸统计完成, 退出。')
        return

    class_map = build_class_mapping(data_dir)
    log_info(f"发现类别数: {len(class_map)} ({', '.join(class_map.keys())})")

    output_dir = Path(args.output) if args.output else input_dir.parent / 'dota_yolo_format1'
    output_dir.mkdir(parents=True, exist_ok=True)

    classes_file = output_dir / 'classes.txt'
    with classes_file.open('w', encoding='utf-8') as fp:
        for name, idx in sorted(class_map.items(), key=lambda item: item[1]):
            fp.write(f"{name}\n")
    log_info(f"已写类别文件: {classes_file}")

    total_processed = 0
    total_skipped = 0
    for split in splits:
        processed, skipped = process_split(split, output_dir, class_map, args)
        total_processed += processed
        total_skipped += skipped
        log_info(f"{split.name}: 处理 {processed} 张, 跳过 {skipped} 张")

    log_info(f"转换结束, 输出 {output_dir}")
    log_info(f"总计: 处理 {total_processed} 张, 跳过 {total_skipped} 张")


if __name__ == '__main__':
    main()
