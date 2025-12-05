#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 随机样本可视化脚本

核心: 从 YOLO 数据集随机抽取 n 张带标注的图片，绘制标注并保存到指定输出目录.
扩展: 支持指定类别文件、随机种子、按数据集子集（train/val/test）保存子目录。
默认: 未提供输出路径时写入运行目录下的 `outputs/visualization/`。

使用示例:
  # 随机抽取 6 张样本并保存到默认 outputs/visualization
  python visualize_yolo_samples.py -d /path/to/yolo/dataset -n 6 --seed 42

  # 指定输出目录和类别文件
  python visualize_yolo_samples.py -d /path/to/yolo/dataset -n 8 -o my_outputs -c classes.txt
"""

from __future__ import annotations
import argparse
import random
import os
from pathlib import Path
from typing import List, Dict, Tuple
import cv2

# 将项目根加入 sys.path 以便复用 utils
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.logging_utils import tee_stdout_stderr, log_info, log_warn, log_error
from utils.yolo_utils import discover_class_names, read_class_names

# 将 stdout/stderr 同步到 logs
_LOG_FILE = tee_stdout_stderr('logs')

import numpy as np
from matplotlib import colors as mcolors


def discover_labeled_images(root: Path) -> List[Dict[str, str]]:
    """扫描数据集，返回包含图片与对应标签路径的列表.

    返回元素: { 'image_path': str, 'label_path': str, 'set_name': str }
    """
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    possible_dirs = [
        root,
        root / 'images',
        root / 'train' / 'images',
        root / 'val' / 'images',
        root / 'test' / 'images',
    ]

    results = []
    for img_dir in possible_dirs:
        if not img_dir.exists():
            continue
        if img_dir.name == 'images':
            label_dir = img_dir.parent / 'labels'
        else:
            label_dir = img_dir / 'labels'
            if not label_dir.exists():
                label_dir = root / 'labels'

        if not label_dir.exists():
            continue

        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in img_exts:
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    # 确保标签非空
                    try:
                        with open(label_file, 'r') as f:
                            lines = [l for l in f.readlines() if l.strip()]
                        if lines:
                            results.append({
                                'image_path': str(img_file),
                                'label_path': str(label_file),
                                'set_name': img_dir.parent.name if img_dir.name == 'images' else 'dataset'
                            })
                    except Exception as e:
                        log_warn(f"读取标注失败: {label_file} -> {e}")

    results.sort(key=lambda x: x['image_path'])
    return results


def _color_name_to_bgr(color_name: str) -> Tuple[int,int,int]:
    try:
        rgb = mcolors.to_rgb(color_name)
        return tuple(int(255 * c) for c in rgb[::-1])
    except Exception:
        return (255, 255, 255)


def save_annotated_image_cv(img_path: str, label_path: str, class_names: Dict[int,str], colors: List[str], out_path: str) -> bool:
    """用 OpenCV 在图片上绘制 YOLO 框并保存."""
    img = cv2.imread(img_path)
    if img is None:
        log_warn(f"无法读取图片: {img_path}")
        return False
    h, w = img.shape[:2]

    anns = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    x_c = float(parts[1]); y_c = float(parts[2]); ww = float(parts[3]); hh = float(parts[4])
                    anns.append((class_id, x_c, y_c, ww, hh))
    except Exception as e:
        log_warn(f"读取标签失败: {label_path} -> {e}")

    for ann in anns:
        class_id, x_c, y_c, ww, hh = ann
        x_c_px = int(x_c * w); y_c_px = int(y_c * h)
        bw = int(ww * w); bh = int(hh * h)
        x1 = int(x_c_px - bw/2); y1 = int(y_c_px - bh/2); x2 = x1 + bw; y2 = y1 + bh
        color_name = colors[class_id % len(colors)] if colors else 'red'
        bgr = _color_name_to_bgr(color_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness=2)
        label_text = class_names.get(class_id, f"Class_{class_id}") + f" ({class_id})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.5; ft = 1
        ((tw, th), _) = cv2.getTextSize(label_text, font, fs, ft)
        tx = max(x1, 0); ty = max(y1 - 6, th + 4)
        cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty), bgr, -1)
        cv2.putText(img, label_text, (tx + 2, ty - 2), font, fs, (0,0,0), ft, cv2.LINE_AA)

    out_dir = Path(out_path).parent
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cv2.imwrite(out_path, img)
        return True
    except Exception as e:
        log_warn(f"写出图片失败: {out_path} -> {e}")
        return False


DEFAULT_COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'lime', 'teal']


def main() -> None:
    parser = argparse.ArgumentParser(description='随机抽取 YOLO 标注样本并保存可视化', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--dataset', required=True, help='YOLO 数据集根目录')
    parser.add_argument('-n', '--num', type=int, default=9, help='随机抽取样本数 (默认 9)')
    parser.add_argument('-o', '--output', default='outputs/visualization', help='保存目录 (默认 outputs/visualization)')
    parser.add_argument('-c', '--classes', help='可选类别文件路径 (每行一个类别名)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--overwrite', action='store_true', help='存在同名输出时覆盖')
    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        log_error(f"数据集路径不存在: {root}")
        raise SystemExit(1)

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)

    images = discover_labeled_images(root)
    if not images:
        log_warn("未找到任何带标注的图片，退出.")
        return

    samples = random.sample(images, min(args.num, len(images)))

    # 优先使用 -c 指定的类别文件，否则自动在数据集中查找
    if args.classes:
        class_names_list = read_class_names(args.classes)
    else:
        class_names_list, source = discover_class_names(root)
        if class_names_list:
            log_info(f"自动发现类别文件: {source}")

    class_names = {i: n for i, n in enumerate(class_names_list)}

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    saved = 0
    for item in samples:
        img_path = item['image_path']
        label_path = item['label_path']
        set_name = item.get('set_name', 'dataset')
        img_name = Path(img_path).name
        out_dir = out_root / set_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / img_name
        if out_file.exists() and not args.overwrite:
            log_info(f"跳过已存在文件: {out_file}")
            continue
        ok = save_annotated_image_cv(img_path, label_path, class_names, DEFAULT_COLORS, str(out_file))
        if ok:
            saved += 1
            log_info(f"保存: {out_file}")
        else:
            log_warn(f"保存失败: {out_file}")

    log_info(f"完成，可视化图片已导出: {saved} 张 到 {out_root}")


if __name__ == '__main__':
    main()
