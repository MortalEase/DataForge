#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VEDAI -> YOLO 格式转换脚本

核心: 解析 VEDAI 原始标注（旋转矩形 OBB 格式）并转为 YOLO 格式
VEDAI 标注格式: Image_ID Center_X Center_Y Orientation X1 Y1 X2 Y2 X3 Y3 X4 Y4 Class_ID Contained Occluded
支持 9 个类别 (1-9: Plane/Boat/Camping Car/Car/Pick-up/Tractor/Truck/Van/Other)
扩展: 支持 RGB (_co.png) 与 NIR (_ir.png) 波段、可选 HBB 或 OBB 格式、可选 8:1:1 格式一划分
默认: 生成 standard 结构(images/+labels/)，可通过 --no-split 禁用自动划分
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import shutil
import subprocess
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import tee_stdout_stderr, log_info, log_warn, log_error

_LOG_FILE = tee_stdout_stderr('logs')

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']


def parse_vedai_annotation(annotation_path: Path) -> List[Tuple[float, float, float, float, int]]:
    """解析 VEDAI 标注文件
    
    VEDAI 标注格式（一行一个车辆）:
    Cx Cy Orient Class_ID Contained Occluded X1 X2 X3 X4 Y1 Y2 Y3 Y4
    (共 14 个字段)
    
    返回: [(cx, cy, w, h, class_id), ...]，坐标为像素值（中心+宽高）
    """
    bboxes = []
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                
                # 过滤掉可能的 Image_ID (如果是 15 列)
                data = parts
                if len(parts) == 15:
                    data = parts[1:]
                
                if len(data) < 14:
                    log_warn(f"{annotation_path.name} 第 {line_idx + 1} 行字段数不足: {line}")
                    continue
                
                try:
                    # 字段 0-1: center coordinates
                    cx = float(data[0])
                    cy = float(data[1])
                    
                    # 字段 3: class_id
                    class_id = int(float(data[3]))
                    
                    # 字段 6-9: 4个角点的 X 坐标
                    xs = [float(x) for x in data[6:10]]
                    # 字段 10-13: 4个角点的 Y 坐标
                    ys = [float(y) for y in data[10:14]]
                    
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    # 重新计算中心和宽高
                    calc_cx = (min_x + max_x) / 2
                    calc_cy = (min_y + max_y) / 2
                    w = max_x - min_x
                    h = max_y - min_y
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    bboxes.append((calc_cx, calc_cy, w, h, class_id))
                except ValueError as e:
                    log_warn(f"{annotation_path.name} 第 {line_idx + 1} 行解析失败: {e}")
                    continue
    except Exception as e:
        log_error(f"读取标注文件失败 {annotation_path}: {e}")
    
    return bboxes


def normalize_bbox_coords(cx: float, cy: float, w: float, h: float, 
                          img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """将像素坐标 (中心x,中心y,宽,高) 归一化为 [0,1] 范围"""
    if img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    norm_cx = cx / img_w
    norm_cy = cy / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    # 限制范围 [0, 1]
    norm_cx = max(0.0, min(1.0, norm_cx))
    norm_cy = max(0.0, min(1.0, norm_cy))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))
    
    return norm_cx, norm_cy, norm_w, norm_h


def get_image_dimensions(img_path: Path) -> Tuple[int, int, bool]:
    """读取图片尺寸（使用 cv2 或 PIL）
    
    返回: (width, height, success)
    """
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            return w, h, True
    except Exception:
        pass
    
    # 备选方案：PIL
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            w, h = img.size
            return w, h, True
    except Exception:
        pass
    
    log_warn(f"无法读取图片尺寸: {img_path}")
    return 0, 0, False


def convert_vedai_to_yolo(images_dir: Path, annotations_dir: Path, output_dir: Path,
                          with_nir: bool = False, verbose: bool = False, obb_mode: bool = False,
                          max_images: Optional[int] = None) -> Tuple[int, int, int]:
    """将 VEDAI 原始数据（分离的 images 和 annotations 目录）转为 YOLO standard 结构
    
    VEDAI 文件结构:
    - images_dir: Vehicules1024/ (包含 *_co.png 和 *_ir.png)
    - annotations_dir: Annotations1024/ (包含 *.txt)
    
    参数:
    - obb_mode: True 保留 OBB (定向框+角度), False 转为 HBB (水平框)
    - max_images: 最大转换图片数量（用于测试）
    
    返回: (converted_count, skipped_count, nir_count)
    """
    images_dir = Path(images_dir).resolve()
    annotations_dir = Path(annotations_dir).resolve()
    output_dir = Path(output_dir).resolve()
    
    if not images_dir.exists():
        log_error(f"图像目录不存在: {images_dir}")
        return 0, 0, 0
    if not annotations_dir.exists():
        log_error(f"标注目录不存在: {annotations_dir}")
        return 0, 0, 0
    
    # 创建输出目录
    images_out = output_dir / 'images'
    labels_out = output_dir / 'labels'
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    
    log_info(f"图像目录: {images_dir}")
    log_info(f"标注目录: {annotations_dir}")
    log_info(f"输出目录: {output_dir}")
    mode_str = "OBB (定向框+角度)" if obb_mode else "HBB (水平框)"
    log_info(f"边界框模式: {mode_str}")
    if with_nir:
        log_info("NIR 模式: 启用 (将复制 _ir.png 近红外图像)")
    if max_images is not None:
        log_info(f"最大转换数量限制: {max_images}")
    
    # 扫描标注文件
    annotation_files = sorted([f for f in annotations_dir.glob('*.txt') 
                               if f.name not in ['classes.txt', 'obj.names', 'names.txt']])
    if not annotation_files:
        log_error(f"未在 {annotations_dir} 中找到任何标注文件")
        return 0, 0, 0
    
    log_info(f"找到 {len(annotation_files)} 个标注文件")
    
    converted = 0
    skipped = 0
    nir_count = 0
    
    # 处理每个标注文件
    for annotation_path in annotation_files:
        if max_images is not None and converted >= max_images:
            log_info(f"达到最大转换数量限制 ({max_images})，停止转换。")
            break

        base_name = annotation_path.stem
        
        # 查找对应的可见光图像 (*_co.png)
        rgb_files = list(images_dir.glob(f'{base_name}_co.*'))
        if not rgb_files:
            if verbose:
                log_warn(f"未找到图像: {base_name}_co.* (跳过)")
            skipped += 1
            continue
        
        rgb_path = rgb_files[0]
        
        # 读取标注
        bboxes = parse_vedai_annotation(annotation_path)
        if not bboxes:
            if verbose:
                log_warn(f"标注文件为空或无效: {annotation_path.name} (跳过)")
            skipped += 1
            continue
        
        # 获取图像尺寸
        img_w, img_h, success = get_image_dimensions(rgb_path)
        if not success or img_w <= 0 or img_h <= 0:
            log_warn(f"无法获取图像尺寸，跳过: {rgb_path}")
            skipped += 1
            continue
        
        # 复制可见光图像（保留 _co 后缀以区别 NIR 波段）
        dst_rgb = images_out / rgb_path.name
        try:
            shutil.copy2(rgb_path, dst_rgb)
        except Exception as e:
            log_warn(f"复制图像失败: {rgb_path} -> {dst_rgb}: {e}")
            skipped += 1
            continue
        
        # 若启用 NIR，查找并复制近红外图像 (*_ir.png)
        if with_nir:
            nir_files = list(images_dir.glob(f'{base_name}_ir.*'))
            if nir_files:
                nir_path = nir_files[0]
                dst_nir = images_out / nir_path.name
                try:
                    shutil.copy2(nir_path, dst_nir)
                    nir_count += 1
                except Exception as e:
                    log_warn(f"复制 NIR 图像失败: {nir_path} -> {dst_nir}: {e}")
        
        # 生成 YOLO 标签
        yolo_lines = []
        
        # 类别映射 (基于用户分析)
        # ID 31=飞机；ID 1=汽车；ID 4=拖拉机；ID 7=摩托车；ID 11=皮卡；ID 8=Bus；
        # ID 23=Boat；ID 9=Van；ID 5=房车；ID 2=Truck；ID 10=牵引车
        CLASS_MAPPING = {
            1: 0,   # Car
            2: 1,   # Truck
            11: 2,  # Pickup
            4: 3,   # Tractor
            5: 4,   # Camping Car
            23: 5,  # Boat
            7: 6,   # Motorcycle
            8: 7,   # Bus
            9: 8,   # Van
            31: 9,  # Plane
            10: 10, # Truck-Tractor
        }
        OTHER_CLASS_INDEX = 11
        
        for cx, cy, w, h, class_id in bboxes:
            norm_cx, norm_cy, norm_w, norm_h = normalize_bbox_coords(cx, cy, w, h, img_w, img_h)
            
            # 映射 ID，未知的归为 Other
            yolo_class_id = CLASS_MAPPING.get(class_id, OTHER_CLASS_INDEX)
            
            # YOLO 格式: class_id cx cy w h （HBB模式，使用AABB）
            yolo_lines.append(f"{yolo_class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        # 写标签文件（添加 _co 后缀保持与 RGB 图片名一致）
        label_file = labels_out / f"{base_name}_co.txt"
        
        # 若启用 NIR，也为其生成对应的标签文件（共用同一份标注）
        label_files_to_write = [label_file]
        if with_nir and list(images_dir.glob(f'{base_name}_ir.*')):
            label_files_to_write.append(labels_out / f"{base_name}_ir.txt")
        # 为所有标签文件写入内容
        for lbl_file in label_files_to_write:
            try:
                with open(lbl_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(yolo_lines) + ('\n' if yolo_lines else ''))
            except Exception as e:
                log_error(f"写标签文件失败: {lbl_file}: {e}")
                skipped += 1
                break
        else:
            converted += 1
            if verbose and converted % 100 == 0:
                log_info(f"已转换: {converted} 个标注")
    
    
    # 写 classes.txt
    classes_file = output_dir / 'classes.txt'
    try:
        with open(classes_file, 'w', encoding='utf-8') as f:
            # 定义最终的类别名称列表
            classes = [
                'Car',           # 0
                'Truck',         # 1
                'Pickup',        # 2
                'Tractor',       # 3
                'Camping Car',   # 4
                'Boat',          # 5
                'Motorcycle',    # 6
                'Bus',           # 7
                'Van',           # 8
                'Plane',         # 9
                'Truck-Tractor', # 10
                'Other'          # 11
            ]
            f.write('\n'.join(classes) + '\n')
        log_info(f"已写类别文件: {classes_file} (共 {len(classes)} 类)")
    except Exception as e:
        log_warn(f"写类别文件失败: {e}")
    
    return converted, skipped, nir_count


def maybe_split_to_format1(raw_dir: Path, output_dir: Path, 
                            train_ratio: float = 0.8, 
                            val_ratio: float = 0.1, 
                            test_ratio: float = 0.1,
                            seed: int = 42) -> bool:
    """调用 yolo_dataset_split.py 将 standard 结构划分为 format1
    
    返回: 成功与否
    """
    script_path = Path(__file__).resolve().parents[2] / 'tools-general' / 'processing' / 'yolo_dataset_split.py'
    
    if not script_path.exists():
        log_warn(f"找不到划分脚本: {script_path}，跳过自动划分")
        return False
    
    log_info(f"调用划分脚本: {script_path.name}")
    
    cmd = [
        sys.executable, str(script_path),
        '-i', str(raw_dir),
        '-o', str(output_dir),
        '--train_ratio', str(train_ratio),
        '--val_ratio', str(val_ratio),
        '--test_ratio', str(test_ratio),
        '--seed', str(seed),
        '--output_format', '1',
    ]
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        if result.returncode == 0:
            log_info("数据集划分完成")
            return True
        else:
            log_warn(f"划分脚本返回非零状态码: {result.returncode}")
            return False
    except Exception as e:
        log_error(f"调用划分脚本失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='将 VEDAI 转换为 YOLO 格式（支持 RGB + NIR 多光谱）',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
使用示例:
  # 仅 RGB，输出 standard 结构（HBB 水平框，默认）
  python vedai2yolo.py -i ~/datasets/VEDAI -o VEDAI-YOLO-raw
  
  # 保留 OBB 定向框格式（含角度）
  python vedai2yolo.py -i ~/datasets/VEDAI -o VEDAI-YOLO-raw --obb
  
  # 包含 NIR 波段（会自动搜索 *_ir.png 等文件进行配对）
  python vedai2yolo.py -i ~/datasets/VEDAI -o VEDAI-YOLO-raw --with-nir
  
  # 转换后自动进行 8:1:1 划分为格式一
  python vedai2yolo.py -i ~/datasets/VEDAI -o VEDAI-YOLO

输出文件结构:
  若指定 --no-split，输出为 standard 结构:
    VEDAI-YOLO-raw/images/  (所有图片)
    VEDAI-YOLO-raw/labels/  (所有标签)
    VEDAI-YOLO-raw/classes.txt
  
  若启用自动划分（默认），还会生成:
    VEDAI-YOLO-format1/train/images + labels
    VEDAI-YOLO-format1/val/images + labels
    VEDAI-YOLO-format1/test/images + labels
    VEDAI-YOLO-format1/data.yaml
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='VEDAI 原始数据根目录（应包含 Vehicules1024/ 和 Annotations1024/ 或 Vehicules512/ 和 Annotations512/）')
    parser.add_argument('-o', '--output', default='outputs/VEDAI-YOLO',
                        help='输出根目录前缀（脚本会生成 *-raw 和可选 *-format1）')
    parser.add_argument('--with-nir', action='store_true',
                        help='启用 NIR（近红外）波段搜索与复制（默认: 仅 RGB）')
    parser.add_argument('--obb', action='store_true',
                        help='保留 OBB 定向框格式（包含角度）。默认转为 HBB 水平框')
    parser.add_argument('--no-split', action='store_true',
                        help='不执行 8:1:1 划分，仅保留 standard 结构（默认: 自动划分）')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='训练集比例（默认: 0.8）')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='验证集比例（默认: 0.1）')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                        help='测试集比例（默认: 0.1）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认: 42）')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    parser.add_argument('--max-images', type=int, default=None,
                        help='最大转换图片数量（用于测试，默认无限制）')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        log_error(f"输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 验证比例和
    if not args.no_split:
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            log_error(f"训练/验证/测试比例总和应为 1.0，当前为 {total_ratio}")
            sys.exit(1)
    
    log_info("===== VEDAI -> YOLO 转换工具 =====")
    log_info(f"输入目录: {input_dir}")
    
    # 自动检测 VEDAI 子目录
    vehicules_dir = None
    annotations_dir = None
    
    # 优先查找 1024 版本
    if (input_dir / 'Vehicules1024').exists() and (input_dir / 'Annotations1024').exists():
        vehicules_dir = input_dir / 'Vehicules1024'
        annotations_dir = input_dir / 'Annotations1024'
        log_info("检测到 1024x1024 分辨率数据")
    elif (input_dir / 'Vehicules512').exists() and (input_dir / 'Annotations512').exists():
        vehicules_dir = input_dir / 'Vehicules512'
        annotations_dir = input_dir / 'Annotations512'
        log_info("检测到 512x512 分辨率数据")
    elif (input_dir / 'vehicules').exists() and (input_dir / 'annotations').exists():
        vehicules_dir = input_dir / 'vehicules'
        annotations_dir = input_dir / 'annotations'
        log_info("检测到小写命名目录")
    else:
        # 尝试直接把输入目录作为图像目录，查找同级的标注目录
        if (input_dir.parent / 'Annotations1024').exists():
            vehicules_dir = input_dir
            annotations_dir = input_dir.parent / 'Annotations1024'
        else:
            log_error(f"未在 {input_dir} 中找到 Vehicules* 和 Annotations* 子目录")
            sys.exit(1)
    
    # 第一步：转换为 YOLO standard 结构
    # 确保输出目录名不重复 "-raw" 后缀
    output_name = str(args.output).rstrip('/')
    if not output_name.endswith('-raw'):
        raw_output = Path(output_name + '-raw')
    else:
        raw_output = Path(output_name)
    log_info(f"\n第一步: 转换为 YOLO standard 结构")
    log_info(f"输出目录: {raw_output}")
    
    converted, skipped, nir_count = convert_vedai_to_yolo(
        vehicules_dir, annotations_dir, raw_output,
        with_nir=args.with_nir, 
        verbose=args.verbose,
        obb_mode=args.obb,
        max_images=args.max_images
    )
    
    log_info(f"\n转换完成统计:")
    log_info(f"  已转换: {converted} 张图片")
    log_info(f"  已跳过: {skipped} 张图片")
    if args.with_nir:
        log_info(f"  NIR 配对: {nir_count} 张")
    
    if converted == 0:
        log_error("未转换任何图片，停止")
        sys.exit(1)
    
    # 第二步：可选划分为 format1
    if args.no_split:
        log_info(f"\n已禁用自动划分。输出位置: {raw_output}")
    else:
        log_info(f"\n第二步: 按 {args.train_ratio}:{args.val_ratio}:{args.test_ratio} 划分为格式一")
        format1_output = Path(args.output)  # 不再加 -format1 后缀，直接用 output 作为最终输出
        
        success = maybe_split_to_format1(
            raw_output, format1_output,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        
        if success:
            log_info(f"\n最终输出位置: {format1_output}")
            log_info(f"目录结构:")
            log_info(f"  {format1_output.name}/train/images + labels")
            log_info(f"  {format1_output.name}/val/images + labels")
            log_info(f"  {format1_output.name}/test/images + labels")
            log_info(f"  {format1_output.name}/data.yaml")
        else:
            log_warn(f"划分失败，但 raw 数据已保存到: {raw_output}")
    
    log_info("\n===== 转换结束 =====")


if __name__ == '__main__':
    main()
