#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DIOR -> YOLO 格式转换脚本

核心: 解析 DIOR 原始标注（XML Pascal VOC 格式）并转为 YOLO 格式
DIOR 标注格式: Pascal VOC XML 格式，包含水平边界框(HBB)和定向边界框(OBB)
支持 20 个类别，使用官方划分 (Main/train.txt, val.txt, test.txt)
默认: 生成格式1结构 (train/val/test 各自含 images/ 和 labels/)
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import tee_stdout_stderr, log_info, log_warn, log_error

_LOG_FILE = tee_stdout_stderr('logs')

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']


def parse_dior_xml_annotation(annotation_path: Path) -> List[Tuple[float, float, float, float, int, Optional[float]]]:
    """解析 DIOR XML 标注文件
    
    返回: [(cx, cy, w, h, class_id, angle), ...]，坐标为像素值（中心+宽高）
    """
    result = []
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # 解析所有对象
        for obj in root.findall('object'):
            class_name = obj.find('name')
            if class_name is None:
                continue
            
            class_name_str = normalize_class_name(class_name.text)
            if class_name_str not in CLASS_NAME_TO_ID:
                continue
            
            class_id = CLASS_NAME_TO_ID[class_name_str]
            angle = None
            
            # 优先解析 OBB (robndbox)
            robndbox = obj.find('.//robndbox')
            if robndbox is not None:
                try:
                    cx = float(robndbox.find('cx').text)
                    cy = float(robndbox.find('cy').text)
                    w = float(robndbox.find('w').text)
                    h = float(robndbox.find('h').text)
                    angle_elem = robndbox.find('angle')
                    if angle_elem is not None:
                        angle = float(angle_elem.text)
                    result.append((cx, cy, w, h, class_id, angle))
                    continue
                except (ValueError, AttributeError):
                    pass
            
            # 回退到 HBB (bndbox)
            bndbox = obj.find('.//bndbox')
            if bndbox is not None:
                try:
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin
                    result.append((cx, cy, w, h, class_id, None))
                except (ValueError, AttributeError):
                    pass
    except Exception as e:
        pass
    
    return result


def convert_dior_to_yolo(input_dir: Path, annotations_dir: Path, output_dir: Path,
                         obb_mode: bool = False, max_images: int = None) -> Tuple[int, int, int]:
    """转换 DIOR 数据集为 YOLO 格式
    
    返回: (已转换图片数, 已跳过图片数, 总目标数)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_output = output_dir / 'images'
    labels_output = output_dir / 'labels'
    images_output.mkdir(exist_ok=True)
    labels_output.mkdir(exist_ok=True)
    
    # 收集所有图像
    all_images = []
    for ext in IMAGE_EXTS:
        all_images.extend(sorted(input_dir.rglob(f'*{ext}')))
        all_images.extend(sorted(input_dir.rglob(f'*{ext.upper()}')))
    all_images = list(set(all_images))
    
    if max_images:
        all_images = all_images[:max_images]
    
    converted = 0
    skipped = 0
    total_objects = 0
    
    for img_path in tqdm(all_images, desc='转换中'):
        stem = img_path.stem
        ann_path = annotations_dir / f'{stem}.xml'
        
        if not ann_path.exists():
            skipped += 1
            continue
        
        objects = parse_dior_xml_annotation(ann_path)
        if not objects:
            skipped += 1
            continue
        
        # 复制图像
        shutil.copy2(img_path, images_output / img_path.name)
        
        # 生成标签（获取图像尺寸用于归一化）
        try:
            from PIL import Image
            img = Image.open(img_path)
            img_w, img_h = img.size
        except:
            try:
                import cv2
                img = cv2.imread(str(img_path))
                img_h, img_w = img.shape[:2]
            except:
                skipped += 1
                continue
        
        # 写入标签
        label_path = labels_output / f'{stem}.txt'
        with open(label_path, 'w') as f:
            for cx, cy, w, h, class_id, angle in objects:
                # 归一化到 [0, 1]
                norm_cx = cx / img_w
                norm_cy = cy / img_h
                norm_w = w / img_w
                norm_h = h / img_h
                
                if obb_mode and angle is not None:
                    f.write(f'{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {angle:.6f}\n')
                else:
                    f.write(f'{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n')
                
                total_objects += 1
        
        converted += 1
    
    # 保存 classes.txt
    classes_path = output_dir / 'classes.txt'
    with open(classes_path, 'w') as f:
        for class_id in sorted(CLASS_ID_TO_NAME.keys()):
            f.write(f'{class_id} {CLASS_ID_TO_NAME[class_id]}\n')
    
    return converted, skipped, total_objects


def use_original_split(input_dir: Path, raw_dir: Path, output_dir: Path) -> bool:
    """使用官方划分文件 (Main/train.txt, val.txt, test.txt)"""
    # 尝试多个可能的位置
    split_base_dirs = [
        input_dir / 'Main',
        input_dir / 'ImageSets' / 'Main',
    ]
    
    split_base = None
    for base_dir in split_base_dirs:
        if (base_dir / 'train.txt').exists():
            split_base = base_dir
            break
    
    if not split_base:
        return False
    
    split_files = {
        'train': split_base / 'train.txt',
        'val': split_base / 'val.txt',
        'test': split_base / 'test.txt',
    }
    
    # 检查所有划分文件是否存在
    for split, split_path in split_files.items():
        if not split_path.exists():
            return False
    
    # 读取文件名列表
    split_files_dict = {}
    for split, split_path in split_files.items():
        with open(split_path) as f:
            split_files_dict[split] = [line.strip() for line in f if line.strip()]
    
    # 查找图像目录
    img_dirs = [input_dir / 'JPEGImages-trainval', input_dir / 'JPEGImages-test']
    
    def find_image_file(filename_stem):
        """在多个目录中查找图像"""
        for ext in IMAGE_EXTS:
            for img_dir in img_dirs:
                img_path = img_dir / f'{filename_stem}{ext}'
                if img_path.exists():
                    return img_path
        return None
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 复制文件到对应的划分目录
    for split, filenames in tqdm(split_files_dict.items(), desc='应用划分'):
        for filename in tqdm(filenames, desc=f'{split}', leave=False):
            img_path = find_image_file(filename)
            label_path = raw_dir / 'labels' / f'{filename}.txt'
            
            if not img_path or not label_path.exists():
                continue
            
            # 复制图像和标签
            shutil.copy2(img_path, output_dir / split / 'images' / img_path.name)
            shutil.copy2(label_path, output_dir / split / 'labels' / label_path.name)
    
    # 生成 data.yaml
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n")
        f.write(f"nc: {len(CLASS_ID_TO_NAME)}\n")
        f.write(f"names: {{{', '.join(f'{i}: {CLASS_ID_TO_NAME[i]}' for i in sorted(CLASS_ID_TO_NAME.keys()))}}}\n")
    
    return True


# DIOR 类别映射
_CLASS_ALIASES = {
    'airplane': 0,
    'airport': 1,
    'baseballfield': 2,
    'basketballcourt': 3,
    'bridge': 4,
    'chimney': 5,
    'dam': 6,
    'expressway-service-area': 7,
    'expressway-toll-station': 8,
    'golffield': 9,
    'groundtrackfield': 10,
    'harbor': 11,
    'overpass': 12,
    'ship': 13,
    'stadium': 14,
    'storagetank': 15,
    'tenniscourt': 16,
    'trainstation': 17,
    'vehicle': 18,
    'windmill': 19,
}

def normalize_class_name(class_name: str) -> str:
    """规范化类别名称（小写、去除多余空格）"""
    return class_name.lower().strip()

CLASS_NAME_TO_ID = {normalize_class_name(k): v for k, v in _CLASS_ALIASES.items()}
CLASS_ID_TO_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}


def main():
    parser = argparse.ArgumentParser(
        description='将 DIOR 转换为 YOLO 格式',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='DIOR 原始数据根目录')
    parser.add_argument('-o', '--output', default='outputs/DIOR-YOLO',
                        help='输出根目录')
    parser.add_argument('--obb', action='store_true',
                        help='使用 OBB 定向框格式')
    parser.add_argument('--max-images', type=int, default=None,
                        help='最大转换图片数量（用于测试）')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input).resolve()
    if not input_dir.exists():
        log_error(f"输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 验证目录结构
    annotations_dir = input_dir / 'Annotations' / 'Horizontal Bounding Boxes'
    if args.obb:
        obb_dir = input_dir / 'Annotations' / 'Oriented Bounding Boxes'
        if obb_dir.exists():
            annotations_dir = obb_dir
    
    if not annotations_dir.exists():
        log_error(f"标注目录不存在: {annotations_dir}")
        sys.exit(1)
    
    # 第一步：转换为 YOLO standard 结构
    output_name = str(args.output).rstrip('/')
    raw_output = Path(output_name + '-raw')
    
    converted, skipped, total_objects = convert_dior_to_yolo(
        input_dir, annotations_dir, raw_output,
        obb_mode=args.obb,
        max_images=args.max_images
    )
    
    if converted == 0:
        log_error("未转换任何图片")
        sys.exit(1)
    
    # 第二步：应用官方划分
    output_dir = Path(args.output)
    if use_original_split(input_dir, raw_output, output_dir):
        log_info(f"转换完成: {converted} 张图片，{total_objects} 个目标")
        log_info(f"输出目录: {output_dir}")
    else:
        log_error("应用原始划分失败")
        sys.exit(1)


if __name__ == '__main__':
    main()
