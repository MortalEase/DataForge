#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEDAI 类别采样脚本
功能：从原始数据集中为每个类别抽取 3 张包含该类别的图片，并转换为 YOLO 格式，
以便可视化确认类别含义。
"""

import os
import sys
import shutil
import glob
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到 sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import log_info, log_warn, log_error

# 目标类别 ID 列表 (基于之前的分析)
TARGET_CLASS_IDS = [1, 2, 4, 5, 7, 8, 9, 10, 11, 23, 31]
SAMPLES_PER_CLASS = 3

def parse_annotation_for_classes(annotation_path: Path) -> List[int]:
    """快速解析标注文件包含的类别 ID"""
    class_ids = set()
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                
                # 过滤掉可能的 Image_ID (如果是 15 列)
                data = parts
                if len(parts) == 15:
                    data = parts[1:]
                
                if len(data) < 14:
                    continue
                
                try:
                    # 字段 3: class_id
                    cid = int(float(data[3]))
                    class_ids.add(cid)
                except ValueError:
                    continue
    except Exception:
        pass
    return list(class_ids)

def convert_sample(image_path: Path, annotation_path: Path, output_dir: Path, id_map: Dict[int, int]):
    """转换单张图片和标注"""
    images_out = output_dir / 'images'
    labels_out = output_dir / 'labels'
    
    # 复制图片
    dst_img = images_out / image_path.name
    if not dst_img.exists():
        shutil.copy2(image_path, dst_img)
    
    # 获取图片尺寸
    img_w, img_h = 0, 0
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        print(f"Error reading image size for {image_path}: {e}")
        return # 无法读取图片尺寸，跳过

    # 转换标注
    yolo_lines = []
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                data = parts
                if len(parts) == 15:
                    data = parts[1:]
                if len(data) < 14:
                    continue
                
                try:
                    cx = float(data[0])
                    cy = float(data[1])
                    class_id = int(float(data[3]))
                    xs = [float(x) for x in data[6:10]]
                    ys = [float(y) for y in data[10:14]]
                    
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    calc_cx = (min_x + max_x) / 2
                    calc_cy = (min_y + max_y) / 2
                    w = max_x - min_x
                    h = max_y - min_y
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    # 归一化
                    norm_cx = max(0.0, min(1.0, calc_cx / img_w))
                    norm_cy = max(0.0, min(1.0, calc_cy / img_h))
                    norm_w = max(0.0, min(1.0, w / img_w))
                    norm_h = max(0.0, min(1.0, h / img_h))
                    
                    # 映射 ID
                    # 如果 ID 不在目标列表中，归为 Other (最后一个索引)
                    yolo_class_id = id_map.get(class_id, len(id_map))
                    
                    yolo_lines.append(f"{yolo_class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
                except ValueError as e:
                    print(f"ValueError parsing line in {annotation_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error processing annotation {annotation_path}: {e}")
        pass
        
    # 写入标签
    label_file = labels_out / f"{image_path.stem}.txt"
    with open(label_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_lines) + '\n')

def main():
    input_dir = Path(os.path.expanduser('~/datasets/VEDAI')).resolve()
    output_dir = Path('outputs/VEDAI-samples').resolve()
    
    # 自动定位目录
    if (input_dir / 'Vehicules1024').exists():
        img_dir = input_dir / 'Vehicules1024'
        ann_dir = input_dir / 'Annotations1024'
    else:
        log_error("未找到 Vehicules1024 目录")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 准备 ID 映射
    # 映射: ID -> Index
    id_map = {uid: i for i, uid in enumerate(TARGET_CLASS_IDS)}
    
    # 记录每个类别已收集的样本数
    collected_counts = {uid: 0 for uid in TARGET_CLASS_IDS}
    
    # 扫描所有标注文件
    ann_files = sorted(list(ann_dir.glob('*.txt')))
    
    print(f"开始扫描 {len(ann_files)} 个文件，为每个类别抽取 {SAMPLES_PER_CLASS} 个样本...")
    
    for ann_file in ann_files:
        if ann_file.name in ['classes.txt', 'annotation1024.txt'] or 'fold' in ann_file.name:
            continue
            
        # 检查该文件包含哪些类别
        present_classes = parse_annotation_for_classes(ann_file)
        
        # 检查是否有我们需要收集的类别
        needed = False
        for cid in present_classes:
            if cid in collected_counts and collected_counts[cid] < SAMPLES_PER_CLASS:
                needed = True
                break
        
        if needed:
            # 找到对应的图片
            base_name = ann_file.stem
            img_file = img_dir / f"{base_name}_co.png"
            if not img_file.exists():
                continue
            
            # 转换并保存
            convert_sample(img_file, ann_file, output_dir, id_map)
            
            # 更新计数
            for cid in present_classes:
                if cid in collected_counts and collected_counts[cid] < SAMPLES_PER_CLASS:
                    collected_counts[cid] += 1
                    print(f"  收集到 ID {cid} 的样本: {base_name} (当前 {collected_counts[cid]}/{SAMPLES_PER_CLASS})")
        
        # 检查是否所有类别都收集齐了
        if all(c >= SAMPLES_PER_CLASS for c in collected_counts.values()):
            print("所有类别样本收集完毕！")
            break
            
    # 生成 classes.txt
    classes_file = output_dir / 'classes.txt'
    class_names = [f"ID_{uid}" for uid in TARGET_CLASS_IDS] + ["Other"]
    with open(classes_file, 'w') as f:
        f.write('\n'.join(class_names) + '\n')
        
    print(f"采样完成。输出目录: {output_dir}")
    print(f"请运行: python tools-general/visualization/visualize_yolo_samples.py -d {output_dir}")

if __name__ == '__main__':
    main()
