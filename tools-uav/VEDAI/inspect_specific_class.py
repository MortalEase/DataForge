#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEDAI 特定类别详细检查脚本
功能：专门提取指定类别（如 ID 10）的样本，生成：
1. 裁剪后的物体特写图（方便辨认细节）
2. 带有标注框的完整图
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import shutil

# 添加项目根目录到 sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import log_info, log_warn, log_error

def parse_vedai_annotation(annotation_path: Path, target_class_id: int):
    """解析标注文件，返回指定类别的所有实例坐标"""
    instances = []
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
                    # 字段 3: class_id
                    class_id = int(float(data[3]))
                    
                    if class_id != target_class_id:
                        continue

                    # 字段 6-9: 4个角点的 X 坐标
                    xs = [float(x) for x in data[6:10]]
                    # 字段 10-13: 4个角点的 Y 坐标
                    ys = [float(y) for y in data[10:14]]
                    
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    instances.append((min_x, min_y, max_x, max_y))
                except ValueError:
                    continue
    except Exception:
        pass
    return instances

def main():
    target_id = 4 # Van
    max_samples = 20  # 提取多少个样本
    
    input_dir = Path(os.path.expanduser('~/datasets/VEDAI')).resolve()
    output_dir = Path(f'outputs/VEDAI-inspection/class_{target_id}').resolve()
    
    img_dir = input_dir / 'Vehicules1024'
    ann_dir = input_dir / 'Annotations1024'
    
    if not img_dir.exists():
        print(f"Error: {img_dir} not found")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    crops_dir = output_dir / 'crops'
    full_dir = output_dir / 'full'
    crops_dir.mkdir(parents=True, exist_ok=True)
    full_dir.mkdir(parents=True, exist_ok=True)
    
    ann_files = sorted(list(ann_dir.glob('*.txt')))
    
    count = 0
    print(f"正在搜索 ID {target_id} 的样本...")
    
    for ann_file in ann_files:
        if count >= max_samples:
            break
            
        instances = parse_vedai_annotation(ann_file, target_id)
        if not instances:
            continue
            
        # 读取图片
        base_name = ann_file.stem
        img_path = img_dir / f"{base_name}_co.png"
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h_img, w_img = img.shape[:2]
        
        # 处理该图片中的所有目标实例
        for i, (min_x, min_y, max_x, max_y) in enumerate(instances):
            if count >= max_samples:
                break
                
            # 1. 保存裁剪图 (加一点 padding)
            pad = 20
            x1 = max(0, int(min_x - pad))
            y1 = max(0, int(min_y - pad))
            x2 = min(w_img, int(max_x + pad))
            y2 = min(h_img, int(max_y + pad))
            
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crop_name = f"{base_name}_obj{i}.png"
                cv2.imwrite(str(crops_dir / crop_name), crop)
            
            # 2. 在原图上画框
            cv2.rectangle(img, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 0, 255), 2)
            cv2.putText(img, f"ID {target_id}", (int(min_x), int(min_y)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            count += 1
            print(f"  提取样本: {base_name} (对象 {i})")
            
        # 保存完整图
        cv2.imwrite(str(full_dir / f"{base_name}_annotated.png"), img)

    print(f"\n完成！已提取 {count} 个样本。")
    print(f"裁剪特写保存在: {crops_dir}")
    print(f"完整标注图保存在: {full_dir}")

if __name__ == '__main__':
    main()
