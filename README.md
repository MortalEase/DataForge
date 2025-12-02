# DataForge

DataForge 是一个专注于目标检测数据集处理的工具箱，提供从格式转换、数据划分、类别管理到统计分析的全流程解决方案。

核心理念：**以 YOLO 格式为核心进行所有数据处理**，仅在最后阶段根据需要转换为 COCO 等其他格式。

## 目录结构与工具说明

### 1. 通用工具 (tools-general)

#### analysis/ (分析工具)

| 脚本 | 作用 | 关键参数 |
|------|------|---------|
| `yolo_dataset_analyzer.py` | YOLO 数据集结构/缺失/统计分析 | `-d`, `--stats` |
| `yolo_dataset_viewer.py` | 可视化查看/筛选/统计 | `-d`, `--filter-classes` |
| `average_color.py` | 计算数据集平均颜色 | `-i` |

#### conversion/ (转换工具)

| 脚本 | 作用 | 关键参数 |
|------|------|---------|
| `voc2yolo.py` | VOC XML -> YOLO 转换 | `-i`, `-o`, `--structure` |
| `yolo2coco.py` | YOLO -> COCO 转换 | `-d`, `-o`, `--split` |
| `yolo_format_convert.py` | YOLO 结构重排 (format1 ↔ format2) | `-d`, `-o`, `--to` |

#### processing/ (处理工具)

| 脚本 | 作用 | 关键参数 |
|------|------|---------|
| `yolo_dataset_split.py` | YOLO 数据集划分 (train/val/test) | `-i`, `-o`, `--train_ratio` |
| `yolo_class_manager.py` | YOLO 类别增删改/重排/清理 | `delete`, `rename`, `reindex` |

### 2. 特定数据集工具

针对特定公开数据集（如 DOTA, VEDAI, 医学影像等）的转换与处理脚本，位于 `tools-uav/` 或其他特定目录下。这些脚本通常用于将原始数据集转换为标准的 YOLO 格式，以便接入上述通用工作流。

---

## 推荐工作流

1.  **格式转换**: 使用特定转换脚本（如 `voc2yolo.py`, `dota2yolo.py`）将原始数据转换为 YOLO 格式。
2.  **数据清洗**: 使用 `yolo_class_manager.py` 统一类别名称、删除无用类别或重排类别 ID。
3.  **数据划分**: 使用 `yolo_dataset_split.py` 将数据集划分为 train/val/test 子集。
4.  **质量检查**: 使用 `yolo_dataset_analyzer.py` 和 `yolo_dataset_viewer.py` 检查标注正确性与数据集分布。
5.  **格式导出**: 如需 COCO 格式（例如用于训练某些模型），使用 `yolo2coco.py` 将处理好的 YOLO 数据集转换为 COCO JSON。

---

## YOLO数据集格式说明

未划分时：

| 格式 | 目录结构 | 特点 |
|------|----------|------|
| **标准** | `dataset/`<br/>`├── images/`<br/>`└── labels/`<br/>`└── classes.txt`<br/>`└── data.yaml` | 单一集合 (未预分割)，常用于后续再划分 |
| **混合** | `dataset/`<br/>`├── *.jpg/*.png`<br/>`├── *.txt`<br/>`└── classes.txt`<br/>`└── data.yaml` | 图片与标签同目录混放，快速整理或小规模数据 |

YOLO数据集支持以下两种主要组织形式：

| 格式 | 目录结构 | 特点 |
|------|----------|------|
| **格式一** | `yolo_dataset/`<br/>`├── train/`<br/>`│   ├── images/`<br/>`│   └── labels/`<br/>`├── val/`<br/>`│   ├── images/`<br/>`│   └── labels/`<br/>`├── test/`<br/>`│   ├── images/`<br/>`│   └── labels/`<br/>`└── classes.txt`<br/>`└── data.yaml` | 按数据集划分分组 (train/val/test 顶级) |
| **格式二** | `yolo_dataset/`<br/>`├── images/`<br/>`│   ├── train/ val/ test/`<br/>`├── labels/`<br/>`│   ├── train/ val/ test/`<br/>`└── classes.txt`<br/>`└── data.yaml` | 按文件类型分组 (images 与 labels 顶级) |

---

## COCO数据集格式说明

| 组成 | 路径/文件 | 说明 |
|------|-----------|------|
| 注释目录 | `annotations/` | 存放所有任务/分割的 JSON 注释 |
| 检测训练集 | `instances_train2017.json` | 目标检测 (images/annotations/categories) 训练集 |
| 检测验证集 | `instances_val2017.json` | 目标检测验证集 |
| 关键点训练 | `person_keypoints_train2017.json` | 人体关键点训练注释 |
| 关键点验证 | `person_keypoints_val2017.json` | 人体关键点验证注释 |
| 描述训练 | `captions_train2017.json` | 图像描述训练注释 |
| 描述验证 | `captions_val2017.json` | 图像描述验证注释 |
| 训练图片 | `train2017/` | 训练图片目录 (file_name 对应) |
| 验证图片 | `val2017/` | 验证图片目录 |
| 测试图片 | `test2017/` | 测试图片目录 (常无公开标注) |

最小检测任务必需：`annotations/instances_train*.json` + `annotations/instances_val*.json` + 对应图片目录。

典型结构：
```
coco_dataset/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── captions_train2017.json
│   ├── captions_val2017.json
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
│
├── train2017/
└── val2017/
```

---

## 详细使用说明

### yolo_dataset_analyzer.py
YOLO数据集分析工具 - 支持多种数据集结构

```bash
# 分析数据集完整性（检查图片与标签对应关系）
python tools-general/analysis/yolo_dataset_analyzer.py -d 数据集根目录

# 显示详细统计信息（包含表格形式的类别分布）
python tools-general/analysis/yolo_dataset_analyzer.py -d 数据集根目录 --stats
```

### yolo_dataset_split.py
YOLO数据集划分工具

```bash
# 基础划分 (默认输出格式一，3个集合)
python tools-general/processing/yolo_dataset_split.py -i 输入数据集目录 -o 输出目录

# 只划分为2个集合 (train/val，不要test)
python tools-general/processing/yolo_dataset_split.py -i 输入数据集目录 -o 输出目录 --no-test --train_ratio 0.8 --val_ratio 0.2
```

### yolo_class_manager.py
YOLO数据集类别管理工具 - 支持删除、重命名类别和备份管理

```bash
# 查看数据集类别信息
python tools-general/processing/yolo_class_manager.py -d 数据集目录 info

# 删除指定类别
python tools-general/processing/yolo_class_manager.py -d 数据集目录 delete -c 1 7 5

# 重命名类别名称
python tools-general/processing/yolo_class_manager.py -d 数据集目录 rename -r "old_name:new_name"
```

### yolo_dataset_viewer.py
YOLO数据集交互式遍历查看器

```bash
# 交互式查看模式
python tools-general/analysis/yolo_dataset_viewer.py -d 数据集根目录

# 批量查看模式
python tools-general/analysis/yolo_dataset_viewer.py -d 数据集根目录 --batch -n 12
```

### yolo2coco.py
YOLO转COCO格式转换工具

```bash
# 1) 多分割 YOLO 结构 (格式一 / 格式二) -> 直接输出各自 JSON
python tools-general/conversion/yolo2coco.py -d path/to/format1_dataset --output_dir output_coco_dir

# 2) 标准结构 / 混合结构 -> 输出单一 COCO 文件
python tools-general/conversion/yolo2coco.py -d path/to/standard_dataset --output_dir coco.json
```

### yolo_format_convert.py
YOLO 目录结构重排工具

```bash
# 自动识别输入结构并转换为相反结构 (format1 <-> format2)
python tools-general/conversion/yolo_format_convert.py -d path/to/yolo_dataset -o path/to/out_dir
```

### voc2yolo.py
VOC (Pascal VOC XML) 转 YOLO 标注转换工具

```bash
python tools-general/conversion/voc2yolo.py -i VOC_ROOT --output_dir YOLO_OUT
```

---

## 日志输出说明

本仓库的所有入口脚本已统一启用日志重定向。每次运行脚本时，标准输出与标准错误会同时：
- 原样打印到控制台；
- 复制写入到项目根目录下的 `logs/` 目录中的日志文件。
