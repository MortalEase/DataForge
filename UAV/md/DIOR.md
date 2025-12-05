# DIOR 数据集

> **全称**：Dataset for Object detection in optical Remote sensing images (DIOR)
> **作者/来源**：西北工业大学
> **发布时间**：2019 年
> **任务类型**：遥感图像中的目标检测基准

---

## 一、简介

DIOR 是一个面向遥感图像中目标检测的大规模基准数据集，旨在为遥感目标检测算法提供可复现的评测场景。数据集包含20个常见目标类别，涵盖飞机、机场、棒球场、篮球场、桥梁、烟囱、水坝、高速公路、立交桥、码头、体育场、储罐、网球场、火车站、车辆、船舶等。

数据集的特色：
- 大规模：包含23,463张图像，超过192,500个目标实例
- 多样性：涵盖多种场景、尺度、方向和光照条件
- 标准化：提供标准的训练/验证/测试划分
- 多格式：同时提供水平边界框(HBB)和定向边界框(OBB)标注

---

## 二、元信息（摘要）

| 维度 | 模态 | 任务类型 | 场景 | 类别数 | 图像尺寸 | 文件格式 |
|------|------|----------|------|--------|----------|----------|
| 2D | 遥感 RGB | 目标检测（Object Detection） | 遥感目标场景 | 20 | 800×800 | JPEG/XML |

> 注：具体条目（例如类别数与图像总量）请以本地下载包为准，可使用命令行统计（见下节）。

---

## 三、目录与标注（概览）

### 原始文件组织

```
DIOR/
├── Annotations/                    # 标注文件
│   ├── Horizontal Bounding Boxes/ # 水平边界框标注（XML格式）
│   └── Oriented Bounding Boxes/   # 定向边界框标注（XML格式）
├── JPEGImages-trainval/           # 训练验证图像
├── ImageSets.zip                  # 图像划分信息
└── Layout/Main/                   # 训练/验证/测试划分
```

### 推荐转换为 YOLO 格式（统一处理）

我们建议将原始 DIOR 转换为 YOLO 标准格式以便与仓库内工具链配合：

```
DIOR-YOLO-raw/
├── images/               # 所有图像
├── labels/               # YOLO 格式 .txt 标签
└── classes.txt           # 类别名列表
```

若需要训练/验证/测试划分，可进一步使用仓库脚本按格式一生成：

```
DIOR-YOLO-format1/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── classes.txt
└── data.yaml
```

---

## 四、如何用 CLI 快速读取本地 DIOR 信息

在你的本地机器上（已下载到 `~/datasets/DIOR/`）运行下面命令可快速获得目录与数量统计：

```bash
# 查看顶层目录结构（只显示前几行）
ls -R ~/datasets/DIOR | head -n 80

# 统计图片与标签总数
find ~/datasets/DIOR -type f -name "*.jpg" | wc -l
find ~/datasets/DIOR/Annotations -type f -name "*.xml" | wc -l
```

若已经转换为 YOLO 原始格式 (`DIOR-YOLO-raw`) 或格式一 (`DIOR-YOLO-format1`)，可使用仓库内工具获取更详细统计：

```bash
# 显示类别与使用统计
python tools-general/processing/yolo_class_manager.py --dataset_dir DIOR-YOLO-format1 info

# 批量查看若干样例（交互式或批量九宫格）
python tools-general/analysis/yolo_dataset_viewer.py -d DIOR-YOLO-format1/train --batch -n 9
```

---

## 五、如何将 DIOR 转为 YOLO（示例流程）

1. 解压原始包并统一放到 `~/datasets/DIOR/`。
2. 编写或使用转换脚本（示例脚本位置建议 `tools-uav/DIOR/dior2yolo.py`）。脚本支持 `--obb` 开关以决定是否使用定向边界框（默认使用水平边界框）。
3. 生成 `DIOR-YOLO-raw/`（images + labels + classes.txt）。
4. 若原始未划分，使用仓库分割工具按 8:1:1 生成格式一：

```bash
python tools-general/processing/yolo_dataset_split.py \
  -i DIOR-YOLO-raw \
  -o DIOR-YOLO-format1 \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

脚本会复制图像、标签并生成 `data.yaml`（含 `path/train/val/test/nc/names`）。

---

## 六、标注格式详解

### 原始 DIOR 标注格式

DIOR 标注文件（`.xml`）采用 Pascal VOC 格式，每行代表一个检测框：

```xml
<annotation>
    <folder>JPEGImages</folder>
    0001.jpg</filename>
    <source>
        <database>DIOR</database>
        <annotation>DIOR</annotation>
        <image>flickr</image>
        <flickrid>0</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>DIOR</name>
    </owner>
    <size>
        <width>800</width>
        <height>800</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>airplane</name>
        <pose>unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>100</xmin>
            <ymin>200</ymin>
            <xmax>300</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
    <object>
        <name>airplane</name>
        <pose>unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>500</xmin>
            <ymin>600</ymin>
            <xmax>700</xmax>
            <ymax>750</ymax>
        </bndbox>
    </object>
</annotation>
```

| 字段 | 含义 | 说明 |
|------|------|------|
| filename | 图像文件名 | 对应的图像文件名称 |
| width, height | 图像尺寸 | 图像的宽度和高度（像素） |
| name | 类别名称 | 目标类别名称 |
| xmin, ymin | 左上角坐标 | 边界框左上角坐标 |
| xmax, ymax | 右下角坐标 | 边界框右下角坐标 |
| difficult | 困难样本标志 | 1=困难样本，0=普通样本 |
| truncated | 截断标志 | 1=截断样本，0=完整样本 |

### 定向边界框格式（OBB）

在 `Oriented Bounding Boxes/` 目录中，标注文件包含额外的角度信息：

```xml
<object>
    <name>airplane</name>
    <pose>unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <robndbox>
        <cx>200</cx>        <!-- 中心点x坐标 -->
        <cy>300</cy>        <!-- 中心点y坐标 -->
        <w>150</w>          <!-- 宽度 -->
        <h>100</h>          <!-- 高度 -->
        <angle>45</angle>   <!-- 旋转角度（度） -->
    </robndbox>
</object>
```

### 类别映射

DIOR 包含 20 个目标类别：

| 类别ID | 名称（英文） | 名称（中文） | 说明 |
|--------|------------|------------|------|
| 0 | airplane | 飞机 | 各种类型的飞机 |
| 1 | airport | 机场 | 机场建筑和设施 |
| 2 | baseballfield | 棒球场 | 棒球场地 |
| 3 | basketballcourt | 篮球场 | 篮球场地 |
| 4 | bridge | 桥梁 | 各种桥梁结构 |
| 5 | chimney | 烟囱 | 工业烟囱 |
| 6 | dam | 水坝 | 水坝建筑 |
| 7 | expressway-service-area | 高速公路服务区 | 高速公路服务设施 |
| 8 | expressway-toll-station | 高速公路收费站 | 高速公路收费站 |
| 9 | golffield | 高尔夫球场 | 高尔夫球场 |
| 10 | groundtrackfield | 田径场 | 运动场跑道 |
| 11 | harbor | 港口 | 港口设施 |
| 12 | overpass | 立交桥 | 立体交叉桥 |
| 13 | ship | 船舶 | 各种船只 |
| 14 | stadium | 体育场 | 大型体育场 |
| 15 | storagetank | 储油罐 | 圆柱形储罐 |
| 16 | tenniscourt | 网球场 | 网球场地 |
| 17 | trainstation | 火车站 | 火车站建筑 |
| 18 | vehicle | 车辆 | 各类车辆 |
| 19 | windmill | 风力发电机 | 风力发电设备 |

### 转换为 YOLO 格式

我们的转换脚本 `tools-uav/DIOR/dior2yolo.py` 自动执行如下映射：

1. **坐标转换**：从 XML 坐标 (xmin, ymin, xmax, ymax) 转为 YOLO 归一化格式
2. **类别映射**：DIOR 类别名 → YOLO class_id (0-19)
3. **格式**（默认 HBB 水平框）：`class_id x_center y_center width height`
4. **可选 OBB 模式**：`class_id x_center y_center width height angle`（保留方向角）

转换后的 `classes.txt`：
```
airplane
airport
baseballfield
basketballcourt
bridge
chimney
dam
expressway-service-area
expressway-toll-station
golffield
groundtrackfield
harbor
overpass
ship
stadium
storagetank
tenniscourt
trainstation
vehicle
windmill
```

---

## 七、标签格式建议（已弃用 - 参见"标注格式详解"）

- 标签采用 YOLO 格式：每行 `class_id x_center y_center width height`（坐标归一化）。
- `classes.txt` 列出类别顺序，每行一个名称（见上表）。
- 若使用 OBB，建议在文件命名中保留区分（如 `*_obb.txt`）以便区分 HBB 与 OBB 标签。

---

## 八、使用与训练建议

- 多尺度训练：DIOR 包含多种尺度的目标，建议使用多尺度训练策略
- 数据增强：使用随机裁剪、旋转、颜色抖动等增强技术
- 输入尺寸：建议 800×800 或 1024×1024，与原始图像尺寸保持一致
- 类别平衡：注意不同类别间的样本数量差异，可使用类别权重或采样策略

---

## 九、引用

若在研究中使用 DIOR，请引用原始文章：

```bibtex
@article{li2020object,
  title   = {Object detection in optical remote sensing images: A survey and a new benchmark},
  author  = {Li, Ke and Wan, Gang and Cheng, Gong and Meng, Liqiu and Han, Junwei},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {159},
  pages   = {296-306},
  year    = {2020}
}
```

---
