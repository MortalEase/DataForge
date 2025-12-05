# VEDAI 数据集

> **全称**：Vehicle Detection in Aerial Imagery (VEDAI)
> **作者/来源**：Sébastien Razakarivony 和 Frédéric Jurie
> **发布时间**：2014 年
> **任务类型**：航拍图像中的车辆检测基准（小目标检测）

---

## 一、简介

VEDAI 是一个面向航拍图像中车辆检测的小目标基准数据集，旨在为自动目标识别算法提供可复现的评测场景。数据集中车辆目标通常体积较小，并存在多方向、光照/阴影变化、反射与遮挡等可变性。

数据集的特色：
- 小目标：车辆在整幅航拍图中占比小，适合评估小目标检测能力
- 多变性：多角度、多场景（城市、停车场、道路）与复杂背景
- 多波段（视版本而定）：部分镜像提供近红外（NIR）波段

作者提供了 development kit 与实验协议，便于实现可比的实验设置与基线评价。

---

## 二、元信息（摘要）

| 维度 | 模态 | 任务类型 | 场景 | 类别数 | 图像尺寸 | 文件格式 |
|------|------|----------|------|--------|----------|----------|
| 2D | 航拍 RGB（可选 NIR） | 目标检测（Object Detection） | 航拍车辆场景 | 视版本（通常为 vehicle 或多类车辆细分） | 512×512 / 1024×1024 | TIFF/PNG/JPEG |

> 注：具体条目（例如类别数与图像总量）请以本地下载包为准，可使用命令行统计（见下节）。

---

## 三、目录与标注（概览）

### 原始文件组织（常见）

```
VEDAI/
├── annotations/          # 标注文件（不同分辨率/分包）
├── images_512/           # 512x512 图像分包（part1/part2）
├── images_1024/          # 1024x1024 图像分包
└── devkit/               # development kit, 脚本与说明
```

### 推荐转换为 YOLO 格式（统一处理）

我们建议将原始 VEDAI 转换为 YOLO 标准格式以便与仓库内工具链配合：

```
VEDAI-YOLO-raw/
├── images/               # 所有图像 (RGB 或含 NIR 的图片按命名保留)
├── labels/               # YOLO 格式 .txt 标签
└── classes.txt           # 类别名列表 (如: vehicle 或细分车辆类别)
```

若需要训练/验证/测试划分，可进一步使用仓库脚本按格式一生成：

```
VEDAI-YOLO-format1/
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

## 四、如何用 CLI 快速读取本地 VEDAI 信息

在你的本地机器上（已下载到 `~/datasets/VEDAI/`）运行下面命令可快速获得目录与数量统计：

```bash
# 查看顶层目录结构（只显示前几行）
ls -R ~/datasets/VEDAI | head -n 80

# 统计图片与标签总数
find ~/datasets/VEDAI -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.tif" \) | wc -l
find ~/datasets/VEDAI -type f -name "*.txt" | wc -l
```

若已经转换为 YOLO 原始格式 (`VEDAI-YOLO-raw`) 或格式一 (`VEDAI-YOLO-format1`)，可使用仓库内工具获取更详细统计：

```bash
# 显示类别与使用统计
python tools-general/processing/yolo_class_manager.py --dataset_dir VEDAI-YOLO-format1 info

# 批量查看若干样例（交互式或批量九宫格）
python tools-general/analysis/yolo_dataset_viewer.py -d VEDAI-YOLO-format1/train --batch -n 9
```

---

## 五、如何将 VEDAI 转为 YOLO（示例流程）

1. 解压原始包并统一放到 `~/datasets/VEDAI/`。
2. 编写或使用转换脚本（示例脚本位置建议 `tools-uav/VEDAI/vedai2yolo.py`）。脚本应支持 `--with-nir` 开关以决定是否把 NIR 波段也纳入转换（当原数据包含 NIR 时）。
3. 生成 `VEDAI-YOLO-raw/`（images + labels + classes.txt）。
4. 若原始未划分，使用仓库分割工具按 8:1:1 生成格式一：

```bash
python tools-general/processing/yolo_dataset_split.py \
  -i VEDAI-YOLO-raw \
  -o VEDAI-YOLO-format1 \
  --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

脚本会复制图像、标签并生成 `data.yaml`（含 `path/train/val/test/nc/names`）。

---

## 六、标注格式详解

### 原始 VEDAI 标注格式

VEDAI 标注文件（`.txt`）每行代表一个检测框，格式为：
```
[Image_ID] Center_X Center_Y Orientation X1 Y1 X2 Y2 X3 Y3 X4 Y4 Class_ID Contained Occluded
```
其中 `Image_ID` 字段在部分版本中存在（例如 development kit 中会在每行前加图像编号），本仓库脚本会自动跳过非数值的首字段。

| 字段 | 含义 | 说明 |
|------|------|------|
| Center_X, Center_Y | 中心坐标 | 像素单位，以图片左上角为原点 |
| Orientation | 方向角 | 弧度制（通常 $[-\pi, \pi]$ 范围），表示目标旋转角度 |
| Class_ID | 类别 ID | 1-9，映射见下表 |
| Contained | 完整性标志 | 1=完全在图内，0=被边界截断 |
| Unknown | 未知字段 | 通常为 0（保留字段） |
| X1~X4, Y1~Y4 | 4个角点坐标 | 旋转矩形的四个顶点，顺序通常为左上→右上→右下→左下 |

### 类别映射

VEDAI 包含 9 个车辆类别：

| Class_ID | 名称（英文） | 名称（中文） | 说明 |
|----------|------------|----------|------|
| 1 | Plane | 飞机 | 小型飞机 |
| 2 | Boat | 船 | 小船/木筏 |
| 3 | Camping Car | 房车 | 露营车 |
| 4 | Car | 汽车 | 普通轿车 |
| 5 | Pick-up | 皮卡 | 皮卡车 |
| 6 | Tractor | 拖拉机 | 农业用拖拉机 |
| 7 | Truck | 卡车 | 大型卡车 |
| 8 | Van | 货车 | 厢式货车 |
| 9 | Other | 其他 | 不属于上述类别的车辆 |

### 转换为 YOLO 格式

我们的转换脚本 `tools-uav/VEDAI/vedai2yolo.py` 自动执行如下映射：

1. **坐标转换**：从像素坐标 (cx, cy, w, h) 转为 YOLO 归一化格式
2. **类别映射**：VEDAI Class_ID (1-9) → YOLO class_id (0-8)，即 `class_id = VEDAI_Class_ID - 1`
3. **格式**（默认 HBB 水平框）：`class_id x_center y_center width height`
4. **可选 OBB 模式**：`class_id x_center y_center width height angle`（保留方向角）

转换后的 `classes.txt`：
```
Plane
Boat
Camping Car
Car
Pick-up
Tractor
Truck
Van
Other
```

---

## 七、标签格式建议（已弃用 - 参见"标注格式详解"）

- 标签采用 YOLO 格式：每行 `class_id x_center y_center width height`（坐标归一化）。
- `classes.txt` 列出类别顺序，每行一个名称（见上表）。
- 若使用 NIR，建议在文件命名中保留区分（如 `*_co.png`/`*_ir.png`）以便区分 RGB 与近红外通道。

---

## 八、使用与训练建议

- 小目标增强：使用多尺度训练、随机缩放、微小目标增强（mosaic/oversample 小目标）等技巧
- 输入尺寸：建议 640×640 或 1024×1024，视 GPU 内存与目标大小调整
- 多光谱实验：可尝试 4 通道输入（RGB+NIR），需自定义数据加载与模型第一层

---

## 九、引用

若在研究中使用 VEDAI，请引用原始文章：

```bibtex
@article{razakarivony2015vedai,
  title   = {Vehicle Detection in Aerial Imagery: A small target detection benchmark},
  author  = {Razakarivony, S{\'e}bastien and Jurie, Fr{\'e}d{\'e}ric},
  journal = {Journal of Visual Communication and Image Representation},
  year    = {2015}
}
```

---