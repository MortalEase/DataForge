# DOTA-v1.0 数据集

> **作者/来源**：武汉大学  
> **版本**：DOTA-v1.0（本仓库当前使用版本）  
> **发布时间**：2017年11月
> **官方主页**：https://captain-whu.github.io/DOTA/index.html  
> **数据集镜像下载**：https://hyper.ai/datasets/4920
> **任务类型**：航拍图像中的旋转目标检测（OBB，定向边界框）

---

## 一、简介

DOTA 数据集是用于航拍图像中的目标检测的大型图像数据集。它可用于发现和评估航拍图像中的物体。无论从数量还是质量上来说，在同类型数据集中都具有很大优势。

DOTA 数据集是一个大规模遥感图像数据集，用于挑战计算机视觉领域的目标检测难题。目标检测一直都是计算机视觉中的一个重要而富有挑战性的问题。尽管过去十年，我们已经见证了目标检测在自然场景的重大进步，但在航拍图像领域，进展却一直很缓慢。

相对于自然图像目标检测任务，例如 COCO、VOC 数据集，其中的目标几乎都是因为重力的原因，具有比较统一的方向。但是在遥感图像目标检测中，目标是以任意方向出现，并不容易完成精确的目标检测，例如车辆、飞机、舰船等。遥感图像数据一般具有目标比例差异、样本不平衡、目标方向/比率差异等特征。

武汉大学于 2017 年 11 月 28 日，于 arXiv 上发布论文《DOTA: A Large-scale Dataset for Object Detection in Aerial Images》，提出了这个新的遥感数据集图像目标检测数据集 DOTA 数据集，之后于 2018 年 6 月在 IEEE 计算机视觉和模式识别会议（CVPR）上发布。

该数据集包含 2806 幅航拍图，每张图像的像素尺寸在 800×800 到 4000×4000 的范围内，其中包含不同尺度、方向和形状的物体。这些图像经由专家使用 15 个常见目标类别进行标注，包括：飞机、轮船、储罐、棒球场、网球场、篮球场、地面跑道、港口、桥梁、大型车辆、小型车辆、直升机、环形交叉路口、足球场和篮球场。完全标注的 DOTA 图像包含 188282 个实例，每个实例均由任意四边形进行标记。

![DOTA 数据集样例](./images/DOTA_example.png "DOTA 数据集样例")

---

## 二、数据集元信息（v1.0）

| 维度 | 模态 | 任务类型 | 传感器来源 | 类别数 | 数据量 | 标注格式 | 数据量大小 |
|------|------|----------|-----------|--------|--------|----------|-----------|
| 2D | 航拍/遥感图像 | 旋转目标检测 (OBB) | Google Earth / GF-2 / JL-1 / 航拍平台 | 15 类 | 2806 张图像 / 188282 个实例 | 有向边界框(四边形) | 约 35 GB |

### 图像尺寸统计

| 指标 | 值 |
|------|-----|
| 图像尺寸范围 | 800 × 800 ~ 4000 × 4000 |
| 平均尺寸 | 约 1600 × 1600 |

> 注意：DOTA 中图像尺寸差异较大，实际转换/训练前建议根据需求统一重采样或缩放到目标分辨率。

### 类别信息（v1.0）

DOTA-v1.0 数据集包含 15 个常见航拍目标类别（你当前下载和使用的版本）：

| 类别 ID | 类别名称 | 类别描述 |
|--------|---------|---------|
| 0 | plane | 飞机 |
| 1 | ship | 轮船 |
| 2 | storage tank | 储罐 |
| 3 | baseball diamond | 棒球场 |
| 4 | tennis court | 网球场 |
| 5 | basketball court | 篮球场 |
| 6 | ground track field | 地面跑道 |
| 7 | harbor | 港口 |
| 8 | bridge | 桥梁 |
| 9 | large vehicle | 大型车辆 |
| 10 | small vehicle | 小型车辆 |
| 11 | helicopter | 直升机 |
| 12 | roundabout | 环形交叉路口 |
| 13 | soccer ball field | 足球场 |
| 14 | swimming pool | 游泳池 |

> 注：上表为 DOTA-v1.0 官方英文类别名称及其中文含义，对应你当前下载的 v1.0 数据版本。

其他版本补充信息（仅作对比，当前仓库未使用）：

- **DOTA-v1.5**：
  - 图像：与 DOTA-v1.0 相同的 2,806 张图像；
  - 类别：在 v1.0 的 15 个类别基础上，增加 *container crane*（集装箱起重机）；
  - 标注：新增了大量极小目标（< 10 像素）标注，总实例数约 403,318；
  - 用途：曾作为 DOAI 2019 航拍图像目标检测挑战赛的官方数据。

- **DOTA-v2.0**：
  - 图像：11,268 张图像；
  - 类别：18 个类别，在 v1.5 的基础上新增 *airport*（机场）、*helipad*（直升机停机坪）；
  - 标注：约 1,793,658 个实例；
  - 划分：train/val/test-dev/test-challenge 等多种拆分方式，用于更大规模的航拍目标检测评测。

---

## 三、数据集文件结构与标注格式

DOTA 数据集通常包含以下目录结构：

```
DOTA-v1.0/
├── train/
│   ├── images/
│   │   ├── P0010.png
│   │   ├── P0011.png
│   │   └── ...
│   ├── labelTxt/
│   │   ├── P0010.txt
│   │   ├── P0011.txt
│   │   └── ...
│   └── Task2_gt.txt (可选)
├── val/
│   ├── images/
│   └── labelTxt/
└── test/
    └── images/
```

### 标注格式（原始 DOTA OBB 格式）

每个 `.txt` 标签文件对应一张图像，包含图像元信息和目标标注。格式如下：

```
'acquisition dates': acquisition_dates
imagesource: GoogleEarth
gsd: 0.125792256955
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
...
```

**字段说明**：
- `acquisition dates`：图像采集日期；缺失时标记为 `None`
- `imagesource`：图像来源（GoogleEarth、GF-2、JL-1、CycloMedia 等），需遵守各平台使用条款
- `gsd`：地面分辨率（Ground Sample Distance），单位米/像素；缺失时标记为 `None`
- `x1 y1 ... x4 y4`：四边形四个顶点的像素坐标（定向边界框，按顺时针顺序，从黄点顶点开始）
- `class_name`：目标类别名称（如 plane, ship 等）
- `difficulty`：难度标志（0=正常，1=难检）

> 许可与用途：DOTA 图像来自 Google Earth、GF-2、JL-1 和 CycloMedia 等多源数据。使用 Google Earth 图像必须遵守 Google Earth 使用条款；所有图像及其标注仅限学术用途，禁止任何商业使用。

---

## 四、数据来源

这些图像来源包含不同传感器和平台，具体为：

- **Google Earth**：谷歌地球卫星数据
- **JL-1 卫星**：中国商业遥感卫星
- **GF-2 卫星**：中国资源卫星数据和应用中心的高分二号卫星数据

多源异构的遥感影像使 DOTA 数据集具有更高的通用性和鲁棒性。

---

## 五、作者与机构

- **发布机构**：武汉大学
- **论文作者**：(详见论文)

---

## 六、来源与引用

- **官方网站/下载链接**：https://hyper.ai/datasets/4920
- **论文地址**：https://arxiv.org/pdf/1711.10398.pdf
- **论文标题**：DOTA: A Large-scale Dataset for Object Detection in Aerial Images
- **发表会议**：CVPR 2018
- **初版发布**：arXiv 2017年11月28日

### 引用格式（BibTeX）

```bibtex
@inproceedings{xia2018dota,
  title={DOTA: A Large-scale Dataset for Object Detection in Aerial Images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}
```

---

## 七、关键词

航拍图像、遥感数据集、旋转目标检测、有向边界框、卫星影像、无人机数据

---

## 八、转换说明

本仓库提供 `dota2yolo.py` 脚本，可将 DOTA 原始格式转换为 YOLO 格式一：

```bash
python dota2yolo.py -i ~/datasets/DOTA-v1.0 -o ./dota_yolo_format1
```

转换过程中，四边形有向边界框会被转换为中心点坐标、宽高和旋转角度，便于 YOLO 旋转目标检测模型的训练。

脚本支持的可选行为：

- `--hbb`：直接导出 axis-aligned HBB（angle=0），适合只支持 HBB 的训练框架。
- `--tile --tile-size <size> --tile-overlap <overlap>`：按指定 tile 大小与重叠切分原图，tile 中的目标实例会按交集重建为 OBB 能兼容原始格式。假如某个目标只有一部分落在 tile 中，脚本会计算交集 / 原始面积的比率，当比率低于 `--min-area-ratio`（默认 0.3）时，自动将该实例标记为难例，以便训练时排除或加权。
- `--min-area-pixels`：控制最小可保留面积，例如默认 16 代表显著裁剪的 tiny 蛋糕将被忽略。

切分流程依赖 `shapely` 进行多边形交集与最小旋转矩形拟合，请确保在环境中安装：

```bash
pip install shapely
```

