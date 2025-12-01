# ChestX-ray8 数据集

> **作者/来源**：Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers
> **机构**：Department of Radiology and Imaging Sciences; National Center for Biotechnology Information; National Institutes of Health (NIH)
> **发布时间**：2017
> **数据集地址**：https://nihcc.app.box.com/v/ChestXray-NIHCC
> **任务类型**：多标签胸部疾病分类与弱监督病灶定位

---

## 一、简介

ChestX-ray8（又称 CXR8）是由美国国立卫生研究院（NIH）发布的大规模胸部 X 光影像数据库，包含超过十万张影像，覆盖数万名患者。标签由自然语言处理（NLP）方法从放射科报告中提取，标注了包括肺不张（Atelectasis）、心脏肥大（Cardiomegaly）、积液（Effusion）等多种常见胸部异常。

该数据集以其大规模、多样性与弱监督标注特点著称，支持多标签分类、病灶定位（部分图片含边界框标注）以及影像-文本联合学习等研究方向。ChestX-ray8 为深度学习模型训练与评估提供了重要资源，广泛应用于自动化医疗诊断研究。
![数据集样例](./images/ChestX-ray8_example.png "数据集样例")
---

## 二、数据集元信息

| 维度 | 模态 | 任务类型 | 解剖结构 | 解剖区域 | 类别数 | 数据量 | 文件格式 |
|------|------|----------|----------|----------|--------|--------|----------|
| 2D | X-ray | 多标签分类 & 弱监督病灶定位 | 胸腔 | 胸部 | 8 类（常用摘要；实际标注包含 14 类/No Finding） | 108,948+ 张 | PNG / DICOM / CSV |

注：常见文献中也统计为 14 类（包括 Pneumonia、Nodule 等），此表以核心项目摘要为准，具体类别请查看标签文件。

### 图像尺寸统计

| 指标 | 值 |
|------|-----|
| 最大图像分辨率（示例） | 1024 × 1024 |
| 常见分辨率 | 1024×1024 或接近（某些原始为更高分辨率） |

> 注意：ChestX-ray8 的图片分辨率与来源多样，实际转换/训练前建议统一重采样或缩放到目标分辨率。

---

## 三、标签信息与统计文件结构

标注主要包含两类：

- 图片级标签（Data_Entry_2017_v2020.csv）：对每张图像给出一组疾病标签，多个标签以 `|` 分隔；含 `No Finding`（无异常）项。
- 边界框标注（BBox_List_2017.csv）：仅部分图片（约数千张）包含病灶的像素/框位置信息，用于定位评估。

仓库示例的文件夹结构（节选）：

```
...
├── PruneCXR/
│   ├── README.txt
│   ├── miccai2023_nih-cxr-lt_labels_train.csv
│   ├── miccai2023_nih-cxr-lt_labels_test.csv
│   └── miccai2023_nih-cxr-lt_labels_val.csv
├── LongTailCXR/
│   ├── README.txt
│   ├── nih-cxr-lt_single-label_balanced-test.csv
│   ├── nih-cxr-lt_single-label_train.csv
│   ├── nih-cxr-lt_image_ids.csv
│   ├── nih-cxr-lt_single-label_test.csv
│   └── nih-cxr-lt_single-label_balanced-val.csv
├── images/
│   ├── batch_download_zips.py
│   ├── images_001.tar.gz
│   ├── images_002.tar.gz
│   └── ...
├── docs files ..
├── metadata files ..
├── notebooks files
└── ...
```

---

## 四、作者与机构

- Xiaosong Wang: Department of Radiology and Imaging Sciences, Clinical Center, National Institutes of Health (NIH)
- Yifan Peng: National Center for Biotechnology Information, National Library of Medicine, National Institutes of Health (NIH)
- Le Lu: Department of Radiology and Imaging Sciences, Clinical Center, National Institutes of Health (NIH)
- Zhiyong Lu: National Center for Biotechnology Information, National Library of Medicine, National Institutes of Health (NIH)
- Mohammadhadi Bagheri: Department of Radiology and Imaging Sciences, Clinical Center, National Institutes of Health (NIH)
- Ronald M. Summers: Department of Radiology and Imaging Sciences, Clinical Center, National Institutes of Health (NIH)

---

## 五、来源与引用

- 官方网站/下载链接: https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
- 文章: Wang et al., "ChestX-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases", CVPR 2017

引用格式（BibTeX）:

```bibtex
@inproceedings{wang2017chestx,
  title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2097--2106},
  year={2017}
}
```