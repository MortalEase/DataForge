"""图片平均颜色计算工具

核心: 提供命令行与函数接口, 输入图片路径返回 RGB 平均颜色.
扩展: 兼容常见图像格式(BMP/PNG/JPEG)并进行简单错误日志输出.
默认: 使用 OpenCV 读取图像, 以 RGB 顺序返回三元组 (R,G,B) 的整数值.

使用示例:
    python average_color.py -i /path/to/image.png

"""

from __future__ import annotations

from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from utils.logging_utils import tee_stdout_stderr, log_info, log_error


def read_image_rgb(image_path: Path) -> np.ndarray:
    """读取图片并返回 RGB 格式的 numpy 数组.

    参数:
        image_path(Path): 图片文件路径.

    返回:
        np.ndarray: HxWx3 的 uint8 数组, 通道顺序为 R,G,B.
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        log_error(f"无法读取图片: {image_path}")
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    # OpenCV 使用 BGR, 转为 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def compute_average_color(image_path: Path) -> Tuple[int, int, int]:
    """计算图片的平均颜色, 以 (R, G, B) 三元组返回整数值.

    平均值使用像素值的算术平均, 最终四舍五入为整数.

    参数:
        image_path(Path): 图片文件路径.

    返回:
        tuple[int,int,int]: 平均颜色 (R, G, B), 值域在 0-255.
    """
    img = read_image_rgb(image_path)
    # 计算每通道平均值
    mean_per_channel = img.mean(axis=(0, 1))  # R, G, B 顺序
    # 四舍五入并转为 int
    mean_rounded = tuple(int(round(float(v))) for v in mean_per_channel)
    return mean_rounded


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """把 (R,G,B) 三元组转换为十六进制字符串, 格式为 "#RRGGBB".

    参数:
        rgb(tuple): 三个 0-255 的整数值, 顺序为 R,G,B.

    返回:
        str: 形如 "#1A2B3C" 的大写十六进制颜色字符串.
    """
    r, g, b = rgb
    # 确保在 0-255
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"#{r:02X}{g:02X}{b:02X}"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="计算图片的平均颜色 (返回 R,G,B).",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="输入图片路径", dest="input")
    return parser


def main() -> None:
    """命令行入口: 解析参数, 计算平均颜色并打印结果."""
    tee_stdout_stderr('logs', script_basename='average_color')
    parser = parse_args()
    args = parser.parse_args()
    img_path = Path(args.input)
    if not img_path.exists():
        log_error(f"输入路径不存在: {img_path}")
        raise FileNotFoundError(f"输入路径不存在: {img_path}")

    try:
        avg = compute_average_color(img_path)
        hex_color = rgb_to_hex(avg)
        log_info(f"平均颜色 (R,G,B): {avg} | HEX: {hex_color}")
    except Exception as exc:
        log_error(f"计算平均颜色失败: {exc}")
        raise


if __name__ == '__main__':
    main()
