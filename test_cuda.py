import torch
print('CUDA版本:', torch.version.cuda)
print('Pytorch版本:', torch.__version__)
print('显卡是否可用:', '可用' if torch.cuda.is_available() else '不可用')
print('显卡数量:', torch.cuda.device_count())
print('是否支持BF16:', '支持' if torch.cuda.is_bf16_supported() else '不支持')
print('当前显卡型号:', torch.cuda.get_device_name(0))
print('当前显卡的CUDA算力:', torch.cuda.get_device_capability(0))
print('当前显卡的总显存:', torch.cuda.get_device_properties(0).total_memory/1024/1024/1024, 'GB')