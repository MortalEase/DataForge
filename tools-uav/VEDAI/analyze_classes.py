import glob
import os
from collections import Counter

annotation_dir = os.path.expanduser('~/datasets/VEDAI/Annotations1024')
files = glob.glob(os.path.join(annotation_dir, '*.txt'))

class_counts = Counter()

for file_path in files:
    if 'classes.txt' in file_path or 'names.txt' in file_path or 'annotation1024.txt' in file_path or 'fold' in file_path:
        continue
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            
            # 格式分析:
            # 14列: Cx Cy Orient Class Contained Occluded X1 X2 X3 X4 Y1 Y2 Y3 Y4
            
            class_id = None
            if len(parts) == 14:
                try:
                    class_id = int(parts[3])
                except ValueError:
                    pass
            
            if class_id is not None:
                class_counts[class_id] += 1
                if class_id > 31:
                    # 依然打印异常值以防万一
                    pass

print("Class ID Counts:")
for class_id, count in sorted(class_counts.items()):
    print(f"ID {class_id}: {count}")
