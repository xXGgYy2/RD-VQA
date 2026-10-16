import random
from pathlib import Path

# 文件路径
root = Path("/mnt/data/local-disk2/xyy/datasets/FineVD/FineVD")
src = root / "finevd_overall_labels.txt"
f_train = root / "finevd_train.txt"
f_val = root / "finevd_val.txt"
f_test = root / "finevd_test.txt"

random.seed(42)

# 读取非空行
all_lines = []
with open(src, "r", encoding="utf-8") as fp:
    for line in fp:
        l = line.strip()
        if l:
            all_lines.append(l)

random.shuffle(all_lines)
n_total = len(all_lines)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
n_test = n_total - n_train - n_val

train = all_lines[:n_train]
val = all_lines[n_train:n_train+n_val]
test = all_lines[n_train+n_val:]

with open(f_train, "w", encoding="utf-8") as fp:
    fp.write("\n".join(train))
with open(f_val, "w", encoding="utf-8") as fp:
    fp.write("\n".join(val))
with open(f_test, "w", encoding="utf-8") as fp:
    fp.write("\n".join(test))

print(f"Total: {n_total}")
print(f"Train: {len(train)} saved to {f_train}")
print(f"Val:   {len(val)} saved to {f_val}")
print(f"Test:  {len(test)} saved to {f_test}")
