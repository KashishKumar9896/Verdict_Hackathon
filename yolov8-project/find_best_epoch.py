# This script extracts the epoch number from the best.pt checkpoint filename in the weights directory
# and prints the corresponding epoch number. It assumes the best.pt is a copy of epochXX.pt or was saved at a certain epoch.
# If not, it will print a warning and default to epoch 0.

import os
import torch
from ultralytics import YOLO

weights_dir = 'runs/detect/final_long_training/weights'
best_ckpt = os.path.join(weights_dir, 'best.pt')

# Try to find which epoch best.pt corresponds to by comparing file hashes
import hashlib
def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

best_hash = file_hash(best_ckpt)

# Find matching epoch file
best_epoch = None
for fname in os.listdir(weights_dir):
    if fname.startswith('epoch') and fname.endswith('.pt'):
        epoch_path = os.path.join(weights_dir, fname)
        if file_hash(epoch_path) == best_hash:
            best_epoch = int(fname.replace('epoch','').replace('.pt',''))
            break

if best_epoch is not None:
    print(f"BEST CHECKPOINT IS FROM EPOCH: {best_epoch}")
else:
    print("WARNING: Could not determine best.pt epoch. Defaulting to 0.")
    best_epoch = 0

# Print the path for use in training
print(f"BEST_CKPT_PATH={best_ckpt}")
print(f"BEST_EPOCH={best_epoch}")
