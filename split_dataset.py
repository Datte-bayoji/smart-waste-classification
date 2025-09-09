import os, shutil, random

SRC = 'dataset/original'   # source folder
DST = 'dataset'            # output dataset with train/val
VAL_RATIO = 0.2            # 20% images for validation

for cls in os.listdir(SRC):
    cls_path = os.path.join(SRC, cls)
    if not os.path.isdir(cls_path): 
        continue
    files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(files)
    n_val = int(len(files) * VAL_RATIO)
    val_files = files[:n_val]
    train_files = files[n_val:]

    # create train/val dirs
    train_dir = os.path.join(DST, 'train', cls)
    val_dir = os.path.join(DST, 'val', cls)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # copy files
    for f in train_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(val_dir, f))

    print(f"{cls}: train={len(train_files)}, val={len(val_files)}")

print("✅ Done splitting dataset.")
