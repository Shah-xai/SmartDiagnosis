import os
import glob
from typing import Optional
from pathlib import Path
from PIL import Image
import random
import shutil

# Functions used in the preparation phase
def image_resizer(root_path:str, split, size=(224,224)):
    img_dir=os.path.join(root_path,split)
    if not os.path.isdir(img_dir):
        raise ValueError(f"{img_dir} is not a directory")
    for current_dir,_,files in os.walk(img_dir):
        for file in files:
            file_path = os.path.join(current_dir,file)
            try:
                img=Image.open(file_path)
                img=img.resize(size)
                img.save(file_path)
            except Exception as e:
                   raise ValueError(f"failed to resize {file_path}: {e}")
    
def get_data_label(data_dir:str)->dict:
    class_names=sorted(d for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir,d)))
    return {cls_name:idx for idx,cls_name in enumerate(class_names)}

def create_lst_files(data_dir:str,
    split:str,
    lst_path: Optional[str]=None):
    """
    data_dir: path containing 'train' and 'val' folders
              e.g. 'dataset/chest_xray'
    split   : 'train' or 'val'
    lst_path: optional explicit output path; if None, uses
              '<root_dir>/<split>.lst'
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise ValueError(f"{split_dir} does not exist")
    class_to_label=get_data_label(split_dir)
    print(f"[{split}] class mapping:", class_to_label)
    if lst_path is None:
        lst_path=os.path.join(data_dir,f"{split}.lst")
    lst_lines=list()
    lst_idx=0
    for class_name,label in class_to_label.items():
        abs_class_dir=os.path.join(split_dir,class_name)
        img_paths=sorted(
            p for p in glob.glob(os.path.join(abs_class_dir,"*"))
            if os.path.isfile(p)
        )
        for img_path in img_paths:
            rel_img_path = os.path.join(class_name,os.path.basename(img_path))
            lst_lines.append(f"{lst_idx}\t{label}\t{rel_img_path}")
            lst_idx+=1
    with open(lst_path,"w", newline="\n") as f:
        for line in lst_lines:
            f.write(line + "\n")
    print(f"{split} data wrote {len(lst_lines)} entries to {lst_path}")

def train_test_split(train_dir:str,val_dir:str, val_ratio:float=0.05,seed:int=0)->None:
    train_dir = Path(train_dir)
    val_dir=Path(val_dir)
    if not train_dir.is_dir():
        raise ValueError(f"{train_dir} is not a directory")

    if not val_dir.is_dir():
        raise ValueError(f"{val_dir} is not a directory")

    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be between 0 and 1")
    random.seed(seed)

    # Loop through each class folder under train/
    for class_path in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        class_name = class_path.name

        # Make sure matching class folder exists under val/
        val_class_dir = val_dir / class_name

        # List all images under the class in train/
        images = [img for img in class_path.iterdir() if img.is_file()]

        if not images:
            print(f" No images found in train/{class_name}")
            continue

        random.shuffle(images)

        n_total = len(images)
        n_val = max(1, int(n_total * val_ratio))  # at least 1

        val_set = images[:n_val]
    
        # Move validation images into val/<class_name>
        for img in val_set:
            dest = val_class_dir / img.name
            shutil.move(str(img), str(dest))

        print(
            f"Class '{class_name}': total={n_total}, "
            f"moved to val={len(val_set)}, remaining in train={n_total - n_val}")
    

if __name__ == "__main__":
    # Adjust this if your path differs
    root_dir = "/home/sagemaker-user/X_ray_project/dataset/chest_xray"
    train_dir=os.path.join(root_dir,"train")
    val_dir=os.path.join(root_dir,"val")
    train_test_split(train_dir,val_dir)

    for split in ["train", "val","test"]:
        try:
            image_resizer(root_dir,split)
            create_lst_files(root_dir, split)
        except Exception as e:
            print(f"[ERROR] Failed to create .lst for split '{split}': {e}")
    
  