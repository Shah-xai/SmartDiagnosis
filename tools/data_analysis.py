import os
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
import math


def class_balance(data_dir: str):
  
    if not os.path.isdir(data_dir):
        raise ValueError(f"{data_dir} doesn't exist!")

    class_counts = {}

    # Count images per class
    for class_path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(class_path):
            continue
        class_name = os.path.basename(class_path)
        images = glob.glob(os.path.join(class_path, "*"))
        class_counts[class_name] = len(images)

    if not class_counts:
        print("No classes found.")
        return class_counts

    total = sum(class_counts.values())

    print("Train data class balance:")
    print("-" * 40)
    for cls_, count in class_counts.items():
        pct = 100.0 * count / total
        print(f"{cls_:10s}: {count:5d} ({pct:5.1f}%)")
    print("-" * 40)
    print(f"Total images: {total}")

    # CI = majority / minority
    max_c = max(class_counts.values())
    min_c = min(class_counts.values())
    CI = max_c / max(min_c, 1)
    print("CI score:", format(CI, ".3g"))
    if CI<1.5:
        print("The dataset is balanced!")
    elif CI<2.5:
        print("Acceptablly imbalance!")
    elif CI<5:
        print("Prone to bias in the final model!")
    else:
        print("Severely imbalance!")

    return class_counts
def show_random_images(data_dir: str, n_per_class: int = 3, seed: int = 42):
    """
    Show up to n_per_class random images from each class folder under data_dir.
  
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"{data_dir} doesn't exist!")

    random.seed(seed)

    # Find class folders
    class_paths = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*"))
        if os.path.isdir(p)
    )
    if not class_paths:
        raise ValueError(f"No class folders found in {data_dir}")

    # Sample images per class
    samples = {}  # class_name -> list of image paths
    for class_path in class_paths:
        class_name = os.path.basename(class_path)
        imgs = glob.glob(os.path.join(class_path, "*"))
        if not imgs:
            continue
        k = min(n_per_class, len(imgs))
        samples[class_name] = random.sample(imgs, k)

    if not samples:
        raise ValueError(f"No images found under {data_dir}")

    n_rows = len(samples)
    n_cols = n_per_class

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 3 * n_rows)
    )

    # axes can be 1D in some shapes; normalize to 2D
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (cls_, img_paths) in enumerate(samples.items()):
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(img_paths):
                img_path = img_paths[col_idx]
                img = Image.open(img_path)
                img_format=img.format
                img_size=img.size
                img_mode=img.mode
                img=img.resize((224,224))
                ax.imshow(img, cmap="gray")
                ax.set_title(
                    f"{cls_}",
                    fontsize=8
                )
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    print ( "Image format:", img_format)
    print("Image size:", img_size)
    print("Image_mode :" ,img_mode)
if __name__=="__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    train_dir = os.path.join(project_root, "dataset", "chest_xray", "train")
    class_balance(train_dir)
    show_random_images(train_dir, n_per_class=2,seed=0)