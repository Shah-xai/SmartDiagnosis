import boto3
import os
from PIL import Image


def s3_upload_file(local_path: str,
                   bucket_name: str,
                   s3_key: str):
    """ 
    Upload one file to an existing S3 bucket.
    """
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket_name, s3_key)


def s3_upload_folder(local_folder: str,
                     bucket_name: str,
                     s3_prefix: str):
    """ 
    Resize and upload all files in a local folder (recursively) to S3 bucket.
    """
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)

            # relative path inside the folder, e.g. NORMAL/n1.jpeg
            rel_path = os.path.relpath(local_path, local_folder)

            # full S3 key, e.g. chest-xray/train/NORMAL/n1.jpeg
            s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            s3_upload_file(local_path, bucket_name, s3_key)

    print("All files uploaded successfully from", local_folder)


def upload_dataset(root_dir: str,
                   bucket_name: str,
                   base_prefix: str = "chest-xray"):
    """
    Uploads:
      - train/ images
      - val/ images
      - train.lst
      - val.lst

    to an existing S3 bucket.

    Local layout (root_dir):
        train/
        val/
        train.lst
        val.lst

    S3 layout:
        s3://bucket_name/base_prefix/train/...
        s3://bucket_name/base_prefix/validation/...
        s3://bucket_name/base_prefix/train_lst/train.lst
        s3://bucket_name/base_prefix/validation_lst/val.lst
    """
    # Build local paths
    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "val")
    train_lst = os.path.join(root_dir, "train.lst")
    val_lst   = os.path.join(root_dir, "val.lst")

    # (Optional) basic safety checks
    for path in [train_dir, val_dir, train_lst, val_lst]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path not found: {path}")

    # 1) Upload train images
    s3_upload_folder(
        local_folder=train_dir,
        bucket_name=bucket_name,
        s3_prefix=os.path.join(base_prefix, "train")
    )

    # 2) Upload val images
    s3_upload_folder(
        local_folder=val_dir,
        bucket_name=bucket_name,
        s3_prefix=os.path.join(base_prefix, "validation")
    )

    # 3) Upload train.lst
    s3_upload_file(
        local_path=train_lst,
        bucket_name=bucket_name,
        s3_key=os.path.join(base_prefix, "train_lst", "train.lst").replace("\\", "/")
    )

    # 4) Upload val.lst
    s3_upload_file(
        local_path=val_lst,
        bucket_name=bucket_name,
        s3_key=os.path.join(base_prefix, "validation_lst", "val.lst").replace("\\", "/")
    )

    print("All chest X-ray data and .lst files uploaded.")

# The entire dataset will be deleted after uploading into s3 to save money
import os
import shutil

def delete_path(path: str):
    """Delete a file or folder safely."""
    if not os.path.exists(path):
        print(f"Path does not exist, skipping: {path}")
        return

    if os.path.isfile(path):
        os.remove(path)
        print(f"Deleted file: {path}")

    elif os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Deleted folder: {path}")
def cleanup_dataset(root_dir: str):
    """
    Delete:
      train/
      val/
      train.lst
      val.lst
    after successful upload.
    """
    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "val")
    train_lst = os.path.join(root_dir, "train.lst")
    val_lst   = os.path.join(root_dir, "val.lst")

    print("\nStarting cleanup...")

    delete_path(train_dir)
    delete_path(val_dir)
    delete_path(train_lst)
    delete_path(val_lst)

    print("Cleanup completed.\n")

if __name__ == "__main__":

    print("\n=== DEBUG MODE: Testing uploader functions ===\n")

    #  MODIFY THESE FOR YOUR DEBUGGING
    test_root_dir = "/home/sagemaker-user/X_ray_project/dataset/chest_xray"
    test_bucket   = "sagemaker-us-east-1-155576114785"
    test_prefix   = "debug-upload"

    print(f"Debug root_dir: {test_root_dir}")
    print(f"Debug bucket:   {test_bucket}")
    print(f"Debug prefix:   {test_prefix}")

    #  TEST only ONE class folder to avoid long uploads
    sample_folder = os.path.join(test_root_dir, "train", "NORMAL")

    if os.path.exists(sample_folder):
        print("\nUploading SAMPLE folder only for testing...\n")
        s3_upload_folder(sample_folder, test_bucket, test_prefix + "/sample-normal")
    else:
        print(f"Sample folder not found: {sample_folder}")

    print("\n=== DEBUG COMPLETE ===\n")