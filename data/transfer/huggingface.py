import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi


def main(transfer_path, target_path):
    api = HfApi()
    api.upload_large_folder(folder_path=transfer_path, repo_id=target_path, repo_type="dataset", num_workers =64)


if __name__ == "__main__":
    transfer_path = "/home/CONNECT/yfang870/yunhengwang/WorldVLN/dataset/cache/train_packed"
    target_path = "YunhengWang/WorldVLN_v_0.2"
    main(transfer_path, target_path)