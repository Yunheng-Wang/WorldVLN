





# import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# from huggingface_hub import HfApi
# api = HfApi()

# api.upload_folder(
#     folder_path="/home/CONNECT/yfang870/yunhengwang/WorldVLN/dataset/cache/train",
#     repo_id="YunhengWang/WorldVLN_v_0.2",
#     repo_type="dataset",
#     multi_commits=True,  # 关键：由于文件太多，自动拆分成多次 commit
#     multi_commits_verbose=True
# )