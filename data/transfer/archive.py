import os
import tarfile
from tqdm import tqdm


def main(source_dir, output_dir, batch_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("正在读取文件夹列表...")
    all_subfolders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
    all_subfolders.sort() # 排序确保顺序一致
    total_folders = len(all_subfolders)
    print(f"找到 {total_folders} 个文件夹，计划分为 { (total_folders + batch_size - 1) // batch_size } 组")

    for i in tqdm(range(0, total_folders, batch_size)):
        batch = all_subfolders[i : i + batch_size]
        batch_num = i // batch_size
        tar_filename = os.path.join(output_dir, f"train_{batch_num:04d}.tar")
        if os.path.exists(tar_filename):
            continue
        with tarfile.open(tar_filename, "w") as tar:
            for folder in batch:
                folder_path = os.path.join(source_dir, folder)
                tar.add(folder_path, arcname=folder)

    print(f"\n打包完成！打包后的文件存放在: {output_dir}")


if __name__ == "__main__":
    source_dir = "/home/CONNECT/yfang870/yunhengwang/WorldVLN/dataset/cache/train"
    output_dir = "/home/CONNECT/yfang870/yunhengwang/WorldVLN/dataset/cache/train_packed"
    batch_size = 1000
    main(source_dir, output_dir, batch_size)
