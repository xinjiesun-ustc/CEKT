import os
import shutil
from tqdm import  tqdm

# 指定起始目录
start_dir = r'F:\教育数据集\Project_CodeNet\Project_CodeNet\data'
# 指定桌面路径，这里以Windows为例，需要根据实际情况调整
desktop_path = os.path.join(os.path.expanduser("~"), r"F:\教育数据集\C\C")

def copy_py_files(src_dir, dst_dir):
    """
    从源目录拷贝所有.c文件到目标目录
    """
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.c'):
                shutil.copy(os.path.join(root, file), dst_dir)

for root, dirs, files in tqdm(os.walk(start_dir)):
    # 获取当前目录的名称
    current_dir_name = os.path.basename(root)
    # 如果当前目录名大于"p02534"，则跳过
    if current_dir_name > 'p02534':
        continue
    if 'C' in dirs:
        python_dir = os.path.join(root, 'C')
        # 获取包含Python文件夹的上级目录的名称
        parent_dir_name = os.path.basename(root)
        # 在桌面上为这个上级目录创建（或使用已存在的）目标文件夹
        target_dir = os.path.join(desktop_path, parent_dir_name)
        os.makedirs(target_dir, exist_ok=True)
        # 拷贝文件
        copy_py_files(python_dir, target_dir)
        # 一旦找到并处理了python文件夹，就从dirs中移除，避免重复遍历
        dirs.remove('C')

print("拷贝完成")
