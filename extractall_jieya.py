import os
import tarfile
import re
import sys

def extract_and_rename_tar_files():
    # 获取当前目录
    current_dir = os.getcwd()
    
    # 目标文件夹名称
    target_folder = "UR001_bag"
    
    # 创建目标文件夹
    target_path = os.path.join(current_dir, target_folder)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print(f"创建文件夹: {target_folder}")
    else:
        print(f"文件夹 {target_folder} 已存在，将继续使用")
    
    # 递归搜索所有子目录中以 '_s' 开头的 .tar 文件
    tar_files = []
    target_path_abs = os.path.abspath(target_path)
    for root, dirs, files in os.walk(current_dir):
        # 跳过目标文件夹本身（使用绝对路径比较）
        root_abs = os.path.abspath(root)
        if root_abs == target_path_abs or root_abs.startswith(target_path_abs + os.sep):
            continue
        
        for filename in files:
            # 匹配文件名中包含 '_s' 的 .tar 文件（如：茶包放入茶杯_s100d77c61ee4382bc52e7ee3d3d985f.tar）
            if '_s' in filename and filename.endswith('.tar'):
                full_path = os.path.join(root, filename)
                tar_files.append(full_path)
    
    # 对文件进行排序，按照文件名中的数字部分排序
    def extract_number(filepath):
        # 提取文件名中的数字部分（在 's' 后面的第一个数字）
        filename = os.path.basename(filepath)
        match = re.search(r'_s(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    tar_files.sort(key=extract_number)
    
    # 限制最多处理200个文件
    if len(tar_files) > 200:
        tar_files = tar_files[:200]
        print(f"发现超过200个tar文件，只处理前200个")
    
    # 如果没有找到tar文件
    if not tar_files:
        print("未找到包含'_s'的.tar文件")
        return
    
    print(f"找到 {len(tar_files)} 个.tar文件需要处理")
    
    # 解压并重命名
    successful_count = 0
    error_count = 0
    
    for i, tar_file_path in enumerate(tar_files, 1):
        try:
            # 构建目标文件夹路径
            target_subfolder = os.path.join(target_path, str(i))
            
            # 创建目标子文件夹
            os.makedirs(target_subfolder, exist_ok=True)
            
            # 打开并解压tar文件
            tar_filename = os.path.basename(tar_file_path)
            print(f"正在解压: {tar_filename} -> {i}")
            
            with tarfile.open(tar_file_path, 'r') as tar:
                # 解压所有文件到目标子文件夹
                tar.extractall(path=target_subfolder, filter='data')
            
            successful_count += 1
            print(f"  成功解压到: {target_subfolder}")
            
        except Exception as e:
            error_count += 1
            print(f"  解压 {tar_file_path} 时出错: {str(e)}")
    
    # 输出总结
    print(f"\n处理完成!")
    print(f"成功解压: {successful_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"解压后的文件保存在: {target_path}")

if __name__ == "__main__":
    print("开始批量解压.tar文件...")
    print("=" * 50)
    extract_and_rename_tar_files()
