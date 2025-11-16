import pandas as pd
import os
from tqdm import tqdm

# ======== 参数设置 ========
input_dir = '/nfs2/group/chun/img_missT/iter3model'   # 本地文件夹路径
output_csv = 'merged_output_missT.csv'
log_file = 'processed_files_missT.txt'

# ======== 获取所有 CSV 文件 ========
all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

# ======== 读取已处理文件列表 ========
processed_files = set()
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        processed_files = set(line.strip() for line in f.readlines())

write_header = not os.path.exists(output_csv)

# ======== 合并处理 ========
for file in tqdm(all_files):
    if file in processed_files:
        continue  # 跳过已处理文件

    try:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)

        df.to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False  # 后续不再写 header

        # 记录已处理文件
        with open(log_file, 'a') as f:
            f.write(file + '\n')

        print(f"Processed: {file}")

    except Exception as e:
        print(f"Failed to process {file}: {e}")
