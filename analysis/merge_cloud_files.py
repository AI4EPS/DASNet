import pandas as pd
import gcsfs
import os
from tqdm import tqdm


fs = gcsfs.GCSFileSystem(project='quakeflow-385822')
output_csv = 'merged_output-3.csv'
gcs_prefix = 'gs://dasnet-iter2/'

all_files = fs.glob(gcs_prefix + '*.csv')

processed_files = set()
if os.path.exists('processed_files-3.txt'):
    with open('processed_files-3.txt', 'r') as f:
        processed_files = set(line.strip() for line in f.readlines())

write_header = not os.path.exists(output_csv)

for file in tqdm(all_files):
    if file in processed_files:
        continue  # 跳过已处理文件

    try:
        with fs.open(file, 'r') as f:
            df = pd.read_csv(f)
            df['source_file'] = os.path.basename(file)

        df.to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False  # 后续不再写 header

        # 记录已处理文件
        with open('processed_files-3.txt', 'a') as f:
            f.write(file + '\n')

        print(f"Processed: {file}")

    except Exception as e:
        print(f"Failed to process {file}: {e}")