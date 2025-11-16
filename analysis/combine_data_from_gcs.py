import pandas as pd
import gcsfs

# 设置你的 GCS 路径（前提是这些文件是公开或你有权限访问）
gcs_path = 'gs://dasnet/MBARI_F8_STU_GL20m_OCP5m_FS200Hz_2023-10-27T174910Z.h5.csv'

# 创建 GCS 文件系统对象
fs = gcsfs.GCSFileSystem(project='quakeflow-385822')

# 读取 CSV
with fs.open(gcs_path, 'r') as f:
    df = pd.read_csv(f)

print(df.head())
