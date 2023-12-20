from torchvision.datasets import CIFAR10
import pandas as pd
import numpy as np
import os

# Download CIFAR10 dataset
cifar10 = CIFAR10(root='/scratch/ssm10076/pytorch-example/data', train=True, download=True)

# Convert dataset to DataFrame
data = []
for image, label in cifar10:
    image_array = np.array(image).flatten()  # Convert PIL Image to NumPy array and flatten
    data.append([*image_array, label])

columns = [f'pixel_{i}' for i in range(3072)] + ['label_col']  # 3072 = 32*32*3
df = pd.DataFrame(data, columns=columns)

# Split DataFrame into 5 parts and save as parquet files
num_files = 5
chunk_size = len(df) // num_files

for i in range(num_files):
    start_index = i * chunk_size
    end_index = None if i == num_files - 1 else (i + 1) * chunk_size
    df_chunk = df.iloc[start_index:end_index]
    file_name = f'cifar10_part_{i + 1}.parquet'
    df_chunk.to_parquet(file_name)

