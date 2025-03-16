# preprocess.py
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
data_dir = 'data/images'
metadata_path = 'data/HAM10000_metadata.csv'
base_dir = 'base_dir'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')

# Create directories
os.makedirs(base_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create class folders
classes = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# Load metadata
df_data = pd.read_csv(metadata_path)

# Identify duplicates
df_data['duplicates'] = df_data['lesion_id'].duplicated(keep=False).map({True: 'has_duplicates', False: 'no_duplicates'})

# Split data into train and validation sets
df_train = df_data[df_data['duplicates'] == 'no_duplicates']
df_val, _ = train_test_split(df_train, test_size=0.17, random_state=101, stratify=df_train['dx'])

# Copy images to respective folders
for _, row in df_train.iterrows():
    src = os.path.join(data_dir, row['image_id'] + '.jpg')
    dst = os.path.join(train_dir, row['dx'], row['image_id'] + '.jpg')
    shutil.copyfile(src, dst)

for _, row in df_val.iterrows():
    src = os.path.join(data_dir, row['image_id'] + '.jpg')
    dst = os.path.join(val_dir, row['dx'], row['image_id'] + '.jpg')
    shutil.copyfile(src, dst)

print("Preprocessing completed!")