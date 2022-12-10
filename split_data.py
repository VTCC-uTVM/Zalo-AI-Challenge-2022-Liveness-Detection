import pandas as pd
import os
import shutil


input_folder = "train/videos"
file_path = "train/label.csv"
df = pd.read_csv(file_path)

real = "videos/real"
fake = "videos/fake"
os.makedirs(real, exist_ok=True)
os.makedirs(fake, exist_ok=True)

for index in range(len(df)):
    row = df.iloc[index]
    video_name = row["fname"]
    label = row["liveness_score"]
    if label:
        shutil.copyfile(
            os.path.join(input_folder, video_name),
            os.path.join(real, video_name)
        )
    else:
        shutil.copyfile(
            os.path.join(input_folder, video_name),
            os.path.join(fake, video_name)
        ) 
 
