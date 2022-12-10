import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn import model_selection as sk_model_selection
import numpy as np
import os
from PIL import Image
import pandas as pd
import hashlib
import cv2
import shutil
# import imagehash

hash_set = {}
def hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()

def find_classes(directory: str):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

directory = "train"
classes, class_to_idx = find_classes(directory)

print(classes, class_to_idx)
video_paths = []
labels = []
label_idx = 0
file_names = set()
duplicate_count = 0
for label in classes:
    class_directory = os.path.join(directory, label)
    for videoname in os.listdir(class_directory):
        try:
            f = os.path.join(class_directory, videoname)
            video_paths.append(f)
            labels.append(label_idx)
        except:
            continue
    label_idx += 1

video_paths = np.array(video_paths)
labels = np.array(labels)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(video_paths, labels)):
    x_train, y_train = video_paths[train_idx], labels[train_idx]
    x_train = np.expand_dims(x_train, 1)
    y_train = np.expand_dims(y_train, 1)
    print(x_train.shape)
    x_val, y_val = video_paths[valid_idx], labels[valid_idx]
    x_val = np.expand_dims(x_val, 1)
    y_val = np.expand_dims(y_val, 1)
    train_df = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=["Video_path", "Label"])
    train_df.to_csv('code/train_fold{}.csv'.format(n_fold), index=False)
    val_df = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=["Video_path", "Label"])
    val_df.to_csv('code/val_fold{}.csv'.format(n_fold), index=False) 
