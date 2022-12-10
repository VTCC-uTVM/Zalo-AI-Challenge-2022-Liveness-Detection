import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from sklearn import model_selection as sk_model_selection
import numpy as np
import os
from PIL import Image
import pandas as pd
import hashlib
import cv2
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
image_paths = []
labels = []
label_idx = 0
file_names = set()
duplicate_count = 0
for label in classes:
    class_directory = os.path.join(directory, label)
    for filename in os.listdir(class_directory):
        try:
            f = os.path.join(class_directory, filename)
            # img = Image.open(f)
            #img = img.convert('L')
            # h = hash(img)
            # h = imagehash.average_hash(img)
            # if h in hash_set.keys():
            #     print(f, hash_set[h])
            # else:
            #     hash_set[h] = f
            # print(os.path.basename(os.path.dirname(f)))
            # if filename in file_names:
            #     duplicate_count += 1
            # else:
            #     file_names.add(filename)
            # checking if it is a file
            # if os.path.isfile(f):
            img = cv2.imread(f)
            if img is not None:
                image_paths.append(f)
                labels.append(label_idx)
            else:
                os.remove(f)
        except:
            continue
    label_idx += 1
# print(duplicate_count)
image_paths = np.array(image_paths)
labels = np.array(labels)

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2022)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(image_paths, labels)):
    x_train, y_train = image_paths[train_idx], labels[train_idx]
    x_train = np.expand_dims(x_train, 1)
    y_train = np.expand_dims(y_train, 1)
    print(x_train.shape)
    x_val, y_val = image_paths[valid_idx], labels[valid_idx]
    x_val = np.expand_dims(x_val, 1)
    y_val = np.expand_dims(y_val, 1)
    train_df = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=["Image_path", "Label"])
    train_df.to_csv('train_fold{}.csv'.format(n_fold), index=False)
    val_df = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=["Image_path", "Label"])
    val_df.to_csv('val_fold{}.csv'.format(n_fold), index=False)

# train_df = pd.read_csv(f"duplicate_removed_train.csv")
# data = train_df.iloc[: , :13].to_numpy()
# labels = train_df["Pawpularity"].to_numpy()

# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)

# for n_fold, (train_idx, valid_idx) in enumerate(folds.split(data, labels)):
#     x_train, y_train = data[train_idx], labels[train_idx]
#     y_train = np.expand_dims(y_train, 1)
#     x_val, y_val = data[valid_idx], labels[valid_idx]
#     y_val = np.expand_dims(y_val, 1)
#     train_df = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=["Id", "Label"])
#     train_df.to_csv('train_fold_dup{}.csv'.format(n_fold), index=False)
#     val_df = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=["Id", "Label"])
#     val_df.to_csv('val_fold_dup{}.csv'.format(n_fold), index=False)

# X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=0.33, random_state=42)