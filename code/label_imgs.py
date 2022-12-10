import os
os.environ["CUDA_VISIBLE_DEVICES"]='5'
import cv2
import numpy as np
from model import Net
from dataset import ImageFolder
from torch.utils.data import DataLoader
import torch
import time
import shutil
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net("convnext_tiny_in22ft1k")
model.to(device)
pretrained_dict = torch.load("weights/27-10-2022_09:26:07/0/checkpoint_acc_best_ema.pt")
model.load_state_dict(pretrained_dict)
model.eval()

val_df =  pd.read_csv("raw_dataset.csv")
DATA_PATH = "raw_dataset"
test_data = ImageFolder(val_df, DATA_PATH, 224, None, "test")
test_loader = DataLoader(test_data, batch_size=64, shuffle=False, pin_memory=False, num_workers=8)
predictions = []
img_path_list = []

with torch.no_grad():
    for batch_idx, (imgs, aug_imgs, labels, img_paths) in enumerate(test_loader):   
        imgs = imgs.to(device).float()
        labels = labels.to(device).long()
        logits = model(imgs)
        probs = torch.nn.Softmax(dim=1)(logits)
        _, preds = torch.max(probs, 1)
        predictions += [preds]
        for j in range(len(img_paths)):
            img_path_list.append(img_paths[j])

predictions = torch.cat(predictions).cpu().numpy()
for i in range(predictions.shape[0]):
    old_file_name = os.path.basename(img_path_list[i])
    new_file_path = os.path.join("pseudo_dataset", str(predictions[i]))
    os.makedirs(new_file_path, exist_ok=True)
    new_file_path = os.path.join(new_file_path, old_file_name)
    shutil.copyfile(img_path_list[i], new_file_path)