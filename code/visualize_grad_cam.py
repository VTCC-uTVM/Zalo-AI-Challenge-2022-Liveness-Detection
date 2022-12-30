import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import Net
import torch
import pandas as pd
from dataset import ImageFolder
from torch.utils.data import DataLoader
import cv2
import numpy as np

import json

exp = "exp_33"

f = open(os.path.join('configs', "{}.json".format(exp)))
default_configs = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net("convnext_base_in22ft1k", 1)
model.to(device)
pretrained_dict = torch.load("../weights/exp_33/0/checkpoint_acc_best_ema.pt")
# print(pretrained_dict.keys())
model.load_state_dict(pretrained_dict)
model.eval()

target_layers = [model.model.norm_pre]


# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.
targets = None
# targets = [ClassifierOutputTarget(1)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
val_df =  pd.read_csv("data/grad_cam.csv")
DATA_PATH = "grad_cam"
test_data = ImageFolder(val_df, DATA_PATH, default_configs, None, "submission")
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=False, num_workers=8)
predictions = []
img_path_list = []
k = 0
for batch_idx, (imgs, labels, img_paths) in enumerate(test_loader):  
    imgs = imgs.to(device).float() 
    eval_imgs = imgs[:,0:3,:,:]
    grayscale_cam = cam(input_tensor=eval_imgs, targets=targets)
    for i in range(imgs.shape[0]):
        img_path = os.path.join(img_paths[i], "frame_0.png")
        file_name = os.path.basename(img_paths[i])
        grayscale_cam_new = grayscale_cam[i, :]
        ori_rgb_img = cv2.imread(img_path, 1)
        ori_shape = ori_rgb_img.shape
        rgb_img = ori_rgb_img[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (default_configs["image_size"], default_configs["image_size"]))
        # ori_rgb_img = cv2.resize(ori_rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        cam_image = show_cam_on_image(rgb_img, grayscale_cam_new)
        cam_image = cv2.resize(cam_image, (ori_shape[1], ori_shape[0]))
        # print(ori_rgb_img.shape, cam_image.shape)
        cam_image = cv2.hconcat([ori_rgb_img, cam_image])
        print(labels[i].item())
        img_path = os.path.join('results', str(labels[i].item()))
        os.makedirs(img_path, exist_ok=True)
        img_path = os.path.join(img_path, 'grad_cam_{}_{}.jpg'.format(labels[i].item(), file_name)) 
        cv2.imwrite(img_path, cam_image)
        k += 1