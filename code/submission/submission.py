import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import cv2
import numpy as np
from model import Net
from dataset import ImageFolder
from torch.utils.data import DataLoader
import torch
import time
import shutil
import pandas as pd
from utils import seed_torch
import json
from model import Net, ArcFaceLossAdaptiveMargin, ArcNet

seed_torch()

exp = "exp_50"
pub = "pub_2"

f = open(os.path.join('code/configs', "{}.json".format(exp)))
default_configs = json.load(f)
default_configs["use_TTA"] = True
default_configs["n_eval"] = 11
# default_configs["image_size"] = 288
# model_paths = ["weights/09-11-2022_17:46:28/0/checkpoint_roc_auc_best.pt", "weights/09-11-2022_19:13:37/1/checkpoint_roc_auc_best.pt", \
#                 "weights/09-11-2022_20:38:12/2/checkpoint_roc_auc_best.pt", "weights/09-11-2022_22:05:27/3/checkpoint_roc_auc_best.pt", \
#                 "weights/09-11-2022_23:28:57/4/checkpoint_roc_auc_best.pt"]


metric = "acc"
train_type = "_ema"
# model_paths = ["weights/10-11-2022_17:32:22/0/checkpoint_{}_best{}.pt".format(metric, train_type), "weights/10-11-2022_19:02:27/1/checkpoint_{}_best{}.pt".format(metric, train_type), \
#                 "weights/10-11-2022_20:23:56/2/checkpoint_{}_best{}.pt".format(metric, train_type), "weights/10-11-2022_21:45:52/3/checkpoint_{}_best{}.pt".format(metric, train_type), \
#                 "weights/10-11-2022_23:05:56/4/checkpoint_{}_best{}.pt".format(metric, train_type)
# ]

# model_paths = ["weights/11-11-2022_16:57:21/0/checkpoint_{}_best{}.pt".format(metric, train_type), "weights/11-11-2022_18:33:53/1/checkpoint_{}_best{}.pt".format(metric, train_type), \
#                 "weights/11-11-2022_20:09:02/2/checkpoint_{}_best{}.pt".format(metric, train_type), "weights/11-11-2022_21:48:18/3/checkpoint_{}_best{}.pt".format(metric, train_type), \
#                 "weights/11-11-2022_23:26:48/4/checkpoint_{}_best{}.pt".format(metric, train_type)
# ]

# model_paths = ["weights/{}/0/checkpoint_{}_best{}.pt".format(exp, metric, train_type), "weights/{}/1/checkpoint_{}_best{}.pt".format(exp, metric, train_type), \
#                 "weights/{}/2/checkpoint_{}_best{}.pt".format(exp, metric, train_type, "weights/{}/3/checkpoint_{}_best{}.pt".format(exp, metric, train_type))
# ]

model_paths = ["weights/{}/0/checkpoint_{}_best{}.pt".format(exp, metric, train_type), "weights/{}/1/checkpoint_{}_best{}.pt".format(exp, metric, train_type), \
                "weights/{}/2/checkpoint_{}_best{}.pt".format(exp, metric, train_type), "weights/{}/3/checkpoint_{}_best{}.pt".format(exp, metric, train_type), \
                "weights/{}/4/checkpoint_{}_best{}.pt".format(exp, metric, train_type)
]
if pub == "pub_1":
    val_df =  pd.read_csv("code/data/public_test.csv")
elif pub == "pub_2":
    val_df =  pd.read_csv("code/data/public_test_2.csv")
DATA_PATH = "public_test"
test_data = ImageFolder(val_df, DATA_PATH, default_configs, None, "submission")
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, pin_memory=False, num_workers=8)

final_scores = np.zeros((len(val_df), 1))

# now = datetime.now()
for fold in range(len(model_paths)):
    model_path = model_paths[fold]
    # weight_path = os.path.join("submission_weights", now)
    # os.makedirs(weight_path, exist_ok=True)
    # new_model_path = os.path.join(weight_path, fold)
    # os.makedirs(new_model_path, exist_ok=True)
    # new_model_path = os.path.join(new_model_path, "checkpoint_{}_best{}.pt".format(exp, metric, train_type))
    # os.makedirs(new_model_path, exist_ok=True)
    # shutil.copyfile(model_path, new_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if default_configs["use_arcface"]:
        model = ArcNet(default_configs["backbone"], default_configs["n_frames"])
    else:    
        model = Net(default_configs["backbone"], default_configs["n_frames"])
    model.to(device)
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)
    model.eval()
    
    predictions = []
    video_id_list = []

    with torch.no_grad():
        for batch_idx, (imgs, labels, img_paths) in enumerate(test_loader):  
            # print("BATCH ID: ", batch_idx) 
            imgs = imgs.to(device).float()
            labels = labels.to(device).long()
            average_scores = torch.zeros((imgs.shape[0],)).cuda()
            if default_configs["use_TTA"]:
                n_eval = 3*default_configs["n_eval"]
            else:
                n_eval = default_configs["n_eval"]
            for k in range(n_eval):
                eval_imgs = imgs[:,k*3:(k+1)*3,:,:]
                logits = model(eval_imgs)
                probs = torch.nn.Softmax(dim=1)(logits)
                _, preds = torch.max(probs, 1)
                # print(preds)
                scores = probs[:,1]
                average_scores += scores
            
            predictions += [average_scores/n_eval]
            for j in range(len(img_paths)):
                video_id = os.path.basename(img_paths[j]) + ".mp4"
                video_id_list.append(video_id)

    predictions = torch.cat(predictions).cpu().numpy()
    predictions = np.expand_dims(predictions, 1)
    final_scores += predictions
    video_ids = np.array(video_id_list)

final_scores /= len(model_paths)
pseudo_preds = []
pseudo_videos = []
for i in range(final_scores.shape[0]):
    if final_scores[i][0] <= 0.7 and final_scores[i][0] >= 0.3:
        print(video_ids[i], final_scores[i])
    if final_scores[i][0] <= 0.1:
        video_path = os.path.join("public_test/videos", video_ids[i].replace(".mp4", ""))
        pseudo_videos.append(video_path)
        pseudo_preds.append(0)
    if final_scores[i][0] > 0.9:
        video_path = os.path.join("public_test/videos", video_ids[i].replace(".mp4", ""))
        pseudo_videos.append(video_path)
        pseudo_preds.append(1)
    # predictions.append()
video_ids = np.expand_dims(video_ids, 1)
print(final_scores.shape, video_ids.shape)
df = pd.DataFrame(np.concatenate((video_ids, final_scores), axis=1), columns=["fname", "liveness_score"])
if default_configs["use_TTA"]:
    use_TTA = "3TTA"
else:
    use_TTA = "noTTA"
csv_path = 'Predict_{}_{}{}_{}_{}_{}_{}.csv'.format(exp, metric, train_type, pub, default_configs['n_eval'], default_configs["image_size"], use_TTA)
print("Save file to ", csv_path)
df.to_csv(csv_path, index=False)

# pseudo_preds = np.array(pseudo_preds)
# pseudo_videos = np.array(pseudo_videos)
# pseudo_preds = np.expand_dims(pseudo_preds, 1)
# pseudo_videos = np.expand_dims(pseudo_videos, 1)
# print(pseudo_preds.shape, pseudo_videos.shape)
# df = pd.DataFrame(np.concatenate((pseudo_videos, pseudo_preds), axis=1), columns=["Video_path", "Label"])
# df.to_csv('code/pseudo_label.csv', index=False)