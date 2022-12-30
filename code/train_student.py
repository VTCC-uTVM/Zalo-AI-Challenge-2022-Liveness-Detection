import os
# import optuna
# import wandb
import mlflow
# from losses.label_smoothing import LabelSmoothingCrossEntropy
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from data_augmentations.mixup import mixup, cutmix
from optimizer.sam import SAM
from optimizer.adan import Adan
from optimizer.ranger21.ranger21 import Ranger21
from datetime import datetime
from data_augmentations.grid import GridMask
import json
from sklearn import metrics
from kd.KD import DistillKL
# try:
#     from apex import amp
#     APEX_AVAILABLE = True
# except ModuleNotFoundError:
APEX_AVAILABLE = False
print(APEX_AVAILABLE)


# os.environ["CUDA_VISIBLE_DEVICES"]='3'

# importing the libraries
import pandas as pd
import numpy as np
import time
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
#matplotlib inline
from torch import nn
# for creating validation set
from torch.optim import lr_scheduler

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
from datetime import datetime

import logging 
import torchvision.transforms as transforms
from timm.utils import ModelEma
from model import Net, ArcFaceLossAdaptiveMargin, ArcNet

#import cv2

from torch.optim import Adam, SGD
from tqdm import tqdm

from utils import seed_torch, count_parameters
from dataset import ImageFolder, BatchSampler, RandomSampler, ImageTeacher
from torch.utils.data import DataLoader

from torch.autograd import Variable
import torch
from torchvision.transforms import Compose 

from eval import eval
import argparse
import gc
import torch.nn.utils.prune as prune
from timm.utils import get_state_dict


weights_path = [None, None, None, None, None]

default_configs = {}

from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def prune_model_global_unstructured(model, layer_type, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, layer_type):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=proportion
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model

def remove_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def lr_func(epoch):
    LR_WARMUP_EPOCHS = 3
    LR_MAX = default_configs["lr"]
    LR_START = 1e-5
    # LR_SUSTAIN_EPOCHS = 65
    # LR_STEP_DECAY = 0.8
    if epoch < LR_WARMUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_WARMUP_EPOCHS * epoch + LR_START
    else:
        lr = LR_MAX
    # else:
    #     lr = LR_MAX*pow(LR_STEP_DECAY, epoch - LR_WARMUP_EPOCHS - LR_SUSTAIN_EPOCHS)
    return lr

exp = "exp_43"
metric = "acc"
train_type = "_ema"

teacher_weight_path = ["weights/{}/0/checkpoint_{}_best{}.pt".format(exp, metric, train_type), "weights/{}/1/checkpoint_{}_best{}.pt".format(exp, metric, train_type), \
                "weights/{}/2/checkpoint_{}_best{}.pt".format(exp, metric, train_type), "weights/{}/3/checkpoint_{}_best{}.pt".format(exp, metric, train_type), \
                "weights/{}/4/checkpoint_{}_best{}.pt".format(exp, metric, train_type)
]


def train_one_fold(fold, train_loader, test_loader):
    print("FOLD: ", fold)
    grid = GridMask(96, 224, 360, 0.6, 1, 0.8)
    DATA_PATH = "train"
    now = datetime.now()
    weight_path = os.path.join("weights", default_configs["a_name"])
    os.makedirs(weight_path, exist_ok=True)
    weight_path = os.path.join(weight_path, str(fold))
    os.makedirs(weight_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start_epoch = 0
    if default_configs["use_arcface"]:
        criterion_cls = ArcFaceLossAdaptiveMargin(m=default_configs["arcface_m"], s=default_configs["arcface_s"])
    else:
        criterion_cls = LabelSmoothingCrossEntropy(default_configs["smoothing_value"])
    criterion_div = DistillKL(default_configs["kd_T"])
    criterion_kd = DistillKL(default_configs["kd_T"])

    criterion_cls.to(device)
    criterion_div.to(device)
    criterion_kd.to(device)
    
    if default_configs["use_arcface"]:
        model = ArcNet(default_configs["backbone"], default_configs["n_frames"])
    else:    
        # model = Net(default_configs["backbone"], default_configs["n_frames"])
        model_s = Net(default_configs["student_backbone"], default_configs["n_frames"])
        model_t = Net(default_configs["teacher_backbone"], default_configs["n_frames"])
#     model = make_model(4)
    # base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"])
    if default_configs["optimizer"] == "SAM":
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        optimizer_model = SAM(model.parameters(), base_optimizer, lr=default_configs["lr"], momentum=0.9, weight_decay=default_configs["weight_decay"], adaptive=True)
    # optimizer_model = torch.optim.AdamW(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"])
    elif default_configs["optimizer"] == "Ranger21":
        optimizer_model = Ranger21(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"], 
        num_epochs=default_configs["num_epoch"], num_batches_per_epoch=len(train_loader))
    elif default_configs["optimizer"] == "SGD":
        optimizer_model = torch.optim.SGD(model.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"], momentum=0.9)
    else:
        optimizer_model = Adan(model_s.parameters(), lr=default_configs["lr"], weight_decay=default_configs["weight_decay"])
    if default_configs["optimizer"] != "Ranger21":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_model, 5, T_mult=1, eta_min=1e-5, last_epoch=- 1, verbose=False)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=args.num_epochs)
#     count_parameters(model.src_embed)
    model_s.to(device)
    model_t.to(device)
    
    scaler = torch.cuda.amp.GradScaler()
    
    pretrained_dict = torch.load(teacher_weight_path[fold])
    model_t.load_state_dict(pretrained_dict)

    best_metric = {"loss": 10000, "acc": 0, "roc_auc": 0, "tpr_fpr_5e-4": 0, "tpr_fpr_1e-4": 0, "eer": 10000}
    best_metric_ema = {"loss": {"score": 10000, "list": []}, "acc": {"score": 0, "list": []}, "roc_auc": {"score": 0, "list": []}, \
        "tpr_fpr_5e-4": {"score": 0, "list": []}, "tpr_fpr_1e-4": {"score": 0, "list": []}, "eer": {"score": 10000, "list": []}}
    best_model_path = ""

    input_list, output_list = [], []
    iter_size = 1

    for epoch in range(start_epoch, default_configs["num_epoch"]):
        print("\n-----------------Epoch: " + str(epoch) + " -----------------")
        grid.set_prob(epoch, default_configs["num_epoch"])
        for param_group in optimizer_model.param_groups:
            # logger.debug("LR = {}".format(param_group['lr']))
            # param_group['lr'] = lr_func(epoch)
            mlflow.log_metric("lr", param_group['lr'], step=epoch)
            print("LR: ", param_group['lr'])
        if epoch == default_configs['start_ema_epoch']:
            print("Start ema......................................................")
            model_ema = ModelEma(
                model_s,
                decay=default_configs["model_ema_decay"],
                device=device, resume='')
        running_loss = 0
        j = 0
        start = time.time()
        n_images = 0
        n_correct = 0
        optimizer_model.zero_grad()
        for batch_idx, (teacher_imgs, student_imgs, labels, img_paths) in enumerate(tqdm(train_loader)):
            model_s.train()
            model_t.eval()
            teacher_imgs = teacher_imgs.to(device).float()
            student_imgs = student_imgs.to(device).float()
            labels = labels.to(device).long()
            # print(labels.shape)    

            if torch.rand(1)[0] < 0.5 and (default_configs["use_mixup"] or default_configs["use_cutmix"]):
                rand_prob = torch.rand(1)[0]

                if default_configs["use_mixup"] == True and default_configs["use_cutmix"] == False:
                    mix_images, target_a, target_b, lam = mixup(imgs, labels, alpha=default_configs["mixup_alpha"])
                if default_configs["use_mixup"] == False and default_configs["use_cutmix"] == True:
                    mix_images, target_a, target_b, lam = cutmix(imgs, labels, alpha=default_configs["mixup_alpha"])
                if default_configs["use_mixup"] == True and default_configs["use_cutmix"] == True: 
                    if rand_prob < 0.5:
                        mix_images, target_a, target_b, lam = mixup(imgs, labels, alpha=default_configs["mixup_alpha"])
                    else:
                        mix_images, target_a, target_b, lam = cutmix(imgs, labels, alpha=default_configs["mixup_alpha"])
                
                with torch.no_grad():
                    logits_t = model_t(mix_images)

                with torch.cuda.amp.autocast():
                    logits_s = model_s(mix_images)
                    loss_cls = criterion_cls(logits_s, target_a) * lam + \
                    (1 - lam) * criterion_cls(logits_s, target_b)
                    loss_div = criterion_div(logits_s, logits_t)
                    # loss = loss_cls
                    loss = default_configs["kd_gamma"] * loss_cls + default_configs["kd_alpha"] * loss_div
                    loss /= default_configs["accumulation_steps"]
                
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    # scaler.unscale_(optimizer_model)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), default_configs["max_norm"])
                    scaler.step(optimizer_model)
                    scaler.update()
                    if epoch >= default_configs['start_ema_epoch']:
                        model_ema.update(model)
                    optimizer_model.zero_grad()
            else:
                with torch.no_grad():
                    logits_t = model_t(teacher_imgs)
                with torch.cuda.amp.autocast():
                    # if default_configs["use_gridmask"] and torch.rand(1)[0] < 0.5:
                    #     imgs = grid(imgs)
                    logits_s = model_s(student_imgs)
                    loss_cls = criterion_cls(logits_s, labels)
                    loss_div = criterion_div(logits_s, logits_t)
                    loss = default_configs["kd_gamma"] * loss_cls + default_configs["kd_alpha"] * loss_div
                    loss /= default_configs["accumulation_steps"]
                scaler.scale(loss).backward()
                if ((batch_idx + 1) % default_configs["accumulation_steps"] == 0) or ((batch_idx + 1) == len(train_loader)):
                    # scaler.unscale_(optimizer_model)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), default_configs["max_norm"])
                    scaler.step(optimizer_model)
                    scaler.update()
                    if epoch >= default_configs['start_ema_epoch']:
                        model_ema.update(model_s)
                    optimizer_model.zero_grad()
        
            running_loss += loss.item() * student_imgs.size(0)
            j = j + student_imgs.size(0)
            n_images += len(labels)
            if default_configs["optimizer"] != "Ranger21":
                scheduler.step()

        # train_acc = n_correct/n_images
        end = time.time()
        mlflow.log_metric("train_loss_f{}".format(fold), running_loss/j, step=epoch)
        mlflow.log_metric("train_elapsed_time_f{}".format(fold), end - start, step=epoch)
        
        # start = time.time()
        # val_metric, ground_truths, scores = eval(test_loader, model, criterion_class, device, default_configs["n_eval"], fold, epoch, is_ema=False)
        # end = time.time()
        # mlflow.log_metric("val_elapsed_time_f{}".format(fold), end - start, step=epoch)

        val_metric_type_list = ["loss", "acc", "roc_auc", "tpr_fpr_5e-4", "tpr_fpr_1e-4", "eer"]
        # for val_metric_type in val_metric_type_list:
        #     print("Val {}: {}".format(val_metric_type, val_metric[val_metric_type]))
        #     mlflow.log_metric("val_{}".format(val_metric_type), val_metric[val_metric_type], step=epoch)
        #     flag = False
        #     if val_metric_type in ["loss", "eer"]:
        #         if(val_metric[val_metric_type] < best_metric[val_metric_type]):
        #             flag = True
        #     else:
        #         if(val_metric[val_metric_type] > best_metric[val_metric_type]):
        #             flag = True 
        #     if flag == True:
        #         best_model_path = os.path.join(weight_path, 'checkpoint_{}_best.pt'.format(val_metric_type))
        #         torch.save(model.state_dict(), best_model_path)
        #         mlflow.log_artifact(best_model_path)
        #         best_metric[val_metric_type] = val_metric[val_metric_type]
        # print("\n=============================================================================")
        if epoch >= default_configs['start_ema_epoch']:
            val_metric, ground_truths, scores = eval(test_loader, model_ema.ema, criterion_cls, device, default_configs["n_eval"], fold, epoch, True, default_configs)
            for val_metric_type in val_metric_type_list:
                print("Val ema {}: {}".format(val_metric_type, val_metric[val_metric_type]))
                mlflow.log_metric("val_{}_ema".format(val_metric_type), val_metric[val_metric_type], step=epoch)
                flag = False
                if val_metric_type in ["loss", "eer"]:
                    if(val_metric[val_metric_type] < best_metric_ema[val_metric_type]["score"]):
                        flag = True
                else:
                    if(val_metric[val_metric_type] > best_metric_ema[val_metric_type]["score"]):
                        flag = True 
                if flag == True:
                    best_model_path = os.path.join(weight_path, 'checkpoint_{}_best_ema.pt'.format(val_metric_type))
                    try:
                        os.remove(best_model_path)
                    except Exception as e:
                        print(e)
                    torch.save(get_state_dict(model_ema), best_model_path)
                    mlflow.log_artifact(best_model_path)
                    best_metric_ema[val_metric_type] = {"score": val_metric[val_metric_type], "list": [ground_truths, scores]}
                    # print("Save best model ema: ", best_model_path, val_metric[val_metric_type])
    
    # model = prune_model_global_unstructured(model, nn.Conv2d, 0.1)
    # test_loss, test_score = eval(val_loader, model, criterion_class, device, False, fold, 15)
    # print("After pruning, test score: ", test_score)
    # torch.save(model.state_dict(), "model_prune_{}.pt".format(fold))
    # mlflow.log_params({"pruned_{}".format(fold): test_score})
    mlflow.log_artifact(best_model_path)
    del model_s
    del model_t
    torch.cuda.empty_cache()
    gc.collect()

    return best_metric_ema


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--exp", type=str, default="exp_1")
    args = parser.parse_args()

    f = open(os.path.join('code/configs', "{}.json".format(args.exp)))
    default_configs = json.load(f)
    f.close()
    print(default_configs)

    seed_torch()
    DATA_PATH = "train"
    avg_score = {"loss": 0, "acc": 0, "roc_auc": 0, "eer": 0}
    n_fold = 5
    
    train_loader_list = {}
    test_loader_list = {}
    for fold in range(n_fold):
        if default_configs["pseudo_label"]:
            train_df = pd.read_csv("code/data/train_fold{}.csv".format(fold))
            pseudo_df = pd.read_csv("code/data/pseudo_label.csv")
            train_df = pd.concat([train_df, pseudo_df])
        else:
            train_df = pd.read_csv("code/data/train_fold{}.csv".format(fold))

        val_df =  pd.read_csv("code/data/val_fold{}.csv".format(fold))
        train_data_1 = ImageTeacher(train_df, DATA_PATH, default_configs, {default_configs["image_size"]: 9}, "train")
        test_data_1 = ImageFolder(val_df, DATA_PATH, default_configs, None, "submission")
        # test_data_2 = ImageFolder(val_df, DATA_PATH, default_configs["image_size"], 11, "test", None)
        if default_configs["batch_size"] <= 2:
            train_loader = DataLoader(train_data_1, batch_size=default_configs["batch_size"], shuffle=True, drop_last=True, pin_memory=True, num_workers=default_configs["num_workers"])
        else:
            train_loader = DataLoader(train_data_1, batch_size=default_configs["batch_size"], shuffle=True, drop_last=True, pin_memory=True, num_workers=default_configs["num_workers"])
        if default_configs["batch_size"] <= 2:
            test_loader = DataLoader(test_data_1, batch_size=8, shuffle=False, pin_memory=True, num_workers=default_configs["num_workers"])
        else:
            test_loader = DataLoader(test_data_1, batch_size=default_configs["batch_size"]*2, shuffle=False, pin_memory=True, num_workers=default_configs["num_workers"])
        train_loader_list[fold] = train_loader
        test_loader_list[fold] = test_loader
    
    # wandb.define_metric("fold")
    # wandb.define_metric("metric", step_metric="fold")
    avg_ground_truths = [] 
    avg_scores = []
    for fold in range(n_fold):
        with mlflow.start_run():
            mlflow.log_params({"Fold": fold})
            mlflow.log_params(default_configs)
            mlflow.log_artifacts("/data/azc/code") 
            # mlflow.set_tags({"TRIAL": trial.number})
            score = train_one_fold(fold, train_loader_list[fold], test_loader_list[fold]) 
            for k, v in avg_score.items():
                avg_score[k] += score[k]["score"]
                mlflow.log_metric("metric_{}".format(k), score[k]["score"])
            ground_truths, scores = score["eer"]["list"]
            avg_ground_truths = avg_ground_truths + ground_truths.tolist()
            avg_scores = avg_scores + scores.tolist()
            
            # mlflow.log_params({"CV": avg_score/n_fold})
            mlflow.end_run()
    
    for k, v in avg_score.items():
        print("CV_{}: ".format(k), avg_score[k]/n_fold)
    
    roc_auc = metrics.roc_auc_score(avg_ground_truths, avg_scores)
    fpr, tpr, threshold = metrics.roc_curve(avg_ground_truths, avg_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print("Correction CV_EER: ", EER)
    print("Correction CV_ROC_AUC: ", roc_auc)
    