from torch.autograd import Variable
import torch
import numpy as np
import time
import torch.nn.functional as F
from sklearn import metrics
import os
import shutil
# from plot_confusion_matrix import cm_analysis
import mlflow
import pandas as pd


#Get the threshold under several fpr
def get_thresholdtable_from_fpr(scores,labels, fpr_list):
    """Calculate the threshold score list from the FPR list
    Args:
      score_target: list of (score,label)
    Returns:
      threshold_list: list, the element is threshold score calculated by the
      corresponding fpr
    """
    threshold_list = []
    live_scores = []
    for score, label in zip(scores,labels):
        if label == 0:
            live_scores.append(float(score))
    live_scores.sort(reverse=True)
    live_nums = len(live_scores)
    
    for fpr in fpr_list:
        i_sample = int(fpr * live_nums)
        i_sample = max(1, i_sample)
        threshold_list.append(live_scores[i_sample - 1])
    print("Score list: ", live_scores[:i_sample - 1])
    return threshold_list

#Get the threshold under thresholds
def get_tpr_from_threshold(scores,labels, threshold_list):
    """Calculate the recall score list from the threshold score list.
    Args:
      score_target: list of (score,label)
      threshold_list: list, the threshold list
    Returns:
      recall_list: list, the element is recall score calculated by the
                   correspond threshold
    """
    tpr_list = []
    hack_scores = []
    for score, label in zip(scores,labels):
        if label == 1:
            hack_scores.append(float(score))
    hack_scores.sort(reverse=True)
    hack_nums = len(hack_scores)
    for threshold in threshold_list:
        hack_index = 0
        while hack_index < hack_nums:
            if hack_scores[hack_index] <= threshold:
                break
            else:
                hack_index += 1
        if hack_nums != 0:
            tpr = hack_index * 1.0 / hack_nums
        else:
            tpr = 0
        tpr_list.append(tpr)
    return tpr_list

counting = 0
def eval(val_loader, model, criterion, device, n_eval, fold, epoch, is_ema, default_configs):
    if is_ema:
        print("EMA EVAL")
    else:
        print("NORMAL EVAL")
    running_loss = 0
    k = 0
    model.eval()
    predictions = []
    thresh_predictions = []
    ground_truths = []
    scores = []
    n_images = 0
    n_correct = 0
    img_path_list = []
    val_metric = {"loss": 0, "acc": 0, "roc_auc": 0, "tpr_fpr_5e-4": 0, "tpr_fpr_1e-4": 0, "eer": 0}
    if default_configs["use_TTA"]:
        n_eval *= 2
    with torch.no_grad():
        for batch_idx, (imgs, labels, img_paths) in enumerate(val_loader):   
            imgs = imgs.to(device).float()
            labels = labels.to(device).long()

            average_probs = torch.zeros((imgs.shape[0], 2)).cuda()
            for k in range(n_eval):
                eval_imgs = imgs[:,k*3:(k+1)*3,:,:]
                if default_configs["use_arcface"] == True:
                    logits = model(eval_imgs)*default_configs["arcface_s"]
                else:
                    logits = model(eval_imgs)
                probs = torch.nn.Softmax(dim=1)(logits)
                average_probs += probs
            
            average_probs /= n_eval
            loss = 0
            running_loss += loss * imgs.size(0)

            k = k + imgs.size(0)

            _, preds = torch.max(average_probs, 1)
            predictions += [preds]

            scores += [average_probs[:,1]]
            ground_truths += [labels.detach().cpu()]
            n_images += len(labels)
            for j in range(len(img_paths)):
                img_path_list.append(img_paths[j])
                

        predictions = torch.cat(predictions).cpu().numpy()
        scores = torch.cat(scores).cpu().numpy()
        # print("Eval score mean: ", np.mean(scores))
        ground_truths = torch.cat(ground_truths).cpu().numpy()
        cm = metrics.confusion_matrix(ground_truths, predictions)
        acc = metrics.accuracy_score(ground_truths, predictions)
        roc_auc = metrics.roc_auc_score(ground_truths, scores)

        fpr, tpr, threshold = metrics.roc_curve(ground_truths, scores, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        val_metric['eer'] = EER
        
        
        fpr_list = [0.01, 0.005, 0.001]
        threshold_list = get_thresholdtable_from_fpr(scores, ground_truths, fpr_list)
        print(threshold_list)
        thresh = threshold_list[2]
        for i in range(scores.shape[0]):
            if scores[i] > thresh:
                thresh_predictions.append(1)
            else:
                thresh_predictions.append(0)
        cm_thresh = metrics.confusion_matrix(ground_truths, thresh_predictions)        
        tpr_list = get_tpr_from_threshold(scores, ground_truths, threshold_list)
        
        # Show the result into score_path/score.txt  
        print("EER: ", EER)
        print('TPR@FPR=10E-3: {}\n'.format(tpr_list[0]))
        print('TPR@FPR=5E-3: {}\n'.format(tpr_list[1]))
        print('TPR@FPR=10E-4: {}\n'.format(tpr_list[2]))
        print('Acc: ', acc)
        print("confusion matrix: ", cm)
        print("thresh confusion matrix: ", cm_thresh)
        print("ROC AUC: ", roc_auc)

    val_metric['loss'] = running_loss/k
    val_metric['acc'] = acc
    val_metric['tpr_fpr_5e-4'] = tpr_list[1]
    val_metric['tpr_fpr_1e-4'] = tpr_list[2]    
    val_metric['roc_auc'] = roc_auc   
         
    return val_metric, ground_truths, scores

