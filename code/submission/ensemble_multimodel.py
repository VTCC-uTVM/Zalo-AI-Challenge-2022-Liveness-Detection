import pandas as pd
import numpy as np
import math
import os
import cv2

metric = "acc"
n_eval = "11"

low_res_weight = [1/10, 3/10, 2/10, 1/10, 3/10]
high_res_weight = [3/10, 1/10, 2/10, 3/10, 1/10]
# Best model
# model_list = ["Predict_exp_43_{}_ema_pub_2_{}_384_3TTA.csv".format(metric, n_eval),
#                 "Predict_exp_40_{}_ema_pub_2_{}_288_3TTA.csv".format(metric, n_eval),
#                 "Predict_exp_39_{}_ema_pub_2_{}_448_3TTA.csv".format(metric, n_eval),
#                 "Predict_acc_ema_B5_3TTA.csv",
#                 "Predict_exp_49_{}_ema_pub_2_{}_224_3TTA.csv".format(metric, n_eval),
#                 ]


model_list = ["Predict_exp_43_{}_ema_pub_2_{}_384_3TTA.csv".format(metric, n_eval),
                "Predict_exp_40_{}_ema_pub_2_{}_288_3TTA.csv".format(metric, n_eval),
                "Predict_exp_39_{}_ema_pub_2_{}_448_3TTA.csv".format(metric, n_eval),
                "Predict_acc_ema_B5_3TTA.csv",
                "Predict_exp_48_{}_ema_pub_2_{}_224_3TTA.csv".format(metric, n_eval),
                ]
model_df_list = []

for i in range(len(model_list)):
    model_df_list.append(pd.read_csv(model_list[i]))

model_dict_list = []
model_ensemble_fname = []
model_ensemble_liveness_score = []
for i in range(len(model_list)):
    model_dict_list.append({})
    for index, row in model_df_list[i].iterrows():
        model_dict_list[i][row['fname']] = row['liveness_score']

statistic_dict = {}
model_ensemble_liveness_score_dict = {}

for index, row in model_df_list[0].iterrows():
    model_ensemble_fname.append(row['fname'])
    file_path = os.path.join("public_test_2_frames/videos", row['fname'].replace(".mp4", ""))
    file_path = os.path.join(file_path, "frame_0.png")
    # print(file_path)
    # img = cv2.imread(file_path)
    # h, w, _ = img.shape
    weight = [1/5, 1/5, 1/5, 1/5, 1/5]
    # if h > 700 and w > 375:
    #     weight = high_res_weight
    # else:
    #     weight = low_res_weight
    score = 0
    # print(weight)
    min_v = 10000
    max_v = 0
    mean_v = 0
    score_list = []
    for i in range(len(model_list)):
        score += weight[i]*model_dict_list[i][row['fname']]
        if model_dict_list[i][row['fname']] < min_v:
            min_v = model_dict_list[i][row['fname']]
        if model_dict_list[i][row['fname']] > max_v:
            max_v = model_dict_list[i][row['fname']]
        mean_v += model_dict_list[i][row['fname']]
        score_list.append(model_dict_list[i][row['fname']])
    mean_v /= len(model_list)
    statistic_dict[row['fname']] = {"min_v": min_v, "max_v": max_v, "mean_v": mean_v, "score_list": score_list}
    model_ensemble_liveness_score.append(score)
    model_ensemble_liveness_score_dict[row['fname']] = score
    # model_ensemble_liveness_score.append(score/len(model_list))

for k, v in statistic_dict.items():
    if v["max_v"] - v["min_v"] > 0.3:
        print(k, v, model_ensemble_liveness_score_dict[k])
model_ensemble_fname = np.array(model_ensemble_fname)
model_ensemble_liveness_score = np.array(model_ensemble_liveness_score)
model_ensemble_fname = np.expand_dims(model_ensemble_fname, 1)
model_ensemble_liveness_score = np.expand_dims(model_ensemble_liveness_score, 1)
print(model_ensemble_fname.shape, model_ensemble_liveness_score.shape)
df = pd.DataFrame(np.concatenate((model_ensemble_fname, model_ensemble_liveness_score), axis=1), columns=["fname", "liveness_score"])
df.to_csv('Predict_ensemble_{}_{}_{}_3TTA.csv'.format(len(model_list), metric, n_eval), index=False)
    