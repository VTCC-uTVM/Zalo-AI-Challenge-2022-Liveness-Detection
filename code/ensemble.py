import pandas as pd
import numpy as np
import math
model_1_df =  pd.read_csv("Predict_ensemble_5_acc_11_3TTA.csv")
model_2_df =  pd.read_csv("Predict_ensemble_acc_11_weight_3TTA.csv")


model_1_dict = {}
model_2_dict = {}
model_ensemble_fname = []
model_ensemble_liveness_score = []

for index, row in model_1_df.iterrows():
    model_1_dict[row['fname']] = row['liveness_score']

for index, row in model_2_df.iterrows():
    model_2_dict[row['fname']] = row['liveness_score']

for index, row in model_1_df.iterrows():
    diff = abs(model_1_dict[row['fname']] - model_2_dict[row['fname']])
    if diff > 0.05:
        print(row['fname'], model_1_dict[row['fname']], model_2_dict[row['fname']])
    model_ensemble_fname.append(row['fname'])
    model_ensemble_liveness_score.append((model_1_dict[row['fname']] + model_2_dict[row['fname']])/2)

model_ensemble_fname = np.array(model_ensemble_fname)
model_ensemble_liveness_score = np.array(model_ensemble_liveness_score)
model_ensemble_fname = np.expand_dims(model_ensemble_fname, 1)
model_ensemble_liveness_score = np.expand_dims(model_ensemble_liveness_score, 1)
print(model_ensemble_fname.shape, model_ensemble_liveness_score.shape)
df = pd.DataFrame(np.concatenate((model_ensemble_fname, model_ensemble_liveness_score), axis=1), columns=["fname", "liveness_score"])
df.to_csv('tmp_4.csv', index=False)
    