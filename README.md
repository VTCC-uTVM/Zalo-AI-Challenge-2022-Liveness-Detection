# Prepare dataset
Training dataset mount to /code/train. Public test 2 dataset mount to /code/public_test_2. Public test 1 dataset mount to /code/public_test

Split videos to 2 folders fake and real
``
python split_data.py 
``

Extract all even frames from videos
```
python extract_frames.py --config extract_frames_train_config.json
python extract_frames.py --config extract_frames_pub1_config.json
python extract_frames.py --config extract_frames_pub2_config.json
```

Split k-fold
```
python split_kfold.py
```

Detect face
```
cd /code/yolov7-face-main
python detect.py
```

# Training
```
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_39 > nohup_exp_39.out &
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_40 > nohup_exp_40.out &
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_43 > nohup_exp_43.out &
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_49 > nohup_exp_49.out &
```
# Experiment result

|Exp|Pub 2| Private|
|--------|----|--------|
|Ensemble exp_39, 40, 43, 49, B5, 3TTA, 11 frames|0.00855|
|Ensemble exp_39, 40, 43, 49, 2TTA, 8 frames|0.019|
|Ensemble exp_39, 40, 43, 49, 2TTA, 8 frames, fp16|0.02137|0.06373|
|Ensemble exp_40, 43, 49, B5, 2TTA, 11 frames, fp32|...|0.05882|
|Exp_39, 3TTA, 7 frames, best eer|0.02778|
|Exp_40, 3TTA, 11 frames, best acc|0.02991|
|Exp_40, 3TTA, 11 frames, best eer|0.02381|
|Exp_43, 3TTA, 11 frames, best acc|0.02381|
|Exp_49, 3TTA, 11 frames, best acc|0.02778|
|B5, 3TTA, 11 frames, best acc|0.02564|

# Submission
Gen submission csv file
```
python code/submission/submission.py
```

Ensemble
```
python code/submission/ensemble_multimodel.py
```

Visualize grad-cam
```
python code/visualize_grad_cam.py 
```
![image](code/results/0/grad_cam_0_38.jpg?raw=true)
