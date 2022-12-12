# Prepare dataset
# Training dataset mount to /code/train
# Public test 2 dataset mount to /code/public_test_2
# Public test 1 dataset mount to /code/public_test

cd /code

# Split videos to 2 folders fake and real
python split_data.py 

# Extract all even frames from videos
python extract_frames.py --config extract_frames_train_config.json
python extract_frames.py --config extract_frames_pub1_config.json
python extract_frames.py --config extract_frames_pub2_config.json

# Split k-fold
python split_kfold.py

# Detect face
cd /code/yolov7-face-main
python detect.py

# Training
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_39 > nohup_exp_39.out &
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_40 > nohup_exp_40.out &
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_43 > nohup_exp_43.out &
CUDA_VISIBLE_DEVICES=0 nohup python code/train_teacher.py --exp exp_49 > nohup_exp_49.out &

# Submission
# Gen submission csv file
python code/submission.py

# Ensemble
python code/ensemble_multimodel.py

# Visualize grad-cam
python code/visualize_grad_cam.py 

![framework](code/results/0/grad_cam_0_38.jpg)

