import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torchvision
video_path = "/data/namnv78/daklac.mp4"
reader = torchvision.io.VideoReader(video_path, "video")
reader.seek(2.0)
while 1:
    frame = next(reader)
    print("hehe")