import os
import cv2
import numpy as np
import time
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train code")
    parser.add_argument("--config", type=str, default="extract_frames_train_config.json")
    args = parser.parse_args()

    f = open(args.config)
    configs = json.load(f)
    f.close()

    directory = configs["directory"]
    labels = configs["labels"]

    for label in labels:
        video_directory = os.path.join(directory, label)
        for root, dirs, files in os.walk(video_directory):
            for filename in files:
                video_path = os.path.join(root, filename)
                base_name = filename.replace(".mp4", "")
                cap = cv2.VideoCapture(video_path)

                # Check if camera opened successfully
                if (cap.isOpened()== False): 
                    print("Error opening video stream or file ", video_path)
                frame_count = 0
                count = 0
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        if frame_count % 2 == 0:
                            frame_path = os.path.join(configs["destination"], label)
                            frame_path = os.path.join(frame_path, base_name)
                            os.makedirs(frame_path, exist_ok=True)
                            frame_path = os.path.join(frame_path, "frame_" + str(frame_count) + ".png")
                            cv2.imwrite(frame_path, frame)
                            count += 1
                        frame_count += 1
                    else:
                        break
      
                cap.release()
 

