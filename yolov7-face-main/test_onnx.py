import os
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm
import torch
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="./yolov5s6_pose_640_ti_lite_54p9_82p2.onnx")
parser.add_argument("--img-path", type=str, default="./sample_ips.txt")
parser.add_argument("--dst-path", type=str, default="./sample_ops_onnxrt")
args = parser.parse_args()


def read_img(img_file, img_mean=127.5, img_scale=1/127.5):
    ori_img = cv2.imread(img_file)
    img = ori_img[:, :, ::-1]
    ori_shape = img.shape
    img = cv2.resize(img, (960,960), interpolation=cv2.INTER_LINEAR)
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img, ori_shape, ori_img


def model_inference(model_path=None, input=None):
    #onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input})
    
    return output[0]


def model_inference_image_list(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    os.makedirs(args.dst_path, exist_ok=True)
    img_file_list = list(open(img_path))
    pbar = enumerate(img_file_list)
    max_index = 20
    pbar = tqdm(pbar, total=min(len(img_file_list), max_index))
    for img_index, img_file  in pbar:
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        img_file = img_file.rstrip()
        input, ori_shape, ori_img = read_img(img_file, mean, scale)

        output = model_inference(model_path, input)
        output = torch.tensor(output)
        pred = non_max_suppression(output, 0.5, 0.4, classes=None, agnostic=False, kpt_label=5)

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                scale_coords(torch.Size([960, 960]), det[:, :4], ori_shape, kpt_label=False)
                scale_coords(torch.Size([960, 960]), det[:, 6:], ori_shape, kpt_label=5, step=3)
                det = det.detach().cpu().numpy()
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    kpts = det[det_index, 6:]
                    print(xyxy, ori_img.shape, ori_shape)
                    ori_img = cv2.resize(ori_img, (960, 960))

                    crop_img = ori_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    cv2.imwrite("result{}.png".format(det_index), crop_img)
                    # plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])


def main():
    model_inference_image_list(model_path="face_detection.onnx", img_path="image_paths.txt",
                               mean=0.0, scale=0.00392156862745098,
                               dst_path="result")

if __name__== "__main__":
    main()