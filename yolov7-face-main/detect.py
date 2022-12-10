import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def inference(model, imgs, paths, im0s_list, device, half, kpt_label, save_dir, class_label, videoname, opt):
    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs).to(device)
    imgs = imgs.half() if half else imgs[0].float()  # uint8 to fp16/32
    imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
    # if img.ndimension() == 3:
    #     img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    preds = model(imgs, augment=opt.augment)[0]
    preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
    for k in range(imgs.shape[0]):
        img = imgs[k]
        img = img.unsqueeze(0)
        path = paths[k]
        det = preds[k]

        im0s = im0s_list[k]
        p, s, im0 = path, '', im0s.copy()
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        # print("Hehe: ", img.shape[2:])
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # print("DEtect: ", len(det))
        if len(det):
            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
            scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)
            # Printmes[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            best_xyxy = 0
            best_C = 0
            for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                c = int(cls)  # integer class
                kpts = det[det_index, 6:]
                x_min = int(xyxy[0].item())
                y_min = int(xyxy[1].item())
                x_max = int(xyxy[2].item())
                y_max = int(xyxy[3].item())
                w = x_max - x_min
                h = y_max - y_min
                
                if w + h > best_C:
                    best_C = w + h
                    best_xyxy = (x_min, y_min, x_max, y_max)
            h_img = im0s.shape[0]
            w_img = im0s.shape[1]
            
            x_min, y_min, x_max, y_max = best_xyxy
            w = x_max - x_min
            h = y_max - y_min
            # x_min = max(x_min-w//8, 0)
            # x_max = min(x_max+w//8, w_img)
            # y_max = min(y_max + (y_max - y_min)//3, h_img)

            crop_img = im0s[y_min:y_max, x_min:x_max]
            new_video_path = os.path.join("/code/azc/train/frames", class_label)
            new_video_path = os.path.join(new_video_path, videoname)
            os.makedirs(new_video_path, exist_ok=True)
            filename = os.path.basename(path).replace(".png", ".txt")
            new_image_path = os.path.join(new_video_path, filename)
            f = open(new_image_path, "w")
            f.write("{} {} {} {}".format(x_min, y_min, x_max, y_max))
            f.close()
                        # cv2.imwrite(new_image_path, crop_img)

def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.eval()
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16


    for class_label in ["real", "fake"]:
        class_folder = os.path.join(source, class_label)
        for videoname in os.listdir(class_folder):
            # if videoname != "1667":
            #     continue
            video_path = os.path.join(class_folder, videoname)
            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(video_path, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(video_path, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()

            imgs = []; paths = []; im0s_list = []
            for path_t, img_t, im0s_t, vid_cap in dataset:
                imgs.append(img_t)
                paths.append(path_t)
                im0s_list.append(im0s_t)
                if len(imgs) == 32:
                    inference(model, imgs, paths, im0s_list, device, half, kpt_label, save_dir, class_label, videoname, opt)
                    imgs = []; paths = []; im0s_list = []
            if len(imgs) > 0:
                inference(model, imgs, paths, im0s_list, device, half, kpt_label, save_dir, class_label, videoname, opt)


    # print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-face.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/code/azc/train/frames', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs= '+', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', type=int, default=5, help='number of keypoints')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
