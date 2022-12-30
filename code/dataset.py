import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
# from data_augmentations.rand_augment import preprocess_input
from torch.utils.data import Sampler,RandomSampler,SequentialSampler
import random

count = 0

class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None, df_img_size = None, img_sizes = None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes
        self.df_img_size = df_img_size

    def __iter__(self):
        num_batch = 0
        batch = []
        size = self.df_img_size
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size        

class ImageFolder(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, data_path, default_configs, randaug_magnitude, mode):
        super().__init__()
        # df = df.sample(frac=0.01)
        df['video_path'] = df["Video_path"]
        self.df = df.reset_index(drop=True)
        self.labels = df["Label"].values
        self.data_path = data_path
        self.mode = mode
        self.randaug_magnitude = randaug_magnitude
        self.df_imgsize = default_configs["image_size"]
        self.n_frames = default_configs["n_frames"]
        self.n_eval = default_configs["n_eval"]
        self.n_mul_dataset = default_configs["n_mul_dataset"]
        self.using_crop_face = default_configs["crop_face"]
        self.p_cutout = 0.5 if default_configs["use_cutout"] else 0
        self.only_augment_real = default_configs["only_augment_real"]
        self.use_hsv = default_configs["use_hsv"]
        self.use_random_aug = default_configs["use_random_augment"]
        self.TTA = default_configs["use_TTA"]
        if default_configs["only_augment_real"] == False:
            if self.use_random_aug == True:
                train_aug_list = [
                    A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                    # A.RandomCrop(width=df_imgsize, height=df_imgsize, p=1),
                    A.CoarseDropout(max_holes=2, max_height=int(self.df_imgsize/4), max_width=int(self.df_imgsize/4), fill_value=127, p=self.p_cutout),
                    A.Flip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                    A.OneOf([
                        A.JpegCompression(),
                    ], p=0.2),
                    # A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5), # Code bi thay doi cho nay
                    # A.OneOf([
                    #     A.OpticalDistortion(p=0.3),
                    #     A.GridDistortion(p=.1),
                    #     A.IAAPiecewiseAffine(p=0.3),
                    # ], p=0.2),
                ]
            else:
                train_aug_list = [
                    A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                    # A.RandomCrop(width=df_imgsize, height=df_imgsize, p=1),
                    A.CoarseDropout(max_holes=2, max_height=int(self.df_imgsize/8), max_width=int(self.df_imgsize/8), fill_value=0, p=self.p_cutout),
                    A.Flip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    
                    # A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5), # Code bi thay doi cho nay
                    # A.OneOf([
                    #     A.OpticalDistortion(p=0.3),
                    #     A.GridDistortion(p=.1),
                    #     A.IAAPiecewiseAffine(p=0.3),
                    # ], p=0.2),
                ]
            
            if default_configs["use_random_resize_crop"] == True:
                train_aug_list.insert(0, A.RandomResizedCrop(width=self.df_imgsize, height=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            else:
                train_aug_list.insert(0, A.Resize(width=self.df_imgsize, interpolation=cv2.INTER_CUBIC, height=self.df_imgsize, p=1))
            
            self.train_transform = A.Compose(train_aug_list)
        else:
            real_aug_list = [
                A.CoarseDropout(max_holes=2, max_height=int(self.df_imgsize/8), max_width=int(self.df_imgsize/8), p=self.p_cutout),
                A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                ], p=0.5),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5)]
            
            if default_configs["use_random_resize_crop"] == True:
                real_aug_list.insert(0, A.RandomResizedCrop(width=self.df_imgsize, height=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            else:
                real_aug_list.insert(0, A.Resize(width=self.df_imgsize, height=self.df_imgsize, p=1))
            self.real_transform = A.Compose(real_aug_list)

            fake_aug_list = [
                A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CoarseDropout(max_holes=2, max_height=int(self.df_imgsize/8), max_width=int(self.df_imgsize/8), p=self.p_cutout),
                A.Flip(p=0.5),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5)
            ]
            if default_configs["use_random_resize_crop"] == True:
                fake_aug_list.insert(0, A.RandomResizedCrop(width=self.df_imgsize, height=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            else:
                fake_aug_list.insert(0, A.Resize(width=self.df_imgsize, height=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            self.fake_transform = A.Compose(fake_aug_list)
        self.test_transform = A.Compose([
            # A.CenterCrop(height=df_imgsize, width=df_imgsize)
            A.Resize(height=self.df_imgsize, width=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1)
        ]
        )
        self.i_batch = 0
        self.i_epoch = 0
        self.n_batch = (len(self.df)//default_configs["batch_size"] + 1)*self.n_mul_dataset
        if self.mode == "test":
            self.frame_list = []
            self.label_list = []
            for index, row in self.df.iterrows():
                video_path = row.video_path
                for root, dirs, files in os.walk(video_path):
                    for filename in files:
                        if filename.endswith(".png"):
                            file_path = os.path.join(root, filename)
                            self.frame_list.append(file_path)
                            self.label_list.append(row.Label)
        
    def __len__(self):
        if self.mode == "train":
            return len(self.df)*self.n_mul_dataset
        elif self.mode == "test":
            return len(self.label_list)
        elif self.mode == "submission":
            return len(self.df)

    def __getitem__(self, index):
        self.i_batch += 1
        
        if self.i_batch % self.n_batch == 0:
            self.i_batch = 0
            self.i_epoch += 1

        IMAGENET_DEFAULT_MEAN =[0.485, 0.456, 0.406]
        IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            # index, input_size = index
            index = index % len(self.df)
            row = self.df.loc[index]
            video_path = row.video_path
            label = self.labels[index]
        elif self.mode == 'test':
            # set the default image size here
            input_size = self.df_imgsize
            label = self.label_list[index]
        elif self.mode == 'submission':
            row = self.df.loc[index]
            video_path = row.video_path
            label = self.labels[index]
        
        frames = []
        if self.mode == 'train':
            file_paths = []
            for root, dirs, files in os.walk(video_path):
                for filename in files:
                    if filename.endswith(".png"):
                        file_path = os.path.join(root, filename)
                        file_paths.append(file_path)
            file_paths.sort()
            if(len(file_paths) == 0):
                print(video_path)
            frames_path = random.sample(file_paths, self.n_frames)
            n_files = len(file_paths)
            # frames_path = [file_paths[self.i_epoch%n_files]]
            frames_path.sort()
            for frame_path in frames_path:
                frame = cv2.imread(frame_path)
                if self.using_crop_face:
                    bbox_path = frame_path.replace(".png", ".txt")
                    try:
                        f = open(bbox_path, "rt")
                        text = f.read()
                        x_min, y_min, x_max, y_max = text.split(" ")
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                        f.close()
                        if np.random.rand() < 0.5:
                            frame[y_min:y_max, x_min:x_max] = 127
                            # cv2.imwrite("ttt.png", frame)
                    except Exception as e:
                        # print(e)
                        pass
                if self.only_augment_real == True:
                    if label == 1:
                        frame = self.real_transform(image=frame)["image"]
                    else:
                        frame = self.fake_transform(image=frame)["image"]
                else:
                    frame = self.train_transform(image=frame)["image"]
                    # if self.use_random_aug:
                    #     frame = preprocess_input(frame, self.n_frames, randaug_magnitude=9)
                if self.use_hsv == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame/255
                frame = (frame - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
                frames.append(frame)
        elif self.mode == 'test':
            video_path = self.frame_list[index]
            frame = cv2.imread(self.frame_list[index])
            frame = self.test_transform(image=frame)["image"]
            if self.use_hsv == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame/255
            frame = (frame - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
            frames.append(frame)
        elif self.mode == 'submission':
            file_paths = []
            for root, dirs, files in os.walk(video_path):
                for filename in files:
                    if filename.endswith(".png"):
                        file_path = os.path.join(root, filename)
                        file_paths.append(file_path)
            file_paths.sort()
            if(len(file_paths) == 0):
                print(video_path)
           
            n_files = len(file_paths)
            step = n_files //self.n_eval
            frames_path = []
            count = 0
            for i in range(0, n_files, step):
                frames_path.append(file_paths[i])
                count += 1
                if count >= self.n_eval:
                    break
                
            for frame_path in frames_path:
                frame = cv2.imread(frame_path)
                frame = self.test_transform(image=frame)["image"]
                if self.TTA:
                    h_frame = A.HorizontalFlip(p=1)(image=frame)["image"]
                    v_frame = A.VerticalFlip(p=1)(image=frame)["image"]

                if self.use_hsv == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                    if self.TTA:
                        h_frame = cv2.cvtColor(h_frame, cv2.COLOR_BGR2YCR_CB)
                        v_frame = cv2.cvtColor(v_frame, cv2.COLOR_BGR2YCR_CB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if self.TTA:
                        h_frame = cv2.cvtColor(h_frame, cv2.COLOR_BGR2RGB)
                        v_frame = cv2.cvtColor(v_frame, cv2.COLOR_BGR2RGB)
                
                frame = frame/255
                frame = (frame - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
                frames.append(frame)
                if self.TTA:
                    h_frame = h_frame/255
                    h_frame = (h_frame - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
                    frames.append(h_frame)

                    v_frame = v_frame/255
                    v_frame = (v_frame - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
                    frames.append(v_frame)
                    
        frames_np = np.concatenate(frames, axis=2)
        # print(frames_np.shape)
        frames_np_normalize = np.moveaxis(frames_np, 2, 0)
        # img = self.transforms(img)
        return frames_np_normalize, label, video_path

import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from rand_augment import preprocess_input
from torchvision.io import read_image
from torch.utils.data import Sampler,RandomSampler,SequentialSampler
import random

count = 0

class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None, df_img_size = None, img_sizes = None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes
        self.df_img_size = df_img_size

    def __iter__(self):
        num_batch = 0
        batch = []
        size = self.df_img_size
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size        

class ImageTeacher(data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, data_path, default_configs, randaug_magnitude, mode):
        super().__init__()
        # df = df.sample(frac=0.01)
        df['video_path'] = df["Video_path"]
        self.df = df.reset_index(drop=True)
        self.labels = df["Label"].values
        self.data_path = data_path
        self.mode = mode
        self.randaug_magnitude = randaug_magnitude
        self.df_imgsize = default_configs["image_size"]
        self.teacher_imgsize = default_configs["teacher_image_size"]
        self.n_frames = default_configs["n_frames"]
        self.n_eval = default_configs["n_eval"]
        self.n_mul_dataset = default_configs["n_mul_dataset"]
        self.using_crop_face = default_configs["crop_face"]
        self.p_cutout = 0.5 if default_configs["use_cutout"] else 0
        self.only_augment_real = default_configs["only_augment_real"]
        self.use_hsv = default_configs["use_hsv"]
        self.use_random_aug = default_configs["use_random_augment"]
        self.TTA = default_configs["use_TTA"]

        if default_configs["only_augment_real"] == False:
            if self.use_random_aug == True:
                train_aug_list = [
                    A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                    # A.RandomCrop(width=df_imgsize, height=df_imgsize, p=1),
                    A.CoarseDropout(max_holes=2, max_height=int(self.teacher_imgsize/4), max_width=int(self.teacher_imgsize/4), fill_value=127, p=self.p_cutout),
                    A.Flip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                    A.OneOf([
                        A.JpegCompression(),
                    ], p=0.2),
                    # A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5), # Code bi thay doi cho nay
                    # A.OneOf([
                    #     A.OpticalDistortion(p=0.3),
                    #     A.GridDistortion(p=.1),
                    #     A.IAAPiecewiseAffine(p=0.3),
                    # ], p=0.2),
                ]
            else:
                train_aug_list = [
                    A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                    # A.RandomCrop(width=df_imgsize, height=df_imgsize, p=1),
                    A.CoarseDropout(max_holes=2, max_height=int(self.teacher_imgsize/8), max_width=int(self.teacher_imgsize/8), fill_value=0, p=self.p_cutout),
                    A.Flip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                    
                    # A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5), # Code bi thay doi cho nay
                    # A.OneOf([
                    #     A.OpticalDistortion(p=0.3),
                    #     A.GridDistortion(p=.1),
                    #     A.IAAPiecewiseAffine(p=0.3),
                    # ], p=0.2),
                ]
            
            if default_configs["use_random_resize_crop"] == True:
                train_aug_list.insert(0, A.RandomResizedCrop(width=self.teacher_imgsize, height=self.teacher_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            else:
                train_aug_list.insert(0, A.Resize(width=self.teacher_imgsize, interpolation=cv2.INTER_CUBIC, height=self.teacher_imgsize, p=1))
            
            self.train_transform = A.Compose(train_aug_list)
        else:
            real_aug_list = [
                A.CoarseDropout(max_holes=2, max_height=int(self.teacher_imgsize/8), max_width=int(self.teacher_imgsize/8), p=self.p_cutout),
                A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                ], p=0.5),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5)]
            
            if default_configs["use_random_resize_crop"] == True:
                real_aug_list.insert(0, A.RandomResizedCrop(width=self.teacher_imgsize, height=self.teacher_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            else:
                real_aug_list.insert(0, A.Resize(width=self.teacher_imgsize, height=self.teacher_imgsize, p=1))
            self.real_transform = A.Compose(real_aug_list)

            fake_aug_list = [
                A.ShiftScaleRotate(rotate_limit=20, interpolation=cv2.INTER_CUBIC, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.CoarseDropout(max_holes=2, max_height=int(self.teacher_imgsize/8), max_width=int(self.teacher_imgsize/8), p=self.p_cutout),
                A.Flip(p=0.5),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5)
            ]
            if default_configs["use_random_resize_crop"] == True:
                fake_aug_list.insert(0, A.RandomResizedCrop(width=self.teacher_imgsize, height=self.teacher_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            else:
                fake_aug_list.insert(0, A.Resize(width=self.teacher_imgsize, height=self.teacher_imgsize, interpolation=cv2.INTER_CUBIC, p=1))
            self.fake_transform = A.Compose(fake_aug_list)
        self.test_transform = A.Compose([
            # A.CenterCrop(height=df_imgsize, width=df_imgsize)
            A.Resize(height=self.teacher_imgsize, width=self.teacher_imgsize, interpolation=cv2.INTER_CUBIC, p=1)
        ]
        )
        self.i_batch = 0
        self.i_epoch = 0
        self.n_batch = (len(self.df)//default_configs["batch_size"] + 1)*self.n_mul_dataset
        if self.mode == "test":
            self.frame_list = []
            self.label_list = []
            for index, row in self.df.iterrows():
                video_path = row.video_path
                for root, dirs, files in os.walk(video_path):
                    for filename in files:
                        if filename.endswith(".png"):
                            file_path = os.path.join(root, filename)
                            self.frame_list.append(file_path)
                            self.label_list.append(row.Label)
        
    def __len__(self):
        if self.mode == "train":
            return len(self.df)*self.n_mul_dataset
        elif self.mode == "test":
            return len(self.label_list)
        elif self.mode == "submission":
            return len(self.df)

    def __getitem__(self, index):
        self.i_batch += 1
        
        if self.i_batch % self.n_batch == 0:
            self.i_batch = 0
            self.i_epoch += 1

        IMAGENET_DEFAULT_MEAN =[0.485, 0.456, 0.406]
        IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
        if self.mode == 'train':
            # index, input_size = index
            index = index % len(self.df)
            row = self.df.loc[index]
            video_path = row.video_path
            label = self.labels[index]
        elif self.mode == 'test':
            # set the default image size here
            input_size = self.df_imgsize
            label = self.label_list[index]
        elif self.mode == 'submission':
            row = self.df.loc[index]
            video_path = row.video_path
            label = self.labels[index]
        
        teacher_frames = []
        student_frames = []
        if self.mode == 'train':
            file_paths = []
            for root, dirs, files in os.walk(video_path):
                for filename in files:
                    if filename.endswith(".png"):
                        file_path = os.path.join(root, filename)
                        file_paths.append(file_path)
            file_paths.sort()
            if(len(file_paths) == 0):
                print(video_path)
            frames_path = random.sample(file_paths, self.n_frames)
            n_files = len(file_paths)
            # frames_path = [file_paths[self.i_epoch%n_files]]
            frames_path.sort()
            for frame_path in frames_path:
                frame = cv2.imread(frame_path)
                if self.using_crop_face:
                    bbox_path = frame_path.replace(".png", ".txt")
                    try:
                        f = open(bbox_path, "rt")
                        text = f.read()
                        x_min, y_min, x_max, y_max = text.split(" ")
                        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                        f.close()
                        if np.random.rand() < 0.5:
                            frame[y_min:y_max, x_min:x_max] = 127
                            # cv2.imwrite("ttt.png", frame)
                    except Exception as e:
                        # print(e)
                        pass
                if self.only_augment_real == True:
                    if label == 1:
                        frame = self.real_transform(image=frame)["image"]
                    else:
                        frame = self.fake_transform(image=frame)["image"]
                else:
                    frame = self.train_transform(image=frame)["image"]
                    # if self.use_random_aug:
                    #     frame = preprocess_input(frame, self.n_frames, randaug_magnitude=9)
                if self.use_hsv == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                teacher_frame = frame
                student_frame = A.Resize(height=self.df_imgsize, width=self.df_imgsize, interpolation=cv2.INTER_CUBIC, p=1)(image=frame)["image"]
                teacher_frame = teacher_frame/255
                teacher_frame = (teacher_frame - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
                teacher_frames.append(teacher_frame)

                student_frame = student_frame/255
                student_frame = (student_frame - IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
                student_frames.append(student_frame)


        frames_teacher_np = np.concatenate(teacher_frames, axis=2)            
        frames_student_np = np.concatenate(student_frames, axis=2)
        # print(frames_np.shape)
        frames_teacher_np = np.moveaxis(frames_teacher_np, 2, 0)
        frames_student_np = np.moveaxis(frames_student_np, 2, 0)
        # img = self.transforms(img)
        return frames_teacher_np, frames_student_np, label, video_path

