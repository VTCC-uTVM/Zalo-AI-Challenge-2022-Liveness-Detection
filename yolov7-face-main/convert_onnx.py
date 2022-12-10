import os 
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import argparse
import sys
import time

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.add_nms import RegisterNMS

import torch
import torch.nn as nn

import onnx
import onnxsim

device = torch.device("cpu")
print(device)
model = attempt_load("yolov7-w6-face.pt", map_location=device)

model.eval()
labels = model.names

# Checks
gs = int(max(model.stride))  # grid size (max stride)
# opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

# Input
img = torch.zeros(1, 3, 960, 960)  # image size(1,3,320,192) iDetection
# img = img.type(torch.uint8)
# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv):  # assign export-friendly activations
        if isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()
        elif isinstance(m.act, nn.SiLU):
            m.act = SiLU()
model.model[-1].export = False
# for i in range(1000):
#     t1 = time.time()
# model = prune(model)
model.model[-1].include_nms = False
model.eval()
model.to(device)
y = model(img)  # dry run
# print(y)
    # print(time.time() - t1)

# ONNX export
print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
f = 'face_detection.onnx'  # filename

output_names = ['output']
torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
    output_names=output_names,          # the ONNX version to export the model to
    export_params=True,
    do_constant_folding=True,
    dynamic_axes = {
                    'images': {
                    0: 'batch_size',},
                    'output': {
                    0: 'batch_size',},
    })

onnx_model = onnx.load(f)
print("Generate simply onnx model")
simplified_onnx_model, success = onnxsim.simplify(onnx_model)
# assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
simplified_onnx_model_path =  f'face_detection.simplified.onnx'

print(f'Generating {simplified_onnx_model_path} ...')
onnx.save(simplified_onnx_model, simplified_onnx_model_path)
print('done')

# print('Registering NMS plugin for ONNX...')
# mo = RegisterNMS(simplified_onnx_model_path)
# mo.register_nms()
# mo.save(simplified_onnx_model_path)


# import onnxmltools
# from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxconverter_common.float16 import convert_float_to_float16
# from onnxconverter_common import auto_mixed_precision
# from onnxmltools.utils.float16_converter import convert_float_to_float16

input_onnx_model = 'face_detection.simplified.onnx'
output_onnx_model = 'face_detection.fp16.simplified.onnx'

# onnx_model = onnxmltools.utils.load_model(input_onnx_model)
# fp16_onnx_model = convert_float_to_float16(onnx_model,
#                              keep_io_types=True)
onnx_model = onnx.load(input_onnx_model)
# def validate(res1, res2):
#     for r1, r2 in zip(res1, res2):
#         if not np.allclose(r1, r2, rtol=0.01, atol=0.001):
#             return False
#     return True
# model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(model, img, validate, keep_io_types=True)
model_fp16 = convert_float_to_float16(onnx_model, min_positive_val=1e-7, max_finite_val=1e3,
                             keep_io_types=True)
onnx.save(model_fp16, output_onnx_model)
