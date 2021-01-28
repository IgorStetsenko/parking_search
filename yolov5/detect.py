import argparse
import time
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detection_function(frame, source="yolov5/66521.jpg", stop_detection=True, view_img='store_true', save_txt=True,
                       imgsz=640, save_img=False):
    """The main detection function"""
    box = []
    weights = "yolov5/yolov5s.pt"
    stop_detection = True
    print(type(source), "type____")
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    #      ('rtsp://', 'rtmp://', 'http://'))

    #dataset = LoadImages(source, img_size=imgsz)

    # Directories
    save_dir = Path(increment_path(Path("runs/detect") / 'exp', exist_ok='store_true'))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, device)  # load FP32 model\
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    img0 = frame
    img = letterbox(img0, new_shape=800)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    print(img.ndimension(), "++++")
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

        # Inference
        # t1 = time_synchronized()

        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None)
        # t2 = time_synchronized()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        # save_path = str(save_dir / p.name)
        # txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
        # s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format

                    points = plot_one_box(xyxy, img0, color=colors[int(cls)], line_thickness=3)
                    box.append(points)
    return box


#
#
#
#
#
#
#
#
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
# #     parser.add_argument('--source', type=str, default='66521.jpg', help='source')  # file/folder, 0 for webcam
# #     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
# #     parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
# #     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
# #     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# #     parser.add_argument('--view-img', action='store_true', help='display results')
# #     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
# #     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
# #     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
# #     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# #     parser.add_argument('--augment', action='store_true', help='augmented inference')
# #     parser.add_argument('--update', action='store_true', help='update all models')
# #     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
# #     parser.add_argument('--name', default='exp', help='save results to project/name')
# #     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
# #     opt = parser.parse_args()
# #     print(opt)
# #
# #     with torch.no_grad():
# #         if opt.update:  # update all models (to fix SourceChangeWarning)
# #             for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
# #                 detect()
# #                 strip_optimizer(opt.weights)
# #         else:
# #             detect()

from yolov5.utils.datasets import *


#
def initialize_and_load_model(device='', out='inference/output', half='store_true',
                              weights='yolov5/weights/yolov5s.pt'):
    # Initialize
    device = select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # Load model
    model = torch.load(weights, map_location=device)['model']
    model.to(device).eval()
    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    return model, half, device
#
# def detection_function(frame,model, device, half):
#
#     imgsz = 800
#     augment = False
#     conf_thres=0.4
#     iou_thres=0.5
#     classes = None
#     agnostic_nms = 'store_true'
#
#     img = letterbox(frame, new_shape=imgsz)[0]
#     im0s = frame
#
#     # Convert
#     img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).to(device)
#     img = img.half() if half else img.float()  # uint8 to fp16/32
#     img /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     pred = model(img, augment=augment)[0]
#     # Apply NMS
#     pred = pred.float()
#     pred = non_max_suppression(pred, conf_thres, iou_thres,
#                                fast=True, classes=classes, agnostic=agnostic_nms)
#
#     boxes_array = []
#     # Process detections
#     for i, det in enumerate(pred):  # detections per image
#         if det is not None and len(det):
#             # Rescale boxes from img_size to im0 size
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
#             for *xyxy, conf, cls in det:
#                 boxes_array.append(xyxy)
#     return np.array(boxes_array,dtype = int )
