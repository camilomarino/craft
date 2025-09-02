"""  
Copyright (c) 2019-present NAVER Corp.parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_image', default='test_image.jpg', type=str, help='path to input image')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')

args = parser.parse_args()

# Get script directory for loading models, but save results in current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
result_folder = os.getcwd()  # Current working directory where script is executed

# Set model paths in the same directory as the script
trained_model_path = os.path.join(script_dir, 'craft_mlt_25k.pth')
refiner_model_path = os.path.join(script_dir, 'craft_refiner_CTW1500.pth')

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # Get original image dimensions
    original_height, original_width = image.shape[:2]

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # Convert individual heatmaps and resize to original size
    score_text_heatmap = imgproc.cvt2HeatmapImg(score_text)
    score_link_heatmap = imgproc.cvt2HeatmapImg(score_link)
    
    score_text_resized = cv2.resize(score_text_heatmap, (original_width, original_height))
    score_link_resized = cv2.resize(score_link_heatmap, (original_width, original_height))

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, score_text_resized, score_link_resized



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + trained_model_path + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model_path, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + refiner_model_path + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model_path)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model_path, map_location='cpu')))

        refine_net.eval()
        # args.poly = True

    t = time.time()

    # Process single image
    image_path = args.test_image
    print("Processing image: {}".format(os.path.basename(image_path)))
    
    image = imgproc.loadImage(image_path)
    
    # Measure inference time
    inference_start = time.time()
    bboxes, polys, score_text_heatmap, score_link_heatmap = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
    inference_time = time.time() - inference_start
    
    print("Inference time: {:.2f}s".format(inference_time))

    # Save individual heatmaps
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    
    text_heatmap_file = os.path.join(result_folder, "res_" + filename + '_1.png')
    link_heatmap_file = os.path.join(result_folder, "res_" + filename + '_2.png')
    
    cv2.imwrite(text_heatmap_file, score_text_heatmap)
    cv2.imwrite(link_heatmap_file, score_link_heatmap)
    
    print("Text heatmap saved: {}".format(text_heatmap_file))
    print("Link heatmap saved: {}".format(link_heatmap_file))

    # Save result with bboxes - ensure directory exists
    print(result_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
    
    # Check if the result image was created
    result_img_file = os.path.join(result_folder, "res_" + filename + '.png')
    if os.path.exists(result_img_file):
        print("Result image with bboxes saved: {}".format(result_img_file))
    else:
        print("Warning: Result image with bboxes was not created!")

    print("Results saved in: {}".format(result_folder))
