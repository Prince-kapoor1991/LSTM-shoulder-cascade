import numpy as np
import tensorflow as tf
import json
import os
import cv2 as cv
import random
import itertools

from data_utils import annotation_to_h5
from scipy.misc import imread, imresize
import annolist as al
from rect import Rect

def load_idl_tf(train_dir):
    image_height=480
    image_width=640
    grid_height=15
    grid_width=20
    rnn_len=1
    annotation=al.parse(train_dir)
    annos=[]
    for anno in annotation:
        anno.imageName = os.path.join(
            os.path.dirname(os.path.realpath(train_dir)), anno.imageName)
        annos.append(anno)
    random.seed(0)
    for epoch in itertools.count():
        random.shuffle(annos)
        for anno in annos:
            I = imread(anno.imageName)
	    #Skip Greyscale images
            if len(I.shape) < 3:
                continue
            if I.shape[2] == 4:
                I = I[:, :, :3]
            boxes, flags = annotation_to_h5(anno,
                                            grid_width,
                                            grid_height,
                                            rnn_len)

            yield {"image": I, "boxes": boxes, "flags": flags}
def make_sparse(n, d):
    v = np.zeros((d,), dtype=np.float32)
    v[n] = 1.
    return v
def load_data_gen(train_dir):
    grid_size=15*20
    data = load_idl_tf(train_dir)

    for d in data:
        output = {}

        rnn_len = 1
        flags = d['flags'][0, :, 0, 0:rnn_len, 0]
        boxes = np.transpose(d['boxes'][0, :, :, 0:rnn_len, 0], (0, 2, 1))
        assert(flags.shape == (grid_size, rnn_len))
        assert(boxes.shape == (grid_size, rnn_len, 4))

        output['image'] = d['image']
        output['confs'] = np.array([[make_sparse(int(detection), d=2) for detection in cell] for cell in flags])
        output['boxes'] = boxes
        output['flags'] = flags

        yield output

def add_rectangles(orig_image, confidences, boxes, use_stitching=False, rnn_len=1, min_conf=0.1, show_removed=True, tau=0.25, show_suppressed=True):
    grid_width=20
    grid_height=15
    num_classes=2
    region_size=32
    image = np.copy(orig_image[0])
    num_cells = grid_height * grid_width
    boxes_r = np.reshape(boxes, (-1,
                                 grid_height,
                                 grid_width,
                                 rnn_len,
                                 4))
    confidences_r = np.reshape(confidences, (-1,
                                             grid_height,
                                             grid_width,
                                             rnn_len,
                                             num_classes))
    cell_pix_size = region_size
    all_rects = [[[] for _ in range(grid_width)] for _ in range(grid_height)]
    for n in range(rnn_len):
        for y in range(grid_height):
            for x in range(grid_width):
                bbox = boxes_r[0, y, x, n, :]
                abs_cx = int(bbox[0]) + cell_pix_size/2 + cell_pix_size * x
                abs_cy = int(bbox[1]) + cell_pix_size/2 + cell_pix_size * y
                w = bbox[2]
                h = bbox[3]
                conf = np.max(confidences_r[0, y, x, n, 1:])
                all_rects[y][x].append(Rect(abs_cx,abs_cy,w,h,conf))

    all_rects_r = [r for row in all_rects for cell in row for r in cell]
    if use_stitching:
        from stitch_wrapper import stitch_rects
        acc_rects = stitch_rects(all_rects, tau)
    else:
        acc_rects = all_rects_r


    pairs = [(acc_rects, (0, 255, 0))]
    if show_suppressed:
        pairs.append((all_rects_r, (255, 0, 0)))
    for rect_set, color in pairs:
        for rect in rect_set:
            if rect.confidence > min_conf:
                cv.rectangle(image,
                    (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                    (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                    color,
                    2)

    rects = []
    for rect in acc_rects:
        r = al.AnnoRect()
        r.x1 = rect.cx - rect.width/2.
        r.x2 = rect.cx + rect.width/2.
        r.y1 = rect.cy - rect.height/2.
        r.y2 = rect.cy + rect.height/2.
        r.score = rect.true_confidence
        rects.append(r)

    return image, rects
