import os
import cv2
import re
import sys
import argparse
import numpy as np
import copy
import annolist as al

def annotation_to_h5(a, cell_width, cell_height, max_len):
    region_size = 32
    image_height=480
    image_width=640
    focus_size=1.8
    biggest_box_px=10000
    assert region_size == image_height/ cell_height
    assert region_size == image_width / cell_width
    cell_regions = get_cell_grid(cell_width, cell_height, region_size)
    cells_per_image = len(cell_regions)
    box_list = [[] for idx in range(cells_per_image)]

    for cidx, c in enumerate(cell_regions):
        box_list[cidx] = [r for r in a.rects if all(r.intersection(c))]

    boxes = np.zeros((1, cells_per_image, 4, max_len, 1), dtype = np.float)
    box_flags = np.zeros((1, cells_per_image, 1, max_len, 1), dtype = np.float)

    for cidx in xrange(cells_per_image):
        #assert(cur_num_boxes <= max_len)

        cell_ox = 0.5 * (cell_regions[cidx].x1 + cell_regions[cidx].x2)
        cell_oy = 0.5 * (cell_regions[cidx].y1 + cell_regions[cidx].y2)

        unsorted_boxes = []
        for bidx in xrange(min(len(box_list[cidx]), max_len)):

            # relative box position with respect to cell
            ox = 0.5 * (box_list[cidx][bidx].x1 + box_list[cidx][bidx].x2) - cell_ox
            oy = 0.5 * (box_list[cidx][bidx].y1 + box_list[cidx][bidx].y2) - cell_oy

            width = abs(box_list[cidx][bidx].x2 - box_list[cidx][bidx].x1)
            height= abs(box_list[cidx][bidx].y2 - box_list[cidx][bidx].y1)

            if (abs(ox) < focus_size * region_size and abs(oy) < focus_size * region_size and
                    width < biggest_box_px and height < biggest_box_px):
                unsorted_boxes.append(np.array([ox, oy, width, height], dtype=np.float))

        for bidx, box in enumerate(sorted(unsorted_boxes, key=lambda x: x[0]**2 + x[1]**2)):
            boxes[0, cidx, :, bidx, 0] = box
            box_flags[0, cidx, 0, bidx, 0] = max(box_list[cidx][bidx].silhouetteID, 1)

    return boxes, box_flags

def get_cell_grid(cell_width, cell_height, region_size):

    cell_regions = []
    for iy in xrange(cell_height):
        for ix in xrange(cell_width):
            cidx = iy * cell_width + ix
            ox = (ix + 0.5) * region_size
            oy = (iy + 0.5) * region_size

            r = al.AnnoRect(ox - 0.5 * region_size, oy - 0.5 * region_size,
                            ox + 0.5 * region_size, oy + 0.5 * region_size)
            r.track_id = cidx

            cell_regions.append(r)


    return cell_regions
