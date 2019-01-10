import cv2 as cv
import numpy as np
import sys
import os
import pickle
import yaml

def edge_box( file_name, number_of_boxes ):
    im = cv.imread( file_name )
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection1.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection1.computeOrientation(edges)
    edges = edge_detection1.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(number_of_boxes)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes, edges

def window_extractor(image, boxes):
    parts = []
    for window in boxes:
        a, b, c, d = window
        parts.append(image[b: b + d, a: a + c])
    
    return parts

def plot_windows(image, boxes, edges):
    a, b, c, d = box
        cv2.rectangle(image, (a, b), (a + c, b + d), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("edges", edges)
    cv2.imshow("edgeboxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    




