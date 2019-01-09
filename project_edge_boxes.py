#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This sample demonstrates structured edge detection and edgeboxes.
Usage:
  edgeboxes_demo.py [<model>] [<input_image>]
'''

import cv2 as cv
import numpy as np
import sys
import os
import pickle
import yaml

modelname = "models/model.yml"
edge_detection1 = cv.ximgproc.createStructuredEdgeDetection( modelname )

def edge_box( file_name ):
    im = cv.imread( file_name )
    rgb_im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    edges = edge_detection1.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection1.computeOrientation(edges)
    edges = edge_detection1.edgesNms(edges, orimap)

    edge_boxes = cv.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(50)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes:
        x, y, w, h = b
        cv.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 1, cv.LINE_AA)

    #cv.imshow("edges", edges)
    cv.imshow("edgeboxes", im)
    cv.waitKey(0)
    #cv.destroyAllWindows()

image_name = "padded_test_images/74.JPEG"
edge_box( image_name )

#for image_path in os.listdir("padded_test_images/"):
#    if image_path[0] == '.' : # extracting hided files that start with '.'
#        continue
#    image_name =  "padded_test_images/" + image_path
#    edge_box( image_name )

