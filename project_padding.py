import numpy as np
from PIL import Image
import scipy.misc
import scipy
import imageio
import sys
import os

def imgPadding( img_directory ):
    for folder in os.listdir( img_directory ):
        if folder[0] == '.' or folder[-4:]==".txt":
            continue
        for image_path in os.listdir( img_directory + "/" + folder ):
            if image_path[0] == '.' : # extracting hided files that start with '.'
                continue
            
            image = Image.open( img_directory + "/" + folder + "/" + image_path).convert("RGB") # load an image
            image = np.asarray(image) # convert to a numpy array
            rowSize = np.size(image,0)
            colSize = np.size(image,1)
            #print(rowSize,colSize)
            
            if rowSize > colSize:
            	diff = rowSize-colSize
            	zr = np.zeros([rowSize,int(diff/2),3], dtype = int)
            	zr2 = np.zeros([rowSize,diff-int(diff/2),3], dtype = int)
            	image = np.concatenate((zr,image), axis = 1)
            	image = np.concatenate((image,zr2), axis = 1)
            
            else:
            	diff = colSize-rowSize
            	zr = np.zeros([int(diff/2),colSize, 3], dtype = int)
            	zr2 = np.zeros([diff-int(diff/2),colSize, 3], dtype = int)
            	image = np.concatenate((zr,image), axis = 0)
            	image = np.concatenate((image,zr2), axis = 0)
            
            imageio.imwrite('outfile.jpg', image)
            image = np.float32(image)
            image = np.resize(image, (224, 224, 3))

            image2 = Image.open("outfile.jpg").convert("RGB")
            image2 = image2.resize((224,224))
            image2 = np.asarray(image2)
            
            imageio.imwrite("padded_images/" + img_directory + "/" + image_path , image2)
