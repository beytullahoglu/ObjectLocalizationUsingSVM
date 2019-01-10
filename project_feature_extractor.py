from PIL import Image
import numpy as np
import scipy.misc
import scipy
import imageio
import sys
import os
import torch
import resnet
import torchvision
import math
import pickle

def vectorExtractor( image ):
    image = np.asarray(image) # convert to a numpy array
    image = np.reshape(image, [1, 224, 224, 3])
    # model takes as input images of size [batch_size, 3, im_height, im_width]
    image = np.transpose(image, [0, 3, 1, 2])
    image = np.float32(image)
    print(image.dtype)
    # convert the Numpy image to torch.FloatTensor
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor)
    # extract features
    model = torchvision.models.resnet50(pretrained=True)
    # model = model.type(torch.cuda.FloatTensor)
    feature_vector = model(image)
    # feature_vector = feature_vector.cpu()
    # convert the features of type torch.FloatTensor to a Numpy array
    # so that you can either work with them within the sklearn environment
    # or save them as .mat files
    feature_vector = feature_vector.detach().numpy()
    
    sum = 0
    for i in range(0,np.size(feature_vector,1)):
        x = (feature_vector[0,i])
        sum = sum + x*x
    sum = math.sqrt(sum) + 1e-4
    return feature_vector/sum

def vectorArrayExtractor():
    feature_vector_array = np.zeros((1,1000), dtype = int)
    feature_vector_array = np.float32(feature_vector_array)
    
    for folder in os.listdir("padded_images"):
        if folder[0] == '.':
            continue
        for image_path in os.listdir("padded_images/" + folder):
            if image_path[0] == '.': # extracting hided files that start with '.'
                continue
            feature_vector = vectorExtractor("padded_images/" + folder + "/" + image_path)
            feature_vector_array = np.vstack((feature_vector_array, feature_vector))
        
    fl = open("test.txt", 'wb')
    pickle.dump( feature_vector_array, fl )
    fl.close()
