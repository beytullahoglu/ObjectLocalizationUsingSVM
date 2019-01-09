#Project Testing Code

import numpy as np
from PIL import Image
import scipy.misc
import scipy
import imageio
import sys
import torch
import resnet
import torchvision
import math
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import pickle

modelname = "models/model1.sav"
mdl = pickle.load(open(modelname, 'rb'))
mdl.get_params()

for image_path in os.listdir("padded_test_images/"):
    if image_path[0] == '.' : # extracting hided files that start with '.'
        continue
    img = Image.open("padded_test_images/" + image_path).convert("RGB")
    img = np.asarray(img)
        # we append an augmented dimension to indicate batch_size, which is one
    img = np.reshape(img, [1, 224, 224, 3])
        # model takes as input images of size [batch_size, 3, im_height, im_width]
    img = np.transpose(img, [0, 3, 1, 2])
    img = np.float32(img)
        #print(image.dtype)
        # convert the Numpy image to torch.FloatTensor
    img = torch.from_numpy(img)
    img = img.type(torch.FloatTensor)
        # extract features
    model = torchvision.models.resnet50(pretrained=True)
    model = model.type(torch.FloatTensor)
    feature_vector = model(img)
        #feature_vector = feature_vector.cpu()
        # convert the features of type torch.FloatTensor to a Numpy array
        # so that you can either work with them within the sklearn environment
        # or save them as .mat files
    feature_vector = feature_vector.detach().numpy()
    sum = 0
    for i in range(0,np.size(feature_vector,1)):
        x = (feature_vector[0,i])
        sum = sum + x*x 
    sum = math.sqrt(sum) + 1e-4
    feature_vector = feature_vector/sum    
    prediction = mdl.predict(feature_vector)
    def numbers_to_objects(argument): 
        switcher = { 
                0.: "bird", 
                1.: "bison", 
                2.: "cat",
                3.: "chimpanzee",
                4.: "dog",
                5.: "elephant",
                6.: "gazelle",
                7.: "star",
                8.: "tiger",
                9.: "zebra",
                } 
  
    # get() method of dictionary data type returns  
    # value of passed argument if it is present  
    # in dictionary otherwise second argument will 
    # be assigned as default value of passed argument 
        return switcher.get(argument, "nothing") 
    print(image_path, " ", numbers_to_objects((prediction[0])))
