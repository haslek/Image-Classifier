import numpy as np
import pandas as pd
import os
import json
from PIL import Image
from torch import nn
import torch.nn.functional as F
import torchvision.models as t_model
from torchvision import datasets,transforms
from train import *
from gen_util import *
from get_input_args import get_input_args
import time
import torch
from gen_util import *
import torchvision
from collections import OrderedDict


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def main():
    input_args = get_input_args()
    use_gpu = input_args.use_gpu
    if use_gpu == 1:
        if torch.cuda.is_available():
            print('GPU available and will be used')
            
        else:
            print('GPU not available, CPU will be used instead')
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    else:
        device = torch.device("cpu")
    
        #print(input_args)
        #with  Image.open(input_args.image_path) as image:
            #plt.imshow(image)
    with open(input_args.category_file, 'r') as f:
            cat_to_name = json.load(f)
    model = load_checkpoint(input_args.checkpoint)
        #print(model)
    prob, classes = predict(input_args.image_path, model,device)
    
        # TODO: Get the most fitting class
    max_index = np.argmax(prob)
    max_probability = prob[max_index]
    label = classes[max_index]
        
    labels=[]
       
    for clas in classes:
            labels.append(cat_to_name[clas])

    print('The most likely class of this image is {} with probability of {}'.format(cat_to_name[label],max_probability))
    print('Top {} predictions: {}'.format(input_args.topk,list(zip(classes,labels, prob))))
    
    
    
        
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    from PIL import Image
    img = Image.open(image)
    data_transforms = load_transform()
    img = data_transforms['testing_set'](img)
    
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = img.unsqueeze(0)
    img = img.float()
    img = img.to(device)
    
    with torch.no_grad():
        result = model.forward(img)
        
    probs = torch.exp(result).data
    prediction = probs.topk(topk)
    probs = prediction[0][0].cpu().data.numpy()
    classes = prediction[1][0].cpu().data.numpy()
    labels = [idx_to_class[i] for i in classes]
    return probs.tolist(), labels


if __name__ == "__main__":
    main()