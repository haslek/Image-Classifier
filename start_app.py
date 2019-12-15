
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
from predict import *

#def network_init(model_name,input_size):
    
    




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
    if input_args.intent == 1:
        train_dir = input_args.dir + '/train'
        valid_dir = input_args.dir + '/valid'
        test_dir = input_args.dir + '/test'
    
        # TODO: Define your transforms for the training, validation, and testing sets
        data_transforms = load_transform()
        # TODO: Load the datasets with ImageFolder
        image_datasets = load_datasets(train_dir,valid_dir,test_dir,data_transforms)

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        dataloaders = load_dataloaders(image_datasets)
    
        with open(input_args.category_file, 'r') as f:
            cat_to_name = json.load(f)
        
        selected_model = input_args.arch
        m_input = 25088
        
        arch_name = 'vgg'
    
        if selected_model == 1:
            model = t_model.vgg16(pretrained = True)
            
        elif selected_model == 2:
            model = t_model.densenet161(pretrained = True)
            m_input = 2208
            arch_name='densenet'
        elif selected_model == 3:
            model = t_model.resnet50(pretrained = True)
            m_input = 2048
            arch_name = 'resnet'
        else:
            print('Invalid model selected, will default to vgg!!')
            model = t_model.vgg16(pretrained = True)
            
         
        hidden = input_args.hidden_u
        
        for param in model.parameters():
            param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
                    ('drop',nn.Dropout(p=0.5)),
                    ('fc1',nn.Linear(m_input,hidden)),
                    ('relu1',nn.ReLU()),
                    ('drop1',nn.Dropout(p=0.5)),
                    ('fc2',nn.Linear(hidden,102)),
                    ('output', nn.LogSoftmax(dim=1))
                ]))
        if selected_model == 3:
            model.fc = classifier
            optimizer = optim.SGD(model.fc.parameters(), input_args.l_r)
        else:
            model.classifier = classifier
            optimizer = optim.SGD(model.classifier.parameters(), input_args.l_r)

        criterion = nn.CrossEntropyLoss()
        
        #print(model)
        trained_model = model_train(model,optimizer,dataloaders['training_loader'],dataloaders['validation_loader'],device,criterion,
                            input_args.epochs)
        
        checkpoint = {
            'base_model': selected_model,
            'arch_name':arch_name,
            'output':102,
            'state_dict': trained_model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'class_to_idx':image_datasets['training_dataset'].class_to_idx
           }
        if selected_model == 3:
            checkpoint['classifier']= trained_model.fc
        else:
            checkpoint['classifier'] = trained_model.classifier
        save_checkpoint(checkpoint,input_args.save_dir)
    elif input_args.intent == 2:
        
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
        
   
    
    
    
    
if __name__ == "__main__":
    main()
    