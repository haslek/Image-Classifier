import time
import pandas as pd
import os
import json
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from collections import OrderedDict
from gen_util import *
from get_input_args import get_input_args



#torch.cuda.is_available()
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
    
    print('Saved as {}'.format(input_args.save_dir))
    
def validate(model,loader,device,criterion):
    model.eval()
    test_loss = 0
    accuracy = 0
    for images, labels in loader:
        images,labels = images.to(device),labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        prediction = torch.exp(output)
        is_correct = (labels.data == prediction.max(dim=1)[1])
        accuracy += is_correct.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def model_train(model,optimizer,t_loader,v_loader,device,criterion,epochs,save_dir ='checkpoint.pth',inspect_point=40):
    st_time = time.time()
    steps = 0
    print("Training Started: {}".format(time.strftime("%H:%M:%S", time.localtime())))
    model.to(device)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for images,labels in iter(t_loader):
            images,labels = images.to(device),labels.to(device)
            steps +=1
            optimizer.zero_grad()
         
            output = model.forward(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % inspect_point == 0:
                with torch.no_grad():
                    test_loss,accuracy = validate(model,v_loader,device,criterion)
                print("Epoch: {}/{}".format(e+1,epochs),
                  "Training Loss: {:.4f}".format(running_loss/inspect_point),
                  "Validation Loss: {:.4f}.. ".format(test_loss/len(v_loader)),
                  "Validation Accuracy: {:.4f}".format(accuracy/len(v_loader)))
            
            running_loss = 0
                  
    print("Training finished: {}".format(time.time()- st_time))
    return model

def save_checkpoint(checkpoint,checkpoint_name):
    torch.save(checkpoint, checkpoint_name)
    
if __name__ == "__main__":
    main()