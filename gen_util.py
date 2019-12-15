
import torch
import torchvision.models as t_model
from torchvision import datasets,transforms


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path)
        print('You are loading a {} trained network'.format(checkpoint['arch_name']))
      
        if checkpoint['base_model'] == 2:
            model = t_model.densenet161(pretrained = True)
            model.classifier = checkpoint['classifier']
        elif checkpoint['base_model']==3:
            model = t_model.resnet50(pretrained = True)
            model.fc = checkpoint['classifier']
        else:
            model = t_model.vgg16(pretrained = True)
            model.classifier = checkpoint['classifier']
        
        model.class_to_idx = checkpoint['class_to_idx']
        
        model.load_state_dict(checkpoint['state_dict'])
       
        return model
    except:
        print('Wrong checkpoint or path!!')
        
def load_datasets(train_dir,valid_dir,test_dir,data_transforms):
    image_datasets = {
            'training_dataset': datasets.ImageFolder(train_dir,transform = data_transforms['training_set']),
            'validation_dataset': datasets.ImageFolder(valid_dir,transform = data_transforms['validation_set']),
            'testing_dataset': datasets.ImageFolder(test_dir,transform = data_transforms['testing_set'])
        }
    return image_datasets

def load_dataloaders(image_datasets):
    dataloaders = {
            'training_loader': torch.utils.data.DataLoader(image_datasets['training_dataset'],batch_size = 64,                      shuffle= True),
            'validation_loader': torch.utils.data.DataLoader(image_datasets['validation_dataset'],batch_size =                      32),
            'testing_loader': torch.utils.data.DataLoader(image_datasets['testing_dataset'],batch_size = 32)
           }
    return dataloaders


def load_transform():
    data_transforms = {
        'training_set':transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                          ]),
        'validation_set':transforms.Compose([transforms.Resize(224),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485,0.456,0.406),[0.229,0.224,0.225])
                                           ]),
        'testing_set':transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                          ])
        }
    return data_transforms