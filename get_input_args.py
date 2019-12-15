import argparse
import os
import random

def get_input_args():
    parse = argparse.ArgumentParser()
    
    parse.add_argument('--intent',help='Select an action: 1 to train, 2 to predict',type=int,default=1)
    
    parse.add_argument('--use_gpu',help='Select 1 to use gpu if available, 2 to use cpu instead', type=int,default=1 )
   
    parse.add_argument('--dir',default='flowers',type=str,
                           help='Directory for images, make sure it contains train,test and valid directories' )
    parse.add_argument('--arch', default=1,type=int,
                       help='Select architecture to use:1 for vgg16, 2 for densenet (default is vgg16)' )
    parse.add_argument('--category_file', default='cat_to_name.json')
    parse.add_argument('--save_dir',help='Where to save the model checkpoint',default='checkpoint.pth')
    parse.add_argument('--l_r',type=float,help='Model learning rate',default= 0.05)
    parse.add_argument('--epochs',type=int,default=5,help='number of epochs, default is 5')
    parse.add_argument('--hidden_u',type=int,default=1024, help='Number of hidden units, default is 1024')
    parse.add_argument('--image_path',default='flowers/test/23/'+random.choice(os.listdir('flowers/test/23/')),help='Image to be predicted')
    
    parse.add_argument('--topk',default=3, type=int,
                           help='Top number of classes that matches the result. Default is 3' )
    parse.add_argument('--cat_names',default='cat_to_name.json',
                           help='Category file name. Expecting JSON format')
    parse.add_argument('--checkpoint',help='Checkpoint to use',default='checkpoint.pth')
    parse.add_argument('--model',help='model used to train',default=1)
    
  
    args = parse.parse_args()
    return args
