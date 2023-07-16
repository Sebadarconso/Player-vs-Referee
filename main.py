import argparse
import torch
import sys
import os

from torch import optim
from torch.utils.data import DataLoader
from utils_proj import *

sys.path.append("/Users/sebastianodarconso/Desktop/PlayerVsReferee_ultimate/code/detection")
from detection.engine import evaluate


def train(opt):
    ## argument parsing
    data_path = opt.data
    backbone = opt.model
    annotations_path = os.path.join(data_path, opt.mode, 'annotations.json')

    n_classes, classes = ds.get_classes(annot_path=annotations_path)
    print(f'\nClasses found: {classes[0], classes[1]}\n')
    
    ## dataset and dataloader
    train_dataset = ds.Dataset(root=data_path, type=opt.mode, transforms=ds.get_transforms(True))
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, collate_fn=ds.collate_fn)

    ## model and hyperparameters 
    model = md.get_model(backbone=backbone, n_classes=n_classes)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print()
    print('-'*90)
    print('Started training...')
    print('-'*90)
   
    ## training 
    tr.train_new_model(model, opt.epochs, optimizer, train_loader, 'cpu', lr_scheduler, backbone)

    print('Training ended, saving the model...')
    print('-'*90)
    
    ## saving the model
    torch.save(model, os.path.join("code/models", backbone, 'model_' + {opt.epochs} + 'ep.pth'))


def test(opt):
    ## argument parsing
    data_path = opt.data

    ## dataset and dataloader 
    test_dataset = ds.Dataset(root=data_path, type=opt.mode, transforms=ds.get_transforms(False))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=ds.collate_fn)

    ## model loading
    model_trained = torch.load(opt.weights, map_location=torch.device('cpu'))
    model_trained.eval()

    ## evaluation of the model 
    evaluate(model_trained, test_loader, 'cpu')


def validation(opt):
    ## argument parsing
    data_path = opt.data
    annotations_path = os.path.join(data_path, opt.mode, 'annotations.json')

    ## dataset
    validation_set = ds.Dataset(root=data_path, type=opt.mode, transforms=ds.get_transforms(True))

    ## model loading
    model_trained = torch.load(opt.weights, map_location=torch.device('cpu'))
    model_trained.eval()

    ## test on validation set
    annotations_path = os.path.join(data_path, opt.mode, 'annotations.json')
    n_classes, classes = ds.get_classes(annot_path=annotations_path)
    ts.testing_validationset(validation_set, model_trained, classes)


def inference(opt):
    ## argument parsing
    path_to_img = opt.image

    ## model loading and inference 
    model_trained = torch.load(opt.weights, map_location=torch.device('cpu'))
    model_trained.eval()
    classes = ['player', 'referee']

    if os.path.isdir(path_to_img):
        for i, img in enumerate(os.listdir(path_to_img)):
            if img != '.DS_Store':
                ts.test_custom_image(os.path.join(path_to_img, img), model_trained, classes, i, False)

    elif os.path.isfile(path_to_img):
        ts.test_custom_image(path_to_img, model_trained, classes, None, True)


def parse_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'validation', 'inference'], default='train', help='Select the mode: train or test')
    parser.add_argument('--data', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--model', choices=['resnet50', 'mobilenet'], default='resnet50', help='Select the model type: resnet50 or mobilenet')
    parser.add_argument('--weights', type=str, default='models/resnet50/model_20ep.pth', help='Path to load the model')
    parser.add_argument('--epochs', type=int, default='10', help='How many training epochs')
    parser.add_argument('--image', type=str, default=None, help='Path to test image or folder')

    return parser.parse_args()


def run(*kwargs):
    opt = parse_options()
    if opt.mode == 'train':
        train(opt)
    elif opt.mode == 'test':
        test(opt)
    elif opt.mode == 'validation':
        validation(opt)
    elif opt.mode == 'inference':
        inference(opt)

if __name__ == '__main__':
    opt = parse_options()
    run(opt)