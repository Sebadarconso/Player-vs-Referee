import torch
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import sys

from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_bounding_boxes
from PIL import Image

sys.path.append("code/utils_proj/dataset.py")
import utils_proj.dataset as ds

def test_custom_image(path, model, classes,  number, single=True):
    test_img = Image.open(path)
    test_img = np.array(test_img)
    transform = A.Compose([ToTensorV2()])
    test_test = transform(image = test_img)
    with torch.no_grad():
        prediction = model(((test_test['image'])/255).unsqueeze(0))
        pred = prediction[0]
    color = []

    for i in pred['labels']:
        c = i.item()
        if c == 1: color.append((255,0,0))
        else: color.append((0,255,0))

    color = list(color)
    labels = ['background', 'player', 'referee']
    detected = []
    for p in pred['labels'][pred['scores'] > 0.5]:
        detected.append(labels[p.item()])
    
    num_ref = detected.count('referee')
    num_players = detected.count('player')
    print(f"Detected {num_ref} referees and {num_players} players")

    plt.imshow(draw_bounding_boxes(test_test['image'],
        pred['boxes'][pred['scores'] > 0.4],
        list([classes[i-1]+ "  "+ str(round(pred['scores'][j].item(), 2)) for j,i in enumerate(pred['labels'][pred['scores'] > 0.4])]), colors=color,width=2,
     font = "calibri.ttf", font_size=25).permute(1, 2, 0))

    if single:
        plt.show()
    else:
        plt.savefig(f'output/{number}.jpeg', dpi='figure')

def testing_validationset(valset, model, classes):

    for sample in valset:
        fig = plt.figure(figsize=(10, 7))
        fig.add_subplot(1,2,1)
        gt = ds.print_sample(sample, classes, flag = False)
        plt.imshow(gt)
        plt.title("Ground Truth")
        fig.add_subplot(1,2,2)
        sample = np.array(sample)
        img_int = torch.tensor(sample[0] * 255, dtype=torch.uint8)
        with torch.no_grad():
            prediction = model(sample[0].unsqueeze(0))
            pred = prediction[0]
        color = []
        for i in pred['labels']:
            c = i.item()
            if c == 1:
                color.append((255,0,0))
            else:
                color.append((0,255,0))
        color = list(color)

        labels = ['background', 'player', 'referee']
        detected = []
        for p in pred['labels'][pred['scores'] > 0.5]:
            detected.append(labels[p.item()])
        
        num_ref = detected.count('referee')
        num_players = detected.count('player')
        print(f"Detected {num_ref} referees and {num_players} players")

        plt.imshow(draw_bounding_boxes(img_int,
            pred['boxes'][pred['scores'] > 0.5],
            list([classes[i-1] + "  "+ str(round(pred['scores'][j].item(), 2)) for j,i in enumerate(pred['labels'][pred['scores'] > 0.5])]), colors=color,width=2,
            font = "calibri.ttf", font_size=15).permute(1, 2, 0))
            
        plt.title("Prediction")
        plt.show()
        # exit()
