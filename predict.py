import torch
import torchvision.models as models
from torch import nn
import numpy as np
from collections import OrderedDict
from PIL import Image
import os
import json

import argparse


def get_input_args_predict():
    parser = argparse.ArgumentParser()

    parser.add_argument('--top_k', type=int, default=5,
                        help='top classes')

    parser.add_argument('checkpoint_path', type=str, default='checkpoint.pth',
                        help='checkpoint path')

    parser.add_argument('--gpu', type=str,
                        default=False, help='gpu')

    parser.add_argument('path_to_img', type=str,
                        default='aipnd-project/flowers/test/1/image_06743.jpg', help='path to image')

    parser.add_argument('--json_file', type=str,
                        default='aipnd-project/cat_to_name.json', help='json file')
    args = parser.parse_args()

    return args


args = get_input_args_predict()

top_k = args.top_k
json_file = args.json_file
checkpoint_path = args.checkpoint_path
gpu = args.gpu
path_to_img = args.path_to_img

if gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

with open(json_file, 'r') as f:
    cat_to_name = json.load(f)

checkpoint = torch.load(checkpoint_path)


def load_checkpoint(device, filepath):
    model = getattr(models, checkpoint['arch'])(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.to(device)
    return model


model = load_checkpoint(device, checkpoint_path)


def crop_image(im, new_width, new_height):
    width, height = im.size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    return im.crop((left, top, right, bottom))


def process_image(infile):
    img = Image.open(infile)

    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    left = (img.width-224)/2
    bottom = (img.height-224)/2
    right = left + 224
    top = bottom + 224
    img = img.crop((left, bottom, right, top))

    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std

    img = img.transpose((2, 0, 1))

    return img


def predict(image_path, model, topk):
    model.eval()
    model.cpu()

    img = torch.from_numpy(image_path).type(torch.FloatTensor)
    im = img.unsqueeze(0)
    probs = torch.exp(model.forward(im))
    top_probs, top_classes = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_classes = top_classes.detach().numpy().tolist()[0]

    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {val: key for key, val in class_to_idx.items()}
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_classes]

    return top_probs, top_classes, top_flowers


proc_img = process_image(path_to_img)
classes, probs, flowers = predict(proc_img, model, top_k)

print('classes: ', classes)
print('probs: ', probs)
print('flowers: ', flowers)
