import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from collections import OrderedDict
import torchvision.models as models

import argparse


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default=False,
                        help='to save directory')
    parser.add_argument('--arch', type=str, default='densenet121',
                        help='CNN Model Architecture')
    parser.add_argument('--learning_rate', type=int,
                        default=0.001, help='learning rate')

    parser.add_argument('--hidden_units', type=str,
                        default='490', help='hidden units')

    parser.add_argument('--epochs', type=int,
                        default=4, help='epochs')

    parser.add_argument('--gpu', type=bool,
                        default=False, help='gpu')

    args = parser.parse_args()

    return args


args = get_input_args()

data_dir = 'aipnd-project/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

save_dir = args.save_dir
arch = args.arch

if arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    input_size = 9216
elif arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_size = 25088
elif arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_size = 1024

learning_rate = args.learning_rate

hidden_layers = args.hidden_units

if (hidden_layers.find(',') != -1):
    hidden_layers = hidden_layers.split(',')
    hidden_layers = [int(layer) for layer in hidden_layers]
else:
    hidden_layers = [int(hidden_layers)]

output_size = 102
hidden_layers.append(output_size)

epochs = args.epochs

data_train = transforms.Compose([transforms.RandomRotation(30),
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])

data_test = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

data_validate = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=data_train)
test_datasets = datasets.ImageFolder(test_dir, transform=data_test)
valid_datasets = datasets.ImageFolder(valid_dir, transform=data_validate)

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(
    train_datasets, batch_size=32, shuffle=True)
testloaders = torch.utils.data.DataLoader(
    test_datasets, batch_size=32, shuffle=True)
validloaders = torch.utils.data.DataLoader(
    valid_datasets, batch_size=32, shuffle=True)

for param in model.parameters():
    param.requires_grad = False

ordDict = OrderedDict()
hidden_sizes = hidden_layers

hidden_layers.insert(0, input_size)

for i in range(len(hidden_layers) - 1):
    ordDict['fc' + str(i + 1)] = nn.Linear(hidden_layers[i],
                                           hidden_layers[i + 1])
    ordDict['relu' + str(i + 1)] = nn.ReLU()
    ordDict['dropout' + str(i + 1)] = nn.Dropout(p=0.1)

ordDict['output'] = nn.Linear(hidden_layers[i + 1], output_size)
ordDict['softmax'] = nn.LogSoftmax(dim=1)

classifier = nn.Sequential(ordDict)

model.classifier = classifier

model.zero_grad()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

device = "cpu"
gpu = args.gpu

if gpu and torch.cuda.is_available():
    device = "cuda"

model = model.to(device)
print_every = 40
steps = 0


def accuracy(loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for (images, labels) in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return '%.2f' % (correct/total * 100)


print('Start training -------------------')

for e in range(epochs):
    running_loss = 0
    for ii, (images, labels) in enumerate(dataloaders):
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}... ".format(running_loss/print_every),
                  'Accuracy on validation dataset: {}%'.format(accuracy(validloaders)))

            running_loss = 0
            model.train()

print('finished training')
print('Accuracy on test dataset: {}%'.format(accuracy(testloaders)))

model.to('cpu')
model.class_to_idx = train_datasets.class_to_idx

checkpoint = {
    'epoch': epochs,
    'input_size': input_size,
    'output_size': output_size,
    'hidden_layers': hidden_layers,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'classifier': model.classifier,
    'class_to_idx': model.class_to_idx,
    'arch': arch
}

if save_dir:
    checkpoint_path = save_dir + '/checkpoint.pth'
else:
    checkpoint_path = 'checkpoint.pth'

torch.save(checkpoint, checkpoint_path)

print('Saved')
