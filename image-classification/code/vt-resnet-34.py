import argparse
import json
import logging
import os
from pathlib import Path
import sys
import torch, torchvision
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
import tokenizerTrans as tokenizer
import projectorTrans as projector
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import time
from PIL import Image
import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

DATASETS = ['train', 'val']

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transforms = {'train': T.Compose([
    T.RandomResizedCrop(size=224),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.RandomHorizontalFlip(),
    T.Normalize(mean_nums, std_nums)
]), 'val': T.Compose([
    T.Resize(size=224),
    T.CenterCrop(size=224),
    T.ToTensor(),
    T.Normalize(mean_nums, std_nums)
]),}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class VT(nn.Module):
    # Constructor
    def __init__(self, L, CT, C):
        super(VT, self).__init__()
        self.bn = nn.BatchNorm2d(256)
        self.tokenizer = tokenizer.Tokenizer(L=L,CT=CT, C=C)
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=5, num_decoder_layers = 0, dim_feedforward=2048, activation='relu', dropout=0.5)
        self.projector = projector.Projector(CT=CT, C=C)

    def forward(self, x):
        x = self.bn(x)
        x, token = self.tokenizer(x)
        token = self.transformer(token, token)
        out, token = self.projector(x,token)
        return out

def _get_data_loader(batch_size, training_dir):
    logger.info("Get data loaders")
    dataset = {
    d: ImageFolder(f'{training_dir}/{d}', transforms[d]) for d in DATASETS
    }
    dataset_sizes = {d: len(dataset[d]) for d in DATASETS}
    logger.info(f'dataset sizes: {dataset_sizes}')
    data_loaders = {
    d: DataLoader(dataset[d], batch_size=batch_size, shuffle=True) for d in DATASETS
    }
    return data_loaders, dataset_sizes

def create_model(n_classes):
    model = models.resnet34(pretrained=True)
    model.layer4 = VT(L=8, CT=512, C=512)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)
    return model.to(device)

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train() #Convert to train mode
    losses = []
    correct_predictions = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device) #Push array to gpu
        labels = labels.to(device)
        outputs = model(inputs) #get prob of output per class
        _, preds = torch.max(outputs, dim=1) # get max of pred
        loss = loss_fn(outputs, labels) # get loss
        correct_predictions += torch.sum(preds==labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval() #Evaluation mode
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(preds==labels)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses) 

def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.state_dict(), path)
    logger.info(f"Checkpoint: Saved the best model: {path} \n")

    
def train(args):
    logger.info("Training using Visual Transformer ResNet34 model")
    logger.debug("\n Number of gpus available - {}".format(args.num_gpus))
    logger.debug(f"\n Device: {device}")
    train_loader, dataset_sizes = _get_data_loader(args.batch_size, args.data_dir)
    logger.info("Building Visual transformer (vt-resnet34) model from Resnet34 Pre-trained model. \n")
    model = create_model(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss_fn = nn.CrossEntropyLoss().to(device)
    best_accuracy = 0
    corresponding_loss = 0
    corresponding_epoch = 0
    start = time.time()
    for epoch in range(args.epochs):
        logger.info(f'\nEpoch {epoch + 1}/{args.epochs}')
        logger.info('-' * 10)
        train_acc, train_loss = train_epoch(model, train_loader['train'], loss_fn, 
                                            optimizer, device, scheduler, dataset_sizes['train'])
        logger.info(f'Train_loss = {train_loss}; Train_accuracy = {train_acc};')
        val_acc, val_loss = eval_model(model, train_loader['val'], loss_fn, device, dataset_sizes['val'])
        logger.info(f'Valid_loss = {val_loss}; Valid_accuracy = {val_acc};')
        if val_acc >= best_accuracy:
            save_model(model, args.model_dir)
            best_accuracy = val_acc
            corresponding_loss = val_loss
            corresponding_epoch = epoch + 1
    end = time.time()        
    logger.info(f'Best val accuracy: {best_accuracy}')
    logger.info(f'Corresponding loss: {corresponding_loss}')
    logger.info(f'Corresponding epoch: {corresponding_epoch}')
    logger.info(f'Runtime of the model is {round((end - start)/60, 2)} mins')

def model_fn(model_dir):
    logger.info('model_fn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(6)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)

def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        image_data = Image.open(requests.get(url, stream=True).raw)
        image_transform = T.Compose([
            T.Resize(size=256),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return image_transform(image_data)
    else:
        logger.info('raising expception')
        raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)
    return ps

def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    classes = {0: 'buildings', 1: 'forest', 2: 'glacier', 3: 'mountain', 4: 'sea', 5: 'street'}
    topk, topclass = prediction_output.topk(3, dim=1)
    result = []
    for i in range(3):
        pred = {'prediction': classes[topclass.cpu().numpy()[0][i]], 'score': f'{topk.cpu().numpy()[0][i] * 100}%'}
        logger.info(f'Adding pediction: {pred}')
        result.append(pred)
    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--step_size', type=int, default=7, 
                        help='step size for StepLR scheduler (default: 7)')
    parser.add_argument('--gamma', type=float, default=0.1, 
                        help='gamma for StepLR scheduler (default: 0.1)')
    parser.add_argument('--nesterov', type=bool, default=True,
                        help='nesterov for SGD optimizer (default: True)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--num_classes', type=int, default=None, 
                        help='number of classes')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
