#!/usr/bin/env python

import argparse

import torch

from fc_model import create_model, train
from helper import load_data

parser = argparse.ArgumentParser()

parser.add_argument("data_directory", help="path to folder containing training and validation data")
parser.add_argument("--save_dir",
                    help="set directory to save checkpoints")
parser.add_argument("--arch", help="choose architecture", default='vgg13')
parser.add_argument("--learning_rate",
                    help="Set the learning rate for the optimization step",
                    default=0.01,
                    type=float)
parser.add_argument("--hidden_units",
                    help="Set number of hidden_units",
                    default=512,
                    type=int)
parser.add_argument("--epochs",
                    help="Set number of epochs to train for",
                    default=20,
                    type=int)
parser.add_argument("--gpu",
                    help="Use GPU for inference",
                    action="store_true")

args = parser.parse_args()


trainloader, validloader, testloader, class_to_idx = load_data(args.data_directory)

model, criterion, optimizer = create_model(arch=args.arch,
                                           hidden_units=args.hidden_units,
                                           lr=args.learning_rate)
model.class_to_idx = class_to_idx

device = torch.device("cuda" if (torch.cuda.is_available() and args.gpu) else "cpu")

train(model,
      criterion,
      optimizer,
      trainloader,
      validloader,
      device=device,
      epochs=args.epochs,
      save_dir=args.save_dir)
