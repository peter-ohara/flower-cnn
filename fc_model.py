import os

import torch
import torchvision.models as models
from torch import nn
from torch import optim

from workspace_utils import active_session


def train(model, criterion, optimizer, trainloader, validloader, device="cpu", epochs=5, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.to(device)
    with active_session():

        print_metrics_every = 5
        steps = 0
        running_loss = 0
        for epoch in range(epochs):
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(images)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_metrics_every == 0:
                    training_loss = running_loss / print_metrics_every
                    validation_loss, validation_accuracy = validate(validloader, model, criterion, device=device)

                    print("Epoch: {}/{}...".format(epoch + 1, epochs),
                          "Training loss: {:.3f}...".format(training_loss),
                          "Validation loss: {:.3f}...".format(validation_loss),
                          "Validation Accuracy: {:.3f}...".format(validation_accuracy))

                    running_loss = 0

            print("Saving model to checkpoint folder:", save_dir)
            save_checkpoint(f"{save_dir}/checkpoint_epoch_{epoch + 1}.tar", model)
            print("Checkpoint saved!")


def validate(dataloader, model, criterion, device="cpu"):
    model.to(device)

    # Disable dropouts before evaluation
    model.eval()

    running_loss = 0
    accuracy = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Disable gradient calculations when evaluating
        with torch.no_grad():
            # Calculate loss
            logps = model(images)
            running_loss += criterion(logps, labels)

            # Calculate accuracy
            ps = torch.exp(logps)
            top_ps, top_class = ps.topk(1, dim=1)
            matches = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(matches.type(torch.FloatTensor))

    # Enable dropouts after evaluation
    model.train()

    loss = running_loss / len(dataloader)
    accuracy = accuracy / len(dataloader)
    return loss, accuracy


def create_model(arch="vgg13", hidden_units=512, lr=0.01):
    assert (hidden_units > 0)

    image_classification_model = getattr(models, arch)
    model = image_classification_model(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if arch.startswith('densenet'):
        input_units = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif arch.startswith('vgg'):
        input_units = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.01),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif arch.startswith('resnet'):
        input_units = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    else:
        raise RuntimeError("Uknown model architecture")

    criterion = nn.NLLLoss()
    model.arch = arch
    model.hidden_units = hidden_units

    return model, criterion, optimizer


def save_checkpoint(save_dir, model):
    checkpoint = {
        'arch': model.arch,
        'hidden_units': model.hidden_units,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir)


def load_checkpoint(filepath, device="cpu"):
    checkpoint = torch.load(filepath, )

    model, _criterion, _optimizer = create_model(arch=checkpoint['arch'],
                                                 hidden_units=checkpoint['hidden_units'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    model.to(device)
    model.eval()

    return model
