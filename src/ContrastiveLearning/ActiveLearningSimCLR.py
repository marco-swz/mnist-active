"""
This file contains the logic required to transition from contrastive learning
to active learning and to train the classifier model based on a uncertainty based
active learning routine.
"""


# Standard library imports

# Data handling and mathematical operations

# Image processing

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Additional utilities
from tqdm import tqdm

# Experiment tracking
import wandb


# Classifier Training Routine
# TODO: (Lucas) Transition to Active Learning Logic
from src.ContrastiveLearning.ContrastiveLearningModels import Classifier


def train_classifier(model, train_dataloader, val_dataloader, num_epochs, wandb_instance=None, freeze_base_encoder=True,
                     learning_rate=1e-3,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), save_path=None):
    # Decide whether to fine-tune the base encoder or not
    if not freeze_base_encoder:
        for param in model.base_encoder.parameters():
            param.requires_grad = True
        print("Fine-tuning the entire model.")
    else:
        for param in model.base_encoder.parameters():
            param.requires_grad = False
        print("Training only the final layer.")

    # Optimizer and Loss Function
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        model.to(device)
        model.train()  # Set the model to training mode
        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_dataloader)

        # Perform validation only if val_dataloader is provided
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_dataloader)
            print("Epoch [{}/{}], Train Loss: {}, Validation Loss: {}".format(epoch + 1, num_epochs, avg_train_loss,
                                                                              avg_val_loss))

            if wandb_instance is not None:
                # Calculate accuracy on train and validation dataset
                train_acc = evaluate_classifiers(model, train_dataloader, device, dataset_string="train", verbose=False)
                val_acc = evaluate_classifiers(model, val_dataloader, device, dataset_string="validation", verbose=False)

                # Log metrics to wandb
                wandb.log({"Epoch": epoch,
                           "Training Loss": avg_train_loss,
                           "Validation Loss": avg_val_loss,
                           "Training Accuracy": train_acc,
                           "Validation Accuracy": val_acc})

            # Checkpointing
            if avg_val_loss < best_val_loss and save_path is not None:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print("Saved Best Classifier Model at Epoch {}".format(epoch + 1))
        else:
            print("Epoch [{}/{}], Train Loss: {}".format(epoch + 1, num_epochs, avg_train_loss))


# Loading classifier model from checkpoint_path
def load_classifier_model(checkpoint_path, base_encoder, num_classes, device):
    # Recreate the classifier model architecture
    model = Classifier(base_encoder, num_classes)

    # Load the saved state dictionary
    model_state = torch.load(checkpoint_path, map_location=torch.device(device))

    # Load the state dictionary into the model
    model.load_state_dict(model_state)

    print("load_classifier_model(): Successfully loaded classifier!")
    return model


# Evaluating classifier
def evaluate_classifiers(model, test_loader, device, dataset_string='test', verbose=True):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to track the gradients
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    if verbose:
        print('Accuracy on {} set: {}%'.format(dataset_string, accuracy))
    return accuracy