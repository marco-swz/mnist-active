"""
This file contains the logic required for running the contrastive learning algorithm
"""

# Data handling and mathematical operations
import pandas as pd
import numpy as np

# Image processing
from PIL import Image

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

# Additional utilities
from tqdm import tqdm

# Project imports
from . import models
from .. import utils

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


class SimCLRAugmentDataset(Dataset):
    def __init__(self, data, idxs_exclude=[]):
        self.data_instance = data
        images = np.array([d.x for d in self.data_instance])
        flattened_images = images.reshape(images.shape[0], -1)
        self.data_frame = pd.DataFrame(flattened_images)
        self.idxs_exclude = set(idxs_exclude)

        # Filter out excluded indices
        self.indices = [i for i in range(len(self.data_frame)) if i not in self.idxs_exclude]

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("L")),  # Ensure image is in grayscale mode
            transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),  # Random crop within 80-100% of the original size
            transforms.RandomRotation(degrees=15),  # Randomly rotate within a range of -15 to 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            AddGaussianNoise(mean=0.0, std=0.1),  # Add Gaussian noise with clipping to [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.data_frame.iloc[self.indices[idx]].values.astype('float32').reshape(28, 28)
        image = Image.fromarray(image).convert("L")  # Ensure image is in grayscale mode
        image1 = self.transform(image)
        image2 = self.transform(image)
        return image1, image2


def create_augmented_data_loader(data, idxs_eval, batch_size=512):
    augmented_dataset = SimCLRAugmentDataset(data, idxs_eval)
    # Shuffle == True to ensure that negative pairs are different across different epochs
    dataloader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Logic for creating SimCLR Model
def create_SimCLR_model(encoder_type):
    if encoder_type == "cnn":
        base_encoder = models.ActiveMnistCnn()
        projection_input_dim = 128 * 3 * 3
    elif encoder_type == "resnet18":
        base_encoder = models.resnet18(pretrained=False)
        # Adjust the first convolutional layer for 1-channel grayscale images
        base_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        projection_input_dim = base_encoder.fc.in_features
        base_encoder.fc = nn.Identity()
    elif encoder_type == "resnet50":
        base_encoder = models.resnet50(pretrained=False)
        # Adjust the first convolutional layer for 1-channel grayscale images
        base_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        projection_input_dim = base_encoder.fc.in_features
        base_encoder.fc = nn.Identity()
    else:
        raise NotImplementedError("Unsupported encoder type")

    projection_head = models.ProjectionHead(input_dim=projection_input_dim)
    simclr_model = models.SimCLR(base_encoder, projection_head)
    return simclr_model


# Contrastive Learning Loss
class NTXentLoss(nn.Module):
    # Defines the Normalized Temperature-scaled Cross Entropy Loss
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, z_i, z_j):
        N = 2 * z_i.size(
            0)  # Adjust N for the current batch size --> total number of images in the batch (original + augmented)
        z = torch.cat((z_i, z_j), dim=0)  # Concatenate the representations of original and augmented images

        # Compute cosine similarity between all pairs in the batch
        # sim_matrix[i][j] = cos_sim(z_i, z_j)
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # Remove self-similarity by setting diagonal elements to a large negative value
        sim_matrix = sim_matrix - torch.eye(N, device=self.device) * 1e12

        # Extract positive pairs: for each original, its positive pair is the corresponding augmented
        # sim_ij contains similarities of (z_i, z_j) and sim_ji contains similarities of (z_j, z_i)
        sim_ij = torch.diag(sim_matrix, N // 2)
        sim_ji = torch.diag(sim_matrix, -N // 2)
        positive_samples = torch.cat((sim_ij, sim_ji), dim=0).view(N, 1)

        # Extract negative samples for each z_i and z_j
        # Negative samples are all other elements in the same row of sim_matrix
        negative_samples = sim_matrix[torch.eye(N, device=self.device) == 0].view(N, -1)

        # Create labels: positive samples (0 index) vs negative samples
        labels = torch.zeros(N).to(self.device).long()

        # Combine positive and negative samples and apply softmax cross entropy loss
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss


# Training loop for contrastive learning
def train_contrastive_model(model,
                            train_loader,
                            epochs,
                            temperature,
                            device,
                            learning_rate,
                            train_val_split=0.9,
                            save_path=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize NTXentLoss with the specified temperature
    criterion = NTXentLoss(temperature=temperature, device=device)

    best_loss = float('inf')

    # Create val_loader, redefine dataloader
    train_loader, val_loader = split_data_loader(train_loader, train_ratio=train_val_split)

    for epoch in range(epochs):
        total_loss = 0
        model.train()  # Set the model to training mode
        for image_pairs in tqdm(train_loader):
            # Assuming image_pairs is a batch where each element contains both images
            images = torch.cat(image_pairs, dim=0).to(device)

            optimizer.zero_grad()

            features = model(images)
            z_i, z_j = torch.split(features, features.shape[0] // 2, dim=0)

            loss = criterion(z_i, z_j)
            total_loss += loss.item()

            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

        val_loss = validate_contrastive_model(model, val_loader, criterion, device)

        train_loss = total_loss / len(train_loader)
        print("Epoch [{}/{}], Training Loss: {}, Validation Loss: {}".format(epoch + 1, epochs, train_loss, val_loss))

        # Log metrics to wandb
        # wandb.log({"Epoch": epoch,
        #           "Training Loss": train_loss,
        #           "Validation Loss": val_loss})

        # Checkpointing
        if val_loss < best_loss and save_path is not None:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"\nSaved Best Model at Epoch {epoch + 1}")


def validate_contrastive_model(model, dataloader, criterion, device):
    # Method for Validation Monitoring to identify model-states that should be checkpointed
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for image_pairs in tqdm(dataloader):
            # Assuming image_pairs is a batch where each element contains both images
            images = torch.cat(image_pairs, dim=0).to(device)

            features = model(images)
            z_i, z_j = torch.split(features, features.shape[0] // 2, dim=0)

            loss = criterion(z_i, z_j)
            total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    return average_loss


def split_data_loader(data_loader, train_ratio=0.8):
    # Get the original dataset
    dataset = data_loader.dataset

    # Calculate the split sizes
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create new DataLoaders from the split datasets
    train_loader = DataLoader(train_dataset, batch_size=data_loader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=data_loader.batch_size, shuffle=False)

    return train_loader, val_loader


def load_simclr_model(checkpoint_path, encoder_type, device):
    # Create the model architecture (should be the same as used during training)
    model = create_SimCLR_model(encoder_type)

    # Load the saved state dictionary
    model_state = torch.load(checkpoint_path, map_location=torch.device(device))

    # Load the state dictionary into the model
    model.load_state_dict(model_state)

    print("load_simclr_model(): Successfully loaded contrastive learner!")
    return model


class ActiveLearningDataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = np.array([d.x for d in self.data[self.indices[idx]]]).astype("float32") / 255.0  # Convert to float32 and normalize
        label = np.array([d.get_label() for d in self.data[self.indices[idx]]])
        return torch.tensor(image).unsqueeze(0), torch.tensor(label)  # Unsqueeze to add channel dimension


def entropy(probs):
    return -np.sum(probs * np.log(probs), axis=1)


def train_classifier(model, data, idxs_excluded, num_epochs, batch_size=25, search_size=400, learning_rate=1e-3,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     freeze_base_encoder=True, save_path=None, search_pool_size=1000):
    # Decide whether to fine-tune the base encoder or not
    if not freeze_base_encoder:
        for param in model.base_encoder.parameters():
            param.requires_grad = True
        print("Fine-tuning the entire model.")
    else:
        for param in model.base_encoder.parameters():
            param.requires_grad = False
        print("Training only the final layers.")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Set to keep track of labeled indices and the excluded indices
    labeled_indices = set()
    excluded_indices = set(idxs_excluded)

    # Initially label a random set of points excluding the excluded indices
    available_indices = [i for i in range(len(data.x)) if i not in excluded_indices]
    idxs_to_label = np.random.choice(available_indices, batch_size, replace=False)
    labeled_indices.update(idxs_to_label)
    num_iterations = int((search_size - batch_size) / batch_size)

    for iteration in tqdm(range(num_iterations)):
        print(f"Active Learning Iteration {iteration + 1}/{num_iterations}")

        # Create dataset and dataloader for current batch of labeled data
        current_dataset = ActiveLearningDataset(data, list(labeled_indices))
        train_dataloader = DataLoader(current_dataset, batch_size=batch_size, shuffle=True)

        # Train the model
        # Monitoring Training Loss for Early Stopping.
        patience = 10
        best_train_loss = float('inf')
        counter = 0

        for epoch in range(num_epochs):
            total_loss = 0
            model.train()
            for images, labels in train_dataloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)

            # Update Monitoring
            if avg_train_loss > best_train_loss:
                counter += 1
                if counter >= patience:
                    break
            else:
                best_train_loss = avg_train_loss
                counter = 0

        # Select new points based on uncertainty
        # Ensure idxs_search only includes unlabeled indices
        idxs_search = np.random.choice(
            [i for i in range(len(data.x)) if i not in labeled_indices and i not in excluded_indices], search_pool_size,
            replace=False)

        # Calculate entropies directly
        model.eval()
        all_probs = []
        with torch.no_grad():
            for start in range(0, len(idxs_search), batch_size):
                end = min(start + batch_size, len(idxs_search))
                batch_indices = idxs_search[start:end]
                images = np.array([d.x for d in data])[batch_indices].astype("float32") / 255.0  # Convert to float32 and normalize
                images = torch.tensor(images).unsqueeze(1).to(device)  # Add channel dimension and move to device
                outputs = model(images)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        entropies = entropy(all_probs)

        # Identify indices of the most uncertain instances
        idxs_max = np.argsort(entropies)[-batch_size:]
        idxs_to_label = idxs_search[idxs_max]

        # Check if adding these indices will exceed the maximum number of labels
        if len(labeled_indices) + len(idxs_to_label) > search_size or \
                len(labeled_indices) + len(idxs_to_label) > utils.MAX_NUM_LABELED:
            print("Cannot label more than the maximum allowed number of labels.")
            idxs_to_label = np.random.choice(list(set(idxs_to_label) - labeled_indices),
                                             utils.MAX_NUM_LABELED - len(labeled_indices), replace=False)

        # Update the set of labeled indices
        labeled_indices.update(idxs_to_label)
        print("Updated labeled indices. Currently using {} images".format(len(labeled_indices)))

    print("Number of labeled data points: {}".format(len(labeled_indices)))
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print("Model saved at ", save_path)

    return model, labeled_indices


def evaluate_accuracy(model, data, idxs_eval, num_samples=100, batch_size=25,
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    # Create dataset and dataloader for the selected images
    eval_dataset = ActiveLearningDataset(data, idxs_eval)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on {num_samples} evaluation images: {accuracy * 100:.2f}%")
    return accuracy


# Loading classifier model from checkpoint_path
def load_classifier_model(checkpoint_path, base_encoder, num_classes, device):
    # Recreate the classifier model architecture
    model = models.Classifier(base_encoder, num_classes)

    # Load the saved state dictionary
    model_state = torch.load(checkpoint_path, map_location=torch.device(device))

    # Load the state dictionary into the model
    model.load_state_dict(model_state)

    print("load_classifier_model(): Successfully loaded classifier!")
    return model
