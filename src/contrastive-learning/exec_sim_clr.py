"""
This file contains the logic required to combine the contents of:
- ActiveLearningSimCLR
- ContrastiveLearningLogic
- ContrastiveLearningModels
"""

# Standard library imports
import numpy as np

# Data handling and mathematical operations

# Image processing

# PyTorch and related libraries
import torch

# Additional utilities

# Experiment tracking

# Project imports
from . import logic, models
from .. import utils

if __name__ == "__main__":
    # Define indices for testing at the end
    data = utils.Data()
    idxs_eval = np.random.choice(data._idxs_unlabeled, 100, replace=False)
    print("Indices for evaluation: len(idxs_eval) = {}, idxs_eval = {}".format(len(idxs_eval), idxs_eval))

    encoder_type = "cnn"  # CNN ('cnn'), ResNet18 ('resnet18'), or ResNet50 ('resnet50')
    num_epochs_contrastive_learning = 1
    simCLR_save_path = "simCLR_save.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Preparation and Augmentation
    batch_size_SimCLR = 1024
    dataloader = logic.create_augmented_data_loader(idxs_eval, batch_size_SimCLR)

    # SimCLR Model Setup
    simclr_model = logic.create_SimCLR_model(encoder_type)

    # Contrastive Learning
    temperature = 0.5
    learning_rate_simCLR = 1e-3

    # For monitoring define wandb here

    # Contrastive Learning Loop
    logic.train_contrastive_model(
        model=simclr_model,
        train_loader=dataloader,
        epochs=num_epochs_contrastive_learning,
        temperature=temperature,
        device=device,
        learning_rate=learning_rate_simCLR,
        save_path=simCLR_save_path
    )

    # Load the best checkpoint (of contrastive learning)
    simclr_model = logic.load_simclr_model(simCLR_save_path, encoder_type, device)

    # Adjust model for classification tasks
    base_encoder = simclr_model.encoder
    classifier_model = models.Classifier(base_encoder, 10).to(device)

    learning_rate_classifier = 1e-4

    # For monitoring define wandb here

    classifier_model, labeled_indices = logic.train_classifier(
        model=classifier_model,
        idxs_excluded=idxs_eval,
        num_epochs=1,
        batch_size=25,
        search_size=400,
        learning_rate=learning_rate_classifier,
        device=device,
        freeze_base_encoder=True,
        save_path=None
    )

    logic.evaluate_accuracy(classifier_model,
                                               idxs_eval,
                                               num_samples=100,
                                               batch_size=25,
                                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )








