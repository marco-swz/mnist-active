"""
This file contains the logic required to combine the contents of:
- ActiveLearningSimCLR
- ContrastiveLearningLogic
- ContrastiveLearningModels
"""

# Standard library imports
import numpy as np

# PyTorch and related libraries
import torch

# Project imports
from torch.utils.data import DataLoader

from . import logic, models
from .. import utils


class ActiveContrastiveLearner(utils.ActiveModel):
    def __init__(self):
        # Creating Data instance and selecting (i.i.d.) the test samples
        self.data = utils.Data()
        self.idxs_eval = np.random.choice(self.data._idxs_unlabeled, 100, replace=False)

        self.simclr_model = None
        self.encoder_type = None
        self.classifier_model = None
        self.labeled_indices = None

    def fit(self, data: utils.Data, params: dict={}):
        # General parameters
        self.encoder_type = params.get("encoder_type", "cnn")  # CNN ('cnn'), ResNet18 ('resnet18'), or ResNet50 ('resnet50')
        self.simclr_model = logic.create_SimCLR_model(self.encoder_type)

        # Parameters for contrastive learning
        num_epochs_contrastive_learning = params.get("num_epochs_contrastive_learning", 100)
        simCLR_save_path = params.get("simCLR_save_path", "simCLR_save.pth")
        temperature = params.get("temperature", 0.5)
        learning_rate_simCLR = params.get("learning_rate_simCLR", 1e-3)
        batch_size_SimCLR = params.get("batch_size_SimCLR", 2048)
        train_val_split = params.get("train_val_split", 0.9)

        # Parameters for classifier training
        num_epochs_active_learning = params.get("num_epochs_active_learning", 100)
        learning_rate_classifier = params.get("learning_rate_classifier", 1e-4)
        batch_size_active_learning = params.get("batch_size_active_learning", 25)
        search_size = params.get("search_size", 400)
        search_pool_size = params.get("search_pool_size", 10000)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data Preparation
        dataloader = logic.create_augmented_data_loader(self.idxs_eval, batch_size_SimCLR)

        # Contrastive Learning Loop
        logic.train_contrastive_model(
            model=self.simclr_model,
            train_loader=dataloader,
            epochs=num_epochs_contrastive_learning,
            temperature=temperature,
            device=device,
            learning_rate=learning_rate_simCLR,
            train_val_split=train_val_split,
            save_path=simCLR_save_path
        )

        # Load the best checkpoint (of contrastive learning)
        simclr_model = logic.load_simclr_model(simCLR_save_path, self.encoder_type, device)

        # Adjust model for classification tasks
        base_encoder = simclr_model.encoder
        classifier_model = models.Classifier(base_encoder, 10).to(device)

        self.classifier_model, self.labeled_indices = logic.train_classifier(
            model=classifier_model,
            idxs_excluded=self.idxs_eval,
            num_epochs=num_epochs_active_learning,
            batch_size=batch_size_active_learning,
            search_size=search_size,
            learning_rate=learning_rate_classifier,
            device=device,
            freeze_base_encoder=True,
            save_path=None,
            search_pool_size=search_pool_size
        )

    def predict(self, x: utils.X):
        if self.labeled_indices is None or self.classifier_model is None:
            print("ActiveContrastiveLearning.predict(): Please train the model first...")
            raise RuntimeError

        # Setting classifier model to evaluation mode
        self.classifier_model.eval()

        predictions = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = x.astype("float32") / 255.0
        x = np.expand_dims(x, -1)  # Add channel dimension
        tensor_x = torch.tensor(x).permute(0, 3, 1, 2)  # Convert to NCHW format
        dataset = torch.utils.data.TensorDataset(tensor_x)
        pred_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for images in pred_dataloader:
                images = images[0].to(device)
                output = self.classifier_model(images)
                predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        preds = np.argmax(predictions, axis=1)
        return preds

    def evaluate(self):
        logic.evaluate_accuracy(
            self.classifier_model,
            self.idxs_eval,
            num_samples=100,
            batch_size=25,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )


if __name__ == "__main__":
    activeContrastiveLearner = ActiveContrastiveLearner()
    activeContrastiveLearner.fit(utils.Data())
    activeContrastiveLearner.evaluate()

    data = utils.Data()
    xs = data._x[[i for i in range(0, 40)]]
    activeContrastiveLearner.predict(xs)









