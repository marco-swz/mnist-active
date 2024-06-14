# Active-Training on the MNIST-Dataset

In this project we explored some approaches using active-training to build classifiers with very little data labels.

Problem Setup:
- Build the best possible classifier with only 500 image labels
- All of MNIST can be used without labels for unsupervised pre-training
- The 500 labels also include the test data

## Approaches

### Clustering

### Active-CNN

The main idea behind this approach is to make a smart selection of training batches.
To find the next training batch, 5000 randomly selected unlabeled images are evaluated.
Then, the entropy over the class probabilities is computed and the images with the highest entropy are chosen for labeling.
The newly labeled data points are added to the training data and the training continues.

### Active-CNN + Contrastive Pre-Training

## Results

| Model                     | Accuracy | 95% CI |
|---------------------------|----------|--------|
| CNN                       |          |        |
| Clustering                |          |        |
| Active-CNN                |          |        |
| Active-CNN (pre-training) |          |        |
