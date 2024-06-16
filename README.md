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

This approach utilizes a self-supervised learning approach called contrastive learning. In particular, the contrastive learning approach is based on [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). The framework is therefore comprised of two distinct phases: the constrastive learning phase and the active learning phase. The test data, or rather its indices, is extracted prior to training and can therefore no longer be accessed afterwards.

**Contrastive Learning**

SinCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. The contrastive learning framework comprises of four major components:
- A stochastic *data augmentation* module, that transforms any given data example randomly resulting in two correlated views of the same example. Such pairs are considered positive pairs. Here, we are using *random resized cropping*, *random rotation*, *color jitter*, and *random Gaussian noise*.
- A convolutional neural network *base encoder* that extracts representation vectors from augmented data examples.
- A small multi-layer perceptron model *projection head* that maps representations to the space where the contrastive loss is applied.
- A *contrastive loss function* defined for a contrastive prediction task. 


## Results

| Model                     | Accuracy | 95% CI |
|---------------------------|----------|--------|
| CNN                       |          |        |
| Clustering                |          |        |
| Active-CNN                |          |        |
| Active-CNN (pre-training) |          |        |
