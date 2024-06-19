# Active-Training on the MNIST-Dataset

In this project we explored some approaches using active-training to build classifiers with very little data labels.

Problem Setup:
- Build the best possible classifier with only 500 image labels
- All of MNIST can be used without labels for unsupervised pre-training
- The 500 labels also include the test data (used split 400/100)

## Approaches

### Clustering

The clustering approach is very simple and mostly used to get a comparison for the other approaches.
The method performs a kMeans-clustering on the full trainings set and has different possibilities for data preparation. 
Afterward, the true labels of the $n$ nearest points to the cluster centers are used to get the most common label of this cluster. The number of labels per cluster is calculated with $n = \frac{\text{num traininglables}}{\text{num clusters}}$ and for the distance between the center and the points the 2-norm is used. 

Data Preparation:
 - Standard Scaler or No Scaler 
 - PCA or TSNE (also uses PCA for dimensionality reduction)

### Active-CNN

The main idea behind this approach is to make a smart selection of training batches.
To find the next training batch, 5000 randomly selected unlabeled images are evaluated.
Then, the entropy over the class probabilities is computed and the images with the highest entropy are chosen for labeling.
The newly labeled data points are added to the training data and the training continues.

### Active-CNN + Contrastive Pre-Training

This approach utilizes a self-supervised learning approach called contrastive learning. In particular, the contrastive learning approach is based on [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709). The framework is therefore comprised of two distinct phases: the constrastive learning phase and the active learning phase. The test data, or rather its indices, is extracted prior to training and can therefore no longer be accessed afterwards.

**Contrastive Learning**

SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. The contrastive learning framework comprises of four major components:
- A stochastic *data augmentation* module, that transforms any given data example randomly resulting in two correlated views of the same example. Such pairs are considered positive pairs. Here, we are using *random resized cropping*, *random rotation*, *color jitter*, and *random Gaussian noise*.
- A convolutional neural network *base encoder* that extracts representation vectors from augmented data examples.
- A small multi-layer perceptron model *projection head* that maps representations to the space where the contrastive loss is applied.
- A *contrastive loss function* defined for a contrastive prediction task. 

The contrastive loss function is defined as:

$$l_{i, j} = - log \frac{exp(sim(z_i, z_j)/\tau}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} exp(sim(z_i, z_j)/\tau},$$
where $`z_i, z_j`$ are examples that have been put through the base encoder and the projection head, $`sim(u, v) = u^T v / ||u||||v||`$ denotes the dot product between $`l_2`$ normalized $`u`$ and $`v`$. Further, $`\mathbb{1}_{[k \neq i]}`$ is an indicator function evaluating to $`1`$ if $`k \neq 1`$ and $`0`$ else. $`\tau`$ denotes a temperature parameter. 

Once the contrastive learning terminates, the best-found model, based on the calculated contrastive loss, is loaded from disk and used for further downstream tasks.

**Active Learning**

Given the existing architecture from the contrastive learning phase, the model has to be restructured. This is done by removing the projection head and adding a new, deeper, MLP model to the "back" of the base encoder. From here on out, the active learning, based on a common supervised learning mechanism, starts. 

We start by sampling 25 images at random at by requesting their labels. The network will then be trained on them, before a random batch of 1000 random unseen images is taken and evaluated. Based on the shannon entropy that is perceived, these are sorted and those 25 samples that maximize the entropy are added to the training set. This is done in hope, that the samples which we know least of, are also those which we can learn most of. This process continues until the maximum number of samples, 400, is reached. 

Finally, the previously segregated 100 test samples will spring into action for a final evaluation of the model.


## Results

| Model                     | Accuracy | 95% CI         |
|---------------------------|----------|----------------|
| CNN                       | 0.693    | (0.506, 0.880) |
| Clustering (PCA)          | 0.873    | (0.851, 0.893) |
| Clustering (T-SNE)        | 0.883    | (0.861, 0.902) |
| Active-CNN                | 0.960    | (0.946, 0.971) |
| Active-CNN (pre-training) | 0.975    | (0.959, 0.986) |
