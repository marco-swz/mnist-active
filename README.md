# Active-Training on the MNIST-Dataset

In this project we explored some approaches using active-training to build classifiers with very little data labels.

Problem Setup:
- Build the best possible classifier with only 500 image labels
- All of MNIST can be used without labels for unsupervised pre-training
- The 500 labels also include the test data (used split 400/100)

## Approaches

### Clustering

The clustering approach is very simple and mostly used to get a comparison for the other approaches.
The method performs a kMeans-clustering on the full trainings set and has different possibilities for data preperation. 
Afterwards the true lables of the $n$ nearest points to the cluster centers are used to get the most common label of this cluster. The number of labels per cluster is calculated with $n= \frac{\#\ \text{trainglables}}{\#\ \text{clusters}}$ and for the distance between the center and the points the 2-norm is used. 

Data Preperation:
 - Standard Scaler or No Scaler 
 - PCA or TSNE (also uses PCA for dimmensionaly reduction)

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
| Clustering (PCA / TSNE)   |          |        |
| Active-CNN                |          |        |
| Active-CNN (pre-training) |          |        |
