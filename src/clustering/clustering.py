
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from math import sqrt



# Method Plot Clusters
def plot_2D_Clusters (train_data, kmean):
    kmean_Y = kmean.predict(train_data)
    centers = kmean.cluster_centers_
    plt.scatter(train_data[:, 0], train_data[:, 1], c=kmean_Y, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Get array of different labels
def getDiffLabels(labels):
    diffLab = []
    for l in labels:
        if np.isin(l,diffLab) == False: 
            diffLab = np.append(diffLab,l)
    return diffLab

def creatIndexArray(n):
    array = []
    for i in range(n):
        array.append(i)
    return array


def distance (center, data, nr_cluster):
    def formular(x):
        d= 0
        for n in range(dimmension):
            d += (center[n]-data[x,n])**2
        return abs(d)
    return formular

# Run Active Learning to label Data
def active_lern (nr_cluster, centers, data_X, data_Y, labels, nr_Points_per_cluster):
    usedPoints = []
    matchedLabels = {}
    for c in range (0,nr_cluster):
        center = centers[c,:]
        clusterPoints = np.where(labels == c)[0]
        formular = distance(center, data_X, nr_cluster)
        sortedByDistance = sorted(clusterPoints, key = formular)
        trueLabels = []
        for i in range(nr_Points_per_cluster):
            trueLabels.append(data_Y[sortedByDistance[i]])
            usedPoints.append(sortedByDistance[i])

        count= Counter(trueLabels)
        # Most Common Label
        mostCommonLabel, _ = count.most_common(1)[0]
        matchedLabels[c] =  mostCommonLabel
    return matchedLabels, usedPoints

# Predicted label for Data
def predict(data_X, kmean, map):
    predKmean = kmean.predict(data_X)
    predLabels = []
    for n in predKmean:
        predLabels.append(map[n])
    return predLabels     

# Get Error and Confidence for predicted Labels
def getConfidence(predLabels, trueLabels, c):
    n = len(predLabels)
    k = 0
    for i in range(n):
        if(predLabels[i]!=trueLabels[i]): 
            k+=1
    err = k/n
    interval = c* sqrt(err*(1-err)/n)+ c**2/n
    return err, interval 

# Create TestDataSet of size n
def createTestData(n,given_X, given_Y, notUsed):
    random_Index = random.sample(notUsed, k=n)
    test_X = []
    test_Y = []
    for i in random_Index:
        test_X.append(given_X[i,:])
        test_Y.append(given_Y[i])
    return test_X, test_Y

# Defined variables
nr_cluster = 20
dimmension = 10
nr_train = 400
nr_test = 100
iter = 100
c = 1.96

# Load Mnist Dataset
(mnist_X, mnist_Y), (_, _) = mnist.load_data()
train_X = np.reshape(mnist_X, (60000,784))

# Create array with indexes of Data
notUsedData = creatIndexArray(mnist_X.shape[0])

# Alternative T-SNE          
# Run PCA on Dataset 
pca = PCA(n_components=dimmension )
train_X_pca = pca.fit_transform(train_X)

# Train Cluster with KMeans 
kmean = KMeans(init='k-means++',n_clusters=nr_cluster, n_init=iter)
kmean.fit(train_X_pca)

# get Map of Matched Labels
matchedMap, usedData = active_lern(nr_cluster, kmean.cluster_centers_, train_X_pca, mnist_Y, kmean.labels_,int(nr_train/nr_cluster))
# Update used Data
notUsedData = list(set(notUsedData).symmetric_difference(usedData))

print(matchedMap)
diffLabels = getDiffLabels(matchedMap.values())
diffLabels.sort()
print(diffLabels)

trainSet_X, trainSet_Y = createTestData(40, train_X_pca, mnist_Y, usedData)

test_X1, test_Y1 = createTestData(nr_test, train_X_pca, mnist_Y, notUsedData)
test_X2, test_Y2 = createTestData(nr_test, train_X_pca, mnist_Y, notUsedData)
test_X3, test_Y3 = createTestData(nr_test, train_X_pca, mnist_Y, notUsedData)

predTrain = predict(trainSet_X,  kmean, matchedMap)
predTest1 = predict(test_X1, kmean, matchedMap)
predTest2 = predict(test_X2, kmean, matchedMap)
predTest3 = predict(test_X3, kmean, matchedMap)

error, interval = getConfidence(predTrain, trainSet_Y, c)
print(str(error)+" +- " +str(interval))

error1, interval1 = getConfidence(predTest1, test_Y1, c)
print(str(error1)+" +- " +str(interval1))
error2, interval2 = getConfidence(predTest2, test_Y2, c)
print(str(error2)+" +- " +str(interval2))
error3, interval3 = getConfidence(predTest3, test_Y3, c)
print(str(error3)+" +- " +str(interval3))







    

