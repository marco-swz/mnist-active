
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from math import sqrt
from numpy.typing import NDArray, ArrayLike

from utils import ActiveModel, MAX_NUM_LABELED, load_data, NUM_TEST, split_data, load_data

class ActiveCluster(ActiveModel):
    model: KMeans
    modelMap: map
    params: dict
    num_train: int
    num_test: int

    def __init__(self, **params):
        params["num_cluster"] = params.get("num_cluster", 20) # nr_cluster = 40
        params["pca_dimmension"] = params.get("pca_dimmension", 10) # dimmension = 10
        params["num_iter"] = params.get("num_iter", 80) # iter = 100
        self.params = params
        
        self.num_train = MAX_NUM_LABELED - NUM_TEST
        self.num_test = NUM_TEST
        self.modelMap = {}

    def fit(self, data_unlabeled: ArrayLike):
        x = np.array([d.x for d in data_unlabeled])
        x = self.perf_PCA(x)
        # Train Cluster with KMeans 
        self.model = KMeans(init='k-means++',n_clusters=self.params["num_cluster"], n_init=self.params["num_iter"])
        self.model.fit(x)
        # Get Map of Matched Labels
        self.active_lern(x, data_unlabeled)

    # Get Data after performed PCA   
    def perf_PCA (self, x: NDArray): # Alternative T-SNE  
        data = np.reshape(x, (len(x),784))       
        pca = PCA(n_components=self.params["pca_dimmension"])
        pca_data = pca.fit_transform(data)
        return pca_data

    # Run Active Learning to get Map for Predictions
    def active_lern(self, x, dataPoints):
        labl_per_cluster = int(self.num_train/self.params["num_cluster"])
        labels = self.model.labels_
        centers = self.model.cluster_centers_
        map = {}
        for c in range (0,self.params["num_cluster"]):
            
            center = centers[c,:]
            clusterPoints = np.where(labels == c)[0]
            formular = self._get_distance(center, x)
            sortedByDistance = sorted(clusterPoints, key = formular)
            idx_to_labeles = sortedByDistance[0:labl_per_cluster]
            data_get_labele = dataPoints[idx_to_labeles]
            trueLabels = np.array([d.get_label() for d in data_get_labele])
            count = Counter(trueLabels)
            # Most Common Label
            mostCommonLabel, _ = count.most_common(1)[0]
            self.modelMap[c] =  mostCommonLabel
        
    def predict(self, x: NDArray) -> NDArray: 
        pca_x = self.perf_PCA(x)
        predKmean = self.model.predict(pca_x)
        predLabels = [self.modelMap[n] for n in predKmean]          
        return predLabels 

    def get_params(self, deep: bool=True):
        return self.params

    def set_params(self, **params):
        self.params = params
        return self

    # Get array of different labels in Modelmap
    def getDiffLabels(self):
        diffLab = []
        for l in self.modelMap.values():
            if np.isin(l,diffLab) == False: 
                diffLab = np.append(diffLab,l)
        return diffLab

    # Calculate the distance between points
    def _get_distance (self, center, data: NDArray):
        def formular(x):
            d = 0
            for n in range(self.params["pca_dimmension"]):
                d += (center[n]-data[x,n])**2
            return d
        return formular

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

if __name__ == "__main__":
    c = 1.96
    data = load_data()
    data_train, data_test = split_data(data, 0.2)
    x_test = np.array([d.x for d in data_test])
    y_test = np.array([d.get_label() for d in data_test])
    model = ActiveCluster()
    model.fit(data_train)
    predTest = model.predict(x_test)
    error, interval = getConfidence(predTest, y_test, c)
    print(str(error)+" +- " +str(interval))


#Method Plot Clusters
def plot_2D_Clusters (train_data, kmean):
    kmean_Y = kmean.predict(train_data)
    centers = kmean.cluster_centers_
    plt.scatter(train_data[:, 0], train_data[:, 1], c=kmean_Y, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()





