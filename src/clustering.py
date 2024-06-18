
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from math import sqrt
from numpy.typing import NDArray, ArrayLike
import skopt

from utils import ActiveModel, MAX_NUM_LABELED, load_data, NUM_TEST, split_data, eval_model, optimize_model

class ActiveCluster(ActiveModel):
    model: KMeans
    modelMap: map
    pca: PCA
    scaler: StandardScaler
    params: dict
    num_train: int
    num_test: int
    train_data: ArrayLike

    def __init__(self, **params):
        params["bool_tsne"] = params.get("perform_tsne",False) # false: perform PCA, true: perform TSNE
        params["bool_scale"] = params.get("perform_scale",False) # false: perform PCA, true: perform TSNE
        params["num_cluster"] = params.get("num_cluster", 20) # nr_cluster = 40
        params["num_iter"] = params.get("num_iter", 80) # iter = 100
        params["pca_dimmension"] = params.get("pca_dimmension", 50) # dimmension = 50 for preparing the tsne
        if(params["bool_scale"]):
            params["pca_dimmension"] = params.get("pca_dimmension", 50) # dimmension = 50 for preparing the tsne
        else:
            params["pca_dimmension"] = params.get("pca_dimmension", 10) # dimmension = 10
        params["tsne_dimmension"] = params.get("pca_dimmension",2) 
        self.params = params
        
        self.num_train = MAX_NUM_LABELED - NUM_TEST
        self.num_test = NUM_TEST
        self.modelMap = {}
        #print("Done Initializing")

    def fit(self, data_unlabeled: ArrayLike):
        x = np.array([d.x for d in data_unlabeled])
        self.train_data = x
        if(self.params["bool_scale"]):
            self.fit_Sacler(x)
            #print("Scaled Data") 
        
        if (self.params["bool_tsne"]): # Fit PCA for preprocessing the Data later
            self.fit_PCA(x) 
            self.train_data = data_unlabeled
        else:
            self.fit_PCA(x)
            x = self.perf_PCA(x)
            #print("Perf PCA")
            # Train the clusters
            self.train_kmeans(x)
            # Get Map of Matched Labels
            self.active_lern(x, data_unlabeled)

    def train_kmeans(self, x:NDArray):
        # Train Cluster with KMeans 
        self.model = KMeans(init='k-means++',n_clusters=self.params["num_cluster"], n_init=self.params["num_iter"])
        self.model.fit(x)
        if(self.params["bool_tsne"] and self.params["tsne_dimmension"]==2):
            plot_2D_Clusters(x, self.model)
        #print("Done Fitting")
  
    def fit_Sacler(self, x:NDArray):
        data = np.reshape(x, (len(x),784))  
        self.scaler = StandardScaler().fit(data)

    # Fit and Perform TSNE together
    def fit_perf_TSNE (self, x: NDArray): # Alternative T-SNE  
        data_pca_reduced = self.perf_PCA(x)
        #print("Perf PCA in TSNE")
        tsne_data = TSNE(n_components=self.params["tsne_dimmension"], n_iter=300, learning_rate=1000).fit_transform(data_pca_reduced)
        #print("Perf TSNE")
        return tsne_data
    
    # Fit PCA with given x
    def fit_PCA(self, x: NDArray): 
        data = np.reshape(x, (len(x),784)) 
        if(self.params["bool_scale"]):
            data = self.scaler.transform(data)    
        self.pca = PCA(n_components=self.params["pca_dimmension"]).fit(data)

    # Get Data after performed PCA   
    def perf_PCA (self, x: NDArray): # Alternative T-SNE  
        data = np.reshape(x, (len(x),784)) 
        if(self.params["bool_scale"]):
            data = self.scaler.transform(data)      
        pca_data = self.pca.transform(data)
        return pca_data

    # Run Active Learning to get Map for Predictions
    def active_lern(self, x, dataPoints):
        labl_per_cluster = int(self.num_train/self.params["num_cluster"])
        labels = self.model.labels_
        centers = self.model.cluster_centers_
        map = {}
        for c in range (0,self.params["num_cluster"]):
            center = centers[c,:] # Center of cluster c
            idx_ClusterPoints = np.where(labels == c)[0] # indixes of ClusterPoints in cluster c
            idx_sortedByDistance = self.get_center_near_points(center, x[idx_ClusterPoints]) # sort ClusterPoints by Distance, get Sorted Indexes
            idx_sortedClusterPoints = idx_ClusterPoints[idx_sortedByDistance] # sort Indexes of ClusterPoints
            idx_to_label = idx_sortedClusterPoints[0:labl_per_cluster] # get only the nearest labl_per_cluster Points
            data_get_label = dataPoints[idx_to_label] # get DataPoints you want label
            trueLabels = np.array([d.get_label() for d in data_get_label]) # get labels
            count = Counter(trueLabels) # Count labels per cluster
            # Most Common Label
            mostCommonLabel, _ = count.most_common(1)[0]
            self.modelMap[c] =  mostCommonLabel
        #print("Got MAP")
    
    # Calculate distance between points and center of cluster
    def get_center_near_points(self, center, points:NDArray):
        distances= np.linalg.norm(points-center, axis=1)
        return np.argsort(distances)
        
    def predict(self, x: NDArray) -> NDArray: 
        if (self.params["bool_tsne"]): # For TSNE you need to train and predict together 
            train_data = np.array([d.x for d in self.train_data])
            data = np.vstack((train_data, x))
            all_trans = self.fit_perf_TSNE(data)
            train_x = all_trans[:-len(x)] # Trainings data for Kmeans
            self.train_kmeans(train_x) 
            self.active_lern(train_x, self.train_data)
            transform_x = all_trans[-len(x):] # test data to be predicted
        else:
            transform_x = self.perf_PCA(x)
        predKmean = self.model.predict(transform_x)
        predLabels = [self.modelMap[n] for n in predKmean]          
        return predLabels 

    def get_params(self, deep: bool=True):
        return self.params

    def set_params(self, **params):
        self.__init__(**params);
        return self

    # Get array of different labels in Modelmap
    def getDiffLabels(self):
        diffLab = []
        for l in self.modelMap.values():
            if np.isin(l,diffLab) == False: 
                diffLab = np.append(diffLab,l)
        return diffLab

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

# Method Plot 2D Clusters
def plot_2D_Clusters (train_data, kmean):
    kmean_Y = kmean.predict(train_data)
    centers = kmean.cluster_centers_
    plt.scatter(train_data[:, 0], train_data[:, 1], c=kmean_Y, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def run():
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

def evaluate():
    data = load_data()
    model = ActiveCluster(
        bool_tsne=False,
        bool_scale=False,
        num_cluster=100,
        pca_dimmension=150,
        num_iter=5,
    )
    eval_model(
        model=model,
        data=data,
        num_retrains=10,
        model_name='cluster_pca',
    )

def optimize():
    data = load_data()

    model = ActiveCluster()

    optimize_model(
        model=model,
        opt_params=dict(
            num_cluster=skopt.space.Integer(10, 100),
            pca_dimmension=skopt.space.Integer(2, 150),
            num_iter=skopt.space.Integer(5, 200),
            bool_tsne=skopt.space.Categorical([True]),
            bool_scale=skopt.space.Categorical([True, False]),
        ),
        data=data,
    )

if __name__ == "__main__":
    #run()
    optimize()
    #evaluate()

