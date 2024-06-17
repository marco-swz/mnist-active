
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
    train_data: NDArray

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
        
        if (self.params["bool_tsne"]):
            #self.fit_TSNE(x)
            x = self.perf_TSNE(x)
        else:
            self.fit_PCA(x)
            x = self.perf_PCA(x)
            #print("Perf PCA")
    
        # Train Cluster with KMeans 
        self.model = KMeans(init='k-means++',n_clusters=self.params["num_cluster"], n_init=self.params["num_iter"])
        self.model.fit(x)
        #print("Done Fitting")
        # Get Map of Matched Labels
        self.active_lern(x, data_unlabeled)

    def fit_Sacler(self, x:NDArray):
        data = np.reshape(x, (len(x),784))  
        self.scaler = StandardScaler().fit(data)

    # Fit TSNE with given x
    def fit_TSNE(self, x: NDArray): 
        self.fit_PCA(x)
        data_pca_reduced = self.perf_PCA(x)
        self.tsne = TSNE(n_components=self.params["tsne_dimmension"], n_iter=300).fit_and_transfrom(data_pca_reduced)

    # Get Data after performed TSNE  
    def perf_TSNE (self, x: NDArray): # Alternative T-SNE  
        self.fit_PCA(x)
        data_pca_reduced = self.perf_PCA(x)    
        #print("Perf PCA in TSNE")
        tsne_data = TSNE(n_components=self.params["tsne_dimmension"], n_iter=300).fit_transform(data_pca_reduced)
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
        #print("Got MAP")
        
    def predict(self, x: NDArray) -> NDArray: 
        if (self.params["bool_tsne"]):
            data = np.vstack((self.train_data, x))
            all_trans = self.perf_TSNE(data)
            transform_x = all_trans[-len(x):]
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

    # Calculate the distance between points
    def _get_distance (self, center, data: NDArray):
        def formular(x):
            d = 0
            for n in range(len(center)):
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
            bool_tsne=skopt.space.Categorical([True, False]),
            bool_scale=skopt.space.Categorical([True, False]),
        ),
        data=data,
    )

if __name__ == "__main__":
    #run()
    #optimize()
    evaluate()

