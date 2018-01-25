from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

from sklearn import metrics
from sklearn import random_projection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as EM
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import cdist
from sklearn.random_projection import SparseRandomProjection
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

Alldata = pd.read_csv('input/GOT.csv')

#splitting features and target from dataset
X = Alldata.values[:, 0:5]
Y = Alldata.values[:, 5]
Y = Y.reshape(1, -1)
labels = Y
data = X

n_samples, n_features  = data.shape
n_digits = len(np.unique(Y))

sample_size = 300
np.random.seed(4)

# K MEANS ###################################################################
#ELBOW CURVE TO FIND # CLUSTERS HOMO
##Nc = range(1, 100)
##kmeans = [KMeans(n_clusters=i, init='k-means++') for i in Nc]
##score = []
##for i in range(len(kmeans)):
##    kmeans[i].fit(data)
##    score.append(metrics.homogeneity_score(labels, kmeans[i].labels_))
##
##plt.plot(Nc,score)
##plt.xlabel('Number of Clusters')
##plt.ylabel('Homogeneity Score')
##plt.title('Elbow Curve with Homogeneity Score')
##plt.show()

 
#ELBOW CURVE TO FIND # CLUSTERS COMPLETE
##Nc = range(1, 100)
##kmeans = [KMeans(n_clusters=i, init='k-means++') for i in Nc]
##score = []
##for i in range(len(kmeans)):
##    kmeans[i].fit(data)
##    score.append(metrics.completeness_score(labels, kmeans[i].labels_))
##
##plt.plot(Nc,score)
##plt.xlabel('Number of Clusters')
##plt.ylabel('Completeness Score')
##plt.title('Elbow Curve with Completeness Score')
##plt.show()

#ELBOW CURVE TO FIND # CLUSTERS SIL
##Nc = range(2, 100)
##kmeans = [KMeans(n_clusters=i, init='k-means++') for i in Nc]
##score = []
##for i in range(len(kmeans)):
##    kmeans[i].fit(data)
##    score.append(metrics.silhouette_score(data, kmeans[i].labels_, metric='euclidean', sample_size=sample_size))
##
##plt.plot(Nc,score)
##plt.xlabel('Number of Clusters')
##plt.ylabel('Silhouette Score')
##plt.title('Elbow Curve with Silhouette Score')
##plt.show()


#ELBOW CURVE TO FIND # CLUSTERS SSE
##Nc = range(1, 100)
##kmeans = [KMeans(n_clusters=i, init='k-means++') for i in Nc]
##score = []
##for i in range(len(kmeans)):
##    kmeans[i].fit(data)
##    min = np.min(np.square(cdist(X, kmeans[i].cluster_centers_, 'euclidean')), axis = 1)
##    value = np.mean(min)
##    score.append(value)
##plt.plot(Nc,score)
##plt.xlabel('Number of Clusters')
##plt.ylabel('SSE Value')
##plt.title('Elbow Curve with SSE')
##plt.show()

# Visualize the results of KMEANS
##km = KMeans(n_clusters=2).fit(data)
##km_2d = km.transform(data)
##
##for i in range(0, km_2d.shape[0]):
##    if km.labels_[i] == 0:
##        c1 = plt.scatter(km_2d[i,0], km_2d[i,1],c='r', marker='x')
##    elif km.labels_[i] == 1:
##        c2 = plt.scatter(km_2d[i,0], km_2d[i,1],c='g', marker='o')
##
##plt.legend([c1, c2], ['Dead', 'Alive'])
##plt.title('GOT Dataset with 2 clusters')
##plt.show()


# Expectation Maximazation ##################################################
# Fit a Gaussian mixture with EM
##Nc = range(1, 100)
##em = [EM(n_components=i, covariance_type='full') for i in Nc]
##bic = []
##for i in range(len(em)):
##    em[i].fit(data)
##    bic.append(em[i].bic(data))
##plt.plot(Nc,bic)
##plt.xlabel('Number of Components')
##plt.ylabel('BIC Score Value')
##plt.title('Expectation Maximization BIC Score')
##plt.show()
##
### Expectation Maximazation Log
##Nc = range(1, 100)
##em = [EM(n_components=i, covariance_type='full') for i in Nc]
##log = []
##for i in range(len(em)):
##    em[i].fit(data)
##    log.append(em[i].score(data))
##plt.plot(Nc,log)
##plt.xlabel('Number of Components')
##plt.ylabel('Log-likelihood Value')
##plt.title('Expectation Maximization Log-Likelihood')
##plt.show()

####### PCA ##################################################
##Nc = range(1, 6)
##pca = [PCA(n_components=i) for i in Nc]
##score = []
##for i in range(len(pca)):
##    pca[i].fit(X)
##    score.append(pca[i].explained_variance_) #the largest eigenvalue is added to this score array
##
##print "Eigenvalue as clusters increase"
##for i in range(len(score)):
##    print score[i]

# #############################################################################
##Visualize the results on PCA-reduced data on KMEANS
##reduced_data = PCA(n_components=2).fit_transform(data)
##kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
##kmeans.fit(reduced_data)
##
### Step size of the mesh. Decrease to increase the quality of the VQ.
##h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
##
### Plot the decision boundary. For that, we will assign a color to each
##x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
##y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
##xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
##
### Obtain labels for each point in mesh. Use last trained model.
##Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
##
### Put the result into a color plot
##Z = Z.reshape(xx.shape)
##plt.figure(1)
##plt.clf()
##plt.imshow(Z, interpolation='nearest',
##           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
##           cmap=plt.cm.Paired,
##           aspect='auto', origin='lower')
##
##plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
### Plot the centroids as a white X
##centroids = kmeans.cluster_centers_
##plt.scatter(centroids[:, 0], centroids[:, 1],
##            marker='x', s=169, linewidths=3,
##            color='w', zorder=10)
##plt.title('K-means clustering on the GOT Dataset (PCA-reduced data)\n'
##          'Centroids are marked with white cross')
##plt.xlim(x_min, x_max)
##plt.ylim(y_min, y_max)
##plt.xticks(())
##plt.yticks(())
##plt.show()

##Visualize the results on PCA-reduced data on KMEANS
##reduced_data = PCA(n_components=2).fit_transform(data)
##kmeans = KMeans(n_clusters=2)
##kmeans.fit(reduced_data)
##km_2d = kmeans.transform(reduced_data)
##
##for i in range(0, km_2d.shape[0]):
##    if kmeans.labels_[i] == 0:
##        c1 = plt.scatter(km_2d[i,0], km_2d[i,1],c='r', marker='x')
##    elif kmeans.labels_[i] == 1:
##        c2 = plt.scatter(km_2d[i,0], km_2d[i,1],c='g', marker='o')
##
##plt.legend([c1, c2], ['Dead', 'Alive'])
##plt.title('GOT Dataset with KMeans clustering k =2, PCA Reduced Data')
##plt.show()

# #############################################################################
## Visualize the results on PCA-reduced data on EM
##reduced_data = PCA(n_components=2).fit_transform(data)
##kmeans = EM(n_components=2) # EM, just called kmeans to copy paste code
##kmeans.fit(reduced_data)
##
### Step size of the mesh. Decrease to increase the quality of the VQ.
##h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
##
### Plot the decision boundary. For that, we will assign a color to each
##x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
##y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
##xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
##
### Obtain labels for each point in mesh. Use last trained model.
##Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
##
### Put the result into a color plot
##Z = Z.reshape(xx.shape)
##plt.figure(1)
##plt.clf()
##plt.imshow(Z, interpolation='nearest',
##           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
##           cmap=plt.cm.Paired,
##           aspect='auto', origin='lower')
##
##plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
##
##plt.title('EM clustering on the GOT Dataset (PCA-reduced data)')
##plt.xlim(x_min, x_max)
##plt.ylim(y_min, y_max)
##plt.xticks(())
##plt.yticks(())
##plt.show()

# #############################################################################
## RANDOMIZED PROJECTIONS

X = Alldata.values[:, 0:5]
Y = Alldata.values[:, 5]
split = train_test_split(X, Y, test_size = 0.3,
    random_state = 42)
(trainData, testData, trainTarget, testTarget) = split

##accuracies = []
##components = np.int32(np.linspace(1, 5, 5))
##
##model = LinearSVC()
##model.fit(trainData, trainTarget)
##baseline = metrics.accuracy_score(model.predict(testData), testTarget)
##for comp in components:
##    # create the random projection
##    sp = SparseRandomProjection(n_components = comp)
##    X = sp.fit_transform(trainData)
##    # train a classifier on the sparse random projection
##    model = LinearSVC()
##    model.fit(X, trainTarget) 
##    # evaluate the model and update the list of accuracies
##    test = sp.transform(testData)
##    accuracies.append(metrics.accuracy_score(model.predict(test), testTarget))
##
### create the figure
##plt.figure()
##plt.suptitle("Accuracy of Randomized Projection on GOT")
##plt.xlabel("# of Components")
##plt.ylabel("Accuracy")
##plt.xlim([1, 5])
##plt.ylim([0, 1.0])
## 
### plot the baseline and random projection accuracies
##plt.plot(components, accuracies)
##plt.show()


#############################################################################
####Visualize the results on RP-reduced data on KMEANS
##reduced_data = SparseRandomProjection(n_components=2).fit_transform(data)
##kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
##kmeans.fit(reduced_data)
##
### Step size of the mesh. Decrease to increase the quality of the VQ.
##h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
##
### Plot the decision boundary. For that, we will assign a color to each
##x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
##y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
##xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
##
### Obtain labels for each point in mesh. Use last trained model.
##Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
##
### Put the result into a color plot
##Z = Z.reshape(xx.shape)
##plt.figure(1)
##plt.clf()
##plt.imshow(Z, interpolation='nearest',
##           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
##           cmap=plt.cm.Paired,
##           aspect='auto', origin='lower')
##
##plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
### Plot the centroids as a white X
##centroids = kmeans.cluster_centers_
##plt.scatter(centroids[:, 0], centroids[:, 1],
##            marker='x', s=169, linewidths=3,
##            color='w', zorder=10)
##plt.title('K-means clustering on the GOT Dataset (RP-reduced data)\n'
##          'Centroids are marked with white cross')
##plt.xlim(x_min, x_max)
##plt.ylim(y_min, y_max)
##plt.xticks(())
##plt.yticks(())
##plt.show()


# #############################################################################
#####Visualize the results on RP-reduced data on EM
##reduced_data = SparseRandomProjection(n_components=2).fit_transform(data)
##kmeans = EM(n_components=2) # EM, just called kmeans to copy paste code
##kmeans.fit(reduced_data)
##
### Step size of the mesh. Decrease to increase the quality of the VQ.
##h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
##
### Plot the decision boundary. For that, we will assign a color to each
##x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
##y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
##xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
##
### Obtain labels for each point in mesh. Use last trained model.
##Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
##
### Put the result into a color plot
##Z = Z.reshape(xx.shape)
##plt.figure(1)
##plt.clf()
##plt.imshow(Z, interpolation='nearest',
##           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
##           cmap=plt.cm.Paired,
##           aspect='auto', origin='lower')
##
##plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
##
##plt.title('EM clustering on the GOT Dataset (RP-reduced data)')
##plt.xlim(x_min, x_max)
##plt.ylim(y_min, y_max)
##plt.xticks(())
##plt.yticks(())
##plt.show()
