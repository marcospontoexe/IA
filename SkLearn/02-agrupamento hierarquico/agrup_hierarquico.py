import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster, datasets
from sklearn.datasets import make_blobs

plt.figure(figsize=(10,5))
X = np.array(
    [[1,1],
    [1.5,1],
    [3,3],
    [3.5,3.3],
    [2.5,3.4],
    [2.5,2.5]]
)
#----plota os valores no grafico--------------
plt.scatter(X[:,0], X[:,1], c='black', s=200)
for i in range(X.shape[0]):
    plt.text(X[i,0]+0.1, X[i,1], str(i+1), horizontalalignment='right')

plt.show()
#---------------------------------------------

#----calculando a distância euclidiana entre os pontos e obter a matriz de distância
D = np.zeros((X.shape[0], X.shape[0]))
for i in range(0, X.shape[0]):
    for j in range(0, X.shape[0]):
        x = X[i,:]
        y = X[j, :]
        d = np.sqrt( (x[0]-y[0])**2 + (x[1]-y[1])**2 )
        D[i,j] = int(d*100)/100

print(f'Matriz com a distância entre os pontos: \n{D}')
#------------------------------------------------------------------------

#-------agrupamento hierarquico----------------------
plt.plot(figsize=(10, 6))
Z = linkage(X, "ward")      # critério de clusterização = Ward
dendrogram(Z)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
#---------------------------------


#----Usando outros critérios de clusterização -----
'''
MIM (single linkage)
MAX (complete linkage)
média dos grupos
distância entre sentróides
ward (erro quadrático)
'''
n_samples = 100
data = datasets.make_moons(n_samples=n_samples, noise=0.05)        # cria um dataset
X = data[0]
k=2
plt.figure(figsize=(10, 5))

#----teste com método 'ward'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward')
y_hr = clustering.fit_predict(X)
plt.subplot(2,2,1)
plt.scatter(X[:,0], X[:,1], c=y_hr, cmap='viridis', s=50)
plt.title('critérios de clusterização: ward')
#---------------------------------------

#----teste com método 'complete'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='complete')
y_hr = clustering.fit_predict(X)
plt.subplot(2,2,2)
plt.scatter(X[:,0], X[:,1], c=y_hr, cmap='viridis', s=50)
plt.title('critérios de clusterização: complete linkage')
#---------------------------------------


#----teste com método 'average'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='average')
y_hr = clustering.fit_predict(X)
plt.subplot(2,2,3)
plt.scatter(X[:,0], X[:,1], c=y_hr, cmap='viridis', s=50)
plt.title('critérios de clusterização: average linkage')
#---------------------------------------

#----teste com método 'single'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='single')
y_hr = clustering.fit_predict(X)
plt.subplot(2,2,4)
plt.scatter(X[:,0], X[:,1], c=y_hr, cmap='viridis', s=50)
plt.title('critérios de clusterização: single linkage')
#---------------------------------------
plt.show()

#------testando com outra dataset-------------
k = 4
n = 200
data = make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=1.5, random_state=50)
X = data[0]

plt.figure(figsize=(10, 5))
#----teste com método 'ward'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='ward')
y_km = clustering.fit_predict(X)
plt.subplot(2,2,1)
plt.scatter(X[:,0], X[:,1], c=y_km, cmap='viridis', s=50)
plt.title('critérios de clusterização: ward')
#---------------------------------------

#----teste com método 'complete'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='complete')
y_km = clustering.fit_predict(X)
plt.subplot(2,2,2)
plt.scatter(X[:,0], X[:,1], c=y_km, cmap='viridis', s=50)
plt.title('critérios de clusterização: complete linkage')
#---------------------------------------

#----teste com método 'average'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='average')
y_km = clustering.fit_predict(X)
plt.subplot(2,2,3)
plt.scatter(X[:,0], X[:,1], c=y_km, cmap='viridis', s=50)
plt.title('critérios de clusterização: average linkage')
#---------------------------------------


#----teste com método 'single'-----------
clustering = cluster.AgglomerativeClustering(n_clusters=k, linkage='single')
y_km = clustering.fit_predict(X)
plt.subplot(2,2,4)
plt.scatter(X[:,0], X[:,1], c=y_km, cmap='viridis', s=50)
plt.title('critérios de clusterização: single linkage')
#---------------------------------------
plt.show()