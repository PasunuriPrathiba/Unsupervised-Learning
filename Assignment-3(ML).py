#!/usr/bin/env python
# coding: utf-8

# In[235]:


#1. Clustering:


# In[236]:


#Part-(a)


# In[237]:


import numpy as np # linear algebra
import pandas as pd # data processing


from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN


# In[238]:


df = pd.read_csv("/home/prathiba/Downloads/DataClustering.csv")
df.head()


# In[239]:


df.shape


# In[240]:


df.info()


# In[241]:


df.describe()


# In[242]:


#correlation
df_corr = df.corr()
df.corr()


# In[243]:


df_corr.style.background_gradient(cmap='coolwarm', axis=None)


# In[244]:


plt.rcParams['figure.figsize'] = (18, 8)

plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['x1'])
plt.title('Distribution of x1', fontsize = 20)
plt.xlabel('Range of x1')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(df['x2'])
plt.title('Distribution of x2', fontsize = 20)
plt.xlabel('Range of x2')
plt.ylabel('Count')


# In[245]:


plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(df['x3'])
plt.title('Distribution of x3', fontsize = 20)
plt.xlabel('Range of x3')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(df['x4'])
plt.title('Distribution of x4', fontsize = 20)
plt.xlabel('Range of x4')
plt.ylabel('Count')


# In[246]:


# From above plots it is clear that all features are left skewed.
#To get Gaussian like distribution we are using log transformation.


# In[247]:


data = np.log(df)


# In[248]:


data.head()


# In[249]:


plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(data['x1'])
plt.title('Distribution of x1', fontsize = 20)
plt.xlabel('Range of x1')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(data['x2'])
plt.title('Distribution of x2', fontsize = 20)
plt.xlabel('Range of x2')
plt.ylabel('Count')


# In[250]:


plt.subplot(1, 2, 1)
sns.set(style = 'whitegrid')
sns.distplot(data['x3'])
plt.title('Distribution of x3', fontsize = 20)
plt.xlabel('Range of x3')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.set(style = 'whitegrid')
sns.distplot(data['x4'])
plt.title('Distribution of x4', fontsize = 20)
plt.xlabel('Range of x4')
plt.ylabel('Count')


# In[251]:


sns.pairplot(data)
plt.title('Pairplot for the Data', fontsize = 20)
plt.show()


# In[252]:


plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(data.corr(), cmap = 'Wistia', annot = True)
plt.title('Heatmap for the Data', fontsize = 20)
plt.show()


# In[253]:


#The Above Graph Showing the correlation between the different features of the dataset.
#This Heat map reflects the most correlated features with Orange Color and least correlated features with yellow color.
#The correlation among the features is less than 0.5.
#We can clearly see that these attributes do not have good correlation among them, 
#that's why we will proceed with all of the features.


# In[254]:


#Part-(b)
#Train k-means, and find the appropriate number of k.


# In[255]:


#Using t-SNE we can reduce our high dimensional features vector to 2 dimensions.
#By using the 2 dimensions as x,y coordinates, the datset can be plotted.

#t-Distributed Stochastic Neighbor Embedding (t-SNE) 
#reduces dimensionality while trying to keep similar instances close and dissimilar instances apart. 
#It is mostly used for visualization, in particular to visualize clusters of instances in high-dimensional space.


# In[256]:


#Dimentionality Reduction with t-SNE :-


# In[257]:


from sklearn.manifold import TSNE

tsne = TSNE(verbose=1, perplexity=100, random_state=42)
data_embedded = tsne.fit_transform(data)


# In[258]:


data_embedded.shape


# In[259]:


data_embedded


# In[260]:


inertia = []
for n in range(1 , 11):
    model = KMeans(n_clusters=n, random_state=21, algorithm='elkan')
    model.fit(data_embedded)
    inertia.append(model.inertia_)


# In[261]:


plt.figure(figsize = (12 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o', c=sns.xkcd_rgb['red pink'])
plt.plot(np.arange(1 , 11) , inertia , '-' ,c=sns.xkcd_rgb['greenish teal'], alpha = 0.8)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('Number of Clusters') 
plt.ylabel('Inertia')
plt.show()


# In[262]:


#Number of clusters = 5


# In[263]:


model = KMeans(n_clusters=5, random_state= 21, algorithm='elkan')
model.fit(data_embedded)
labels = model.labels_


# In[264]:


fig, axs = plt.subplots(figsize=[10,10])
sns.scatterplot(x=data_embedded[:,0],
                y=data_embedded[:,1],
                hue=labels,
                palette=sns.color_palette('husl', 5),
                edgecolor=None,
                alpha=0.8,
                ax=axs)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clusters')
plt.show()


# In[265]:


k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(data)
df['y'] = y_pred


# In[266]:


#Number of clusters = 6


# In[267]:


model = KMeans(n_clusters=6, random_state= 21, algorithm='elkan')
model.fit(data_embedded)
labels = model.labels_


# In[268]:


fig, axs = plt.subplots(figsize=[10,10])
sns.scatterplot(x=data_embedded[:,0],
                y=data_embedded[:,1],
                hue=labels,
                palette=sns.color_palette('husl', 6),
                edgecolor=None,
                alpha=0.8,
                ax=axs)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clusters')
plt.show()


# In[269]:


#Part-(c)
#Train DBSCAN, and see if by varying MinPts and ε, you can get the same number of clusters as k-means


# In[270]:


model = DBSCAN(eps=0.5, min_samples=10)
labels = model.fit_predict(data)


# In[271]:


np.unique(labels)


# In[272]:


fig, axs = plt.subplots(figsize=[10,10])
sns.scatterplot(x=data_embedded[:,0],
                y=data_embedded[:,1],
                hue=labels,
                palette=sns.xkcd_palette([ 'black', 'cyan', 'red pink', 'amber', 'purple']),
                edgecolor=None,
                alpha=0.8,
                ax=axs)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters')
plt.show()


# In[273]:


#Part-(d)
#Using the cluster assignment as the label, visualize the t-sne embedding.


# In[274]:


# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.color_palette("bright", 1)

# plot
sns.scatterplot(data_embedded[:,0], data_embedded[:,1], palette=palette)
plt.title('t-SNE with no Labels')
plt.show()


# In[275]:


#This looks pretty bland. There are some clusters we can immediately detect, 
#but the many instances closer to the center are harder to separate.
#t-SNE did a good job at reducing the dimensionality, but now we need some labels.
#Let's use the clusters found by k-means as labels.
#This will help visually separate different concentrations of topics


# In[276]:


# sns settings
sns.set(rc={'figure.figsize':(15,15)})

# colors
palette = sns.hls_palette(5, l=.4, s=.9)

# plot
sns.scatterplot(data_embedded[:,0], data_embedded[:,1], hue=y_pred, legend='full', palette=palette)
plt.title('t-SNE with Kmeans Labels')
plt.savefig("improved_cluster_tsne.png")
plt.show()


# In[277]:


df.head(100)


# In[278]:


#Now we can see that cluster numbers also added to the dataset


# In[279]:


#2. PCA:


# In[280]:


#part-(a) Visualize the data from the file DataPCA.csv.


# In[281]:


import numpy as np # linear algebra
import pandas as pd # data processing


from warnings import filterwarnings
filterwarnings("ignore")

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


# In[282]:


df1 = pd.read_csv('/home/prathiba/Downloads/DataPCA.csv')


# In[283]:


df1.head()


# In[284]:


df1.shape


# In[285]:


df1.info()


# In[286]:


df1.describe()


# In[287]:


#Correlation
df1_corr = df1.corr()
df1_corr


# In[288]:


df1_corr.style.background_gradient(cmap='coolwarm', axis=None)


# In[289]:


#we can see that data is highly correlated.


# In[290]:


sns.pairplot(df1)
plt.title('Pairplot for the Data', fontsize = 20)
plt.show()


# In[291]:


plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df1.corr(), cmap = 'Wistia', annot = True)
plt.title('Heatmap for the Data', fontsize = 20)
plt.show()


# In[292]:


#part-(b) Train PCA.


# In[293]:


# PCA performs best with a normalized feature set.
#We will perform standard scalar normalization to normalize our feature set.


# In[294]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df1 = sc.fit_transform(df1)


# In[295]:


df1


# In[296]:


#Define two components  
pca=PCA(n_components=2) 
principalComponents=pca.fit_transform(df1) 
principalDf=pd.DataFrame(data=principalComponents,columns=['principal component 1','principal component 2']) 
principalDf.head()


# In[297]:


print(pca.components_)


# In[298]:


#Let's see how much we can reduce the dimensions while still keeping 95% variance.
#We will apply Principle Component Analysis (PCA) to our datassest. 
#The reason for this is that by keeping a large number of dimensions with PCA, 
#we don’t destroy much of the information, but hopefully will remove some noise/outliers from the data,
#and make the clustering problem easier.


# In[299]:


explained_variance = pca.explained_variance_ratio_


# In[300]:


print(explained_variance)


# In[301]:


#It can be seen that first principal component is responsible for 60.9% variance. 
#Similarly, the second principal component causes 31.9% variance in the dataset. 
#Collectively we can say that (60.9 + 31.9) 92.8% percent of the information contained
#in the feature set is captured by the first two principal components.


# In[302]:


#part-(c):-plot the variance explained versus PCA dimensions.


# In[303]:


pca = PCA().fit(principalDf)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# In[304]:


plt.plot(principalDf.columns, explained_variance,)
plt.xlabel('number of components')
plt.ylabel('explained variance')


# In[305]:



percentage_var_explained = pca.explained_variance_ratio_;  
cum_var_explained=np.cumsum(percentage_var_explained)
#plot PCA spectrum   
plt.figure(1,figsize=(6,4))
plt.clf()  
plt.plot(explained_variance, pca.components_, linewidth=2)  
plt.axis('tight')  
plt.grid() 
plt.xlabel('n_components') 
plt.ylabel('Variance_explained')  
plt.show()


# In[306]:


#part-(d):-Reconstruct the data with various numbers of PCA dimensions, and compute the MSE.


# In[307]:


df_new = pca.inverse_transform(principalDf)


# In[308]:


df_new.shape


# In[309]:


from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error



components = [2,3,4,5,6]    
for n in components:
    pca = PCA(n_components=n)
    recon = pca.inverse_transform(pca.fit_transform(df1))
    rmse = mean_squared_error(df1[0], recon[0],squared=False)
    print("RMSE: {} with {} components".format(rmse, n))


# In[310]:


#part-(3). Non-linear dimension reduction:
#part-(a). Visualize the data from the file DataKPCA.csv.


# In[311]:


df2 = pd.read_csv("/home/prathiba/Downloads/DataKPCA.csv")
df2.head()


# In[312]:


df2.shape


# In[313]:


df2.info()


# In[314]:


df2.describe()


# In[315]:


#Correlation
df2_corr = df2.corr()
df2_corr


# In[316]:


df2_corr.style.background_gradient(cmap='coolwarm', axis=None)


# In[331]:


sns.pairplot(df2)
plt.title('Pairplot for the Data', fontsize = 20)
plt.show()


# In[318]:


plt.rcParams['figure.figsize'] = (15, 8)
sns.heatmap(df2.corr(), cmap = 'Wistia', annot = True)
plt.title('Heatmap for the Data', fontsize = 20)
plt.show()


# In[319]:


#From above Heat map, it is clear that the variables are highly correlated.


# In[320]:


#Part-(b). Train KPCA.


# In[321]:


from sklearn.decomposition import PCA,SparsePCA,KernelPCA,NMF
from sklearn.datasets import make_circles


# In[322]:


KPCA = KernelPCA(n_components = 10, kernel="rbf", fit_inverse_transform=True, gamma=10)
KPCA_fit = KPCA.fit(df2)
X_KPCA = KPCA.fit_transform(df2)
X_KPCA=pd.DataFrame(data=X_KPCA,columns=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10']) 
#X_KPCA_back = KPCA.inverse_transform(X_KPCA)


# In[323]:


X_KPCA.shape


# In[324]:


X_KPCA.head()


# In[325]:


#Part-(C). Plot the variance explained versus KPCA dimensions for up to 10 dimensions.


# In[326]:


expl_var_kpca = np.var(X_KPCA, axis=0)


# In[327]:


expl_var_ratio_kpca = expl_var_kpca / np.sum(expl_var_kpca)


# In[328]:


print(X_KPCA.columns)


# In[329]:



plt.plot(np.cumsum(expl_var_kpca),X_KPCA.columns)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[330]:


plt.plot(X_KPCA.columns,expl_var_kpca)
plt.xlabel('number of components')
plt.ylabel('explained variance')

