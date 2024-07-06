import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

df.rename(index=str, columns={'Annual Income (k$)' : 'Income',
                              'Spending Score (1-100)' : 'Score'
                              }, inplace=True)


x = df.drop(['CustomerID', 'Gender'], axis=1)

st.header("isi dataset")
st.write(x)

# menentukan panah elbow
clusters=[]
for i in range(1,11):
    km =KMeans(n_clusters=i).fit(x)
    clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=clusters, ax=ax)
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')

#panah elbow
ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 40000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=3))
ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=3))


st.set_option('deprecation.showPyplotGlobalUse', False)
elbow_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah k")
clust = st.sidebar.slider("Pilih jumlah : ", 2, 10, 3, 1)

def k_mean(n_clust): 
    n_clust = 4
kmean = KMeans(n_clusters=n_clust).fit(x)
x['labels'] = kmean.labels_

plt.figure(figsize=(10, 8))
sns.scatterplot(x='Income', y='Score', hue='labels', style='labels', size='labels', palette=sns.color_palette('hls', n_clust), data=x)

for label in x['labels'].unique():
    plt.annotate(label,
             (x[x['labels'] == label]['Income'].mean(),
              x[x['labels'] == label]['Score'].mean()),
             horizontalalignment='center',
             verticalalignment='center',
             size=20, weight='bold',
             color='black')

plt.show()

st.header(cluster plot)
st.pyplot()
st.write(x)

k_mean