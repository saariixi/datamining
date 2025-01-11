
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import zipfile

zip_file_path = 'C:/Dokumentumok/egyetem/felmondok/student.zip'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.printdir()
    
    file_name = 'student-mat.csv'
    with zip_ref.open(file_name) as my_file:
        data = pd.read_csv(my_file, sep=';')

print(data.head())

X = data.select_dtypes(include=np.number)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
data['Klaszter'] = kmeans.labels_

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='Klaszter', data=data,
    palette='viridis', s=100
)
plt.title('Diákok klaszterei (PCA után)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title='Klaszter')
plt.show()

cluster_analysis = data.groupby('Klaszter').mean()
print("\nKlaszterek jellemzői:")
print(cluster_analysis)

