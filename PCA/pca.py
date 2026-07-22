from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("customer_churn_prediction_dataset.csv")
df = df.drop(columns=['customerID','Churn'])

df_encoded = pd.get_dummies(df, columns= ["gender","Partner",
                                          "Dependents","PhoneService",
                                          "MultipleLines","InternetService",
                                          "OnlineSecurity","OnlineBackup",
                                          "DeviceProtection","TechSupport",
                                          "StreamingTV","StreamingMovies",
                                          "Contract","PaperlessBilling",
                                          "PaymentMethod"], drop_first=True)
print(df_encoded.shape)

#required before PCA since it's sensitive to scale
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
print(df_scaled.describe())

# Fit PCA with all components
pca = PCA()
pca.fit(df_scaled)
print(pca.explained_variance_ratio_) # variance explained by each individual component
print(pca.explained_variance_ratio_.cumsum()) # running total — helps decide how many components to keep


pca_kmeans = PCA(n_components=10)
df_pca_kmeans = pca_kmeans.fit_transform(df_scaled)
print(df_pca_kmeans.shape)

kmeans_3 = KMeans(n_clusters=3, init="k-means++", random_state=42)
labels_3 = kmeans_3.fit_predict(df_pca_kmeans)
df['Cluster_3'] = labels_3
print(df.groupby('Cluster_3')[['tenure','MonthlyCharges','TotalCharges']].mean())
print("Silhouette Score(k=3):", silhouette_score(df_pca_kmeans, labels_3))
print(df["Cluster_3"].value_counts())


for k in [2,3,4,5]:
    km = KMeans(n_clusters=k, init="k-means++",random_state=42)
    labels = km.fit_predict(df_pca_kmeans)
    score = silhouette_score(df_pca_kmeans,labels)
    print(f"K={k}: Silhouette Score = {score:.4f}")


pca_viz = PCA(n_components=2)
df_pca_viz = pca_viz.fit_transform(df_scaled)
