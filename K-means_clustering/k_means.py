import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


df = pd.read_csv("customer_churn_prediction_dataset.csv")

df = df.drop(columns=["Churn","customerID"])

print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.shape)

df_encoded = pd.get_dummies(df, columns= ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"], drop_first=True)
print(df_encoded.shape)

scaler = StandardScaler()
df_scaled = pd.DataFrame( scaler.fit_transform(df_encoded) , columns= df_encoded.columns)
print(df_scaled.describe())

WCSS = []
for i in range (1,11):
    Kmeans = KMeans(n_clusters=i, init = "k-means++", random_state=42)
    Kmeans.fit(df_scaled)
    WCSS.append(Kmeans.inertia_) 

plt.plot(range(1,11),WCSS)
plt.xlabel("Number of CLuster")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.savefig("Elbow_Method.png")
plt.show()

kmeans_3 = KMeans(n_clusters =3, init= "k-means++", random_state=42)
labels_3 = kmeans_3.fit_predict(df_scaled)

df['Cluster_3'] = labels_3
print(df.groupby('Cluster_3')[['tenure','MonthlyCharges','TotalCharges']].mean())
print("Silhouette Score (k=3):", silhouette_score(df_scaled, labels_3))

kmeans_6 = KMeans(n_clusters=6 , init = 'k-means++', random_state=42)
labels_6 = kmeans_6.fit_predict(df_scaled)
df['Cluster_6'] = labels_6
print(df.groupby('Cluster_6')[['tenure','MonthlyCharges','TotalCharges']].mean())
print("Silhouette Score (K=6):", silhouette_score(df_scaled, labels_6))
