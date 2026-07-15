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

# pd.get_dummies() converts categorical (text) columns into 0/1 binary columns (one-hot encoding)
# drop_first=True removes the first category of each column to avoid redundancy
# (e.g., if gender only has Male/Female, we only need one column like gender_Male;
# if it's 0, we already know it means Female — keeping both would be duplicate information)
df_encoded = pd.get_dummies(df, columns= ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"], drop_first=True)
print(df_encoded.shape)

scaler = StandardScaler()
# fit_transform() first "fits" the scaler (calculates mean and std of each column),
# then "transforms" the data using those values (returns a NumPy array, not a DataFrame)
# pd.DataFrame(...) wraps that array back into a DataFrame so we keep our original column names
df_scaled = pd.DataFrame( scaler.fit_transform(df_encoded) , columns= df_encoded.columns)
print(df_scaled.describe())

#numerical column only
numeric_cols = ['tenure','MonthlyCharges', 'TotalCharges','SeniorCitizen']
df_numeric = df[numeric_cols]

# init="k-means++" is a smarter way to pick initial cluster centers
# (spreads them out from each other) instead of placing them randomly,
# which usually leads to faster, more reliable convergence
scaler_numeric = StandardScaler()
df_numeric_Scaled = pd.DataFrame(scaler_numeric.fit_transform(df_numeric), columns=numeric_cols)
print(df_numeric_Scaled.describe())

kmeans_3 = KMeans(n_clusters=3, init ="k-means++", random_state=42)
labels_3 = kmeans_3.fit_predict(df_numeric_Scaled)
df['Cluster_3'] = labels_3
print(df.groupby('Cluster_3')[['tenure','MonthlyCharges','TotalCharges','SeniorCitizen']].mean())
print("Silhouette Score (k=3):", silhouette_score(df_numeric_Scaled, labels_3))

# c=df['Cluster_3'] assigns a different color to each cluster (0, 1, 2)
# cmap='viridis' sets the color palette used for those cluster colors
plt.scatter(df['tenure'],df['TotalCharges'], c=df['Cluster_3'], cmap= 'viridis')
plt.xlabel("Tenure")
plt.ylabel("Total Charges")
plt.title("Customer Cluster (k=3)")
plt.savefig("Tenure_TotalCharges(k=3).png")
plt.show()

WCSS = []
for i in range (1,11):
    Kmeans = KMeans(n_clusters=i, init = "k-means++", random_state=42)
    Kmeans.fit(df_scaled)
     # kmeans.inertia_ gives the WCSS (Within-Cluster Sum of Squares) for this K —
    # a measure of how tightly packed the clusters are (lower = tighter clusters)
    WCSS.append(Kmeans.inertia_) 

plt.plot(range(1,11),WCSS)
plt.xlabel("Number of CLuster")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.savefig("Elbow_Method.png")
plt.show()

# kmeans_3 = KMeans(n_clusters =3, init= "k-means++", random_state=42)
# labels_3 = kmeans_3.fit_predict(df_scaled)

# df['Cluster_3'] = labels_3
# print(df.groupby('Cluster_3')[['tenure','MonthlyCharges','TotalCharges']].mean())
# print("Silhouette Score (k=3):", silhouette_score(df_scaled, labels_3))

# kmeans_6 = KMeans(n_clusters=6 , init = 'k-means++', random_state=42)
# labels_6 = kmeans_6.fit_predict(df_scaled)
# df['Cluster_6'] = labels_6
# print(df.groupby('Cluster_6')[['tenure','MonthlyCharges','TotalCharges']].mean())
# print("Silhouette Score (K=6):", silhouette_score(df_scaled, labels_6))
