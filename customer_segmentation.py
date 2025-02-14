import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load the dataset (make sure the file is in the same directory)
df = pd.read_csv("Mall_Customers.csv")

# Quick peek at the data
print("\nInitial Data Preview:")
print(df.head())

# Fix column names for easier access
df.rename(columns={'Genre': 'Gender', 'Annual Income (k$)': 'AnnualIncome', 'Spending Score (1-100)': 'SpendingScore'},
          inplace=True)
df.columns = ['CustomerID', 'Gender', 'Age', 'AnnualIncome', 'SpendingScore']
df.drop('CustomerID', axis=1, inplace=True)  # Drop CustomerID since it's not useful for clustering

# Convert income from thousands to full dollar values
df['AnnualIncome'] *= 1000

# Double-check transformations
print("\nUpdated Column Names:")
print(df.columns)

print("\nSample Data After Transformation:")
print(df.head())

# Convert gender to numeric values (Male -> 0, Female -> 1)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for missing data types
print("\nColumn Data Types:")
print(df.dtypes)

# Scale the features so they work better with clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Find the best number of clusters using the Elbow Method
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow method to help determine the best K
plt.figure(figsize=(8, 5))

plt.plot(K_range, inertia, marker='o')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.title('Elbow Method for Optimal K')

plt.show(block=False)  # Continue execution without waiting

plt.pause(2)  # Small delay to ensure the plot displays properly

# Choose the best K (based on the elbow method)
optimal_k = 5  # Adjust this if needed
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Show customer clusters based on income and spending score
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['AnnualIncome'], y=df['SpendingScore'], hue=df['Cluster'], palette='viridis')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()


# Streamlit dashboard to interact with the data
def main():
    st.title("Customer Segmentation Dashboard")
    st.write("Explore different customer clusters based on selected features.")

    cluster_choice = st.selectbox("Choose a feature to compare with Spending Score",
                                  ["AnnualIncome", "Age", "SpendingScore"])

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[cluster_choice], y=df['SpendingScore'], hue=df['Cluster'], palette='coolwarm')
    plt.xlabel(cluster_choice)
    plt.ylabel("Spending Score")
    plt.title(f"Customer Segmentation Based on {cluster_choice}")
    st.pyplot(plt)


if __name__ == "__main__":
    main()

