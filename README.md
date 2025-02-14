Customer Segmentation with K-Means Clustering

Overview

This project applies K-Means clustering to segment customers based on their Annual Income, Age, and Spending Score. 
It uses Python, Pandas, Scikit-Learn, Seaborn, and Matplotlib for data analysis and visualization.

# Features

  - Data Preprocessing: Cleans and transforms customer data for clustering.
  - Elbow Method: Determines the optimal number of clusters.
  - K-Means Clustering: Segments customers based on similar behaviors.
  - Visualization: Scatter plots show distinct customer groups.


## Requirements

If You Don’t Have a Python IDE

  - Download Python: Python Download

  - Install PyCharm (Recommended IDE): PyCharm Download

Install Required Libraries
  - pip install pandas numpy matplotlib seaborn scikit-learn ( In IDE terminal input this line, however if your using Pycharm it will let you download it by hovering over the line of code importing the library) 

Ensure that the Mall_Customers.csv file is in the same folder as the Python script (customer_segmentation.py).
The dataset contains customer information such as:
  - Customer ID (removed for clustering)
  - Gender (converted to numeric values)
  - Age
  - Annual Income (scaled for better clustering)
  - Spending Score (used to identify customer segments)


### Output & Visualizations

  - Elbow Method Plot: Helps choose the optimal number of clusters.

  - Customer Segments Scatter Plot: Visualizes clusters by income and spending score.
