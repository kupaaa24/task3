# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Step 1: Create a sample dataset
np.random.seed(42)
data = {
    'Feature1': np.random.rand(100) * 100,  # Random values for feature 1
    'Feature2': np.random.rand(100) * 50,   # Random values for feature 2
    'Label': np.random.choice([0, 1], size=100)  # Binary labels for classification
}
df = pd.DataFrame(data)

# Step 2: Scatter plot with labeled points
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['Feature1'], df['Feature2'], c=df['Label'], cmap='coolwarm', label=df['Label'])

# Adding labels to each data point
for i in range(len(df)):
    plt.text(df['Feature1'][i], df['Feature2'][i], f'({df["Feature1"][i]:.2f}, {df["Feature2"][i]:.2f})', fontsize=8)

plt.colorbar(scatter)
plt.title('Scatter plot with labeled data points')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 3: Linear regression and regression line overlay
X = df[['Feature1']].values
y = df['Feature2'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the values
y_pred = model.predict(X_test)

# Plotting actual data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title('Scatter plot with regression line')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Step 4: Statistical overlay with Seaborn (Joint plot with regression line and KDE)
sns.jointplot(x='Feature1', y='Feature2', data=df, kind='reg', color='purple', height=8)
plt.show()

# Step 5: KMeans clustering integration
kmeans = KMeans(n_clusters=2)
df['Cluster'] = kmeans.fit_predict(df[['Feature1', 'Feature2']])

# Plotting the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='coolwarm')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
