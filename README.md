# Chess-Game-Analysis
Project Description
This project involves the analysis of chess game data using Python. It includes data loading, preliminary analysis, data visualization, and the identification of anomalies in move evaluations. The project utilizes libraries such as Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and others for data processing and analysis.

Project Structure
Library Import: Import necessary libraries for data manipulation, visualization, and machine learning.

Data Loading: Load data from a CSV file and display the first few rows and descriptive statistics.

Preliminary Data Analysis: Check for the presence of required columns and remove rows with null values.

Move Evaluation Distribution Analysis: Visualize the distribution of move evaluations using Matplotlib and Seaborn.

Anomaly Detection: Calculate differences in move evaluations and Z-scores to identify anomalies.

Anomaly Visualization: Visualize anomalous changes in move evaluations.

Installation and Running
Install Dependencies: Ensure you have all necessary libraries installed. You can install them using pip:

bash
Copy
pip install pandas numpy matplotlib seaborn scikit-learn
Data Loading: Place the data file in CSV format in the project's data folder.

Run Jupyter Notebook: Open Jupyter Notebook and run the chess_analysis.ipynb file.

Usage
Data Loading: Execute the first cell to load data from the CSV file.

Preliminary Analysis: Run the following cells for preliminary data analysis.

Move Evaluation Distribution Analysis: Execute the cell to visualize the distribution of move evaluations.

Anomaly Detection and Visualization: Run the remaining cells to detect and visualize anomalies in move evaluations.

Example Results
Move Evaluation Distribution: A histogram showing the distribution of move evaluations.

Anomalous Changes: A table and plot showing games with anomalous changes in move evaluations.

Code Examples
Library Import
python
Copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
Data Loading
python
Copy
data = pd.read_csv('data/moves.csv')
print(data.head())
print(data.describe())
Move Evaluation Distribution Analysis
python
Copy
plt.figure(figsize=(10, 6))
sns.histplot(data['Real Move Eval'], bins=30, kde=True)
plt.title('Move Evaluation Distribution')
plt.xlabel('Evaluation')
plt.ylabel('Frequency')
plt.show()
Anomaly Detection
python
Copy
data['evaluation_diff'] = data['Real Move Eval'].diff()
mean_diff = data['evaluation_diff'].mean()
std_diff = data['evaluation_diff'].std()
data['z_score'] = (data['evaluation_diff'] - mean_diff) / std_diff
anomalies = data[data['z_score'].abs() > 3]
print(anomalies)
