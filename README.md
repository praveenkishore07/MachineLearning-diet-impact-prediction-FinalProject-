
# Child Growth Prediction: Natural vs. Supplement Diet Analysis
Supervised Machine Learning | Regression Analysis | Nutritional Science

# ➤ Project Overview :
This research project implements a Linear Regression framework to analyze and predict child growth rates ($cm/year$) based on dietary intake patterns. By comparing Natural Whole-Food Diets against Supplemented Diets, the model quantifies how different nutritional strategies influence physical development markers across various age groups.

# ➤ Technical Architecture :
The project follows a rigorous statistical pipeline to ensure model generalizability and prevent overfitting:
```
◈ Core Engine: Ordinary Least Squares (OLS) Regression
◈ Validation Strategy: 5-Fold Cross-Validation (via cross_val_score)
◈ Data Split: 80/20 Train-Test partition with a fixed random_state=42 for reproducibility
◈ Inference: Predictive mapping of "Hold-out" data to verify real-world accuracy
```

# ➤ Feature Engineering & Variables :
The model processes a multivariate input matrix to solve for growth velocity:

 Independent Variables (X) : Age_Years,Daily_Calorie_Intake,Diet_Type: (Natural vs. Supplemented).
 
 Dependent Variable (y): Growth_Rate_cm(centimeters gained per year).
 
# ➤ Tech Stack
Data Manipulation: Pandas (DataFrames) & NumPy (Vectorized Math)

Machine Learning: Scikit-Learn (LinearRegression, Metrics)

Visualization: Seaborn (Statistical Density) & Matplotlib (Regression Plotting)

# ➤ Evaluation Metrics :
The model performance is audited using two primary statistical indicators:

◈ Mean Squared Error (MSE): Quantifies prediction precision by penalizing large vertical deviations (residuals).

◈ R-Squared ($R^2$): The Coefficient of Determination, measuring the proportion of growth variance explained by the dietary features.

# ➤ Visual Insights : 
The repository includes advanced statistical visualizations:

1.Actual vs. Predicted Plot: A scatter plot with an Identity Line (y=x) to visualize the residual sum of squares and model "fit."

2.Growth Density (Violin Plot): Utilizes Kernel Density Estimation (KDE) to compare the probability distribution and medians of growth rates between diet groups.

# ➤ Program -
```
Developed by : A PRAVEEN KISHORE
```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Data Ingestion
df = pd.read_csv('child_growth_data.csv')
X = df[['Age_Years', 'Daily_Calorie_Intake', 'Diet_Type']]
y = df['Growth_Rate_cm']

# Model Initialization and Cross-Validation
model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Training and Testing Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation Results
print("--- Cross-Validation Results (5-Folds) ---")
print(f"Scores for each fold: {cv_scores}")
print(f"Average R2 Score: {np.mean(cv_scores):.4f}")
print(f"Score Stability (Std Dev): {np.std(cv_scores):.4f}")

print("\n--- Final Test Set Evaluation ---")
print(f"Final R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

# Visualization: Actual vs. Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, label='Perfect Fit')
plt.xlabel('Actual Growth Rate (cm/year)')
plt.ylabel('Predicted Growth Rate (cm/year)')
plt.title('Actual vs. Predicted Growth Rates')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Visualization: Growth Density (Violin Plot)
plt.figure(figsize=(8, 5))
sns.violinplot(x='Diet_Type', y='Growth_Rate_cm', data=df, palette='muted')
plt.xticks([0, 1], ['Natural Diet', 'Supplemented Diet'])
plt.title('Growth Density: Natural vs. Supplement')
plt.show()
```
# Evalution metrics :
<img width="801" height="228" alt="em" src="https://github.com/user-attachments/assets/ed6c6cc2-f7b7-4784-bd1b-24431feb4ed0" />

# Outcomes:
## Actual vs Predicted --
<img width="1140" height="690" alt="predictg" src="https://github.com/user-attachments/assets/813197fe-dbfc-4007-96af-ba0347089abc" />

## Growth Density --
<img width="957" height="612" alt="growthdensity" src="https://github.com/user-attachments/assets/56cdd7c4-a58f-4dbb-9189-87ffac14e124" />

# ➤ Conclusion :
This project successfully demonstrates that Supervised Learning (Regression) can effectively quantify the impact of nutritional sources on child development. The results suggest that while both diets support growth, [insert your specific conclusion, e.g., "natural diets provide a more consistent growth trajectory"]. This model can serve as a foundational tool for pediatric nutritional planning and caloric optimization.
