
# Child Growth Prediction: Natural vs. Supplement Diet Analysis
Supervised Machine Learning | Regression Analysis | Nutritional Science

# ➤ Project Overview :
This research project implements a Linear Regression framework to analyze and predict child growth rates ($cm/year$) based on dietary intake patterns. By comparing Natural Whole-Food Diets against Supplemented Diets, the model quantifies how different nutritional strategies influence physical development markers across various age groups.

# ➤Technical Architecture :
The project follows a rigorous statistical pipeline to ensure model generalizability and prevent overfitting:

◈ Core Engine: Ordinary Least Squares (OLS) Regression
◈ Validation Strategy: 5-Fold Cross-Validation (via cross_val_score)
◈ Data Split: 80/20 Train-Test partition with a fixed random_state=42 for reproducibility
◈ Inference: Predictive mapping of "Hold-out" data to verify real-world accuracy

# ➤Feature Engineering & Variables :
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

#➤
