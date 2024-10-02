from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

file_path = '/home/jiyli/Data/Image_Attack/icey/ass1_24s2/modified_cleaned_data.csv'
df = pd.read_csv(file_path)
features = "Total_Spent_Nov23_May24"
predictors = df.drop(columns=[features])
x_train, x_test, y_train, y_test = train_test_split(predictors, df[features], test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor model
dt_model = DecisionTreeRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'min_samples_leaf': list(range(1, 10)),
    'min_samples_split': list(range(1, 8)),
    'max_depth': range(1, 6)
}

# Perform grid search with cross-validation to find the best parameters
dt_cv = GridSearchCV(dt_model, param_grid, cv=5, scoring='neg_mean_squared_error')
dt_cv.fit(x_train, y_train)

# Get the best estimator and make predictions
dt_best = dt_cv.best_estimator_
y_pred_dt = dt_best.predict(x_test)

# Calculate performance metrics
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

# Output the results
print(f"Best Decision Tree parameters: {dt_cv.best_params_}")
print(f"Decision Tree - MAE: {mae_dt:.2f}")
print(f"Decision Tree - RMSE: {rmse_dt:.2f}")
print(f"Decision Tree - R²: {r2_dt:.2f}")
