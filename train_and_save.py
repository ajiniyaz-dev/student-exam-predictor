import os
import numpy as np
import pandas as pd
import joblib
import kagglehub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# 1) Load dataset
path = kagglehub.dataset_download("lainguyn123/student-performance-factors")
df = pd.read_csv(os.path.join(path, "StudentPerformanceFactors.csv"))


# 2) Split X and y
X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]


# 3) Define ordinal columns + order
ordinal_cols = [
    "Parental_Involvement",
    "Access_to_Resources",
    "Motivation_Level",
    "Family_Income",
    "Teacher_Quality",
    "Parental_Education_Level"
]

ordinal_categories = [
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["Low", "Medium", "High"],
    ["High School", "College", "Postgraduate"]
]

# 4) Nominal + numeric columns
cat_cols = X.select_dtypes(include="object").columns
nominal_cols = [c for c in cat_cols if c not in ordinal_cols]
num_cols = X.select_dtypes(exclude="object").columns


# 5) Preprocessor (safe for missing + unknown)
preprocessor = ColumnTransformer(
    transformers=[
        ("ord", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ]), ordinal_cols),

        ("nom", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), nominal_cols),

        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ]
)


# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7) ElasticNet pipeline
enet_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", ElasticNet(random_state=42, max_iter=10000))
])


# 8) GridSearchCV fine tuning
param_grid_enet = {
    "regressor__alpha": np.logspace(-4, 1, 20),
    "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid_enet = GridSearchCV(
    enet_pipe,
    param_grid=param_grid_enet,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_enet.fit(X_train, y_train)

print("Best ElasticNet params:", grid_enet.best_params_)
print("Best CV R2:", grid_enet.best_score_)


# 9) Evaluate on test set
final_model = grid_enet.best_estimator_
y_pred = final_model.predict(X_test)

print("\n--- Test Performance (Final ElasticNet) ---")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))


# 10) Save model (this will match your VS Code sklearn version)
joblib.dump(final_model, "final_elasticnet_model.pkl")
print("\nSaved: final_elasticnet_model.pkl")
