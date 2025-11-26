from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load dataset and train models when app starts
df = pd.read_csv("stroke_data.csv")   # <-- Put your dataset name here

# Preprocessing
df.replace(["Unknown", "unknown", ""], np.nan, inplace=True)
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

X = df.drop("stroke", axis=1)
y = df["stroke"].astype(int)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Models
rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
xgb = XGBClassifier(eval_metric="logloss", random_state=42)

rf_pipe = ImbPipeline([
    ("pre", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", rf)
])

xgb_pipe = ImbPipeline([
    ("pre", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", xgb)
])

# Cross-validation to choose best model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_score = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc").mean()
xgb_score = cross_val_score(xgb_pipe, X, y, cv=cv, scoring="roc_auc").mean()

# Train the final best model
best_model = rf_pipe
#by our prediction we choose the best model based on cross-validation scores
#best_model = xgb_pipe if xgb_score > rf_score else rf_pipe
best_model.fit(X_train, y_train)

# Flask App
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get form inputs
    data = {
        "gender": request.form["gender"],
        "age": float(request.form["age"]),
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "ever_married": request.form["ever_married"],
        "work_type": request.form["work_type"],
        "Residence_type": request.form["Residence_type"],
        "avg_glucose_level": float(request.form["avg_glucose_level"]),
        "bmi": float(request.form["bmi"]),
        "smoking_status": request.form["smoking_status"]
    }

    df_input = pd.DataFrame([data])

    # Prediction
    prediction = best_model.predict(df_input)[0]
    probability = best_model.predict_proba(df_input)[0][1]

    return render_template("result.html",
                           pred=int(prediction),
                           prob=round(probability, 3))


if __name__ == "__main__":
    app.run(debug=True)
