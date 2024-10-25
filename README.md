==========================================================
                  Titanic Survival Prediction
==========================================================

This project focuses on predicting the survival of passengers on the Titanic using machine learning techniques. The dataset used is derived from the Titanic dataset on Kaggle, and the model employed is `XGBClassifier` from the XGBoost library.

----------------------------------------------------------
Directory Structure:
----------------------------------------------------------

1. **data/**
   - `train.csv`        : Training data containing features and labels.
   - `x_test.csv`       : Processed test data used for predictions.

2. **notebooks/**
   - `Titanic Survival Prediction.ipynb` : Jupyter Notebook containing code for data preprocessing, model training, and evaluation.

3. **models/**
   - `xgb_model.pkl` : Saved trained model.

4. **README.txt**    : This file, providing an overview of the project.

5. **requirements.txt** : List of required Python packages to run the project.

----------------------------------------------------------
Dataset Information:
----------------------------------------------------------

The dataset includes details on passengers like age, sex, class, family size, and more. The target variable is whether the passenger survived (`0` for No, `1` for Yes).

Key columns:
- PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

----------------------------------------------------------
How to Use the Project:
----------------------------------------------------------

1. **Installation:**
   a. Clone the repository:
      ```
      git clone https://github.com/your-username/titanic-survival-prediction.git
      ```
   b. Navigate to the project directory:
      ```
      cd titanic-survival-prediction
      ```
   c. Install dependencies:
      ```
      pip install -r requirements.txt
      ```

2. **Running the Notebook:**
   Open `notebooks/Titanic Survival Prediction.ipynb` in Jupyter Notebook or Jupyter Lab to see the full workflow of the project, from data preprocessing to model training and evaluation.

3. **Making Predictions on New Data:**
   - Create a new data file matching the training feature format.
   - Use the pre-trained `xgb_model.pkl` and preprocessing pipeline to predict:
     ```python
     import pandas as pd
     from joblib import load

     # Load the model
     model = load("models/xgb_model.pkl")

     # Example: Loading new data for prediction
     new_data = pd.DataFrame({...})  # Replace with actual data

     # Preprocess using the existing pipeline
     new_data_processed = preprocessor.transform(new_data)

     # Predict
     prediction = model.predict(new_data_processed)
     print(f"Prediction: {prediction}")
     ```

4. **Evaluation:**
   - Ensure `y_test.csv` matches `x_test.csv` data rows.
   - Load `x_test.csv` and `y_test.csv` to evaluate model accuracy:
     ```python
     from sklearn.metrics import accuracy_score
     
     # Load data
     X_test = pd.read_csv("data/x_test.csv")
     y_test = pd.read_csv("data/y_test.csv")

     # Preprocess and predict
     X_test_processed = preprocessor.transform(X_test)
     y_pred = model.predict(X_test_processed)

     # Calculate accuracy
     accuracy = accuracy_score(y_test, y_pred)
     print(f"Accuracy on X_test: {accuracy:.2f}")
     ```

----------------------------------------------------------
Feature Engineering:
----------------------------------------------------------

1. **Family_size** = SibSp + Parch + 1
2. **isAlone**     = 1 if Family_size == 1, else 0

----------------------------------------------------------
Important Notes:
----------------------------------------------------------

- **Ensure Consistency**: All data processed for predictions must go through the same preprocessing steps as training data.
- **Parameters**: For any changes, make sure they align with the training pipeline settings to prevent feature mismatches.
- **Ensure Data Availability**: Make sure paths to CSV files are correctly set up.

----------------------------------------------------------
Dependencies:
----------------------------------------------------------

- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- jupyter

Install them using:
