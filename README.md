
# No Churn Telecom

## Project Overview
The "No Churn Telecom" project aims to predict customer churn for a telecom company. By analyzing customer behavior and historical data, this project builds a machine learning model to classify whether a customer is likely to churn or not.

## Workflow
1. **Data Collection**: Reads the dataset (`modified_output (1).csv`) containing customer information and churn labels.
2. **Data Preparation**:
    - Handles missing values using `SimpleImputer`.
    - Encodes categorical features with `LabelEncoder`.
    - Standardizes numerical features using `StandardScaler`.
3. **Exploratory Data Analysis (EDA)**:
    - Visualizes data distribution and relationships using `matplotlib` and `seaborn`.
4. **Model Training**:
    - Trains multiple classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
    - Hyperparameter tuning is performed using GridSearchCV.
    - Saves the best model (`RandomForestClassifier`) to a `.pkl` file.
5. **Model Evaluation**:
    - Computes metrics like accuracy, precision, recall, F1-score, and ROC AUC.

## How to Use
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the notebook (`NCT FINAL.ipynb`) to preprocess data, train the model, and evaluate results.
3. Use the provided model for prediction:
    ```python
    import joblib
    model = joblib.load('best_churn_model.pkl')
    prediction = model.predict(data)  # Replace `data` with your input features
    ```

## Results
- Metrics:
    - Accuracy: 85%
    - Precision: 83%
    - Recall: 80%
    - F1-score: 81.5%
    - ROC AUC: 87%
- Visualizations (stored in `results/plots/`):
    - `confusion_matrix.png`: Confusion matrix visualization.

## Files and Directories
- **`NCT FINAL.ipynb`**: Main notebook containing the code.
- **`best_churn_model.pkl`**: Saved model file.
- **`requirements.txt`**: List of dependencies.
- **`metrics.json`**: Evaluation metrics in JSON format.
- **`results/`**: Directory for storing plots and outputs.

## License
This project is licensed under the [MIT License](LICENSE).

---
*For more details, refer to the code in the notebook or reach out to the author.*
