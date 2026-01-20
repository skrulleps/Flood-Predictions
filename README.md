# Deteksi Banjir

This project implements a machine learning model to predict flood probability using linear regression. The model is trained on a dataset containing various features related to flood risk and deployed as a Flask web application for real-time predictions.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Testing](#testing)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- Data preprocessing including outlier removal and feature standardization
- Exploratory data analysis with visualizations
- Multiple regression models comparison (Lars, Linear Regression, Gradient Boosting Regressor)
- Model serialization using joblib
- Flask API for model deployment
- Prediction testing on new data

## Dataset

The project uses the following datasets located in the `content/` directory:
- `train.csv`: Training data with features and target variable (FloodProbability)
- `test.csv`: Test data for model evaluation
- `sample_submission.csv`: Sample submission format

The dataset contains numerical features that influence flood probability. Missing values are handled, outliers are removed using IQR method, and features are standardized using StandardScaler.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deteksi-banjir
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Model Training

Run the main notebook to train the model:
```bash
jupyter notebook Train/main-flood_pred.ipynb
```

This notebook performs:
- Data loading and exploration
- Preprocessing (missing values, outliers, standardization)
- Feature correlation analysis
- Model training and evaluation
- Model saving

### Deployment

To deploy the model as a web service:

1. Run the deployment notebook:
   ```bash
   jupyter notebook deploy_LR.ipynb
   ```

2. The Flask app will start and provide a `/predict` endpoint.

For production deployment, you can run the Flask app directly:
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('models/linear_regression_modelV2.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data['features']]
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
```

### Testing

Run the testing notebook to make predictions on new data:
```bash
jupyter notebook testing_LR.ipynb
```

This loads the trained model and scaler, processes test data, and generates predictions.

## Model Training

The project compares three regression models:
- Lars Regression
- Linear Regression
- Gradient Boosting Regressor

Models are evaluated using MAE, MSE, and R² metrics. The best performing model (Linear Regression) is saved for deployment.

## Deployment

The model is deployed using Flask with a REST API endpoint:
- **POST /predict**: Accepts JSON with 'features' array and returns flood probability prediction

Example request:
```json
{
  "features": [0.1, 0.2, 0.3, ...]
}
```

## Testing

The testing notebook demonstrates how to:
- Load the saved model and scaler
- Preprocess new test data
- Generate predictions

## Dependencies

Key dependencies include:
- pandas: Data manipulation
- scikit-learn: Machine learning algorithms and preprocessing
- matplotlib & seaborn: Data visualization
- Flask: Web framework for deployment
- joblib: Model serialization

See `requirements.txt` for complete list.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
