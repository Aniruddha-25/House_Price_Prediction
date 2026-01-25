"""
House Price Prediction Flask Application
Converts the Jupyter notebook into a production-ready Flask web application
"""

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, r2_score
import calendar
import os
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.serving import WSGIRequestHandler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Suppress Flask multiple address logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Global variables for model and scaler
best_model = None
scaler = None
feature_columns = None
model_score = None


class HousePricePredictor:
    """Main class for house price prediction"""

    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.model_score = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.feature_columns = None

    def load_data(self, data_path='./Data/combined_data.csv'):
        """Load combined training and testing data"""
        try:
            self.data = pd.read_csv(data_path)
            # Separate train and test based on 'dataset' column or NaN SalePrice
            if 'dataset' in self.data.columns:
                self.len_train = len(self.data[self.data['dataset'] == 'train'])
            else:
                # Alternative: count rows with SalePrice values
                self.len_train = self.data['SalePrice'].notna().sum()
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        # Handle MSZoning
        mszoning_mode = self.data["MSZoning"].mode()[0]
        self.data["MSZoning"] = self.data["MSZoning"].fillna(mszoning_mode)

        # Handle Garage features
        num_garage_feat = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
        cat_garage_feat = ["GarageCars", "GarageArea", "GarageYrBlt"]

        garage_cont = "NA"
        for feat in cat_garage_feat + num_garage_feat:
            if feat in self.data.columns:
                self.data[feat] = self.data[feat].fillna(garage_cont)

        # Handle other missing numerical values with median
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.data[col].isnull().any():
                self.data[col] = self.data[col].fillna(self.data[col].median())

        # Handle other missing categorical values with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().any():
                mode_val = self.data[col].mode()[0] if len(self.data[col].mode()) > 0 else "Unknown"
                self.data[col] = self.data[col].fillna(mode_val)

    def convert_numerical_to_categorical(self):
        """Convert numerical features to categorical"""
        for_num_conv = [
            "MSSubClass",
            # "YearBuilt",  # Keep as numerical - don't convert to categorical
            "YearRemodAdd",
            "GarageYrBlt",
            "MoSold",
            "YrSold",
        ]

        self.data["MoSold"] = self.data["MoSold"].apply(lambda x: calendar.month_abbr[int(x)] if x != 'NA' else 'NA')

        for feat in for_num_conv:
            self.data[feat] = self.data[feat].astype(str)

    def apply_ordinal_encoding(self):
        """Apply ordinal encoding for ordinal categorical features"""
        ordinal_mappings = {
            "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtExposure": ["NA", "No", "Mn", "Av", "Gd"],
            "BsmtFinType1": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "BsmtFinType2": ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
            "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
            "FireplaceQu": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageQual": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageCond": ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
            "PoolQC": ["NA", "Fa", "TA", "Gd", "Ex"],
            "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
            "GarageFinish": ["NA", "Unf", "RFn", "Fin"],
            "PavedDrive": ["N", "P", "Y"],
            "Utilities": ["ELO", "NoSeWa", "NoSeWr", "AllPub"],
        }

        for feature, categories in ordinal_mappings.items():
            if feature in self.data.columns:
                dtype_cat = CategoricalDtype(categories=categories, ordered=True)
                self.data[feature] = self.data[feature].astype(dtype_cat).cat.codes

    def one_hot_encode(self):
        """Apply one-hot encoding for nominal categorical features"""
        object_features = self.data.select_dtypes(include='object').columns.tolist()
        self.data = pd.get_dummies(
            self.data, 
            columns=object_features, 
            prefix=object_features, 
            drop_first=True
        )

    def prepare_train_test_data(self):
        """Split data into training and testing sets"""
        self.feature_columns = [col for col in self.data.columns if col != 'SalePrice']

        self.X_train = self.data[:self.len_train][self.feature_columns]
        self.y_train = self.data["SalePrice"][:self.len_train]
        self.X_test = self.data[self.len_train:][self.feature_columns]

        # Select only numerical features
        numerical_cols = self.X_train.select_dtypes(include=[np.number]).columns.tolist()
        self.X_train = self.X_train[numerical_cols]
        self.X_test = self.X_test[numerical_cols]
        self.feature_columns = numerical_cols

        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_models(self):
        """Train multiple models and select the best one"""
        models = {
            "LinearRegression": LinearRegression(),
            "SVR": SVR(),
            "SGDRegressor": SGDRegressor(random_state=42),
            "KNeighborsRegressor": KNeighborsRegressor(),
            "GaussianProcessRegressor": GaussianProcessRegressor(random_state=42),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "XGBRegressor": XGBRegressor(random_state=42, verbosity=0),
            "IsotonicRegression": IsotonicRegression(),
        }

        models_score = []

        for model_name, model_instance in models.items():
            print(f"Training model: {model_name}")
            try:
                cv = KFold(n_splits=7, shuffle=True, random_state=45)
                r2 = make_scorer(r2_score)
                r2_val_score = cross_val_score(model_instance, self.X_train, self.y_train, cv=cv, scoring=r2)
                score = r2_val_score.mean()
                models_score.append((model_name, score, model_instance))
                print(f"Score of model ({model_name}): {score}")
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                continue

        # Select best model
        if models_score:
            best_model_name, best_score, best_model_instance = max(models_score, key=lambda x: x[1])
            self.best_model = best_model_instance
            self.model_score = best_score

            # Train on full dataset
            self.best_model.fit(self.X_train, self.y_train)

            print(f"\nBest Model: {best_model_name}")
            print(f"Score: {best_score * 100:.2f}%")
            return best_model_name, best_score
        else:
            raise Exception("No models trained successfully")

    def predict(self, input_data):
        """Make prediction on new data"""
        if self.best_model is None:
            return None

        # Scale the input
        input_scaled = self.scaler.transform([input_data])

        # Make prediction
        prediction = self.best_model.predict(input_scaled)
        return prediction[0]


# Initialize predictor
predictor = HousePricePredictor()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the model endpoint with comprehensive error handling"""
    max_retries = 2
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"\n{'='*60}")
            print(f"Training attempt {retry_count + 1}/{max_retries}")
            print(f"{'='*60}")

            print("📂 Loading data...")
            if not predictor.load_data():
                return jsonify({'error': 'Failed to load data. Check if combined_data.csv exists.'}), 400
            print(f"✓ Data loaded successfully ({len(predictor.data)} rows)")

            print("🔧 Handling missing values...")
            predictor.handle_missing_values()
            print("✓ Missing values handled")

            print("🔄 Converting numerical to categorical...")
            predictor.convert_numerical_to_categorical()
            print("✓ Conversion complete")

            print("📊 Applying ordinal encoding...")
            predictor.apply_ordinal_encoding()
            print("✓ Ordinal encoding applied")

            print("🎨 Applying one-hot encoding...")
            predictor.one_hot_encode()
            print("✓ One-hot encoding complete")

            print("📈 Preparing training data...")
            predictor.prepare_train_test_data()
            print(f"✓ Data prepared ({len(predictor.feature_columns)} features)")

            print("🤖 Training models (this may take a minute)...")
            best_model_name, best_score = predictor.train_models()
            print(f"✓ Best model: {best_model_name} with score {best_score*100:.2f}%")

            print(f"{'='*60}")
            print("✅ Training completed successfully!")
            print(f"{'='*60}\n")

            return jsonify({
                'success': True,
                'model_name': best_model_name,
                'accuracy': f"{best_score * 100:.2f}%",
                'score': best_score
            }), 200

        except Exception as e:
            print(f"❌ Training error (attempt {retry_count + 1}): {str(e)}")
            retry_count += 1

            if retry_count >= max_retries:
                error_msg = f"Training failed after {max_retries} attempts: {str(e)}"
                print(f"❌ {error_msg}\n")
                return jsonify({'error': error_msg}), 500

            # Reset predictor state for retry
            predictor.best_model = None
            predictor.data = None
            print("🔄 Retrying training...\n")
            continue

    return jsonify({'error': 'Maximum training attempts exceeded'}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict house price endpoint with safety checks and retry logic"""
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Validate request
            if not request.is_json:
                return jsonify({'error': 'Request must be JSON'}), 400

            data = request.json
            if not data or 'features' not in data:
                return jsonify({'error': 'Missing features in request'}), 400

            # Check if model is trained
            if predictor.best_model is None:
                return (
                    jsonify(
                        {
                            "error": "Model not trained yet. Please train the model first."
                        }
                    ),
                    400,
                )

            # Validate model is ready
            if predictor.scaler is None or predictor.feature_columns is None:
                return (
                    jsonify(
                        {"error": "Model training incomplete. Please train again."}
                    ),
                    400,
                )

            # Prepare input data with proper error handling
            try:
                features_data = data['features']

                # Create a DataFrame to properly handle features
                input_df = pd.DataFrame([[0] * len(predictor.feature_columns)], columns=predictor.feature_columns)

                # Get user inputs
                lot_area = float(features_data.get('LotArea', 0))
                year_built = float(features_data.get('YearBuilt', 0))
                gr_liv_area = float(features_data.get('GrLivArea', 0))
                bedrooms = float(features_data.get('BedroomAbvGr', 0))
                bathrooms = float(features_data.get('FullBath', 0))
                garage_area = float(features_data.get('GarageArea', 0))

                # Validate inputs
                if lot_area < 0 or gr_liv_area < 0 or garage_area < 0 or bedrooms < 0 or bathrooms < 0 or year_built < 0:
                    raise ValueError("Values cannot be negative")

                # Set the numerical features
                for col in input_df.columns:
                    if col == 'LotArea':
                        input_df[col] = lot_area
                    elif col == 'YearBuilt':
                        input_df[col] = year_built
                    elif col == 'GrLivArea':
                        input_df[col] = gr_liv_area
                    elif col == 'BedroomAbvGr':
                        input_df[col] = bedrooms
                    elif col == 'FullBath':
                        input_df[col] = bathrooms
                    elif col == 'GarageArea':
                        input_df[col] = garage_area
                    # All other columns remain 0 (default for one-hot encoded categories)

                # Convert to numpy array for prediction
                input_features = input_df.values.astype(np.float64)

                # Validate input features
                if np.any(np.isnan(input_features)):
                    raise ValueError("Input contains NaN values")
                if np.any(np.isinf(input_features)):
                    raise ValueError("Input contains infinite values")

                # Debug: Log the features being sent
                print(f"📊 Prediction Input: LotArea={lot_area}, YearBuilt={year_built}, "
                      f"GrLivArea={gr_liv_area}, Bedrooms={bedrooms}, "
                      f"Bathrooms={bathrooms}, GarageArea={garage_area}")
                print(f"📊 Feature array shape: {input_features.shape}, dtype: {input_features.dtype}")

            except (ValueError, TypeError, KeyError) as e:
                return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

            # Make prediction with error handling
            try:
                # Use the scaler with feature_names_in_
                input_scaled = predictor.scaler.transform(input_features)
                predicted_price = predictor.best_model.predict(input_scaled)[0]

                # Validate prediction output
                if predicted_price is None:
                    raise ValueError("Prediction returned None")
                if np.isnan(predicted_price) or np.isinf(predicted_price):
                    raise ValueError("Prediction is invalid (NaN or Inf)")
                if predicted_price < 0:
                    predicted_price = abs(predicted_price)  # Safety: absolute value

                print(f"✅ Prediction Result: ₹{predicted_price:,.2f}\n")

                return jsonify({
                    'success': True,
                    'predicted_price': f"₹{predicted_price:,.2f}",
                    'price_value': round(predicted_price, 2)
                }), 200

            except Exception as pred_error:
                print(f"❌ Prediction error (attempt {retry_count + 1}): {str(pred_error)}")
                retry_count += 1

                if retry_count >= max_retries:
                    return jsonify({'error': f'Prediction failed after {max_retries} attempts: {str(pred_error)}'}), 500
                continue

        except Exception as e:
            print(f"❌ Prediction endpoint error (attempt {retry_count + 1}): {str(e)}")
            retry_count += 1

            if retry_count >= max_retries:
                return jsonify({'error': f'Request processing failed: {str(e)}'}), 500
            continue

    return jsonify({'error': 'Maximum retry attempts exceeded'}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if predictor.best_model is None:
            return jsonify({
                'trained': False,
                'message': 'Model not trained yet'
            })

        return jsonify({
            'trained': True,
            'model_type': type(predictor.best_model).__name__,
            'accuracy': f"{predictor.model_score * 100:.2f}%" if predictor.model_score else 'Unknown',
            'features_count': len(predictor.feature_columns) if predictor.feature_columns else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Get analytics data for charts"""
    try:
        # Load data if not already loaded
        if predictor.data is None:
            if not predictor.load_data():
                return jsonify({'error': 'Failed to load data'}), 400
            predictor.handle_missing_values()
        
        # Get training data prices
        if 'dataset' in predictor.data.columns:
            train_data = predictor.data[predictor.data['dataset'] == 'train']
        else:
            train_data = predictor.data[predictor.data['SalePrice'].notna()]
        
        prices = train_data['SalePrice'].dropna().values.tolist()
        
        if not prices or len(prices) == 0:
            return jsonify({
                'average_price': 0,
                'median_price': 0,
                'min_price': 0,
                'max_price': 0,
                'total_records': 0,
                'price_distribution': [],
                'top_correlations': []
            })
        
        # Calculate statistics
        prices_array = np.array(prices)
        avg_price = float(np.mean(prices_array))
        median_price = float(np.median(prices_array))
        min_price = float(np.min(prices_array))
        max_price = float(np.max(prices_array))
        
        # Calculate feature correlations with price
        correlation_dict = {}
        numeric_data = predictor.data.select_dtypes(include=[np.number])
        
        for col in numeric_data.columns:
            if col != 'SalePrice' and col not in ['Id']:
                try:
                    corr = predictor.data[[col, 'SalePrice']].corr().iloc[0, 1]
                    if not pd.isna(corr):
                        correlation_dict[col] = abs(corr)
                except:
                    pass
        
        # Get top correlations
        top_correlations = sorted(correlation_dict.items(), key=lambda x: x[1], reverse=True)[:8]
        top_corr_data = [
            {'name': name.replace('_', ' '), 'correlation': float(corr)}
            for name, corr in top_correlations
        ]
        
        # Calculate price by year (from YearBuilt feature)
        price_by_year = {}
        if 'YearBuilt' in predictor.data.columns:
            try:
                # Group by year and calculate average price
                year_price_groups = train_data.groupby('YearBuilt')['SalePrice'].agg(['mean', 'count']).reset_index()
                for idx, row in year_price_groups.iterrows():
                    year = int(row['YearBuilt']) if isinstance(row['YearBuilt'], str) else row['YearBuilt']
                    price = float(row['mean'])
                    price_by_year[year] = price
            except Exception as e:
                print(f"Error calculating price by year: {e}")
        
        return jsonify({
            'average_price': avg_price,
            'median_price': median_price,
            'min_price': min_price,
            'max_price': max_price,
            'total_records': len(prices),
            'price_distribution': prices,
            'top_correlations': top_corr_data,
            'price_by_year': price_by_year
        })
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({
            'error': str(e),
            'average_price': 0,
            'median_price': 0,
            'min_price': 0,
            'max_price': 0,
            'total_records': 0,
            'price_distribution': [],
            'top_correlations': [],
            'price_by_year': {}
        }), 200


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏠 House Price Predictor - Starting...")
    print("="*60)
    print("📍 Open your browser and go to:")
    print("   http://127.0.0.1:5000 (local)")
    print("   Or on the same network:")
    print("   http://YOUR_IP_ADDRESS:5000")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
