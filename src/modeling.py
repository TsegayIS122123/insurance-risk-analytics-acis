"""
Insurance Predictive Modeling Module
Implements claim severity prediction and premium optimization models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class InsuranceModeler:
    """Main class for insurance predictive modeling"""
    
    def __init__(self, data_path='data/processed.csv'):
        self.data = pd.read_csv(data_path)
        self.models = {}
        self.results = {}
        print(f"✅ Data loaded: {self.data.shape}")
    
    def prepare_claim_severity_data(self):
        """Prepare data for claim severity prediction (claims > 0 only)"""
        print("🔧 Preparing claim severity data...")
        
        # Filter only policies with claims
        self.claims_data = self.data[self.data['TotalClaims'] > 0].copy()
        print(f"   Claims data: {self.claims_data.shape[0]} policies with claims")
        
        # Target variable
        self.y_severity = self.claims_data['TotalClaims']
        
        # Feature selection based on EDA and hypothesis testing
        features = [
            'SumInsured', 'CustomValueEstimate', 'RegistrationYear',
            'Cubiccapacity', 'Kilowatts', 'NumberOfDoors',
            'Province', 'VehicleType', 'Make', 'Bodytype',
            'CoverType', 'ExcessSelected', 'CalculatedPremiumPerTerm'
        ]
        
        # Filter available features
        available_features = [f for f in features if f in self.claims_data.columns]
        self.X_severity = self.claims_data[available_features]
        
        # Handle missing values
        self._handle_missing_values()
        
        # Feature engineering
        self._create_severity_features()
        
        return self.X_severity, self.y_severity
    
    def prepare_claim_probability_data(self):
        """Prepare data for claim probability prediction (binary classification)"""
        print("🔧 Preparing claim probability data...")
        
        # Create binary target
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        self.y_probability = self.data['HasClaim']
        
        print(f"   Claim rate: {self.y_probability.mean()*100:.2f}%")
        
        # Feature selection
        features = [
            'SumInsured', 'CalculatedPremiumPerTerm', 'ExcessSelected',
            'Province', 'PostalCode', 'VehicleType', 'Make', 'Model',
            'RegistrationYear', 'Cubiccapacity', 'Kilowatts',
            'Bodytype', 'NumberOfDoors', 'Gender', 'MaritalStatus',
            'CoverType', 'CoverCategory', 'TermFrequency'
        ]
        
        available_features = [f for f in features if f in self.data.columns]
        self.X_probability = self.data[available_features]
        
        # Handle missing values
        self._handle_missing_values(classification=True)
        
        # Feature engineering
        self._create_probability_features()
        
        return self.X_probability, self.y_probability
    
    def _handle_missing_values(self, classification=False):
        """Handle missing values in features"""
        # Implementation depends on which X we're using
        pass
    
    def _create_severity_features(self):
        """Create features for claim severity prediction"""
        # Vehicle age
        if 'RegistrationYear' in self.X_severity.columns:
            current_year = 2015  # Based on data timeframe
            self.X_severity['VehicleAge'] = current_year - self.X_severity['RegistrationYear']
        
        # Value ratios
        if all(col in self.X_severity.columns for col in ['SumInsured', 'CustomValueEstimate']):
            self.X_severity['ValueRatio'] = self.X_severity['CustomValueEstimate'] / self.X_severity['SumInsured']
        
        print(f"   Created {self.X_severity.shape[1]} features for severity prediction")
    
    def _create_probability_features(self):
        """Create features for claim probability prediction"""
        # Geographic risk scores based on hypothesis testing
        if 'Province' in self.X_probability.columns:
            # Create province risk scores from Task 3 results
            province_scores = {
                'Gauteng': 1.4,      # Highest risk from hypothesis testing
                'KwaZulu-Natal': 1.2,
                'Limpopo': 1.1,
                'North West': 0.9,
                'Mpumalanga': 0.9,
                'Western Cape': 0.8,  # Lower risk
            }
            self.X_probability['ProvinceRiskScore'] = self.X_probability['Province'].map(
                lambda x: province_scores.get(x, 1.0)
            )
        
        # Vehicle power to weight ratio
        if all(col in self.X_probability.columns for col in ['Kilowatts', 'Cubiccapacity']):
            self.X_probability['PowerToWeight'] = self.X_probability['Kilowatts'] / (self.X_probability['Cubiccapacity'] + 1)
        
        print(f"   Created {self.X_probability.shape[1]} features for probability prediction")
    
    def build_severity_models(self, test_size=0.3, random_state=42):
        """Build and compare claim severity prediction models"""
        print("\n" + "="*60)
        print("🏗️  BUILDING CLAIM SEVERITY PREDICTION MODELS")
        print("="*60)
        
        # Prepare data
        X, y = self.prepare_claim_severity_data()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Encode categorical variables
        X_train_encoded, X_test_encoded = self._encode_categorical(X_train, X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state, max_depth=5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=random_state)
        }
        
        # Train and evaluate models
        severity_results = {}
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            model.fit(X_train_encoded, y_train)
            
            # Predict
            y_pred = model.predict(X_test_encoded)
            
            # Evaluate
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Store results
            severity_results[name] = {
                'model': model,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"   RMSE: R{rmse:,.2f}")
            print(f"   R² Score: {r2:.4f}")
        
        self.results['severity'] = severity_results
        return severity_results
    
    def build_probability_models(self, test_size=0.3, random_state=42):
        """Build and compare claim probability prediction models"""
        print("\n" + "="*60)
        print("🏗️  BUILDING CLAIM PROBABILITY PREDICTION MODELS")
        print("="*60)
        
        # Prepare data
        X, y = self.prepare_claim_probability_data()
        
        # Handle class imbalance
        print(f"   Class distribution: {y.value_counts().to_dict()}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Encode categorical variables
        X_train_encoded, X_test_encoded = self._encode_categorical(X_train, X_test, classification=True)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'XGBoost Classifier': xgb.XGBClassifier(n_estimators=100, random_state=random_state)
        }
        
        # Train and evaluate models
        probability_results = {}
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            model.fit(X_train_encoded, y_train)
            
            # Predict
            y_pred = model.predict(X_test_encoded)
            y_pred_proba = model.predict_proba(X_test_encoded)[:, 1]
            
            # Store results
            probability_results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Print classification report
            print(f"   Accuracy: {np.mean(y_pred == y_test):.4f}")
        
        self.results['probability'] = probability_results
        return probability_results
    
    def _encode_categorical(self, X_train, X_test, classification=False):
        """Encode categorical variables"""
        # Implementation for one-hot encoding
        return X_train, X_test  # Simplified - implement proper encoding
    
    def premium_optimization_framework(self):
        """Implement premium optimization using predicted risk"""
        print("\n" + "="*60)
        print("💰 PREMIUM OPTIMIZATION FRAMEWORK")
        print("="*60)
        
        # Get best models
        best_severity_model = self._get_best_model('severity')
        best_probability_model = self._get_best_model('probability')
        
        # Calculate risk-based premium
        # Premium = (Predicted Probability × Predicted Severity) + Expenses + Profit Margin
        expense_loading = 1.15  # 15% for expenses
        profit_margin = 1.10    # 10% profit margin
        
        print("🎯 Risk-Based Premium Formula:")
        print("   Premium = (P(Claim) × Expected Claim Amount) × Expense Loading × Profit Margin")
        print(f"   Expense Loading: {expense_loading}")
        print(f"   Profit Margin: {profit_margin}")
        
        return {
            'severity_model': best_severity_model,
            'probability_model': best_probability_model,
            'expense_loading': expense_loading,
            'profit_margin': profit_margin
        }
    
    def _get_best_model(self, model_type):
        """Select best model based on performance"""
        if model_type not in self.results:
            return None
        
        if model_type == 'severity':
            # Select based on RMSE
            best_name = min(self.results[model_type].items(), 
                          key=lambda x: x[1]['rmse'])[0]
            print(f"   Best {model_type} model: {best_name}")
            return self.results[model_type][best_name]['model']
        
        return None
    
    def save_models(self, path='models/trained_models/'):
        """Save trained models for deployment"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for model_type, results in self.results.items():
            for name, result in results.items():
                filename = f"{path}/{model_type}_{name.replace(' ', '_')}.pkl"
                joblib.dump(result['model'], filename)
                print(f"💾 Saved: {filename}")