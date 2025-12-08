"""
Feature engineering for insurance predictive modeling
Creates derived features based on domain knowledge
"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    """Creates insurance-specific features for modeling"""
    
    @staticmethod
    def create_vehicle_features(df):
        """Create vehicle-related features"""
        features = df.copy()
        
        # Vehicle age - SAFE VERSION
        if 'RegistrationYear' in features.columns:
            try:
                # Convert to numeric first
                features['RegistrationYear'] = pd.to_numeric(features['RegistrationYear'], errors='coerce')
                features['VehicleAge'] = 2015 - features['RegistrationYear']
                
                # Create age groups safely
                features['VehicleAgeGroup'] = pd.cut(
                    features['VehicleAge'], 
                    bins=[-1, 3, 7, 15, 100], 
                    labels=['New', 'Young', 'Middle', 'Old']
                )
            except Exception as e:
                print(f"⚠️  Error creating vehicle age features: {e}")
        
        # Engine features - SAFE VERSION
        if all(col in features.columns for col in ['Cubiccapacity', 'Kilowatts']):
            try:
                # Convert to numeric first
                features['Cubiccapacity'] = pd.to_numeric(features['Cubiccapacity'], errors='coerce')
                features['Kilowatts'] = pd.to_numeric(features['Kilowatts'], errors='coerce')
                
                # Avoid division by zero
                features['PowerToCapacity'] = features['Kilowatts'] / (features['Cubiccapacity'].replace(0, np.nan))
            except Exception as e:
                print(f"⚠️  Error creating engine features: {e}")
        
        # Value features - SAFE VERSION
        if all(col in features.columns for col in ['SumInsured', 'CustomValueEstimate']):
            try:
                features['SumInsured'] = pd.to_numeric(features['SumInsured'], errors='coerce')
                features['CustomValueEstimate'] = pd.to_numeric(features['CustomValueEstimate'], errors='coerce')
                
                # Avoid division by zero
                features['InsuredToValueRatio'] = features['SumInsured'] / (features['CustomValueEstimate'].replace(0, np.nan))
            except Exception as e:
                print(f"⚠️  Error creating value features: {e}")
        
        return features
    
    @staticmethod
    def create_geographic_features(df):
        """Create geographic risk features based on hypothesis testing"""
        features = df.copy()
        
        # Province risk scores from Task 3 - SAFE VERSION
        province_risk_scores = {
            'Gauteng': 1.4,
            'KwaZulu-Natal': 1.2,
            'Limpopo': 1.1,
            'North West': 0.9,
            'Mpumalanga': 0.9,
            'Western Cape': 0.8,
            'Eastern Cape': 0.85,
            'Free State': 0.95,
            'Northern Cape': 0.9
        }
        
        if 'Province' in features.columns:
            # Ensure Province is string type
            features['Province'] = features['Province'].astype(str)
            features['ProvinceRiskScore'] = features['Province'].map(
                lambda x: province_risk_scores.get(str(x).strip(), 1.0)
            )
        
        # Urban vs rural (based on postal code ranges) - SAFE VERSION
        if 'PostalCode' in features.columns:
            try:
                features['PostalCode'] = pd.to_numeric(features['PostalCode'], errors='coerce')
                features['IsUrban'] = features['PostalCode'].apply(
                    lambda x: 1 if pd.notna(x) and (x < 3000 or (8000 <= x < 8300)) else 0
                )
            except Exception as e:
                print(f"⚠️  Error creating postal code features: {e}")
        
        return features
    
    @staticmethod
    def create_policy_features(df):
        """Create policy-related features"""
        features = df.copy()
        
        # Excess relative to sum insured - SAFE VERSION
        if all(col in features.columns for col in ['ExcessSelected', 'SumInsured']):
            try:
                features['ExcessSelected'] = pd.to_numeric(features['ExcessSelected'], errors='coerce')
                features['SumInsured'] = pd.to_numeric(features['SumInsured'], errors='coerce')
                
                # Avoid division by zero
                features['ExcessRatio'] = features['ExcessSelected'] / (features['SumInsured'].replace(0, np.nan))
            except Exception as e:
                print(f"⚠️  Error creating excess ratio: {e}")
        
        # Premium features - SAFE VERSION
        if all(col in features.columns for col in ['CalculatedPremiumPerTerm', 'SumInsured']):
            try:
                features['CalculatedPremiumPerTerm'] = pd.to_numeric(features['CalculatedPremiumPerTerm'], errors='coerce')
                features['SumInsured'] = pd.to_numeric(features['SumInsured'], errors='coerce')
                
                # Avoid division by zero
                features['PremiumToValueRatio'] = features['CalculatedPremiumPerTerm'] / (features['SumInsured'].replace(0, np.nan))
            except Exception as e:
                print(f"⚠️  Error creating premium ratio: {e}")
        
        # Coverage features - SAFE VERSION
        if 'CoverType' in features.columns:
            # Ensure CoverType is string type
            features['CoverType'] = features['CoverType'].astype(str)
            
            # Encode coverage types by risk level
            cover_risk_map = {
                'Comprehensive': 1.0,
                'Third Party': 0.7,
                'Third Party Fire & Theft': 0.8
            }
            features['CoverRiskScore'] = features['CoverType'].map(
                lambda x: cover_risk_map.get(str(x).strip(), 0.9)
            )
        
        return features
    
    @staticmethod
    def convert_all_numeric(df, columns=None):
        """Convert specified columns to numeric safely"""
        if columns is None:
            # Get all potentially numeric columns
            columns = df.select_dtypes(include=['object', 'float', 'int']).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        return df