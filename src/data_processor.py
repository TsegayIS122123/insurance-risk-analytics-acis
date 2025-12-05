import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class InsuranceDataProcessor:
    """Class to process and analyze insurance data with OOP principles"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
    def load_data(self, delimiter: str = '|') -> pd.DataFrame:
        """Load the insurance data from text file"""
        try:
            # First, let's inspect the file
            with open(self.filepath, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
            
            print(f"First line (header): {first_line[:100]}...")
            print(f"Second line (data): {second_line[:100]}...")
            
            # Count pipes in first line to determine if it's header
            pipes_in_first = first_line.count('|')
            pipes_in_second = second_line.count('|')
            
            print(f"Pipes in first line: {pipes_in_first}")
            print(f"Pipes in second line: {pipes_in_second}")
            
            if pipes_in_first == pipes_in_second:
                # First line is data, need to use column names from instructions
                print("First line appears to be data, using manual column names")
                # Load data without header
                self.df = pd.read_csv(self.filepath, delimiter=delimiter, header=None, low_memory=False)
                
                # Use column names from the instructions
                columns = [
                    'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 'IsVATRegistered',
                    'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType',
                    'MaritalStatus', 'Gender', 'Country', 'Province', 'PostalCode',
                    'MainCrestaZone', 'SubCrestaZone', 'ItemType', 'Mmcode', 'VehicleType',
                    'RegistrationYear', 'Make', 'Model', 'Cylinders', 'Cubiccapacity',
                    'Kilowatts', 'Bodytype', 'NumberOfDoors', 'VehicleIntroDate',
                    'CustomValueEstimate', 'AlarmImmobiliser', 'TrackingDevice',
                    'CapitalOutstanding', 'NewVehicle', 'WrittenOff', 'Rebuilt',
                    'Converted', 'CrossBorder', 'NumberOfVehiclesInFleet', 'SumInsured',
                    'TermFrequency', 'CalculatedPremiumPerTerm', 'ExcessSelected',
                    'CoverCategory', 'CoverType', 'CoverGroup', 'Section', 'Product',
                    'StatutoryClass', 'StatutoryRiskType', 'TotalPremium', 'TotalClaims'
                ]
                
                # Assign column names (only if we have the right number)
                if len(self.df.columns) == len(columns):
                    self.df.columns = columns
                else:
                    print(f"Warning: Expected {len(columns)} columns but got {len(self.df.columns)}")
                    # Create generic column names
                    self.df.columns = [f'col_{i}' for i in range(len(self.df.columns))]
                    
            else:
                # First line is header
                print("First line appears to be header")
                self.df = pd.read_csv(self.filepath, delimiter=delimiter, low_memory=False)
            
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return self.df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            # Try alternative approach
            try:
                print("Trying alternative loading method...")
                self.df = pd.read_csv(self.filepath, sep=delimiter, engine='python', low_memory=False)
                print(f"Alternative load successful. Shape: {self.df.shape}")
                return self.df
            except Exception as e2:
                print(f"Alternative load also failed: {e2}")
                return None
    
    def analyze_data_structure(self) -> Dict:
        """Analyze and categorize columns by data type"""
        if self.df is None:
            print("Please load data first!")
            return {}
        
        # Store column types
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Also check columns that should be numeric but are stored as objects
        potential_numeric_cols = []
        numeric_keywords = ['Premium', 'Claims', 'Amount', 'Value', 'Estimate', 'Year', 
                           'Number', 'Count', 'Capacity', 'Kilowatts', 'Cylinders']
        
        for col in self.df.columns:
            if any(keyword.lower() in col.lower() for keyword in numeric_keywords):
                if col not in self.numerical_cols and col in self.categorical_cols:
                    potential_numeric_cols.append(col)
        
        # Check for potential datetime columns
        date_pattern_cols = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time']):
                date_pattern_cols.append(col)
        
        structure_info = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numerical_columns': self.numerical_cols,
            'categorical_columns': self.categorical_cols,
            'potential_numeric_cols': potential_numeric_cols,
            'date_pattern_columns': date_pattern_cols,
            'column_names': list(self.df.columns)
        }
        
        return structure_info
    
    def convert_data_types(self):
        """Convert columns to appropriate data types"""
        if self.df is None:
            return
        
        print("Converting data types...")
        
        # Convert numeric columns
        numeric_keywords = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm',
                           'CustomValueEstimate', 'RegistrationYear', 'Cylinders', 'Cubiccapacity',
                           'Kilowatts', 'NumberOfDoors', 'NumberOfVehiclesInFleet']
        
        for col in self.df.columns:
            for keyword in numeric_keywords:
                if keyword.lower() in col.lower():
                    try:
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                        print(f"Converted {col} to numeric")
                        if col not in self.numerical_cols:
                            self.numerical_cols.append(col)
                        if col in self.categorical_cols:
                            self.categorical_cols.remove(col)
                    except:
                        print(f"Could not convert {col} to numeric")
        
        # Convert date columns
        date_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                    for keyword in ['date', 'month', 'time'])]
        
        for col in date_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"Converted {col} to datetime")
            except:
                print(f"Could not convert {col} to datetime")
        
        # Update column type lists
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
    def create_derived_features(self) -> None:
        """Create derived features for analysis"""
        print("⚙️ Creating derived features...")
        
        # Calculate Loss Ratio (Claims/Premium)
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            # Handle division by zero - create a safe division
            self.df['LossRatio'] = np.where(
                self.df['TotalPremium'] > 0,
                self.df['TotalClaims'] / self.df['TotalPremium'],
                np.nan
            )
            print(f"   ✓ Created LossRatio feature")
        
        # Calculate Claim Frequency indicator
        if 'TotalClaims' in self.df.columns:
            self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
            print(f"   ✓ Created HasClaim feature")
            claim_rate = self.df['HasClaim'].mean() * 100
            print(f"     Claim rate: {claim_rate:.2f}%")
        
        # Calculate Profit Margin
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['ProfitMargin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            print(f"   ✓ Created ProfitMargin feature")
            
            # Calculate average profit margin
            avg_profit = self.df['ProfitMargin'].mean()
            print(f"     Average profit margin: R{avg_profit:.2f}")
        
        # Calculate Claim Severity (average claim amount when claim occurs)
        if 'TotalClaims' in self.df.columns and 'HasClaim' in self.df.columns:
            # Only for policies with claims
            claims_data = self.df[self.df['HasClaim'] == 1]
            if len(claims_data) > 0:
                avg_severity = claims_data['TotalClaims'].mean()
                print(f"   ✓ Average claim severity: R{avg_severity:.2f}")
        
        # Add risk categories based on loss ratio
        if 'LossRatio' in self.df.columns:
            # Define risk categories
            conditions = [
                self.df['LossRatio'] <= 0.3,  # Low risk
                (self.df['LossRatio'] > 0.3) & (self.df['LossRatio'] <= 0.7),  # Medium risk
                self.df['LossRatio'] > 0.7  # High risk
            ]
            choices = ['Low Risk', 'Medium Risk', 'High Risk']
            self.df['RiskCategory'] = np.select(conditions, choices, default='Unknown')
            print(f"   ✓ Created RiskCategory feature")
        
        print(" Derived features created successfully!")
        
    def assess_data_quality(self) -> Dict:
        """Assess data quality including missing values and inconsistencies"""
        if self.df is None:
            return {}
        
        # Calculate missing values
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        # Sort by missing percentage (descending)
        missing_sorted = missing_percentage.sort_values(ascending=False)
        
        # Get top 10 columns with most missing values
        top_missing = missing_sorted[missing_sorted > 0].head(10)
        
        quality_report = {
            'total_missing_values': missing_values.sum(),
            'total_missing_percentage': (missing_values.sum() / (len(self.df) * len(self.df.columns))) * 100,
            'missing_by_column': missing_values,
            'missing_percentage_by_column': missing_percentage,
            'top_missing_columns': top_missing,
            'duplicate_rows': self.df.duplicated().sum(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns[:10]},
            'zero_values': {col: (self.df[col] == 0).sum() for col in self.numerical_cols[:5] if col in self.df.columns}
        }
        
        return quality_report
    def get_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for numerical columns"""
        if len(self.numerical_cols) < 2:
            print("Not enough numerical columns for correlation analysis")
            return pd.DataFrame()
        
        # Select numerical columns (skip any with all NaN)
        valid_cols = []
        for col in self.numerical_cols:
            if col in self.df.columns and self.df[col].notna().sum() > 1:
                valid_cols.append(col)
        
        if len(valid_cols) < 2:
            print(f"Not enough valid numerical columns. Only {len(valid_cols)} columns with data.")
            return pd.DataFrame()
        
        print(f"Calculating correlation matrix for {len(valid_cols)} numerical columns...")
        
        try:
            # Calculate correlation
            corr_matrix = self.df[valid_cols].corr(method=method)
            return corr_matrix
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    def detect_outliers_iqr(self, column: str) -> Dict:
        """Detect outliers using IQR method for a specific column"""
        if column not in self.df.columns or column not in self.numerical_cols:
            print(f"Column {column} not found or not numerical")
            return {}
        
        try:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
            
            return {
                'column': column,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(self.df)) * 100,
                'outliers_sample': outliers[column].head(10).tolist() if len(outliers) > 0 else []
            }
        except Exception as e:
            print(f"Error detecting outliers for {column}: {e}")
            return {}
    
    def calculate_descriptive_stats(self) -> pd.DataFrame:
        """Calculate descriptive statistics for numerical features"""
        if self.df is None or len(self.numerical_cols) == 0:
            print("No numerical data available!")
            return pd.DataFrame()
        
        stats_dict = {}
        for col in self.numerical_cols:
            stats_dict[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                '25%': self.df[col].quantile(0.25),
                'median': self.df[col].median(),
                '75%': self.df[col].quantile(0.75),
                'max': self.df[col].max(),
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis(),
                'missing': self.df[col].isnull().sum(),
                'missing_pct': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
        
        return pd.DataFrame(stats_dict).T
    
    # ... rest of the methods stay the same ...