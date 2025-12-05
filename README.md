
# 🏢 Insurance Risk Analytics & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)
![DVC](https://img.shields.io/badge/Data_Version_Control-DVC-purple)
![CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-yellow)
Insurance Risk Analytics & Predictive Modeling is to analyze 18 months of historical insurance data to identify low-risk customer segments for targeted premium reductions, validate key risk factors through statistical hypothesis testing, develop machine learning models for accurate claim prediction and premium optimization, and deliver actionable recommendations that will enhance AlphaCare Insurance Solutions' marketing strategy and overall profitability in the South African car insurance market.

## 🏦 Business Context

**AlphaCare Insurance Solutions (ACIS)** is revolutionizing car insurance in South Africa through data-driven risk assessment. As a Marketing Analytics Engineer, my mission is to analyze historical claim data to optimize premiums and identify profitable customer segments.

### 📈 Business Objectives
- **Discover Low-Risk Segments**: Identify customers with minimal claim probability for targeted marketing
- **Optimize Premium Pricing**: Implement risk-based pricing models for maximum profitability
- **Enhance Risk Assessment**: Build predictive models to accurately forecast claim likelihood and severity
- **Drive Growth**: Attract new customers through competitive, data-backed pricing strategies

## 🎯 Project Overview

This project implements end-to-end insurance analytics across four key phases:

### **Phase 1: Exploratory Data Analysis** 
*Understanding 18 months of insurance data (Feb 2014 - Aug 2015)*
- Portfolio loss ratio analysis
- Risk pattern identification across provinces and demographics
- Data quality assessment and outlier detection

### **Phase 2: Statistical Hypothesis Testing**
*Validating risk drivers through A/B testing*
- Geographic risk differences (provinces, zip codes)
- Demographic risk factors (gender-based analysis)
- Profitability margin testing across segments

### **Phase 3: Predictive Modeling**
*Building machine learning models for risk assessment*
- Claim severity prediction models
- Premium optimization frameworks
- Feature importance analysis using SHAP/LIME

### **Phase 4: Prescriptive Analytics**
*Data-driven business recommendations*
- Premium adjustment strategies
- Marketing campaign targeting
- Risk management protocols

## 📊 Dataset Overview

**Historical Insurance Data**: February 2014 - August 2015

### Key Data Categories:
- **Policy Information**: UnderwrittenCoverID, PolicyID, Transaction dates
- **Client Details**: Demographics, location, financial status
- **Vehicle Information**: Make, model, age, value, safety features
- **Coverage Details**: Sum insured, premium calculations, cover types
- **Claims History**: Total premium collected, total claims paid

## 🛠️ Technical Architecture

### **Core Technologies**
| Component | Technology Stack |
|-----------|------------------|
| **Data Processing** | `pandas`, `numpy`, `scipy` |
| **Machine Learning** | `scikit-learn`, `xgboost`, `statsmodels` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Model Interpretation** | `shap`, `lime` |
| **Version Control** | `Git`, `DVC (Data Version Control)` |
| **CI/CD** | `GitHub Actions` |
| **Environment** | `Python 3.8+`, `Jupyter Notebooks` |

**Clone the Repository**
   ```bash
   git clone https://github.com/TsegayIS122123/insurance-risk-analytics-acis.git
   cd insurance-risk-analytics-acis
   ```
# Set Up Virtual Environment
```
python -m venv venv
source .venv/Scripts/activate
```
# Install Dependencies
pip install -r requirements.txt

## 📊 TASK 1: EXPLORATORY DATA ANALYSIS 

### 🛠️ Technical Implementation

#### OOP Architecture
- **InsuranceDataProcessor**: Data loading, cleaning, transformation, and analysis
- **InsuranceVisualizer**: Visualization generation for EDA insights

#### Key Features Implemented
-  Data loading with pipe delimiter support
-  Automatic data type conversion for numerical and datetime columns
-  Derived feature creation (LossRatio, HasClaim, ProfitMargin, RiskCategory)
-  Comprehensive data quality assessment
-  Statistical analysis and hypothesis foundation
-  Three creative visualizations as required

---

## 🔍 METHODOLOGY

### 1. Data Loading & Preprocessing
- **Data Source**: Pipe-delimited text file (`|` separator)
- **File Size**: 1,000,099 rows × 52 columns
- **Processing**:
  - Loaded data with manual column assignment (no header row)
  - Converted key columns to proper data types:
    - Numerical: TotalPremium, TotalClaims, SumInsured, etc.
    - Datetime: TransactionMonth, VehicleIntroDate
  - Created derived features for analysis

### 2. Data Quality Assessment
- **Missing Values Analysis**: Identified columns with critical data gaps
- **Duplicate Detection**: Checked for duplicate policy records
- **Data Type Validation**: Ensured proper formatting of categorical and numerical data
- **Outlier Detection**: Used IQR method to identify extreme values

### 3. Statistical Analysis
- **Descriptive Statistics**: Mean, median, std, min, max for all numerical features
- **Distribution Analysis**: Histograms and boxplots for key metrics
- **Correlation Analysis**: Relationships between numerical variables
- **Temporal Analysis**: Trends over the 18-month period

### 4. Risk Metrics Calculation
- **Loss Ratio**: TotalClaims / TotalPremium
- **Claim Frequency**: Percentage of policies with claims
- **Claim Severity**: Average claim amount when claims occur
- **Profit Margin**: TotalPremium - TotalClaims
- **Risk Categories**: Low, Medium, High based on Loss Ratio thresholds

---

## 📈 KEY FINDINGS

### 🚨 CRITICAL BUSINESS INSIGHTS

#### 1. Profitability Crisis
- **Average Profit Margin**: -R2.96 per policy
- **Profitable Policies**: 61.5% (615,554 policies)
- **Loss-making Policies**: 0.3% (3,060 policies) but with catastrophic impact
- **Break-even/Zero**: 38.2% (381,485 policies)

#### 2. Anomalous Claim Patterns
- **Claim Rate**: Only 0.28% of policies have claims (extremely low)
- **Average Claim Severity**: R23,273.39
- **Maximum Claim**: R393,092.11
- **Premium-Risk Mismatch**: High-risk clients pay 7.5x higher premiums but claims are 51x premiums

#### 3. Risk Assessment Gaps
- **Risk Distribution**:
  - Low Risk: 61.5% (615,539 policies)
  - Unknown Risk: 38.2% (381,923 policies) ← **CRITICAL ISSUE**
  - High Risk: 0.3% (2,632 policies)
  - Medium Risk: 0.0% (5 policies)

#### 4. Geographic Risk Variations
- **Province Analysis**: Significant differences in Loss Ratio across regions
- **High-Risk Areas**: Identified provinces with consistently higher loss ratios
- **Low-Risk Areas**: Identified profitable regions for targeted marketing

### 📊 DATA QUALITY ISSUES (CRITICAL)

| Column | Missing % | Impact | Severity |
|--------|-----------|---------|----------|
| NumberOfVehiclesInFleet | 100% | Cannot assess fleet size | 🔴 **Critical** |
| CrossBorder | 99.9% | Missing cross-border risk data | 🔴 **Critical** |
| CustomValueEstimate | 78% | Cannot assess vehicle value accurately | 🔴 **Critical** |
| Rebuilt | 64% | Missing vehicle history data | 🟠 **High** |
| Converted | 64% | Missing vehicle modification data | 🟠 **High** |
| WrittenOff | 64% | Missing write-off history | 🟠 **High** |
| LossRatio | 38.2% | Due to zero/negative premiums | 🟡 **Medium** |
| NewVehicle | 15.3% | Missing vehicle age indicator | 🟡 **Medium** |

**Total Missing Values**: 5,449,585 (9.73% of all data cells)

---

## 🎨 VISUALIZATIONS CREATED

### 1. Profitability Analysis Dashboard
- **Distribution of Profit Margins**: Histogram showing negative skew
- **Profit/Loss Breakdown**: Pie chart of profitable vs loss-making policies
- **Key Insight**: Few catastrophic claims wipe out profits from many small policies

### 2. Missing Data Heatmap
- **Top 15 Columns**: Horizontal bar chart colored by missing percentage
- **Severity Classification**: Red (>50%), Orange (20-50%), Yellow (5-20%), Green (<5%)
- **Key Insight**: Critical risk assessment columns have 64-100% missing data

### 3. Temporal Trends Analysis
- **Monthly Loss Ratio**: Line plot showing variability over time
- **Premium vs Claims**: Comparison of monthly aggregates
- **Policy Count**: Volume trends over 18-month period
- **Key Insight**: Significant month-to-month volatility in performance

### 4. Additional Supporting Visualizations
- **Distribution Plots**: Key numerical features (TotalPremium, TotalClaims, etc.)
- **Boxplots**: Outlier detection for financial metrics
- **Risk Category Distribution**: Breakdown of Low/Medium/High/Unknown risk
- **Geographic Analysis**: Loss Ratio by Province

