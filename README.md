
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

## 🔄 Task 2: Data Version Control with DVC 

### Why DVC?
- **Large Data File**: 503 MB insurance data exceeds GitHub's 100 MB limit
- **Version Control**: Track data changes alongside code
- **Reproducibility**: Ensure consistent analysis results
- **Storage Efficiency**: Store data separately from code repository

### DVC Setup Instructions

#### 1. Install DVC
```bash
pip install dvc
```
# Initialize DVC
dvc init
###  Objectives Achieved
- [x] Install and configure DVC
- [x] Set up local remote storage
- [x] Add large data file to DVC tracking
- [x] Commit .dvc pointer files to Git
- [x] Push data to DVC remote storage
- [x] Create sample data for testing
- [x] Document DVC setup in README

### 🎯 Solution to Data Size Problem
**Problem**: Data file (503 MB) exceeds GitHub's 100 MB limit
**Solution**: DVC manages large files outside Git repository

### 🛠️ Implementation
- **DVC Setup**: Initialized with local storage at `C:\Users\HP\Desktop\dvc-storage`
- **Data Management**: Full dataset tracked by DVC, sample in Git
- **Configuration**: Flexible data source selection (`config/dvc_config.py`)
- **Documentation**: Complete setup instructions in README

# Pipeline Implementation

## 🎯 Objective
Establish reproducible and auditable data pipeline using DVC for insurance risk analytics.

## 🔧 What Was Implemented

### 1. **DVC Initialization & Configuration**
-  DVC initialized with `.dvc/` directory
- Local storage configured at `C:/Users/HP/Desktop/dvc-storage/`
- Data file tracked via DVC (503MB insurance dataset)

### 2. **Complete DVC Pipeline** (`dvc.yaml`)
The pipeline for insurance analytics:

#### **Data Validation**
- Validates dataset structure and quality
- Checks for missing values and duplicates

## 📊 Key Features

### **Reproducibility**
- Complete pipeline with `dvc repro`
- Version-controlled data and code
- Auditable analysis trail

### **Insurance-Specific**
- Correct pipe delimiter handling for dataset
- Insurance KPI calculations
- Business-ready reporting

### **Compliance**
- Data version control for regulatory requirements
- Audit trail for insurance analytics
- Reproducible results for stakeholders

### What was accomplished:
-  **DVC Pipeline Setup**: Created reproducible data processing pipeline
-  **Data Processing**: Processed 1,000,099 insurance records
-  **Business Insights Generated**:
  - Overall claim rate: 0.28%
  - Average claim amount: R23,273.39
  - 61.5% of policies are profitable
  - Premium ratio for claims vs no-claims: 7.5x
-  **Output Files**:
  - `data/processed.csv` (533MB, 56 columns)
  - `data/eda_summary.txt` (key statistics)
-  **DVC Tracking**: All data files version controlled

### Critical Findings:
1. **Data Quality Issues**: Several columns have high missing values (up to 100%)
2. **Risk Distribution**: 61.5% low risk, 0.3% high risk policies
3. **Profitability**: Majority of policies profitable despite some high claims

# 📊 TASK 3: HYPOTHESIS TESTING FOR RISK DRIVERS
# 🎯 Objective
Statistically validate key risk drivers to form the basis of ACIS's new segmentation and pricing strategy through A/B hypothesis testing.

# 📋 Hypotheses Tested
#  Province Risk Differences
- Null Hypothesis (H₀): There are no risk differences across provinces

- Test Results: REJECT H₀ (p = 5.93e-19)

# Key Findings:

- Gauteng: Highest risk province (0.3356% claim rate)

- KwaZulu-Natal: 0.2845% claim rate

- Limpopo: 0.2698% claim rate

- North West: 0.2436% claim rate

Statistical Significance: Confirmed for both claim frequency and severity

# Zip Code Risk Differences
Null Hypothesis (H₀): There are no risk differences between zip codes

Test Results: REJECT H₀ (p ≈ 0)

# Key Findings:

Zip 1863: Highest risk area (0.5084% claim rate)

Zip 7405: Lowest risk area

Statistical Significance: Micro-geographic risk variations are highly significant

## Zip Code Profit Margin Differences
Null Hypothesis (H₀): There is no significant margin (profit) difference between zip codes

Test Results: REJECT H₀ (p ≈ 0)

# Key Findings:

Zip 400: Most profitable (R38.81 average profit)

Zip 1863: Least profitable (R-100.57 average loss)

Profit Spread: R139.38 difference between highest and lowest

#  Gender Risk Differences
Null Hypothesis (H₀): There is no significant risk difference between Women and Men

Test Results: FAIL TO REJECT H₀

# Key Findings:

Male Claim Rate: 0.2195% (94 claims out of 42,817 policies)

Female Claim Rate: 0.2073% (14 claims out of 6,755 policies)

Statistical Result: No significance in frequency (p = 1.000) or severity (p = 0.224)

# 🔬 Statistical Methodology
# Test Selection Matrix
- Hypothesis	Metric	Test Used	Statistical Reason
- Province Risk	Claim Frequency	Chi-square Test	Categorical comparison across groups
- Province Risk	Claim Severity	ANOVA	Mean comparison across multiple groups
- Zip Code Risk	Claim Frequency	Chi-square Test	Categorical comparison across groups
- Zip Code Profit	Profit Margin	Kruskal-Wallis	Non-parametric, handles outliers
- Gender Risk	Claim Frequency	Fisher's Exact Test	Small sample sizes, rare events
- Gender Risk	Claim Severity	Mann-Whitney U Test	Non-parametric, skewed data
- Data Preparation Process

# PROVINCE RISK HIERARCHY (Highest to Lowest):
1. Gauteng → 0.3356% claim rate (HIGHEST RISK)
2. KwaZulu-Natal → 0.2845% claim rate
3. Limpopo → 0.2698% claim rate
4. North West → 0.2436% claim rate
5. Mpumalanga → 0.2428% claim rate
2. Micro-Geographic Patterns Emerge

# ZIP CODE EXTREMES:
• Highest Risk: Zip 1863 (0.5084% claim rate)
• Lowest Risk: Zip 7405 
• Most Profitable: Zip 400 (R38.81 average profit)
• Least Profitable: Zip 1863 (R-100.57 average loss)
3. Gender is NOT a Significant Risk Factor

# GENDER ANALYSIS RESULTS:
• Male: 0.2195% claim rate
• Female: 0.2073% claim rate
• Difference: Not statistically significant (p = 1.000)
• Implication: No gender-based pricing differentiation needed
# 🎯 IMMEDIATE BUSINESS ACTIONS
Action 1: Geographic Pricing Restructuring

# PROPOSED PREMIUM ADJUSTMENTS:
• Gauteng Policies: +15-20% premium increase
• Low-Risk Provinces: -5-10% premium reduction
• Implementation: Phased rollout over 6 months
• Expected Impact: 8-12% improvement in loss ratio
Action 2: Targeted Marketing Strategy

# MARKETING FOCUS AREAS:
• PRIORITY: Zip codes 400, 152, 299 (most profitable areas)
• AVOID: Zip code 1863 (high risk, unprofitable)
• CAMPAIGN: "Safe Driver Zones" for low-risk areas
• BUDGET: Reallocate 30% of marketing spend to profitable regions
Action 3: Risk Assessment Enhancement

# UNDERWRITING IMPROVEMENTS:
1. Add geographic risk scoring to all new policies
2. Implement real-time risk assessment for zip codes
3. Create "Risk Heat Maps" for agent training
4. Develop automated geographic risk alerts
# 📊 Performance Impact Projections
- Initiative	Expected Outcome	Timeline	Success Metrics
- Geographic Pricing	8-12% loss ratio improvement	6 months	Loss Ratio < 85%
- Targeted Marketing	15% acquisition cost reduction	3 months	CAC decrease by R150
- Risk Assessment	25% reduction in high-risk policies	12 months	High-risk policies < 0.2%

### **📊 Statistical Results Overview**
| Hypothesis | Decision | p-value | Business Significance |
|------------|----------|---------|----------------------|
| Province Risk Differences | **REJECT H₀** | 5.93e-19 |  **High Impact** |
| Zip Code Risk Differences | **REJECT H₀** | ≈0 |  **High Impact** |
| Zip Code Profit Differences | **REJECT H₀** | ≈0 |  **High Impact** |
| Gender Risk Differences | **FAIL TO REJECT H₀** | 1.000 |  **Compliance Confirmed** |

### **💡 Key Business Insights**

#### **🚨 Critical Findings**
1. **Geographic Risk Concentration**
   - Gauteng has 0.3356% claim rate (highest risk)
   - 25-40% higher risk than other provinces
   - Statistical significance: p < 0.00001

2. **Micro-Geographic Profitability**
   - Zip 400: R38.81 average profit (most profitable)
   - Zip 1863: R-100.57 average loss (least profitable)
   - Profit gap: R139.38 between extremes

3. **Gender Neutrality**
   - Male: 0.2195% claim rate (94/42,817)
   - Female: 0.2073% claim rate (14/6,755)
   - No statistical significance (p = 1.000)

### **🎯 Actionable Recommendations**

#### **Immediate Actions (0-3 Months)**
- **Premium Adjustments**: 
  - Gauteng: +15-20% premium increase
  - Low-risk provinces: -5-10% discount
  - Expected impact: 8-12% loss ratio improvement

- **Targeted Marketing**:
  - Focus on profitable zip codes (400, 152, 299)
  - Reduce acquisition in loss-making areas (1863)
  - Launch "Safe Driver Zones" campaign

#### **Strategic Initiatives (3-12 Months)**
- **Risk Assessment Enhancement**:
  - Implement geographic risk scoring
  - Develop real-time risk dashboard
  - Train agents on risk factors

- **Compliance & Transparency**:
  - Maintain gender-neutral pricing
  - Document geographic risk rationale
  - Prepare regulatory compliance reports

### **📈 Expected Business Impact**

| Metric | Target | Timeline | Success Criteria |
|--------|--------|----------|------------------|
| Loss Ratio Improvement | 8-12% | 6 months | Loss Ratio < 85% |
| Customer Acquisition Cost | 15% reduction | 3 months | CAC decrease by R150 |
| High-Risk Policy Reduction | 25% reduction | 12 months | High-risk < 0.2% |
| Overall Profitability | 8-12% increase | 12 months | ROI improvement |

