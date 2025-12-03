
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