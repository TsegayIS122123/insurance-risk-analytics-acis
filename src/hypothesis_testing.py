# src/hypothesis_testing.py
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, fisher_exact, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

class InsuranceHypothesisTester:
    """Comprehensive hypothesis testing for insurance risk analysis"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with insurance data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Insurance dataset with columns: Province, PostalCode, Gender, 
            TotalPremium, TotalClaims, etc.
        """
        self.data = data.copy()
        self.results = {}
        
    def prepare_data(self):
        """Prepare data for hypothesis testing"""
        print("🔄 Preparing data for hypothesis testing...")
        
        # Ensure key columns are numeric
        numeric_cols = ['TotalPremium', 'TotalClaims', 'SumInsured']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Create derived metrics
        self.data['HasClaim'] = (self.data['TotalClaims'] > 0).astype(int)
        self.data['ProfitMargin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        
        # Clean categorical columns
        if 'Province' in self.data.columns:
            self.data['Province'] = self.data['Province'].astype(str).str.strip()
        
        if 'Gender' in self.data.columns:
            # Standardize gender values
            gender_map = {
                'M': 'Male', 'Male': 'Male', 'Mr': 'Male',
                'F': 'Female', 'Female': 'Female', 'Mrs': 'Female', 'Miss': 'Female'
            }
            self.data['Gender'] = self.data['Gender'].astype(str).str.strip().str.title()
            self.data['Gender'] = self.data['Gender'].map(lambda x: gender_map.get(x, 'Unknown'))
        
        print(f" Data prepared. Shape: {self.data.shape}")
        print(f"   Claim rate: {self.data['HasClaim'].mean()*100:.4f}%")
        
        return self.data
    
    def test_province_risk_differences(self, alpha=0.05):
        """
        Test H₀: There are no risk differences across provinces
        
        Tests both:
        1. Claim Frequency (proportion with claims)
        2. Claim Severity (average claim amount when claim occurs)
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 1: Province Risk Differences")
        print("="*70)
        
        # Clean and prepare data
        if 'HasClaim' not in self.data.columns:
            self.prepare_data()
        
        # Filter to provinces with sufficient data
        province_counts = self.data['Province'].value_counts()
        min_samples = 100
        valid_provinces = province_counts[province_counts >= min_samples].index.tolist()
        
        if len(valid_provinces) < 2:
            print(f" Not enough provinces with sufficient data")
            return None
        
        df_province = self.data[self.data['Province'].isin(valid_provinces)].copy()
        
        test_results = {
            'test_name': 'Province Risk Differences',
            'n_provinces': len(valid_provinces),
            'total_policies': len(df_province),
            'tests': {}
        }
        
        # TEST 1: Claim Frequency Differences (Chi-square)
        print("\n📊 TEST 1A: Claim Frequency Differences")
        print("-" * 40)
        
        contingency = pd.crosstab(df_province['Province'], df_province['HasClaim'])
        
        if contingency.shape[1] >= 2:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            print(f"Chi-square Test Results:")
            print(f"  Chi² statistic: {chi2:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Degrees of freedom: {dof}")
            
            # Calculate claim rates by province
            claim_rates = df_province.groupby('Province')['HasClaim'].agg(['mean', 'count'])
            claim_rates['ClaimRate%'] = claim_rates['mean'] * 100
            claim_rates = claim_rates.sort_values('ClaimRate%', ascending=False)
            
            print(f"\nClaim Rates by Province:")
            print(claim_rates[['ClaimRate%', 'count']].round(4))
            
            if p_value < alpha:
                print(f"\n RESULT: REJECT H₀ (p = {p_value:.6f})")
                highest = claim_rates['ClaimRate%'].idxmax()
                lowest = claim_rates['ClaimRate%'].idxmin()
                diff = claim_rates['ClaimRate%'].max() - claim_rates['ClaimRate%'].min()
                
                print(f"🎯 BUSINESS INSIGHT: Significant province risk differences found")
                print(f"   Highest risk: {highest} ({claim_rates.loc[highest, 'ClaimRate%']:.4f}%)")
                print(f"   Lowest risk: {lowest} ({claim_rates.loc[lowest, 'ClaimRate%']:.4f}%)")
                print(f"   Risk difference: {diff:.4f} percentage points")
                
                test_results['tests']['claim_frequency'] = {
                    'test': 'chi-square',
                    'p_value': p_value,
                    'reject_h0': True,
                    'highest_risk': highest,
                    'lowest_risk': lowest,
                    'risk_difference': diff
                }
            else:
                print(f"\n RESULT: FAIL TO REJECT H₀ (p = {p_value:.6f})")
                test_results['tests']['claim_frequency'] = {
                    'test': 'chi-square',
                    'p_value': p_value,
                    'reject_h0': False
                }
        
        # TEST 2: Claim Severity Differences (ANOVA/Kruskal-Wallis)
        print("\n📊 TEST 1B: Claim Severity Differences")
        print("-" * 40)
        
        df_claims = df_province[df_province['HasClaim'] == 1]
        
        if len(df_claims) >= 30:
            # Prepare data for statistical test
            severity_groups = []
            province_names = []
            
            for province in valid_provinces:
                province_severity = df_claims[df_claims['Province'] == province]['TotalClaims']
                if len(province_severity) >= 3:
                    severity_groups.append(province_severity.tolist())
                    province_names.append(province)
            
            if len(severity_groups) >= 2:
                try:
                    # Try ANOVA first
                    f_stat, p_value = f_oneway(*severity_groups)
                    test_used = 'ANOVA'
                except:
                    # Use Kruskal-Wallis if ANOVA fails
                    h_stat, p_value = kruskal(*severity_groups)
                    test_used = 'Kruskal-Wallis'
                
                print(f"{test_used} Test Results:")
                if test_used == 'ANOVA':
                    print(f"  F-statistic: {f_stat:.4f}")
                else:
                    print(f"  H-statistic: {h_stat:.4f}")
                print(f"  p-value: {p_value:.6f}")
                
                severity_stats = df_claims.groupby('Province')['TotalClaims'].agg(['mean', 'median', 'count'])
                severity_stats.columns = ['MeanSeverity', 'MedianSeverity', 'ClaimCount']
                
                print(f"\nClaim Severity by Province:")
                print(severity_stats.sort_values('MeanSeverity', ascending=False).round(2))
                
                if p_value < alpha:
                    print(f"\n RESULT: REJECT H₀ (p = {p_value:.6f})")
                    highest_sev = severity_stats['MeanSeverity'].idxmax()
                    lowest_sev = severity_stats['MeanSeverity'].idxmin()
                    diff = severity_stats['MeanSeverity'].max() - severity_stats['MeanSeverity'].min()
                    
                    print(f"🎯 BUSINESS INSIGHT: Significant claim severity differences")
                    print(f"   Highest severity: {highest_sev} (R{severity_stats.loc[highest_sev, 'MeanSeverity']:,.2f})")
                    print(f"   Lowest severity: {lowest_sev} (R{severity_stats.loc[lowest_sev, 'MeanSeverity']:,.2f})")
                    print(f"   Severity difference: R{diff:,.2f}")
                    
                    test_results['tests']['claim_severity'] = {
                        'test': test_used,
                        'p_value': p_value,
                        'reject_h0': True,
                        'highest_severity': highest_sev,
                        'lowest_severity': lowest_sev,
                        'severity_difference': diff
                    }
                else:
                    print(f"\n RESULT: FAIL TO REJECT H₀ (p = {p_value:.6f})")
                    test_results['tests']['claim_severity'] = {
                        'test': test_used,
                        'p_value': p_value,
                        'reject_h0': False
                    }
        
        self.results['province_test'] = test_results
        return test_results
    
    def test_zipcode_risk_differences(self, alpha=0.05, top_n=20):
        """
        Test H₀: There are no risk differences between zip codes
        
        Tests claim frequency differences
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 2: Zip Code Risk Differences")
        print("="*70)
        
        if 'PostalCode' not in self.data.columns:
            print(" PostalCode column not found")
            return None
        
        # Clean postal codes
        self.data['PostalCode_Clean'] = pd.to_numeric(self.data['PostalCode'], errors='coerce')
        
        # Get top N zip codes by policy count
        zip_counts = self.data['PostalCode_Clean'].value_counts()
        top_zips = zip_counts.head(top_n).index.tolist()
        
        if len(top_zips) < 2:
            print(f"⚠️ Not enough zip codes with data")
            return None
        
        df_zip = self.data[self.data['PostalCode_Clean'].isin(top_zips)].copy()
        
        test_results = {
            'test_name': 'Zip Code Risk Differences',
            'n_zips': len(top_zips),
            'total_policies': len(df_zip),
            'tests': {}
        }
        
        print(f"Testing top {len(top_zips)} zip codes by policy count")
        
        # TEST: Claim Frequency Differences (Chi-square)
        print("\n📊 TEST: Claim Frequency by Zip Code")
        print("-" * 40)
        
        contingency = pd.crosstab(df_zip['PostalCode_Clean'], df_zip['HasClaim'])
        
        if contingency.shape[1] >= 2:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            print(f"Chi-square Test Results:")
            print(f"  Chi² statistic: {chi2:.4f}")
            print(f"  p-value: {p_value:.6f}")
            
            claim_rates = df_zip.groupby('PostalCode_Clean')['HasClaim'].agg(['mean', 'count'])
            claim_rates['ClaimRate%'] = claim_rates['mean'] * 100
            claim_rates = claim_rates.sort_values('ClaimRate%', ascending=False)
            
            print(f"\nClaim Rates by Zip Code (Top 10):")
            print(claim_rates.head(10)[['ClaimRate%', 'count']].round(4))
            
            if p_value < alpha:
                print(f"\n RESULT: REJECT H₀ (p = {p_value:.6f})")
                highest = claim_rates['ClaimRate%'].idxmax()
                lowest = claim_rates['ClaimRate%'].idxmin()
                diff = claim_rates['ClaimRate%'].max() - claim_rates['ClaimRate%'].min()
                
                print(f"🎯 BUSINESS INSIGHT: Significant zip code risk differences")
                print(f"   Highest risk: Zip {highest} ({claim_rates.loc[highest, 'ClaimRate%']:.4f}%)")
                print(f"   Lowest risk: Zip {lowest} ({claim_rates.loc[lowest, 'ClaimRate%']:.4f}%)")
                print(f"   Risk difference: {diff:.4f} percentage points")
                
                test_results['tests']['claim_frequency'] = {
                    'test': 'chi-square',
                    'p_value': p_value,
                    'reject_h0': True,
                    'highest_risk': highest,
                    'lowest_risk': lowest,
                    'risk_difference': diff
                }
            else:
                print(f"\n RESULT: FAIL TO REJECT H₀ (p = {p_value:.6f})")
                test_results['tests']['claim_frequency'] = {
                    'test': 'chi-square',
                    'p_value': p_value,
                    'reject_h0': False
                }
        
        self.results['zipcode_risk_test'] = test_results
        return test_results
    
    def test_zipcode_margin_differences(self, alpha=0.05, top_n=20):
        """
        Test H₀: There is no significant margin (profit) difference between zip codes
        
        Tests profit margin differences using Kruskal-Wallis test
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 3: Zip Code Profit Margin Differences")
        print("="*70)
        
        if 'PostalCode' not in self.data.columns or 'ProfitMargin' not in self.data.columns:
            print("❌ Required columns not found")
            return None
        
        # Clean postal codes
        self.data['PostalCode_Clean'] = pd.to_numeric(self.data['PostalCode'], errors='coerce')
        
        # Get top N zip codes by policy count
        zip_counts = self.data['PostalCode_Clean'].value_counts()
        top_zips = zip_counts.head(top_n).index.tolist()
        
        df_zip = self.data[self.data['PostalCode_Clean'].isin(top_zips)].copy()
        
        test_results = {
            'test_name': 'Zip Code Margin Differences',
            'n_zips': len(top_zips),
            'total_policies': len(df_zip),
            'tests': {}
        }
        
        print(f"Testing top {len(top_zips)} zip codes by policy count")
        
        # TEST: Profit Margin Differences (Kruskal-Wallis)
        print("\n📊 TEST: Profit Margin by Zip Code")
        print("-" * 40)
        
        # Prepare data for Kruskal-Wallis test
        margin_groups = []
        zip_names = []
        
        for zip_code in top_zips:
            zip_margins = df_zip[df_zip['PostalCode_Clean'] == zip_code]['ProfitMargin'].dropna()
            if len(zip_margins) >= 5:
                margin_groups.append(zip_margins.tolist())
                zip_names.append(str(zip_code))
        
        if len(margin_groups) >= 2:
            h_stat, p_value = kruskal(*margin_groups)
            
            print(f"Kruskal-Wallis Test Results:")
            print(f"  H-statistic: {h_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            
            profit_stats = df_zip.groupby('PostalCode_Clean')['ProfitMargin'].agg(['mean', 'median', 'count'])
            profit_stats.columns = ['MeanProfit', 'MedianProfit', 'PolicyCount']
            profit_stats = profit_stats.sort_values('MeanProfit', ascending=False)
            
            print(f"\nProfit Margin by Zip Code (Top 10):")
            print(profit_stats.head(10).round(2))
            
            if p_value < alpha:
                print(f"\n RESULT: REJECT H₀ (p = {p_value:.6f})")
                most_profitable = profit_stats['MeanProfit'].idxmax()
                least_profitable = profit_stats['MeanProfit'].idxmin()
                profit_diff = profit_stats['MeanProfit'].max() - profit_stats['MeanProfit'].min()
                
                print(f"🎯 BUSINESS INSIGHT: Significant profit margin differences")
                print(f"   Most profitable: Zip {most_profitable} (R{profit_stats.loc[most_profitable, 'MeanProfit']:.2f})")
                print(f"   Least profitable: Zip {least_profitable} (R{profit_stats.loc[least_profitable, 'MeanProfit']:.2f})")
                print(f"   Profit difference: R{profit_diff:.2f}")
                
                test_results['tests']['profit_margin'] = {
                    'test': 'kruskal-wallis',
                    'p_value': p_value,
                    'reject_h0': True,
                    'most_profitable': most_profitable,
                    'least_profitable': least_profitable,
                    'profit_difference': profit_diff
                }
            else:
                print(f"\n RESULT: FAIL TO REJECT H₀ (p = {p_value:.6f})")
                test_results['tests']['profit_margin'] = {
                    'test': 'kruskal-wallis',
                    'p_value': p_value,
                    'reject_h0': False
                }
        
        self.results['zipcode_margin_test'] = test_results
        return test_results
    
    def test_gender_risk_differences(self, alpha=0.05):
        """
        Test H₀: There is no significant risk difference between Women and Men
        
        Tests both claim frequency and severity
        """
        print("\n" + "="*70)
        print("HYPOTHESIS TEST 4: Gender Risk Differences")
        print("="*70)
        
        if 'Gender' not in self.data.columns:
            print("❌ Gender column not found")
            return None
        
        # Filter to only Male and Female
        df_gender = self.data[self.data['Gender'].isin(['Male', 'Female'])].copy()
        
        if len(df_gender) < 100:
            print(f"⚠️ Not enough gender data")
            return None
        
        male_count = (df_gender['Gender'] == 'Male').sum()
        female_count = (df_gender['Gender'] == 'Female').sum()
        
        print(f"Sample sizes: Male = {male_count:,}, Female = {female_count:,}")
        
        test_results = {
            'test_name': 'Gender Risk Differences',
            'male_count': male_count,
            'female_count': female_count,
            'tests': {}
        }
        
        # TEST 1: Claim Frequency (Fisher's Exact Test for rare events)
        print("\n📊 TEST 4A: Claim Frequency by Gender")
        print("-" * 40)
        
        male_claims = df_gender[df_gender['Gender'] == 'Male']['HasClaim']
        female_claims = df_gender[df_gender['Gender'] == 'Female']['HasClaim']
        
        table = [[male_claims.sum(), len(male_claims) - male_claims.sum()],
                [female_claims.sum(), len(female_claims) - female_claims.sum()]]
        
        oddsratio, p_value = fisher_exact(table)
        
        print(f"Fisher's Exact Test Results:")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Odds Ratio: {oddsratio:.3f}")
        
        male_rate = male_claims.mean() * 100
        female_rate = female_claims.mean() * 100
        
        print(f"\nClaim Rates:")
        print(f"  Male: {male_rate:.4f}%")
        print(f"  Female: {female_rate:.4f}%")
        print(f"  Difference: {abs(male_rate - female_rate):.4f} percentage points")
        
        if p_value < alpha:
            print(f"\n RESULT: REJECT H₀ (p = {p_value:.6f})")
            if oddsratio > 1:
                print(f"🎯 BUSINESS INSIGHT: Males have {oddsratio:.2f}x higher odds of claiming")
                higher_risk = 'Male'
            else:
                print(f"🎯 BUSINESS INSIGHT: Females have {1/oddsratio:.2f}x higher odds of claiming")
                higher_risk = 'Female'
            
            test_results['tests']['claim_frequency'] = {
                'test': 'fisher_exact',
                'p_value': p_value,
                'odds_ratio': oddsratio,
                'reject_h0': True,
                'higher_risk': higher_risk,
                'male_rate': male_rate,
                'female_rate': female_rate
            }
        else:
            print(f"\n RESULT: FAIL TO REJECT H₀ (p = {p_value:.6f})")
            test_results['tests']['claim_frequency'] = {
                'test': 'fisher_exact',
                'p_value': p_value,
                'odds_ratio': oddsratio,
                'reject_h0': False,
                'male_rate': male_rate,
                'female_rate': female_rate
            }
        
        # TEST 2: Claim Severity (Mann-Whitney U Test)
        print("\n📊 TEST 4B: Claim Severity by Gender")
        print("-" * 40)
        
        df_claims_gender = df_gender[df_gender['HasClaim'] == 1]
        
        male_severity = df_claims_gender[df_claims_gender['Gender'] == 'Male']['TotalClaims']
        female_severity = df_claims_gender[df_claims_gender['Gender'] == 'Female']['TotalClaims']
        
        if len(male_severity) >= 5 and len(female_severity) >= 5:
            u_stat, p_value = mannwhitneyu(male_severity, female_severity, alternative='two-sided')
            
            print(f"Mann-Whitney U Test Results:")
            print(f"  U-statistic: {u_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            
            male_median = male_severity.median()
            female_median = female_severity.median()
            
            print(f"\nMedian Claim Severity:")
            print(f"  Male: R{male_median:,.2f}")
            print(f"  Female: R{female_median:,.2f}")
            print(f"  Difference: R{abs(male_median - female_median):,.2f}")
            
            if p_value < alpha:
                print(f"\n RESULT: REJECT H₀ (p = {p_value:.6f})")
                if male_median > female_median:
                    print(f"🎯 BUSINESS INSIGHT: Male claims are R{male_median - female_median:,.2f} higher")
                    higher_severity = 'Male'
                else:
                    print(f"🎯 BUSINESS INSIGHT: Female claims are R{female_median - male_median:,.2f} higher")
                    higher_severity = 'Female'
                
                test_results['tests']['claim_severity'] = {
                    'test': 'mannwhitneyu',
                    'p_value': p_value,
                    'reject_h0': True,
                    'higher_severity': higher_severity,
                    'male_median': male_median,
                    'female_median': female_median
                }
            else:
                print(f"\n RESULT: FAIL TO REJECT H₀ (p = {p_value:.6f})")
                test_results['tests']['claim_severity'] = {
                    'test': 'mannwhitneyu',
                    'p_value': p_value,
                    'reject_h0': False,
                    'male_median': male_median,
                    'female_median': female_median
                }
        
        self.results['gender_test'] = test_results
        return test_results
    
    def run_all_tests(self, alpha=0.05):
        """Run all hypothesis tests"""
        print("🚀 RUNNING ALL HYPOTHESIS TESTS")
        print("="*70)
        
        # Prepare data
        self.prepare_data()
        
        # Run individual tests
        print("\n🔍 Testing Hypothesis 1: Province Risk Differences")
        self.test_province_risk_differences(alpha=alpha)
        
        print("\n🔍 Testing Hypothesis 2: Zip Code Risk Differences")
        self.test_zipcode_risk_differences(alpha=alpha)
        
        print("\n🔍 Testing Hypothesis 3: Zip Code Margin Differences")
        self.test_zipcode_margin_differences(alpha=alpha)
        
        print("\n🔍 Testing Hypothesis 4: Gender Risk Differences")
        self.test_gender_risk_differences(alpha=alpha)
        
        # Summary
        print("\n" + "="*70)
        print("📊 HYPOTHESIS TESTING SUMMARY")
        print("="*70)
        
        rejections = 0
        total_tests = 0
        
        for test_name, result in self.results.items():
            if result and 'tests' in result:
                for subtest_name, subtest_result in result['tests'].items():
                    total_tests += 1
                    if subtest_result.get('reject_h0', False):
                        rejections += 1
        
        print(f"\n📈 Summary Statistics:")
        print(f"   Total Tests Run: {total_tests}")
        print(f"   Null Hypotheses Rejected: {rejections}")
        print(f"   Null Hypotheses Not Rejected: {total_tests - rejections}")
        print(f"   Significance Level (α): {alpha}")
        
        return self.results
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        report_lines = []
        report_lines.append("HYPOTHESIS TESTING REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for test_name, result in self.results.items():
            if result:
                report_lines.append(f"TEST: {result.get('test_name', test_name)}")
                report_lines.append("-"*40)
                
                if 'tests' in result:
                    for subtest_name, subtest_result in result['tests'].items():
                        report_lines.append(f"  {subtest_name.replace('_', ' ').title()}:")
                        report_lines.append(f"    Test: {subtest_result.get('test', 'N/A')}")
                        report_lines.append(f"    p-value: {subtest_result.get('p_value', 'N/A'):.6f}")
                        report_lines.append(f"    Result: {'REJECT H₀' if subtest_result.get('reject_h0') else 'FAIL TO REJECT H₀'}")
                        
                        # Add specific insights
                        if subtest_result.get('reject_h0'):
                            if 'risk_difference' in subtest_result:
                                report_lines.append(f"    Risk Difference: {subtest_result['risk_difference']:.4f}%")
                            if 'severity_difference' in subtest_result:
                                report_lines.append(f"    Severity Difference: R{subtest_result['severity_difference']:,.2f}")
                            if 'profit_difference' in subtest_result:
                                report_lines.append(f"    Profit Difference: R{subtest_result['profit_difference']:.2f}")
                        
                        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        with open('data/hypothesis_testing_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"📄 Report saved to:data/ hypothesis_testing_report.txt")
        return report_text