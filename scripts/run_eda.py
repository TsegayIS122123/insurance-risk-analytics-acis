#!/usr/bin/env python3
"""
Script to run EDA analysis
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

print(f"📂 Project root: {project_root}")
print(f"📂 Python path includes: {sys.path[:3]}...")

try:
    from src.data_processor import InsuranceDataProcessor
    from src.visualizer import InsuranceVisualizer
    print(" Successfully imported OOP classes!")
except ImportError as e:
    print(f" Import error: {e}")
    print("\n📂 Checking if files exist...")
    
    # Check if files exist
    data_processor_path = project_root / "src" / "data_processor.py"
    visualizer_path = project_root / "src" / "visualizer.py"
    
    print(f"   • data_processor.py exists: {data_processor_path.exists()}")
    print(f"   • visualizer.py exists: {visualizer_path.exists()}")
    
    # Try alternative import
    try:
        import importlib.util
        
        # Import data_processor
        spec = importlib.util.spec_from_file_location(
            "data_processor", 
            data_processor_path
        )
        data_processor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_processor)
        InsuranceDataProcessor = data_processor.InsuranceDataProcessor
        
        # Import visualizer
        spec = importlib.util.spec_from_file_location(
            "visualizer", 
            visualizer_path
        )
        visualizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(visualizer)
        InsuranceVisualizer = visualizer.InsuranceVisualizer
        
        print(" Imported via alternative method!")
    except Exception as alt_e:
        print(f" Alternative import also failed: {alt_e}")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("📊 STARTING INSURANCE DATA EDA ANALYSIS")
    print("="*60)
    
    # Initialize processor
    data_path = project_root / "data" / "MachineLearningRating_v3.txt"
    print(f"\n📂 Loading data from: {data_path}")
    
    if not data_path.exists():
        print(f" Data file not found at: {data_path}")
        print("   Please check if the file exists and try again.")
        return
    
    processor = InsuranceDataProcessor(str(data_path))
    
    # Try pipe delimiter FIRST (based on your data structure)
    print("\n🔍 Trying pipe delimiter: '|'")
    df = processor.load_data(delimiter='|')
    
    if df is None or len(df) == 0:
        print("❌ Could not load data with pipe delimiter.")
        print("   Trying other delimiters...")
        # Fall back to other delimiters
        delimiters = ['\t', ',', ';']
        for delimiter in delimiters:
            try:
                print(f"\n🔍 Trying delimiter: {repr(delimiter)}")
                df = processor.load_data(delimiter=delimiter)
                if df is not None and len(df) > 0:
                    print(f"✅ Successfully loaded data with delimiter {repr(delimiter)}")
                    break
            except Exception as e:
                print(f"❌ Failed with delimiter {repr(delimiter)}: {e}")
                continue
    
    if df is None or len(df) == 0:
        print("\n❌ Could not load data with any delimiter.")
        print("   Please check the file format.")
        return
    
    # ADD THIS: Convert data types after loading
    print("\n" + "="*60)
    print("🔄 CONVERTING DATA TYPES")
    print("="*60)
    
    processor.convert_data_types()
    
    if df is None or len(df) == 0:
        print("\n Could not load data with any delimiter.")
        print("   Please check the file format.")
        return
    
    # Analyze structure
    print("\n" + "="*60)
    print("🔍 ANALYZING DATA STRUCTURE")
    print("="*60)
    
    structure_info = processor.analyze_data_structure()
    
    print(f"\n📊 DATA STRUCTURE:")
    print(f"   • Rows: {structure_info['total_rows']:,}")
    print(f"   • Columns: {structure_info['total_columns']}")
    print(f"   • Numerical Columns: {len(structure_info['numerical_columns'])}")
    print(f"   • Categorical Columns: {len(structure_info['categorical_columns'])}")
    
    # Show sample columns
    print(f"\n📋 FIRST 5 COLUMNS:")
    for col in processor.df.columns[:5]:
        print(f"   • {col}")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("📈 CALCULATING DESCRIPTIVE STATISTICS")
    print("="*60)
    
    stats_df = processor.calculate_descriptive_stats()
    
    if not stats_df.empty:
        print(f"\n📊 Statistics calculated for {len(stats_df)} numerical columns")
        
        # Show key metrics if available
        key_metrics = ['TotalPremium', 'TotalClaims', 'SumInsured']
        available_metrics = [m for m in key_metrics if m in stats_df.index]
        
        if available_metrics:
            print("\n📊 KEY INSURANCE METRICS:")
            for metric in available_metrics:
                mean_val = stats_df.loc[metric, 'mean']
                std_val = stats_df.loc[metric, 'std']
                print(f"   • {metric}: Mean = {mean_val:,.2f}, Std = {std_val:,.2f}")
    
    # Create derived features
    print("\n" + "="*60)
    print("⚙️ CREATING DERIVED FEATURES")
    print("="*60)
    
    processor.create_derived_features()
    
    if 'LossRatio' in processor.df.columns:
        loss_ratio_stats = processor.df['LossRatio'].describe()
        print(f"\n📊 LOSS RATIO (Claims/Premium) STATISTICS:")
        print(f"   • Mean: {loss_ratio_stats['mean']:.3f}")
        print(f"   • Std: {loss_ratio_stats['std']:.3f}")
        print(f"   • Min: {loss_ratio_stats['min']:.3f}")
        print(f"   • Max: {loss_ratio_stats['max']:.3f}")
    
    print("\n" + "="*60)
    print("🚨 CRITICAL BUSINESS INSIGHTS")
    print("="*60)

    # 1. Profitability Analysis
    if 'ProfitMargin' in processor.df.columns:
        profitable = (processor.df['ProfitMargin'] > 0).sum()
        loss_making = (processor.df['ProfitMargin'] < 0).sum()
        total = len(processor.df)
        
        print(f"\n💰 PROFITABILITY ANALYSIS:")
        print(f"   • Profitable policies: {profitable:,} ({profitable/total*100:.1f}%)")
        print(f"   • Loss-making policies: {loss_making:,} ({loss_making/total*100:.1f}%)")
        print(f"   • Break-even/zero: {total - profitable - loss_making:,}")

    # 2. Claim Analysis
    if 'HasClaim' in processor.df.columns:
        claim_policies = processor.df[processor.df['HasClaim'] == 1]
        no_claim_policies = processor.df[processor.df['HasClaim'] == 0]
        
        if len(claim_policies) > 0:
            print(f"\n📊 CLAIM ANALYSIS:")
            print(f"   • Policies with claims: {len(claim_policies):,} ({len(claim_policies)/total*100:.2f}%)")
            print(f"   • Average claim amount: R{claim_policies['TotalClaims'].mean():,.2f}")
            print(f"   • Max claim amount: R{claim_policies['TotalClaims'].max():,.2f}")
            
            # Compare premiums
            avg_premium_claim = claim_policies['TotalPremium'].mean()
            avg_premium_no_claim = no_claim_policies['TotalPremium'].mean()
            print(f"   • Avg premium (with claim): R{avg_premium_claim:,.2f}")
            print(f"   • Avg premium (no claim): R{avg_premium_no_claim:,.2f}")
            print(f"   • Premium ratio: {avg_premium_claim/avg_premium_no_claim:.2f}x")

    # 3. Risk Category Analysis
    if 'RiskCategory' in processor.df.columns:
        risk_counts = processor.df['RiskCategory'].value_counts()
        print(f"\n⚠️  RISK CATEGORY DISTRIBUTION:")
        for category, count in risk_counts.items():
            percentage = count / total * 100
            print(f"   • {category}: {count:,} ({percentage:.1f}%)")
        
    # Data quality
    print("\n" + "="*60)
    print("🧹 DATA QUALITY ASSESSMENT")
    print("="*60)
    
    quality_report = processor.assess_data_quality()
    
    print(f"\n📊 OVERALL DATA QUALITY:")
    print(f"   • Total Missing Values: {quality_report['total_missing_values']:,}")
    print(f"   • Overall Missing %: {quality_report['total_missing_percentage']:.2f}%")
    print(f"   • Duplicate Rows: {quality_report['duplicate_rows']:,}")
    
    # Show top 10 columns with missing values
    top_missing = quality_report['top_missing_columns']
    if len(top_missing) > 0:
        print(f"\n📊 TOP 10 COLUMNS WITH MISSING VALUES (>0%):")
        for col, pct in top_missing.items():
            missing_count = quality_report['missing_by_column'][col]
            print(f"   • {col}: {missing_count:,} ({pct:.1f}%)")
    else:
        print(f"\n📊 No columns with missing values!")
    
    print("\n" + "="*60)
    print(" EDA ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Save summary to file
    summary_path = project_root / "reports" / "eda_summary.txt"
    summary_path.parent.mkdir(exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("INSURANCE DATA EDA SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Rows: {structure_info['total_rows']:,}\n")
        f.write(f"Columns: {structure_info['total_columns']}\n")
        f.write(f"Numerical Columns: {len(structure_info['numerical_columns'])}\n")
        f.write(f"Categorical Columns: {len(structure_info['categorical_columns'])}\n")
        if 'LossRatio' in processor.df.columns:
            f.write(f"Average Loss Ratio: {processor.df['LossRatio'].mean():.3f}\n")
    
    print(f"\n📄 Summary saved to: {summary_path}")
    
    return processor.df

if __name__ == "__main__":
    df = main()
    print(f"\n📊 Final DataFrame shape: {df.shape if df is not None else 'N/A'}")