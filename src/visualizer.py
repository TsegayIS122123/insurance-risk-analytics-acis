import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not installed. Interactive plots will use matplotlib.")

class InsuranceVisualizer:
    """Class for creating insurance data visualizations"""
    
    def __init__(self, style: str = 'seaborn'):
        self.style = style
        self.set_style()
    
    def set_style(self):
        """Set visualization style"""
        if self.style == 'seaborn':
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = [12, 8]
            plt.rcParams['font.size'] = 12
        elif self.style == 'ggplot':
            plt.style.use('ggplot')
    
    def plot_numerical_distribution(self, data: pd.Series, title: str, ax=None):
        """Plot distribution of numerical data"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Clean data (remove NaN)
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            ax.text(0.5, 0.5, f'No data for {title}', 
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Histogram with KDE
        sns.histplot(clean_data, kde=True, ax=ax, bins=50, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {title}', fontsize=16, fontweight='bold')
        ax.set_xlabel(title, fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        
        # Add mean and median lines
        mean_val = clean_data.mean()
        median_val = clean_data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend()
        
        # Add statistics annotation
        stats_text = f'n={len(clean_data):,}\nStd={clean_data.std():.2f}\nMin={clean_data.min():.2f}\nMax={clean_data.max():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return ax
    
    def plot_categorical_distribution(self, data: pd.Series, title: str, top_n: int = 10, ax=None):
        """Plot distribution of categorical data (top N categories)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Clean data
        clean_data = data.dropna()
        
        if len(clean_data) == 0:
            ax.text(0.5, 0.5, f'No data for {title}', 
                   ha='center', va='center', transform=ax.transAxes)
            return ax
        
        # Get top N categories
        value_counts = clean_data.value_counts().head(top_n)
        
        # Create bar plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors, edgecolor='black')
        ax.set_title(f'Top {top_n} Categories for {title}', fontsize=16, fontweight='bold')
        ax.set_xlabel(title, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # Add percentage labels
        total = len(clean_data)
        for i, (category, count) in enumerate(value_counts.items()):
            percentage = (count / total) * 100
            ax.text(i, count/2, f'{percentage:.1f}%', ha='center', va='center',
                   color='white', fontweight='bold')
        
        return ax
    
    def plot_boxplots_outliers(self, df: pd.DataFrame, columns: List[str], figsize: Tuple = (15, 10)):
        """Create boxplots for outlier detection"""
        # Filter columns that exist in dataframe
        existing_cols = [col for col in columns if col in df.columns]
        
        if not existing_cols:
            print("❌ None of the specified columns found in dataframe")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No columns found for boxplots', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        n_cols = min(4, len(existing_cols))
        n_rows = (len(existing_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for idx, col in enumerate(existing_cols):
            if idx < len(axes):
                # Clean data
                clean_data = df[col].dropna()
                
                if len(clean_data) == 0:
                    axes[idx].text(0.5, 0.5, f'No data for {col}', 
                                  ha='center', va='center', transform=axes[idx].transAxes)
                    axes[idx].set_title(f'{col} (No Data)')
                    continue
                
                # Create boxplot
                bp = axes[idx].boxplot(clean_data, patch_artist=True)
                
                # Customize boxplot
                bp['boxes'][0].set_facecolor('lightblue')
                bp['medians'][0].set_color('red')
                bp['medians'][0].set_linewidth(2)
                bp['whiskers'][0].set_color('black')
                bp['whiskers'][1].set_color('black')
                bp['caps'][0].set_color('black')
                bp['caps'][1].set_color('black')
                bp['fliers'][0].set(marker='o', color='red', alpha=0.5)
                
                axes[idx].set_title(f'Boxplot of {col}', fontsize=12, fontweight='bold')
                axes[idx].set_ylabel(col, fontsize=10)
                
                # Calculate and display outlier statistics
                Q1 = clean_data.quantile(0.25)
                Q3 = clean_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                
                # Add annotation
                stats_text = f'Outliers: {len(outliers):,}\n({len(outliers)/len(clean_data)*100:.1f}%)'
                axes[idx].text(0.05, 0.95, stats_text, transform=axes[idx].transAxes, 
                              fontsize=9, verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Hide empty subplots
        for idx in range(len(existing_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Outlier Detection - Box Plots', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str = 'Correlation Heatmap'):
        """Plot correlation heatmap"""
        if corr_matrix.empty or len(corr_matrix) < 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Not enough data for correlation matrix', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5},
                   annot=True, fmt=".2f", ax=ax, annot_kws={"size": 9})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_loss_ratio_by_category(self, df: pd.DataFrame, category_col: str, title: str):
        """Plot loss ratio by category"""
        if 'LossRatio' not in df.columns:
            print("❌ LossRatio not found in dataframe!")
            return None
        
        if category_col not in df.columns:
            print(f"❌ {category_col} not found in dataframe!")
            return None
        
        # Clean data
        clean_df = df[[category_col, 'LossRatio']].dropna()
        
        if len(clean_df) == 0:
            print(f"❌ No data available for {category_col} vs LossRatio")
            return None
        
        # Calculate statistics by category
        category_stats = clean_df.groupby(category_col).agg({
            'LossRatio': ['mean', 'count', 'std']
        }).round(3)
        
        category_stats.columns = ['mean_loss_ratio', 'policy_count', 'std_loss_ratio']
        category_stats = category_stats.sort_values('mean_loss_ratio', ascending=False).head(15)
        
        if len(category_stats) == 0:
            print(f"❌ No categories to plot for {category_col}")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Bar plot for mean loss ratio
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(category_stats)))
        bars = ax1.bar(range(len(category_stats)), category_stats['mean_loss_ratio'], 
                      color=colors, edgecolor='black')
        ax1.set_title(f'Average Loss Ratio by {title}', fontsize=16, fontweight='bold')
        ax1.set_xlabel(title)
        ax1.set_ylabel('Loss Ratio (Claims/Premium)')
        ax1.set_xticks(range(len(category_stats)))
        ax1.set_xticklabels(category_stats.index, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Horizontal line for overall average
        overall_avg = clean_df['LossRatio'].mean()
        ax1.axhline(y=overall_avg, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Avg: {overall_avg:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Bar plot for policy count
        ax2.bar(range(len(category_stats)), category_stats['policy_count'], 
               color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_title(f'Policy Count by {title}', fontsize=14, fontweight='bold')
        ax2.set_xlabel(title)
        ax2.set_ylabel('Number of Policies')
        ax2.set_xticks(range(len(category_stats)))
        ax2.set_xticklabels(category_stats.index, rotation=45, ha='right')
        
        # Add count labels
        for i, count in enumerate(category_stats['policy_count']):
            ax2.text(i, count + max(category_stats['policy_count'])*0.01, 
                    f'{int(count):,}', ha='center', va='bottom', fontsize=9)
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_temporal_plot(self, df: pd.DataFrame, date_col: str):
        """Create interactive temporal plot using Plotly"""
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available, creating static plot")
            return self.create_static_temporal_plot(df, date_col)
        
        if date_col not in df.columns:
            print(f"❌ {date_col} not found in dataframe!")
            return None
        
        try:
            # Ensure datetime format
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col])
            
            if len(df_copy) == 0:
                print(f"❌ No valid dates in {date_col}")
                return None
            
            # Group by month
            df_copy['YearMonth'] = df_copy[date_col].dt.to_period('M').astype(str)
            monthly_data = df_copy.groupby('YearMonth').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            
            # Calculate derived metrics
            monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
            monthly_data['ProfitMargin'] = monthly_data['TotalPremium'] - monthly_data['TotalClaims']
            
            # Create interactive plot
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Monthly Loss Ratio Trend', 'Monthly Premium vs Claims', 'Monthly Policy Count'),
                vertical_spacing=0.12
            )
            
            # 1. Loss Ratio line
            fig.add_trace(
                go.Scatter(x=monthly_data['YearMonth'], y=monthly_data['LossRatio'],
                          mode='lines+markers', name='Loss Ratio',
                          line=dict(color='firebrick', width=3),
                          marker=dict(size=8, symbol='circle')),
                row=1, col=1
            )
            
            # Add break-even line
            fig.add_hline(y=1, line_dash="dash", line_color="green", 
                         annotation_text="Break-even", 
                         annotation_position="bottom right",
                         row=1, col=1)
            
            # 2. Premium and Claims bars
            fig.add_trace(
                go.Bar(x=monthly_data['YearMonth'], y=monthly_data['TotalPremium'],
                      name='Total Premium', marker_color='lightseagreen', opacity=0.7),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=monthly_data['YearMonth'], y=monthly_data['TotalClaims'],
                      name='Total Claims', marker_color='lightcoral', opacity=0.7),
                row=2, col=1
            )
            
            # 3. Policy Count
            fig.add_trace(
                go.Bar(x=monthly_data['YearMonth'], y=monthly_data['PolicyID'],
                      name='Policy Count', marker_color='lightblue', opacity=0.7),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=900,
                showlegend=True,
                title_text="Insurance Performance Over Time",
                title_font=dict(size=20),
                template='plotly_white',
                hovermode='x unified'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Month", row=1, col=1, tickangle=45)
            fig.update_xaxes(title_text="Month", row=2, col=1, tickangle=45)
            fig.update_xaxes(title_text="Month", row=3, col=1, tickangle=45)
            
            fig.update_yaxes(title_text="Loss Ratio", row=1, col=1)
            fig.update_yaxes(title_text="Amount (R)", row=2, col=1)
            fig.update_yaxes(title_text="Policy Count", row=3, col=1)
            
            return fig
            
        except Exception as e:
            print(f"❌ Error creating interactive plot: {e}")
            return self.create_static_temporal_plot(df, date_col)
    
    def create_static_temporal_plot(self, df: pd.DataFrame, date_col: str):
        """Create static temporal plot"""
        if date_col not in df.columns:
            print(f"❌ {date_col} not found in dataframe!")
            return None
        
        try:
            # Ensure datetime format
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
            df_copy = df_copy.dropna(subset=[date_col])
            
            if len(df_copy) == 0:
                print(f"❌ No valid dates in {date_col}")
                return None
            
            # Group by month
            df_copy['YearMonth'] = df_copy[date_col].dt.to_period('M').astype(str)
            monthly_data = df_copy.groupby('YearMonth').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            
            # Calculate derived metrics
            monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(14, 12))
            
            # 1. Loss Ratio
            axes[0].plot(monthly_data['YearMonth'], monthly_data['LossRatio'], 
                        'r-o', linewidth=2, markersize=8)
            axes[0].axhline(y=1, color='green', linestyle='--', label='Break-even')
            axes[0].set_title('Monthly Loss Ratio Trend', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Loss Ratio')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].tick_params(axis='x', rotation=45)
            
            # 2. Premium vs Claims
            x = range(len(monthly_data))
            width = 0.35
            axes[1].bar([i - width/2 for i in x], monthly_data['TotalPremium'], width, 
                       label='Premium', color='lightseagreen', alpha=0.7)
            axes[1].bar([i + width/2 for i in x], monthly_data['TotalClaims'], width, 
                       label='Claims', color='lightcoral', alpha=0.7)
            axes[1].set_title('Monthly Premium vs Claims', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Amount (R)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(monthly_data['YearMonth'], rotation=45)
            
            # 3. Policy Count
            axes[2].bar(monthly_data['YearMonth'], monthly_data['PolicyID'], 
                       color='lightblue', alpha=0.7)
            axes[2].set_title('Monthly Policy Count', fontsize=14, fontweight='bold')
            axes[2].set_ylabel('Number of Policies')
            axes[2].set_xlabel('Month')
            axes[2].grid(True, alpha=0.3)
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.suptitle('Insurance Performance Over Time', fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"❌ Error creating static plot: {e}")
            return None
    
    def plot_profitability_analysis(self, df: pd.DataFrame):
        """Plot profitability analysis"""
        if 'ProfitMargin' not in df.columns:
            print("❌ ProfitMargin not found in dataframe!")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Profit Margin Distribution
        profit_data = df['ProfitMargin'].dropna()
        if len(profit_data) > 0:
            axes[0, 0].hist(profit_data, bins=50, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
            axes[0, 0].set_title('Profit Margin Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Profit Margin (R)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Profit/Loss Categories
        if 'ProfitMargin' in df.columns:
            conditions = [
                df['ProfitMargin'] > 0,
                df['ProfitMargin'] < 0,
                df['ProfitMargin'] == 0
            ]
            choices = ['Profit', 'Loss', 'Break-even']
            df['ProfitCategory'] = np.select(conditions, choices, default='Unknown')
            profit_counts = df['ProfitCategory'].value_counts()
            
            colors = ['lightgreen', 'lightcoral', 'lightblue']
            axes[0, 1].pie(profit_counts.values, labels=profit_counts.index, 
                          colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Profit/Loss Distribution', fontweight='bold')
        
        # 3. Premium vs Claims Scatter
        if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
            claim_data = df[df['TotalClaims'] > 0]
            if len(claim_data) > 0:
                scatter = axes[1, 0].scatter(claim_data['TotalPremium'], claim_data['TotalClaims'],
                                           alpha=0.6, c='red', s=20)
                axes[1, 0].axline((0, 0), slope=1, color='green', linestyle='--', label='Break-even')
                axes[1, 0].set_title('Premium vs Claims (Policies with Claims)', fontweight='bold')
                axes[1, 0].set_xlabel('Premium (R)')
                axes[1, 0].set_ylabel('Claim Amount (R)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Risk Category Distribution
        if 'RiskCategory' in df.columns:
            risk_counts = df['RiskCategory'].value_counts()
            colors = ['lightgreen', 'gold', 'lightcoral', 'lightgray']
            axes[1, 1].bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='black')
            axes[1, 1].set_title('Risk Category Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Risk Category')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add count labels
            for i, count in enumerate(risk_counts.values):
                axes[1, 1].text(i, count + max(risk_counts.values)*0.01, 
                               f'{int(count):,}', ha='center', fontsize=9)
        
        plt.suptitle('Profitability and Risk Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_missing_data_heatmap(self, df: pd.DataFrame, top_n: int = 20):
        """Plot missing data heatmap for top N columns with most missing values"""
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_pct = missing_pct.sort_values(ascending=False).head(top_n)
        
        if len(missing_pct) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No missing values found', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color gradient based on missing percentage
        colors = []
        for pct in missing_pct.values:
            if pct > 50:
                colors.append('red')
            elif pct > 20:
                colors.append('orange')
            elif pct > 5:
                colors.append('yellow')
            else:
                colors.append('lightgreen')
        
        bars = ax.barh(range(len(missing_pct)), missing_pct.values, color=colors, edgecolor='black')
        ax.set_yticks(range(len(missing_pct)))
        ax.set_yticklabels(missing_pct.index)
        ax.set_xlabel('Percentage Missing (%)')
        ax.set_title(f'Top {len(missing_pct)} Columns with Missing Values', fontsize=16, fontweight='bold')
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, missing_pct.values)):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', ha='left', va='center', fontsize=9)
        
        # Add reference lines
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='>50% = Critical')
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='>20% = High')
        ax.axvline(x=5, color='yellow', linestyle='--', alpha=0.5, label='>5% = Moderate')
        ax.legend()
        
        plt.tight_layout()
        return fig