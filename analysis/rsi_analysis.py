#!/usr/bin/env python3
"""
RSI Data Analysis Script
1. Plot time series of all RSI timeframes (4h, 12h, 1d, 3d)
2. Analyze RSI distributions and identify oversold/overbought thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RSIAnalyzer:
    """Analyze multi-timeframe RSI data from merged crypto files"""
    
    def __init__(self, data_dir="./"):
        self.data_dir = Path(data_dir)
        self.merged_files = list(self.data_dir.glob("merged_*_4h_with_multi_tf_rsi.csv"))
        self.data = {}
        
    def load_all_data(self):
        """Load all merged RSI data files"""
        print("Loading merged RSI data files...")
        
        for file_path in self.merged_files:
            symbol = file_path.stem.replace('merged_', '').replace('_4h_with_multi_tf_rsi', '')
            print(f"  Loading {symbol}...")
            
            df = pd.read_csv(file_path)
            df['time'] = pd.to_datetime(df['time'])
            df['symbol'] = symbol
            
            self.data[symbol] = df
        
        print(f"Loaded {len(self.data)} datasets")
        return self.data
    
    def plot_rsi_time_series(self, symbols=None, save_plots=True):
        """Plot RSI time series for all timeframes"""
        if not self.data:
            self.load_all_data()
        
        symbols_to_plot = symbols or list(self.data.keys())[:4]  # Plot first 4 if not specified
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(len(symbols_to_plot), 1, figsize=(15, 4*len(symbols_to_plot)))
        if len(symbols_to_plot) == 1:
            axes = [axes]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        rsi_columns = ['RSI_4h', 'RSI_12h', 'RSI_1d', 'RSI_3d']
        labels = ['4H RSI', '12H RSI', '1D RSI', '3D RSI']
        
        for idx, symbol in enumerate(symbols_to_plot):
            df = self.data[symbol]
            ax = axes[idx]
            
            # Plot each RSI timeframe
            for i, (col, label, color) in enumerate(zip(rsi_columns, labels, colors)):
                if col in df.columns:
                    ax.plot(df['time'], df[col], label=label, color=color, alpha=0.8, linewidth=1.5)
            
            # Add horizontal reference lines
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            ax.axhline(y=50, color='gray', linestyle='-', alpha=0.3, label='Neutral (50)')
            
            # Styling
            ax.set_title(f'{symbol.replace("_", " ").title()} - Multi-Timeframe RSI', fontsize=14, fontweight='bold')
            ax.set_ylabel('RSI', fontsize=12)
            ax.set_ylim(0, 100)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45)
        
        plt.xlabel('Time', fontsize=12)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('rsi_time_series_analysis.png', dpi=300, bbox_inches='tight')
            print("📈 Saved time series plot: rsi_time_series_analysis.png")
        
        plt.show()
    
    def analyze_rsi_distributions(self, save_plots=True):
        """Analyze RSI distributions and identify key thresholds"""
        if not self.data:
            self.load_all_data()
        
        # Combine all RSI data across symbols
        all_rsi_data = []
        
        for symbol, df in self.data.items():
            for col in ['RSI_4h', 'RSI_12h', 'RSI_1d', 'RSI_3d']:
                if col in df.columns:
                    timeframe = col.replace('RSI_', '')
                    rsi_values = df[col].dropna()
                    
                    for value in rsi_values:
                        all_rsi_data.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'rsi': value
                        })
        
        rsi_df = pd.DataFrame(all_rsi_data)
        
        # Create comprehensive analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribution by timeframe
        ax1 = axes[0, 0]
        timeframes = ['4h', '12h', '1d', '3d']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for tf, color in zip(timeframes, colors):
            data = rsi_df[rsi_df['timeframe'] == tf]['rsi']
            ax1.hist(data, bins=50, alpha=0.6, label=f'{tf.upper()} RSI', color=color, density=True)
        
        ax1.axvline(x=30, color='green', linestyle='--', alpha=0.7, label='Traditional Oversold (30)')
        ax1.axvline(x=70, color='red', linestyle='--', alpha=0.7, label='Traditional Overbought (70)')
        ax1.set_title('RSI Distribution by Timeframe', fontsize=14, fontweight='bold')
        ax1.set_xlabel('RSI Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot by timeframe  
        ax2 = axes[0, 1]
        sns.boxplot(data=rsi_df, x='timeframe', y='rsi', ax=ax2, palette=colors)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('RSI Box Plot by Timeframe', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timeframe')
        ax2.set_ylabel('RSI Value')
        ax2.grid(True, alpha=0.3)
        
        # 3. Percentile analysis for 4h RSI (most relevant for trading)
        ax3 = axes[1, 0]
        rsi_4h_data = rsi_df[rsi_df['timeframe'] == '4h']['rsi']
        
        # Calculate key percentiles
        percentiles = [1, 5, 10, 20, 25, 75, 80, 90, 95, 99]
        pct_values = np.percentile(rsi_4h_data, percentiles)
        
        ax3.hist(rsi_4h_data, bins=50, alpha=0.7, color='#1f77b4', density=True)
        
        # Mark key percentiles
        for pct, val in zip(percentiles, pct_values):
            if pct <= 25:  # Oversold candidates
                ax3.axvline(x=val, color='green', linestyle=':', alpha=0.8, 
                           label=f'{pct}th percentile ({val:.1f})')
            elif pct >= 75:  # Overbought candidates  
                ax3.axvline(x=val, color='red', linestyle=':', alpha=0.8,
                           label=f'{pct}th percentile ({val:.1f})')
        
        ax3.axvline(x=30, color='green', linestyle='--', linewidth=2, label='Traditional Oversold (30)')
        ax3.axvline(x=70, color='red', linestyle='--', linewidth=2, label='Traditional Overbought (70)')
        
        ax3.set_title('4H RSI Distribution with Percentiles', fontsize=14, fontweight='bold')
        ax3.set_xlabel('RSI Value')
        ax3.set_ylabel('Density')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistical summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics for each timeframe
        stats_data = []
        for tf in timeframes:
            data = rsi_df[rsi_df['timeframe'] == tf]['rsi']
            stats_data.append({
                'Timeframe': tf.upper(),
                'Mean': f"{data.mean():.1f}",
                'Std': f"{data.std():.1f}",
                '5th Pct': f"{np.percentile(data, 5):.1f}",
                '25th Pct': f"{np.percentile(data, 25):.1f}",
                '75th Pct': f"{np.percentile(data, 75):.1f}",
                '95th Pct': f"{np.percentile(data, 95):.1f}",
                'Min': f"{data.min():.1f}",
                'Max': f"{data.max():.1f}"
            })
        
        stats_table = pd.DataFrame(stats_data)
        table = ax4.table(cellText=stats_table.values, colLabels=stats_table.columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('RSI Statistical Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('rsi_distribution_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 Saved distribution plot: rsi_distribution_analysis.png")
        
        plt.show()
        
        # Print key insights
        print("\n" + "="*60)
        print("📈 RSI ANALYSIS INSIGHTS")
        print("="*60)
        
        rsi_4h = rsi_df[rsi_df['timeframe'] == '4h']['rsi']
        
        print(f"\n🎯 4H RSI Key Levels:")
        print(f"   • 5th Percentile (Very Oversold): {np.percentile(rsi_4h, 5):.1f}")
        print(f"   • 10th Percentile (Oversold): {np.percentile(rsi_4h, 10):.1f}")
        print(f"   • 25th Percentile (Weak): {np.percentile(rsi_4h, 25):.1f}")
        print(f"   • 50th Percentile (Median): {np.percentile(rsi_4h, 50):.1f}")
        print(f"   • 75th Percentile (Strong): {np.percentile(rsi_4h, 75):.1f}")
        print(f"   • 90th Percentile (Overbought): {np.percentile(rsi_4h, 90):.1f}")
        print(f"   • 95th Percentile (Very Overbought): {np.percentile(rsi_4h, 95):.1f}")
        
        print(f"\n🔍 Recommended Thresholds:")
        oversold_5pct = np.percentile(rsi_4h, 5)
        oversold_10pct = np.percentile(rsi_4h, 10)
        overbought_90pct = np.percentile(rsi_4h, 90)
        overbought_95pct = np.percentile(rsi_4h, 95)
        
        print(f"   • Aggressive Oversold: ≤ {oversold_5pct:.1f} (5% of time)")
        print(f"   • Conservative Oversold: ≤ {oversold_10pct:.1f} (10% of time)")  
        print(f"   • Conservative Overbought: ≥ {overbought_90pct:.1f} (10% of time)")
        print(f"   • Aggressive Overbought: ≥ {overbought_95pct:.1f} (5% of time)")
        
        print(f"\n📊 Traditional vs Data-Driven Comparison:")
        print(f"   • Traditional Oversold (30): {(rsi_4h <= 30).mean()*100:.1f}% of time")
        print(f"   • Traditional Overbought (70): {(rsi_4h >= 70).mean()*100:.1f}% of time")
        print(f"   • Data suggests more nuanced thresholds based on actual distribution")
        
        return stats_table
    
    def analyze_symbol_comparison(self, save_plots=True):
        """Compare RSI characteristics across different crypto symbols"""
        if not self.data:
            self.load_all_data()
        
        # Prepare data for comparison
        symbol_stats = []
        
        for symbol, df in self.data.items():
            rsi_4h = df['RSI_4h'].dropna()
            
            if len(rsi_4h) > 0:
                symbol_stats.append({
                    'Symbol': symbol.replace('_', ' ').title(),
                    'Mean RSI': rsi_4h.mean(),
                    'Volatility (Std)': rsi_4h.std(),
                    'Oversold (10th pct)': np.percentile(rsi_4h, 10),
                    'Overbought (90th pct)': np.percentile(rsi_4h, 90),
                    'Data Points': len(rsi_4h)
                })
        
        comparison_df = pd.DataFrame(symbol_stats)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mean RSI comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(comparison_df)), comparison_df['Mean RSI'], 
                       color='steelblue', alpha=0.7)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Neutral (50)')
        ax1.set_title('Mean RSI by Symbol', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean RSI')
        ax1.set_xticks(range(len(comparison_df)))
        ax1.set_xticklabels(comparison_df['Symbol'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 2. RSI volatility comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(comparison_df)), comparison_df['Volatility (Std)'],
                       color='orange', alpha=0.7)
        ax2.set_title('RSI Volatility by Symbol', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RSI Standard Deviation')
        ax2.set_xticks(range(len(comparison_df)))
        ax2.set_xticklabels(comparison_df['Symbol'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 3. Oversold/Overbought thresholds
        ax3 = axes[1, 0]
        width = 0.35
        x = np.arange(len(comparison_df))
        
        bars3a = ax3.bar(x - width/2, comparison_df['Oversold (10th pct)'], width,
                        label='Oversold (10th pct)', color='green', alpha=0.7)
        bars3b = ax3.bar(x + width/2, comparison_df['Overbought (90th pct)'], width,
                        label='Overbought (90th pct)', color='red', alpha=0.7)
        
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Traditional Oversold')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Traditional Overbought')
        
        ax3.set_title('RSI Thresholds by Symbol', fontsize=14, fontweight='bold')
        ax3.set_ylabel('RSI Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels(comparison_df['Symbol'], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Data summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Format the comparison table for display
        display_df = comparison_df.copy()
        for col in ['Mean RSI', 'Volatility (Std)', 'Oversold (10th pct)', 'Overbought (90th pct)']:
            display_df[col] = display_df[col].round(1)
        
        table = ax4.table(cellText=display_df.values, colLabels=display_df.columns,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Symbol Comparison Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('rsi_symbol_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 Saved symbol comparison: rsi_symbol_comparison.png")
        
        plt.show()
        
        return comparison_df
    
    def run_full_analysis(self):
        """Run complete RSI analysis workflow"""
        print("🚀 Starting comprehensive RSI analysis...")
        print("="*60)
        
        # 1. Load data
        self.load_all_data()
        
        # 2. Time series analysis
        print("\n📈 1. Creating RSI time series plots...")
        self.plot_rsi_time_series()
        
        # 3. Distribution analysis  
        print("\n📊 2. Analyzing RSI distributions...")
        stats_table = self.analyze_rsi_distributions()
        
        # 4. Symbol comparison
        print("\n🔍 3. Comparing symbols...")
        comparison_df = self.analyze_symbol_comparison()
        
        print("\n✅ RSI analysis complete!")
        print("📁 Generated files:")
        print("   • rsi_time_series_analysis.png")
        print("   • rsi_distribution_analysis.png") 
        print("   • rsi_symbol_comparison.png")
        
        return stats_table, comparison_df


def main():
    """Run RSI analysis"""
    analyzer = RSIAnalyzer()
    
    # Run full analysis
    stats, comparison = analyzer.run_full_analysis()
    
    # Optional: Save results to CSV
    stats.to_csv('rsi_statistics_summary.csv', index=False)
    comparison.to_csv('rsi_symbol_comparison.csv', index=False)
    print("\n💾 Saved CSV reports:")
    print("   • rsi_statistics_summary.csv")
    print("   • rsi_symbol_comparison.csv")


if __name__ == "__main__":
    main()