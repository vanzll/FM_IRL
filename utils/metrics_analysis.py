

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import json
from pathlib import Path

class ExperimentAnalyzer:
    def __init__(self, project_name: str, entity: str = None):
        """
        initialize experiment analyzer
        
        Args:
            project_name: wandb project name
            entity: wandb entity name
        """
        self.project_name = project_name
        self.entity = entity
        self.api = wandb.Api()
        
    def get_runs_data(self, filters: Dict = None) -> pd.DataFrame:
        """
        get experiment run data from wandb
        
        Args:
            filters: filters, e.g. {"config.alg": "drail"}
            
        Returns:
            DataFrame containing all run data
        """
        runs = self.api.runs(f"{self.entity}/{self.project_name}" if self.entity else self.project_name)
        
        data = []
        for run in runs:
            if filters:
                # check if the filter conditions are met
                config = run.config
                if not all(config.get(k.split('.')[-1], None) == v for k, v in filters.items()):
                    continue
                    
            # get history data
            history = run.history()
            if not history.empty:
                # add run information
                history['run_id'] = run.id
                history['run_name'] = run.name
                history['alg'] = run.config.get('alg', 'unknown')
                history['seed'] = run.config.get('seed', 'unknown')
                history['env_name'] = run.config.get('env_name', 'unknown')
                
                data.append(history)
        
        if data:
            return pd.concat(data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def compute_performance_metrics(self, df: pd.DataFrame, 
                                  metric_cols: List[str] = None) -> Dict:
        """
        compute performance metrics statistics
        
        Args:
            df: experiment data DataFrame
            metric_cols: metrics columns to analyze
            
        Returns:
            dictionary containing statistics
        """
        if metric_cols is None:
            metric_cols = ['eval_episode_reward', 'eval_success_rate', 'eval_episode_length']
        
        results = {}
        
        # group by algorithm and compute statistics
        for alg in df['alg'].unique():
            alg_data = df[df['alg'] == alg]
            results[alg] = {}
            
            for metric in metric_cols:
                if metric in alg_data.columns:
                    # compute final performance (average of last 100 episodes)
                    final_performance = alg_data[metric].tail(100).mean()
                    std_performance = alg_data[metric].tail(100).std()
                    
                    # compute convergence speed (number of steps to reach 90% final performance)
                    max_perf = alg_data[metric].max()
                    threshold = 0.9 * max_perf
                    convergence_step = None
                    for i, perf in enumerate(alg_data[metric]):
                        if perf >= threshold:
                            convergence_step = i
                            break
                    
                    results[alg][metric] = {
                        'final_mean': final_performance,
                        'final_std': std_performance,
                        'max_performance': max_perf,
                        'convergence_step': convergence_step,
                        'sample_efficiency': final_performance / (convergence_step + 1) if convergence_step else 0
                    }
        
        return results
    
    def compare_methods(self, results: Dict, metric: str = 'eval_episode_reward') -> pd.DataFrame:
        """
        compare performance of different methods
        
        Args:
            results: performance metrics results
            metric: metric to compare
            
        Returns:
            comparison result DataFrame
        """
        comparison_data = []
        
        for alg, alg_results in results.items():
            if metric in alg_results:
                comparison_data.append({
                    'Method': alg,
                    'Final Performance': alg_results[metric]['final_mean'],
                    'Std': alg_results[metric]['final_std'],
                    'Max Performance': alg_results[metric]['max_performance'],
                    'Convergence Step': alg_results[metric]['convergence_step'],
                    'Sample Efficiency': alg_results[metric]['sample_efficiency']
                })
        
        return pd.DataFrame(comparison_data)
    
    def generate_comparison_plots(self, df: pd.DataFrame, 
                                save_path: str = './results_comparison'):
        """
        生成比较图表
        
        Args:
            df: experiment data DataFrame
            save_path: save path
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. learning curve comparison
        plt.figure(figsize=(12, 8))
        for alg in df['alg'].unique():
            alg_data = df[df['alg'] == alg]
            if 'eval_episode_reward' in alg_data.columns:
                # compute moving average
                reward_ma = alg_data['eval_episode_reward'].rolling(window=50).mean()
                plt.plot(alg_data.index, reward_ma, label=alg, linewidth=2)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Episode Reward')
        plt.title('Learning Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. final performance boxplot
        plt.figure(figsize=(10, 6))
        final_rewards = []
        labels = []
        
        for alg in df['alg'].unique():
            alg_data = df[df['alg'] == alg]
            if 'eval_episode_reward' in alg_data.columns:
                final_rewards.append(alg_data['eval_episode_reward'].tail(100).values)
                labels.append(alg)
        
        plt.boxplot(final_rewards, labels=labels)
        plt.ylabel('Final Episode Reward')
        plt.title('Final Performance Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/final_performance_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. sample efficiency comparison
        plt.figure(figsize=(10, 6))
        sample_efficiency = []
        methods = []
        
        for alg in df['alg'].unique():
            alg_data = df[df['alg'] == alg]
            if 'eval_episode_reward' in alg_data.columns:
                # compute sample efficiency (final performance / convergence steps)
                final_perf = alg_data['eval_episode_reward'].tail(100).mean()
                max_perf = alg_data['eval_episode_reward'].max()
                threshold = 0.9 * max_perf
                
                convergence_step = None
                for i, perf in enumerate(alg_data['eval_episode_reward']):
                    if perf >= threshold:
                        convergence_step = i
                        break
                
                if convergence_step:
                    efficiency = final_perf / (convergence_step + 1)
                    sample_efficiency.append(efficiency)
                    methods.append(alg)
        
        plt.bar(methods, sample_efficiency)
        plt.ylabel('Sample Efficiency')
        plt.title('Sample Efficiency Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_path}/sample_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self, results: Dict, comparison_df: pd.DataFrame, 
                      save_path: str = './results_analysis'):
        """
        export analysis results
        
        Args:
            results: performance metrics results
            comparison_df: comparison result DataFrame
            save_path: save path
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # save detailed results
        with open(f'{save_path}/detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # save comparison results
        comparison_df.to_csv(f'{save_path}/method_comparison.csv', index=False)
        
        # generate LaTeX table
        latex_table = comparison_df.to_latex(index=False, float_format='%.3f')
        with open(f'{save_path}/comparison_table.tex', 'w') as f:
            f.write(latex_table)
        
        print(f"Results saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='experiment result display analysis tool')
    parser.add_argument('--project', type=str, required=True, help='wandb project name')
    parser.add_argument('--entity', type=str, default=None, help='wandb entity name')
    parser.add_argument('--filters', type=str, default='{}', help='filter conditions JSON string')
    parser.add_argument('--save_path', type=str, default='./results_analysis', help='save path')
    
    args = parser.parse_args()
    
    # initialize analyzer
    analyzer = ExperimentAnalyzer(args.project, args.entity)
    
        # parse filter conditions
    filters = json.loads(args.filters) if args.filters else {}
    
    # get data
    print("Getting experiment data from wandb...")
    df = analyzer.get_runs_data(filters)
    
    if df.empty:
        print("No matching experiment data found")
        return
    
    print(f"Found {len(df)} experiment records")
    
    # compute performance metrics
    print("Computing performance metrics...")
    results = analyzer.compute_performance_metrics(df)
    
    # compare methods
    print("Comparing different methods...")
    comparison_df = analyzer.compare_methods(results)
    
    # generate plots
    print("Generating comparison plots...")
    analyzer.generate_comparison_plots(df, args.save_path)
    
    # export results
    print("Exporting results...")
    analyzer.export_results(results, comparison_df, args.save_path)
    
    # print summary
    print("\n=== Experiment result summary ===")
    print(comparison_df.to_string(index=False))
    
    print(f"\nDetailed results saved to: {args.save_path}")

if __name__ == "__main__":
    main() 