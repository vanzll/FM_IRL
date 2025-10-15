
from typing import Dict, Any, Set, List
from collections import defaultdict

class MetricsFilter:
    """Metrics filter class"""
    
    def __init__(self, allowed_metrics: List[str] = None):
        """
        initialize metrics filter
        
        Args:
            allowed_metrics: allowed metrics list, if None then use default configuration
        """
        if allowed_metrics is None:
            # default core metrics
            self.allowed_metrics = {
                'avg_r',              # average episode reward
                'avg_ep_found_goal',  # average episode success rate
                'dist_entropy',       # policy distribution entropy
                'timestamp',          # timestamp
                'step',              # current training step
                'episode'            # current episode
            }
        else:
            self.allowed_metrics = set(allowed_metrics)
        
        print(f"Metrics filter initialized, allowed metrics: {sorted(self.allowed_metrics)}")
    
    def filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        filter metrics dictionary, only keep allowed metrics
        
        Args:
            metrics: original metrics dictionary
            
        Returns:
            filtered metrics dictionary
        """
        filtered = {}
        
        for key, value in metrics.items():
            # strictly match metric name
            if key in self.allowed_metrics:
                filtered[key] = value
        
        return filtered
    
    def is_allowed(self, metric_name: str) -> bool:
        """
            check if metric is allowed
        
        Args:
            metric_name: metric name
            
        Returns:
            bool: True if metric is allowed
        """
        return metric_name in self.allowed_metrics
    
    def add_metric(self, metric_name: str):
        """add allowed metric"""
        self.allowed_metrics.add(metric_name)
        print(f"Added metric: {metric_name}")
    
    def remove_metric(self, metric_name: str):
        """remove allowed metric"""
        if metric_name in self.allowed_metrics:
            self.allowed_metrics.remove(metric_name)
            print(f"Removed metric: {metric_name}")
    
    def get_allowed_metrics(self) -> Set[str]:
        """get current allowed metrics set"""
        return self.allowed_metrics.copy()


# 全局指标过滤器实例
_global_filter: MetricsFilter = None

def init_metrics_filter(allowed_metrics: List[str] = None) -> MetricsFilter:
    """
    initialize global metrics filter
    
    Args:
        allowed_metrics: allowed metrics list
        
    Returns:
        metrics filter instance
    """
    global _global_filter
    _global_filter = MetricsFilter(allowed_metrics)
    return _global_filter

def filter_log_vals(log_vals: Dict[str, Any]) -> Dict[str, Any]:
    """
        filter log metrics values
    
    Args:
        log_vals: original log metrics values
        
    Returns:
        filtered log metrics values
    """
    if _global_filter is None:
        # if not initialized, use default configuration
        init_metrics_filter()
    
    return _global_filter.filter_metrics(log_vals)

def get_filter() -> MetricsFilter:
    """get global metrics filter"""
    if _global_filter is None:
        init_metrics_filter()
    return _global_filter