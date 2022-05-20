from typing import Any, Dict, List, Tuple, Type
import pandas as pd
import numpy as np

from pathlib import Path


def load_results_table(base_dir: str, processed_results_folder: str = 'processed_results') -> pd.DataFrame:
    """Loads the final results table containing accuracies for every seed, domain, lambdas and parameter selection methods.
    Used for presentin in a jupyter noteboook."""
    base_dir = Path(base_dir)
    results_dir = base_dir / processed_results_folder
    acc_file = results_dir / 'accuracies.npz'
    src_tgt_acc_dict = np.load(acc_file, allow_pickle=True)
    list(src_tgt_acc_dict.keys())
    src_acc = src_tgt_acc_dict['src_acc'].item()
    tgt_acc = src_tgt_acc_dict['tgt_acc'].item()
    src_tgt_acc_df = create_results_table(src_acc, tgt_acc)
    return src_tgt_acc_df

def create_ensemble_weights_table(ew_dict: Dict[str, Any], accs_dict: Dict[str, Any]) -> pd.DataFrame:
    """Create a table with the ensemble weights for each domain, seed and ensemble methods for every lambda.

    Args:
        ew_dict (Dict[str, Any]): The ensemble weights dictionary.
        accs_dict (Dict[str, Any]): Source or target accuracy dictionary. Used to get lambdas only
    Returns:
        pd.DataFrame: A table containing the ensemble weights for each lambda.
    """
    seeds, domains, lambdas = _get_idx_lists_from_seed_domain_model_key_dict(accs_dict)
    seeds, domains, ews_keys = _get_idx_lists_from_seed_domain_model_key_dict(ew_dict)
    # this keeps ordering of lambdas (set operation shuffles the list)
    for ensemble_method in ews_keys:
        lambdas.remove(ensemble_method)
    
    ensemble_methods = []
    ew_series_list = []
    for s, domain_dict in ew_dict.items():
        for d, ensemble_method_weights_dict in domain_dict.items():
            if not ensemble_methods:
                ensemble_methods = list(ensemble_method_weights_dict.keys())
            for ensemble_method, ensemble_weights in ensemble_method_weights_dict.items():
                ew_series_list.append(pd.Series(data=ensemble_weights, index=lambdas))
    index = pd.MultiIndex.from_product([seeds, domains, ensemble_methods], names=['seed', 'domains', 'ensemble_methods'])
    results_df = pd.DataFrame(ew_series_list, index=index)
    return results_df

def load_ensemble_weights_table(base_dir: str, processed_results_folder: str = 'processed_results') -> pd.DataFrame:
    base_dir = Path(base_dir)
    results_dir = base_dir / processed_results_folder
    acc_file = results_dir / 'accuracies.npz'
    src_tgt_acc_dict = np.load(acc_file, allow_pickle=True)
    list(src_tgt_acc_dict.keys())
    src_acc = src_tgt_acc_dict['src_acc'].item()
    tgt_acc = src_tgt_acc_dict['tgt_acc'].item()
    ensemble_weights = src_tgt_acc_dict['ensemble_weights'].item()
    ensemble_weights_df = create_ensemble_weights_table(ensemble_weights, src_acc)    
    return ensemble_weights_df

def create_results_table(src_acc: Dict, tgt_acc: Dict) -> pd.DataFrame:
    src_acc_df = _create_results_table(src_acc)
    tgt_acc_df = _create_results_table(tgt_acc)
    src_tgt_acc_df = _combine_src_target_table(src_acc_df, tgt_acc_df)
    return src_tgt_acc_df

def _combine_src_target_table(src_df: pd.DataFrame, tgt_df: pd.DataFrame) -> pd.DataFrame:
    src_tgt_acc_df = pd.concat({'source': src_df.copy(), 'target': tgt_df.copy()}, names=['domain'])
    src_tgt_acc_df = src_tgt_acc_df.swaplevel(0, 2).sort_index(level='seed').sort_index(level='domains')
    return src_tgt_acc_df

def _create_results_table(accs_dict: dict, add_lambda_mean_median: bool = False) -> pd.DataFrame:
    """Create a pandas dataframe from results dict."""
    seeds = list(accs_dict.keys())
    domains = []
    lambdas = []

    accs_series_list = []
    for s, domain_dict in accs_dict.items():
        if not domains:
            domains = list(domain_dict.keys())
        for d, lambda_dict in domain_dict.items():
            if not lambdas:
                lambdas = list(lambda_dict.keys())
            accs_series_list.append(pd.Series(lambda_dict))
    index = pd.MultiIndex.from_product([seeds, domains], names=['seed', 'domains'])
    results_df = pd.DataFrame(accs_series_list, index=index)
    if add_lambda_mean_median:
        results_df = _add_lambda_mean_median_column(results_df)
    return results_df

def _add_lambda_mean_median_column(results_df: pd.DataFrame, parameter_sel_indices: List[str] = ['agg', 'multi_reg', 'bp', 'dev', 'iwv']) -> pd.DataFrame:
    # get dataframe with lambda results only (without parameter selection methods):
    # create a list with the columns to remove
    available_param_sel_methods = list(set(results_df.columns.values) & set(parameter_sel_indices))

    lambda_df = results_df.drop(labels=available_param_sel_methods, axis=1)

    results_df = results_df.copy()
    results_df['lam_mean'] = lambda_df.mean(axis=1)
    results_df['lam_median'] = lambda_df.median(axis=1)
    return results_df

def _get_idx_lists_from_seed_domain_model_key_dict(accs_dict):
    seeds = list(accs_dict.keys())
    domains = []
    lambdas = []
    for s, domain_dict in accs_dict.items():
        if not domains:
            domains = list(domain_dict.keys())
        for d, lambda_dict in domain_dict.items():
            if not lambdas:
                lambdas = list(lambda_dict.keys())
    return seeds, domains, lambdas