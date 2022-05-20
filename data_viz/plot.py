from typing import Any, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Helpers:


def _is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def _get_style(type, distr):
    """type is either a domain (src and target) or an ensemble method.
    distr, specifies source or target accuracy."""
    ls = '-'
    col = None
    marker = ''
    label = f'{type}_{distr}'
    lw = None

    if type == 'agg' and distr == 'target':
        label = 'IWA target'
        col = '#990606'
        lw = 4
    elif type == 'agg' and distr == 'source':
        label = 'IWA source'
        col = '#f59393'
        lw = 4
    elif type == 'source_reg' and distr == 'target':
        label = 'Source-only regression target'
        col = 'grey'
        lw = 4
    elif type == 'dev' and distr == 'target':
        label = 'DEV target'
        col = 'grey'
        lw = 4

    elif type == 'moons_src-moons_tgt':
        if distr == 'source':
            label = 'Individual models source'
            ls = '-'
            col = 'green'
            marker = 's'
            lw = 4
        else:
            label = 'Individual models target'
            ls = '--'
            col = 'orange'
            marker = 'o'
            lw = 4
            pass

    return ls, lw, col, label, marker


# Plotting functions:

def paperplot_lambdas_vs_accuracy_barplot(
        df_acc: pd.DataFrame, df_ew: pd.DataFrame, domains: List[str] = [],
        lambdas: List[str] = [],
        ensemble_methods: List[str] = [],
        title=None, style=None, cmap='Greys', bar_width=0.7, ax=None, figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54)):
    # get domains
    if not domains:
        domains = list(df_acc.index.get_level_values('domains').unique())

    assert len(domains) == 1
    if ax is None:
        f, axs = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
        axs = [axs]
    else:
        f = None
        axs = [ax]

    # get lambdas
    if not lambdas:
        lambdas = [l for l in df_acc.columns if _is_float(l)]
        if not lambdas:
            raise ValueError('No lambda columns in dataframe!')

    # get ensemble methods
    if not ensemble_methods:
        ensemble_methods = ['agg']
        if not ensemble_methods:
            raise ValueError('No ensemble methods in dataframe!')

    plt.setp(axs[0], xticks=np.arange(len(lambdas)), xticklabels=[float(l) for l in lambdas])
    my_cmap = plt.get_cmap(cmap)
    for i, d in enumerate(domains):
        # accuracy
        domain_df_acc_lambdas = df_acc.loc[d][lambdas]
        # ensemble weights
        domain_df_ew_lambdas = df_ew.swaplevel(0, 1).loc[d][lambdas]
        domain_df_acc_ems = df_acc.loc[d][ensemble_methods]
        for domain in ['target']:
            # plot accuracy of lambda models
            vals_acc = domain_df_acc_lambdas.swaplevel().loc[domain]
            vals_ew = domain_df_ew_lambdas.swaplevel(0, 1).loc['agg']
            val_ew = vals_ew.mean(axis=0)
            val = vals_acc.mean(axis=0)
            std = vals_acc.std(axis=0)
            # lower, upper = val - std, val + std

            ls, lw, col, label, marker = _get_style(d, domain)
            if style is not None and d in style.keys():
                ls = style[d].get('ls', ls)
                col = style[d].get('col', col)
                label = style[d].get('label', label)
                marker = style[d].get('marker', marker)

            # axs[0].plot(val, ls=ls, label=label, c=col, marker=marker)
            axs[0].errorbar(x=np.arange(len(val)), y=val, yerr=std, fmt='none', marker=marker,
                            ls=None, c='black', capsize=4)

            norm = mpl.colors.Normalize(vmin=np.min(val_ew), vmax=np.max(val_ew))
            axs[0].bar(x=np.arange(len(val)), height=val,
                       width=bar_width, color=my_cmap(norm(val_ew)))

            # plot accuracy of ensemble methods only aggregation method is left in dataframe
            vals_acc = domain_df_acc_ems.swaplevel().loc[domain]

            # values and std for every ensemble method
            val = vals_acc.mean(axis=0)
            std = vals_acc.std(axis=0)
            lower, upper = val - std, val + std

            xs = np.zeros(len(lambdas)+2)
            xs[0] = -bar_width/2
            xs[1:len(lambdas)+1] = np.arange(len(lambdas))
            xs[-1] = len(lambdas)-1+bar_width/2

            for j, em in enumerate(vals_acc.columns):
                v = val[em] * np.ones(len(lambdas)+2)
                l = lower[em] * np.ones(len(lambdas)+2)
                u = upper[em] * np.ones(len(lambdas)+2)

                ls, lw, col, label, marker = _get_style(em, domain)
                if style is not None and d in style.keys():
                    ls = style[d].get('ls', ls)
                    col = style[d].get('col', col)
                    label = style[d].get('label', label)
                    marker = style[d].get('marker', marker)

                # axs[0].errorbar(x=-0.3 - j * 0.3, y=val[em], yerr=std[em], marker=marker,
                #                 ls=ls, c=col, label=label, linewidth=lw, capsize=4)
                axs[0].plot(xs, v, ls=ls, label=label, c=col, marker=marker)
                axs[0].fill_between(x=xs, y1=l, y2=u, color=col, alpha=0.4)

    axs[0].set_xlabel(r'$\lambda$')
    axs[0].set_ylabel('Target accuracy')
    axs[0].legend()
    # axs[0].grid()
    axs[0].set_ylim(0.7, 1)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Ensemble weight')
    _ = plt.tight_layout()
    axs[0].get_figure().autofmt_xdate()
    return f

def paperplot2_lambdas_vs_accuracy_vs_ensemble_weights_bars(df_acc: pd.DataFrame, df_ew: pd.DataFrame, domains: List[str] = [], lambdas: List[str] = [], 
                                                  ensemble_methods: List[str] = ['agg', 'source_reg'],
                                                  title=None,
                                                  alpha=0.2,
                                                  bar_width=0.7,
                                                  plot_errorbars=False,
                                                  figsize=(2 * 12 * 1 / 2.54, 2 * 12 * 1 / 2.54)):
    # get domains
    if not domains:
        domains = list(df_acc.index.get_level_values('domains').unique())

    assert len(domains) == 1
    f, (ax0, ax1) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    f.suptitle(title)
    axs = [ax0, ax1]
    axs[0].grid(alpha=0.3)
    axs[1].grid(alpha=0.3)

    # get lambdas
    if not lambdas:
        lambdas = [l for l in df_acc.columns if _is_float(l)]
        if not lambdas:
            raise ValueError('No lambda columns in dataframe!')

    # get ensemble methods
    if not ensemble_methods:
        # ensemble_methods = ['agg']
        if not ensemble_methods:
            raise ValueError('No ensemble methods in dataframe!')

    xs = np.arange(len(lambdas))
    plt.setp(axs[0], xticks=xs, xticklabels=[float(l) for l in lambdas])  
    for i, d in enumerate(domains):
        domain_df_lambdas = df_acc.loc[d][lambdas]
        domain_df_ems = df_acc.loc[d][ensemble_methods]
        for domain in ['source', 'target']:
            #! plot accuracy of lambda models
            vals = domain_df_lambdas.swaplevel().loc[domain]

            val = vals.mean(axis=0)
            std = vals.std(axis=0)
            lower, upper = val - std, val + std

            ls, lw, col, label, marker = _get_style(d, domain)

            axs[0].plot(val, ls=ls, label=label, c=col, marker=marker, lw=lw, ms=2.5*lw)
            if plot_errorbars:
                axs[0].fill_between(x=xs, y1=lower, y2=upper, color=col, alpha=alpha)

            # axs[0].errorbar(x=np.arange(len(val)), y=val, yerr=std, marker=marker,
            #                 ls=ls, c=col, label=label, linewidth=lw, capsize=4)

            #! plot accuracy of ensemble methods
            if domain == 'target':
                vals = domain_df_ems.swaplevel().loc[domain]

                # values and std for every ensemble method
                val = vals.mean(axis=0)
                std = vals.std(axis=0)
                lower, upper = val - std, val + std


                # plot every ensemble method
                for j, em in enumerate(vals.columns):
                    v = val[em] * np.ones(len(lambdas))
                    l = lower[em] * np.ones(len(lambdas))
                    u = upper[em] * np.ones(len(lambdas))

                    ls, lw, col, label, marker = _get_style(em, domain)

                    # axs[0].errorbar(x=-0.3 - j * 0.3, y=val[em], yerr=std[em], marker=marker,
                    #                 ls=ls, c=col, label=label, linewidth=lw, capsize=4)
                    axs[0].plot(xs,v, ls=ls, label=label, c=col, marker=marker, lw=lw)
                    if plot_errorbars:
                        axs[0].fill_between(x=xs, y1=l, y2=u, color=col, alpha=alpha)

    
    bar_width = 1. / (len(ensemble_methods)+1)
    ind = np.arange(len(lambdas))

    for i, d in enumerate(domains):
        domain_df_lambdas = df_ew.swaplevel(0,1).loc[d][lambdas]
        for j, em in enumerate(ensemble_methods):
            # plot ensemble weights
            vals = domain_df_lambdas.swaplevel(0,1).loc[em]
            val = vals.mean(axis=0)
            std = vals.std(axis=0)
            lower, upper = val - std, val + std
            ls, lw, col, label, marker = _get_style(em, 'target')
            axs[1].bar(ind+(j*bar_width), val, bar_width, label=em, align='center', color=col)


    axs[1].set_xlabel(r'$\lambda$')
    axs[0].set_ylabel('Accuracy')
    axs[1].set_ylabel('Ensemble weights')
    axs[0].legend(loc=1, prop={'size':13})
    axs[0].set_ylim(0.75,1)
    axs[1].set_yticks([1.0,0,-1.0])
    # axs[1].set_yscale('log')

    _ = plt.tight_layout()
    axs[0].get_figure().autofmt_xdate()
    return f