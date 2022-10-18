import pandas as pd
import matplotlib.pyplot as plt

def plot_sweep_summary(summary_df: pd.DataFrame,
                       x_axis: str,
                       y_axis: str,
                       compare_parameter: str,
                       run_level_name: str = 'values',
                       title=None,
                       ax=None,
                       figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54)):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    # select rows from compare parameter
    comp_param_vals = summary_df[compare_parameter].unique()
    comp_param_vals.sort()
    # sort along x_axis
    summary_df = summary_df.sort_values(by=x_axis, axis=0)

    comp_param_str = compare_parameter.split('.')[-1]

    # get x and y axis
    for cpv in comp_param_vals:
        df = summary_df.loc[summary_df[compare_parameter] == cpv].drop(compare_parameter, axis=1)
        x_vals = df[x_axis].values
        y_vals = df[y_axis].values
        ax.plot(x_vals, y_vals, label=f'{comp_param_str}={cpv}')

    ax.legend()
    ax.grid()
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return f


def _create_concatenated_df(summary_df: pd.DataFrame, x_axis: str, y_axis: str, compare_parameter: str, run_level_name: str = 'values'):
    # select rows from compare parameter
    comp_param_vals = summary_df[compare_parameter].unique()
    comp_param_vals.sort()
    # sort along x_axis
    summary_df = summary_df.sort_values(by=x_axis, axis=0)    

    # create concatenated df from compare_parameter
    names = [compare_parameter, run_level_name]
    comp_dfs = {}

    # get x and y axis
    for cpv in comp_param_vals:
        df = summary_df.loc[summary_df[compare_parameter] == cpv].drop(compare_parameter, axis=1)
        comp_dfs[cpv] = df
    conc_df = pd.concat(comp_dfs, names=names, axis=0)
    return conc_df