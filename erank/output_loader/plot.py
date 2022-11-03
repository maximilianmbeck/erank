from typing import Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_sweep_summary(
    summary_df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    compare_parameter: str,
    compare_parameter_val_selection: List[Any] = [],
    title: str = None,
    ax=None,
    ylim: Tuple[float, float] = (),
    figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54)) -> mpl.figure.Figure:
    """Function for creating a sweep summary plot.
    Allows for selecting the x- and y-axis parameters separately. 
    Typical setup: 
    - x-axis: A sweep parameter, e.g. `data.dataset_kwargs.rotation_angle`
    - y-axis: A metric, e.g. `Accuracy-train_step-0`
    - compare_parameter: Compare different parameter setups, e.g. `init_model_step`

    Args:
        summary_df (pd.DataFrame): The sweep summary dataframe.
        x_axis (str): Parameter to plot on the x-axis.
        y_axis (str): Parameter to plot on the y-axis.
        compare_parameter (str): The compare parameter. Plot a line for each parameter.
        compare_parameter_val_selection (List[Any], optional): If specified, plot only these values. Plots all otherwise. Defaults to [].
        title (str, optional): The title. Defaults to None.
        ax (, optional): The Axes. Defaults to None.
        ylim (Tuple[float, float], optional): 
        figsize (tuple, optional): Size of the Figure. Defaults to (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54).

    Returns:
        Figure: The matplotlib figure.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    # select rows from compare parameter
    comp_param_vals = summary_df[compare_parameter].unique()
    comp_param_vals.sort()
    if compare_parameter_val_selection:
        comp_val_sel = np.array(compare_parameter_val_selection)
        comp_param_vals = np.intersect1d(comp_param_vals, comp_val_sel)
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
    if ylim:
        ax.set_ylim(*ylim)
    return f