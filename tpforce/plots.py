from __future__ import division, unicode_literals
import numpy as np
from pandas import DataFrame
from functools import wraps
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


def make_axes(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get('ax') is None:
            kwargs['ax'] = plt.gca()
            # Delete legend keyword so remaining ones can be passed to plot().
            try:
                legend = kwargs['legend']
            except KeyError:
                legend = None
            else:
                del kwargs['legend']
            try:
                filename = kwargs['filename']
            except KeyError:
                filename = None
            else:
                del kwargs['filename']
            result = func(*args, **kwargs)
            if not (kwargs['ax'].get_legend_handles_labels() == ([], []) or \
                    legend is False):
                plt.legend(loc='best')
            if filename is not None:
                save_ax(result, filename)
            plt.show()
            return result
        else:
            return func(*args, **kwargs)
    return wrapper


def save_ax(ax, filename, **kwargs):
    fig = ax.get_figure()
    _kwargs = dict(bbox_inches="tight", facecolor=fig.get_facecolor(),
                   edgecolor='none')
    _kwargs.update(kwargs)
    _, ext = os.path.splitext(filename)
    if ext == '':
        fig.savefig(filename + '.png', **kwargs)
        fig.savefig(filename + '.pdf', **kwargs)
    else:
        fig.savefig(filename, **kwargs)

@make_axes
def plot_rdf(bins, hist, diameter=None, ax=None, err=None, **plot_kwargs):
    if diameter:
        bins = bins / diameter
    ax.plot(bins, hist, **plot_kwargs)
    ax.plot([0, bins.max()], [1, 1], ls='--', color='green')
    if diameter:
        ax.plot([1, 1], [0, np.max(hist) + 0.1], color='red')
        ax.set_xlabel('r [diameters]')
    else:
        if mpl.rcParams['text.usetex']:
            ax.set_xlabel(r'r [\textmu m]')
        else:
            ax.set_xlabel('r [\xb5m]')
    ax.set_ylabel(r'g(r)')
    ylim = hist.max() + 0.1

    if 'color' in plot_kwargs:
        color = plot_kwargs['color']
    else:
        color = None

    if err is not None:
        lowerbound = (hist - err).clip(0, ylim)
        higherbound = (hist + err).clip(0, ylim)
        ax.fill_between(bins, lowerbound, higherbound, color=color, alpha=0.1)

    ax.set_xlim(0, bins.max())
    ax.set_ylim(0, ylim)
    ax.grid()
    return ax

@make_axes
def plot_energy(x, u, diameter=None, ax=None, err=None, **plot_kwargs):
    if diameter:
        x = x / diameter
    u_finite = u[np.isfinite(u)]
    x_finite = x[np.isfinite(u)]
    ylim = [np.min(u_finite)-0.5, np.max(u_finite)+2]

    ax.plot(x_finite, u_finite, **plot_kwargs)
    if diameter:
        ax.plot([1, 1], ylim, color='red')
        ax.set_xlabel('r [diameters]')
    else:
        if mpl.rcParams['text.usetex']:
            ax.set_xlabel(r'$r [\textmu m]$')
        else:
            ax.set_xlabel('r [\xb5m]')
    if mpl.rcParams['text.usetex']:
        ax.set_ylabel(r'$u [k_B T]$')
    else:
        ax.set_ylabel(r'u [kT]')

    if 'color' in plot_kwargs:
        color = plot_kwargs['color']
    else:
        color = None

    if err is not None:
        lowerbound = err[0].clip(*ylim)
        higherbound = err[1].clip(*ylim)
        higherbound[~np.isfinite(higherbound)] = ylim[1] + 1
        lowerbound[~np.isfinite(lowerbound)] = ylim[0] - 1
        ax.fill_between(x, lowerbound, higherbound, color=color, alpha=0.1)

    ax.set_xlim(0, x.max())
    ax.set_ylim(*ylim)
    ax.grid()
    return ax


@make_axes
def plot_transition(s_bins, trans, dilution=1, ax=None):
    for x in range(trans.shape[0])[::dilution]:
        line = ax.plot([s_bins[x]]*2, [0, 1])[0]
        plt.plot(s_bins, trans[x], color=line.get_color())

    ax.set_xlabel(r'$s, s_0 [\mu m]$')
    ax.set_ylabel('$P(s, dt | s_0, 0)$')
    return ax
