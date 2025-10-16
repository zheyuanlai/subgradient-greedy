import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

ACADEMIC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'Palatino', 'Computer Modern Roman'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 0.8,
    'figure.dpi': 110,
}

PALETTE = ['#0C5DA5', '#00A08A', '#F2AD00', '#F98400', '#5BBCD6', '#B40F20']

def use_academic_style(latex=False):
    plt.rcParams.update(ACADEMIC_PARAMS)
    if latex:
        plt.rcParams.update({
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath,amssymb}'
        })
    sns.set_style("whitegrid", {
        'axes.edgecolor': '0.2',
        'axes.linewidth': 0.8,
        'grid.linestyle': '--',
        'grid.linewidth': 0.4
    })
    sns.set_palette(PALETTE)

def finalize_axes(ax):
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5, linestyle='--')
    ax.tick_params(direction='out', length=4, width=0.7)

def save_figure(fig, path_png, path_pdf=None):
    fig.tight_layout()
    fig.savefig(path_png, dpi=300, bbox_inches='tight')
    if path_pdf:
        fig.savefig(path_pdf, bbox_inches='tight')
    return path_png