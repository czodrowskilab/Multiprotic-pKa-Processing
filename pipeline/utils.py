from collections import defaultdict as ddict
from typing import List, Dict, Any, Iterable, Set, Tuple, Iterator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from rdkit import Chem
from rdkit.Chem import Draw, Mol
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use('agg')


class Node:
    def __init__(self, mol: Mol, sma: str, parent: 'Node' = None, children: List['Node'] = ()):
        self.parent = parent
        self.children = children
        self.mol = mol
        self.sma = sma


def gen_children(node: Node, smarts: Iterable[str]) -> List[Node]:
    children = []
    for sma in smarts:
        sm = Chem.MolFromSmiles(sma, sanitize=False)
        for s in sm.GetSubstructMatches(node.mol):
            if s[0] == 0:
                children.append(Node(sm, sma, parent=node))
                break
    return children


def get_atom_env_smi(mol_: Mol, atom_ix: int, radius: int) -> str:
    if radius == 0:
        return mol_.GetAtomWithIdx(atom_ix).GetSymbol()
    bond_ix = []
    r = radius
    while len(bond_ix) == 0 and r > 0:
        bond_ix = Chem.FindAtomEnvironmentOfRadiusN(mol_, r, atom_ix, useHs=True)
        r -= 1
    if len(bond_ix) == 0:
        raise ValueError('No environment extractable')
    atom_ix_set = set()
    for bix in bond_ix:
        b = mol_.GetBondWithIdx(bix)
        atom_ix_set.add(b.GetBeginAtomIdx())
        atom_ix_set.add(b.GetEndAtomIdx())
    return Chem.MolFragmentToSmiles(mol_, atom_ix_set, bond_ix, rootedAtAtom=atom_ix, allHsExplicit=True)


def get_mol_img(s: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(s, sanitize=False)
    img = Draw.MolToImage(mol)
    img = img.convert('RGBA')
    datas = img.getdata()
    new_data = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    return np.asarray(img)


def offset_image(coord: int, s: str, ax: plt.Axes) -> None:
    img = get_mol_img(s)
    im = OffsetImage(img, zoom=0.2)
    im.image.axes = ax
    ab = AnnotationBbox(im, (coord, 0), xybox=(0., -30.), frameon=False, xycoords='data', boxcoords='offset points',
                        pad=0)
    ax.add_artist(ab)


def topx_plot(s: pd.Series, n: int, pad_xlabels: bool = False, xlabels: bool = True) -> plt.Figure:
    ng = len(s)
    na = s.sum()
    s = s.head(n)
    smis = s.index
    counts = s.values
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = sns.barplot(x=smis, y=counts)
    pad = 50
    xl = 'Group'
    t = 'groups'
    for i, x in enumerate(smis):
        if len(x) == 1:
            pad = 0
            xl = 'Atom'
            t = 'atoms'
            break
        offset_image(i, x, ax)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='major',  # both major and minor ticks are affected
        labelbottom=xlabels,  # labels along the bottom edge are off
        pad=pad)
    ax.set_title(f'Distribution of protonated/deprotonated {t}')
    ax.set_ylabel('Count')
    ax.set_xlabel(xl, labelpad=20 if xlabels else 70)
    yi = ax.yaxis.get_data_interval()[1]
    xi = ax.xaxis.get_data_interval()[1]
    ax.set_ylim((ax.yaxis.get_data_interval()[0], yi + yi / 10))
    pad = yi / 100
    for i, x in enumerate(s.values):
        ax.text(i, x + pad, x, ha='center')
    ax.text(0.98, 0.95, f'Overall groups: {na}\nUnique groups: {ng}', transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='top')
    if pad_xlabels and xlabels:
        for i, tick in enumerate(ax.get_xaxis().get_major_ticks()):
            if i % 2 != 0:
                tick.set_pad(70)
                tick.label1 = tick._get_text1()
    fig.tight_layout()
    return fig


def sat_plot(sl: List[pd.Series], rl: List[int], n: int) -> plt.Figure:
    assert len(sl) > 0
    assert len(sl) == len(rl)
    fig = plt.figure(figsize=(10, 5), dpi=150)
    adf = pd.DataFrame(columns=['Top X', '%', 'Radius'])
    for i in range(len(sl)):
        asp = 100 / sl[i].sum()
        df = pd.DataFrame({'%': [0]}).append(sl[i][:n].to_frame(name='%').copy())
        df['Radius'] = rl[i]
        df['Top X'] = range(0, len(df))
        pv = []
        for j in range(len(df)):
            pv.append(df.loc[df.index[:j + 1], '%'].sum() * asp)
        df['%'] = pv
        adf = adf.append(df)
    adf = adf.astype({'%': float, 'Radius': int, 'Top X': int})
    ax = sns.lineplot(x='Top X', y='%', hue='Radius', data=adf, palette='bright', marker='.', lw=3, ms=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Radius',
              loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.xticks(range(0, n + 1, 5))
    plt.yticks(range(0, 101, 10))
    plt.xlim((-1, n + 1))
    plt.ylim((-5, 105))
    fig.tight_layout()
    return fig


def mol_to_grid_image(mols: List[Mol], nrows: int, ncols: int, figsize: Tuple[int, int] = (16, 8),
                      text: bool = True) -> plt.Figure:
    im_list = [Draw.MolToImage(m, (600, 600)) for m in mols]
    fig = plt.figure(figsize=figsize, dpi=600)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(nrows, ncols),
                     axes_pad=(0.1, 0.3 if text else 0.1),  # pad between axes in inch.
                     label_mode='1'
                     )

    for i, (ax, im) in enumerate(zip(grid, im_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if text:
            ax.text(ax.axes.get_xlim()[1] / 2, ax.axes.get_ylim()[0] + 25,
                    Chem.MolToSmiles(Chem.RemoveHs(mols[i], sanitize=False)), horizontalalignment='center',
                    verticalalignment='center')

    for ax in grid[-len(grid) - len(im_list):]:
        ax.set_facecolor('white')

    return fig


def plot_group_grid(radius: int, topx: int, grp_dict: Dict[int, pd.Series], name: str, text: bool = True) -> None:
    topx_smi = [s for s in list(grp_dict[radius].keys())[:topx]]
    topx_mols = [Chem.MolFromSmiles(s, sanitize=False) for s in topx_smi]
    fig = mol_to_grid_image(topx_mols, 6, 8, text=text)
    fig.savefig(f'{name}_R{radius}_top{topx}.svg', bbox_inches='tight')


def export_smarts_list(path: str, radius: int, topx: int, dict_list: List[Dict[int, pd.Series]]) -> Set[str]:
    sma_set = set()
    for gd in dict_list:
        sma_set.update(gd[radius].head(topx).index.values)

    with open(path, 'w') as f:
        for s in sma_set:
            f.write(f'{s}\n')

    return sma_set


def add_hs_and_calc_2d(df: pd.DataFrame) -> None:
    df.ROMol = df.ROMol.apply(Chem.AddHs)
    for mol in df.ROMol:
        mol.Compute2DCoords()


def get_time_str(s: float) -> str:
    m = s // 60
    s %= 60
    r = f'{int(m)}m ' if m != 0 else ''
    return f'{r}{int(s)}s'


def validation_plot(scores: Dict[str, Dict[str, int]], df_len: int, times: Dict[str, float] = None) -> plt.Figure:
    clear_names = {'R1_oV': 'R1', 'R3_oV': 'R3', 'R1_V-R3': 'R1 with\nR3 valid.', 'R1_V-R4': 'R1 with\nR4 valid.',
                   'R1_V-R5': 'R1 with\nR5 valid.', 'R1_V-R6': 'R1 with\nR6 valid.'}

    ind = np.arange(len(scores))
    width = 0.7
    times_width = 0.2
    keys = list(scores.keys())
    times = [times[k] for k in keys] if times else None

    less = [scores[c]['less'] for c in keys]
    exact = [scores[c]['exact'] for c in keys]
    more = [scores[c]['more'] for c in keys]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    p1 = ax.bar(ind, less, width)
    p2 = ax.bar(ind, exact, width, bottom=less)
    p3 = ax.bar(ind, more, width, bottom=np.sum([exact, less], axis=0))
    ax.set_xticks(ind)
    ax.set_xticklabels([clear_names[k] for k in keys if k in clear_names])
    ax.set_ylabel('Molecules')

    if times:
        divider = make_axes_locatable(ax)
        ax_t = divider.append_axes('top', size='25%', pad=0.05, sharex=ax)
        ax_t.xaxis.set_tick_params(labelbottom=False)
        ax_t.yaxis.set_tick_params(labelbottom=False)
        ax_t.bar(ind, times, times_width)
        ax_t.text(-1.12, ax_t.get_ylim()[1] / 2, 'Times', verticalalignment='center')
        for i in ind:
            ax_t.text(i, ax_t.transLimits.inverted().transform((0, 1.0))[1], get_time_str(times[i]),
                      horizontalalignment='center', verticalalignment='bottom')

    ax.legend((p1[0], p2[0], p3[0]), ('Less', 'Exact', 'More'), loc='right', ncol=1, bbox_to_anchor=(1.1, 0.5))

    for i in ind:
        ax.text(i, int(less[i] / 2), f'{100 / df_len * less[i]:.2f}%',
                horizontalalignment='center', verticalalignment='center')
        ax.text(i, int(exact[i] / 2 + less[i]), f'{100 / df_len * exact[i]:.2f}%',
                horizontalalignment='center', verticalalignment='center')
        ax.text(i, int(more[i] / 2 + exact[i] + less[i]), f'{100 / df_len * more[i]:.2f}%',
                horizontalalignment='center', verticalalignment='center')

    return fig


# Source: https://www.geeksforgeeks.org/python-group-consecutive-list-elements-with-tolerance/
def split_tol(values: List[float], tol: float) -> Iterator[float]:
    res = []
    last = values[0]
    for ele in values:
        if ele - last > tol:
            yield np.round(np.mean(res), 3)
            res = []
        res.append(ele)
        last = ele
    yield np.round(np.mean(res), 3)


def plot_corr(plot_df: pd.DataFrame, reg_line: bool = True, connect_grps: Dict[Any, List[int]] = None, hue: bool = True,
              reg_col: str = 'red', colors: List = None) -> plt.Figure:
    fig = plt.figure(dpi=150, figsize=(5, 5))
    ax = sns.lineplot(x=[1, 13], y=[1, 13], color='black', lw='1')
    ax.lines[0].set_linestyle('--')
    if reg_line:
        sns.regplot(data=plot_df, x='Experimental pKa', y='Marvin pKa', marker='.', ci=None, scatter=False,
                    line_kws=dict(color=reg_col, lw='1'))
    sns.scatterplot(data=plot_df, x='Experimental pKa', y='Marvin pKa', hue='Group Count' if hue else None,
                    legend='full' if hue else False, edgecolor='none', palette=colors if hue else None, s=5)
    plt.xlim(1, 13)
    plt.ylim(1, 13)
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Experimental p$K$ₐ')
    plt.ylabel('Marvin p$K$ₐ')

    if connect_grps is not None:
        grp = plot_df.groupby('ID').agg(list)
        for ix in connect_grps:
            row = grp.loc[ix]
            for iix in connect_grps[ix]:
                for i in range(len(row['Group Count'])):
                    if i != iix:
                        plt.plot([row['Experimental pKa'][iix], row['Experimental pKa'][i]],
                                 [row['Marvin pKa'][iix], row['Marvin pKa'][i]], lw=0.5, c='black', ls='--')
    return fig


def plot_corr_joint(plot_df: pd.DataFrame, reg_line: bool = True, connect_grps: Dict[Any, List[int]] = None,
                    hue: bool = True, reg_col: str = 'red', colors: List = None) -> plt.Figure:
    jg = sns.jointplot(data=plot_df, x='Experimental pKa', y='Marvin pKa', kind='scatter',
                       hue='Group Count' if hue else None, edgecolor='none', palette=colors if hue else None, s=5,
                       xlim=(1, 13), ylim=(1, 13), height=6)
    jg.fig.set_dpi(150)
    ax = sns.lineplot(x=[1, 13], y=[1, 13], color='black', lw='1', ax=jg.ax_joint)
    ax.lines[0].set_linestyle('--')
    if reg_line:
        sns.regplot(data=plot_df, x='Experimental pKa', y='Marvin pKa', marker='.', ci=None, scatter=False,
                    line_kws=dict(color=reg_col, lw='1'), ax=jg.ax_joint)
    jg.ax_joint.set_xlim(1, 13)
    jg.ax_joint.set_ylim(1, 13)
    jg.ax_joint.legend(bbox_to_anchor=(1.035, 1.205), loc=2, borderaxespad=0.0, ncol=2)
    jg.ax_joint.set_xlabel('Experimental p$K$ₐ')
    jg.ax_joint.set_ylabel('Marvin p$K$ₐ')

    if connect_grps is not None:
        grp = plot_df.groupby('ID').agg(list)
        for ix in connect_grps:
            row = grp.loc[ix]
            for iix in connect_grps[ix]:
                for i in range(len(row['Group Count'])):
                    if i != iix:
                        jg.ax_joint.plot([row['Experimental pKa'][iix], row['Experimental pKa'][i]],
                                         [row['Marvin pKa'][iix], row['Marvin pKa'][i]], lw=0.5, c='black', ls='--')
    return jg.fig


def get_plot_data(df: pd.DataFrame, res_column: str, min_grp: int = 1, max_grp: int = 5, min_err: int = 0,
                  stats: bool = True) -> Tuple[pd.DataFrame, Dict[Any, List[int]], List]:
    pred_single_vals = []
    exp_single_vals = []
    n_grp = []
    ids = []
    outlier_ix = ddict(list)
    for ix, row in df.iterrows():
        if len(row[res_column]) < min_grp or len(row[res_column]) > max_grp:
            continue
        for i in range(len(row[res_column])):
            n_grp.append(str(len(row[res_column])))
            pred_single_vals.append(row['predicted'][i])
            exp_single_vals.append(row[res_column][i])
            ids.append(ix)
            diff = abs(row['predicted'][i] - row[res_column][i])
            if diff >= min_err:
                outlier_ix[ix].append(i)

    plot_df = pd.DataFrame({'Group Count': n_grp, 'Experimental pKa': exp_single_vals,
                            'Marvin pKa': pred_single_vals, 'ID': ids})
    if stats:
        print(f'Remaining mols: {len(set(ids))}')
        print(f'Remaining vals: {len(exp_single_vals)}\n')
        print(f'Overall MAE:    {mean_absolute_error(exp_single_vals, pred_single_vals):.3f}')
        print(f'Overall RMSE:   {mean_squared_error(exp_single_vals, pred_single_vals, squared=False):.3f}')
        print(f'Overall R2:     {r2_score(exp_single_vals, pred_single_vals):.3f}')
        print('Groups\tMolecules')
        for i, r in plot_df['Group Count'].value_counts().iteritems():
            print(f'{i}\t{int(r / int(i))}')

    colors = sns.color_palette('bright')[max(min_grp - 1, 0):]
    colors = colors[:len(plot_df['Group Count'].value_counts())]

    return plot_df, outlier_ix, colors
