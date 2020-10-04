import pickle as pkl
from sys import argv

from rdkit import RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger

from .utils import *

RDLogger.logger().setLevel(RDLogger.CRITICAL)
sns.set()


# root_smarts have to be radius 1, end radius size via parameter
def generate_smarts_trees(grp_dict: Dict[int, pd.Series], root_smarts: Iterable[str], end_radius: int) -> List[Node]:
    trees = []
    for root_sma in root_smarts:
        root = Node(mol=Chem.MolFromSmarts(root_sma), sma=root_sma)
        root.children = gen_children(root, grp_dict[end_radius].index)
        trees.append(root)
    return trees


def group_summary_str(s: pd.Series) -> str:
    n = s.sum()
    n_p = 100 / n
    t1s = s.head(1).sum()
    t3s = s.head(3).sum()
    t5s = s.head(5).sum()
    t10s = s.head(10).sum()
    t20s = s.head(20).sum()
    t30s = s.head(30).sum()
    t40s = s.head(40).sum()
    t50s = s.head(50).sum()
    return f'''Unique groups: {s.count()}
    Top1:  {t1s}/{n} => {t1s * n_p:.2f}%
    Top3:  {t3s}/{n} => {t3s * n_p:.2f}%
    Top5:  {t5s}/{n} => {t5s * n_p:.2f}%
    Top10: {t10s}/{n} => {t10s * n_p:.2f}%
    Top20: {t20s}/{n} => {t20s * n_p:.2f}%
    Top30: {t30s}/{n} => {t30s * n_p:.2f}%
    Top40: {t40s}/{n} => {t40s * n_p:.2f}%
    Top50: {t50s}/{n} => {t50s * n_p:.2f}%'''


def count_groups(df: pd.DataFrame, r: int) -> pd.Series:
    uc = Uncharger()
    group_count = ddict(int)
    for ix, row in df.iterrows():
        if row.atoms == '':
            continue
        mol = uc.uncharge(row.ROMol)
        aids = [int(x) - 1 for x in row.atoms.split(',')]
        for ai in aids:
            group_count[get_atom_env_smi(mol, ai, r)] += 1
    return pd.Series(group_count, index=group_count.keys()).sort_values(ascending=False)


def do_radii_test(df: pd.DataFrame, radii: List[int], topx: int,
                  name: str, xlabels: bool = False) -> Dict[int, pd.Series]:
    grp_dict = {}
    for radius in radii:
        print(f'\nRadius {radius}:')
        grp_dict[radius] = count_groups(df, radius)
        print(grp_dict[radius].head(20))
        print(grp_dict[radius].describe())
        print(group_summary_str(grp_dict[radius]))
        xl = xlabels if radius != 0 else True
        pxl = xlabels if radius != 0 else False
        fig = topx_plot(grp_dict[radius], topx, radius, pad_xlabels=pxl, xlabels=xl)
        fig.savefig(f'{name}_R{radius}.svg')
    return grp_dict


def main() -> None:
    dataset = argv[1]
    ddl_prot_smi = argv[2]
    if not (3 <= len(argv) <= 5):
        print('ERROR: Wrong number of arguments')
        exit(1)
    radii_to_test = [int(x) for x in argv[3].split(',')] if len(argv) > 3 else [0, 1, 2, 3, 4, 5, 6]
    trees_to_build = [int(x) for x in argv[4].split(',')] if len(argv) > 4 else [3, 4, 5, 6]

    # ChemAxon Marvin analysis
    marvin_df = PandasTools.LoadSDF(dataset).set_index('ID', verify_integrity=True)
    add_hs_and_calc_2d(marvin_df)
    print(f'Dataset size: {len(marvin_df)}')
    print('\nChemAxon Marvin:')

    grp_dict_marvin = do_radii_test(marvin_df, radii_to_test, 10, 'marvin')

    fig = sat_plot(list(grp_dict_marvin.values()), list(grp_dict_marvin.keys()), 50)
    fig.savefig('marvin_sat_curve.svg')

    # Dimorphite-DL analysis
    print('\nDimorphite-DL:')
    didl_df = pd.read_csv(ddl_prot_smi, sep='\t', header=None, names=['prot_smi', 'smi'])
    didl_df = didl_df.groupby('smi').agg(list).reset_index()
    PandasTools.AddMoleculeColumnToFrame(didl_df, 'smi')
    didl_df.drop(columns='smi', inplace=True)
    print(f'Protonated SMILES size: {len(didl_df)}')

    atoms_col = []
    for _, row in didl_df.iterrows():
        cdf = pd.DataFrame(columns=range(row.ROMol.GetNumAtoms()))
        oro = list(Chem.CanonicalRankAtoms(row.ROMol))
        for i, smi in enumerate(row.prot_smi):
            pm = Chem.MolFromSmiles(smi)
            ro = list(Chem.CanonicalRankAtoms(pm))
            if oro != ro:
                d = dict(zip(ro, range(len(ro))))
                pm = Chem.RenumberAtoms(pm, [d[j] for j in oro])
            cdf.loc[i] = [a.GetFormalCharge() for a in pm.GetAtoms()]
        remaining_cols = [col for col in cdf.columns if len(cdf[col].unique()) != 1]
        atoms_col.append(','.join([str(x + 1) for x in remaining_cols]))
    didl_df['atoms'] = atoms_col
    add_hs_and_calc_2d(didl_df)

    grp_dict_didl = do_radii_test(didl_df, radii_to_test, 10, 'didl')

    fig = sat_plot(list(grp_dict_didl.values()), list(grp_dict_didl.keys()), 50)
    fig.savefig('didl_sat_curve.svg')

    plot_group_grid(3, 48, grp_dict_marvin, 'marvin')
    plot_group_grid(3, 48, grp_dict_didl, 'didl')

    sma_set = export_smarts_list('titratable_groups_SMARTS_R1.csv', 1, 20, [grp_dict_marvin, grp_dict_didl])

    for radius in trees_to_build:
        trees = generate_smarts_trees(grp_dict_marvin, sma_set, radius)
        with open(f'sma_tree_r{radius}.pkl', 'wb') as f:
            pkl.dump(trees, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
