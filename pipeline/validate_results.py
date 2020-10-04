import pickle as pkl
from sys import argv
from time import time

from rdkit import RDLogger
from rdkit.Chem import PandasTools
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger

from pipeline.utils import *

RDLogger.logger().setLevel(RDLogger.CRITICAL)
sns.set()
uncharger = Uncharger()


def get_valid_atom_ids(df_: pd.DataFrame, trees) -> Tuple[List[str], float]:
    t0 = time()
    atom_ids = []
    for mol in df_.ROMol:
        ail = []
        mol_ = uncharger.uncharge(mol)
        for t in trees:
            for match in mol_.GetSubstructMatches(t.mol):
                valid = False
                for c in t.children:
                    for m2 in mol_.GetSubstructMatches(c.mol):
                        if set(match) <= set(m2) and match[0] == m2[0]:
                            valid = True
                            break
                    if valid:
                        break
                if valid:
                    ail.append(match[0] + 1)
        ail = [str(x) for x in sorted(list(set(ail)))]
        atom_ids.append(','.join(ail))
    t1 = time() - t0
    print(f'Time: {t1:.2f}s')
    return atom_ids, t1


def get_matches(df: pd.DataFrame, sma_mols: Iterable[Any], col: str) -> float:
    atom_ids = []
    t0 = time()
    for mol in df.ROMol:
        ail = []
        _mol = uncharger.uncharge(mol)
        for pm in sma_mols:
            for match in _mol.GetSubstructMatches(pm):
                ail.append(match[0] + 1)
        ail = [str(x) for x in sorted(list(set(ail)))]
        atom_ids.append(','.join(ail))
    t0 = time() - t0
    print(f'Time: {t0:.2f}s')
    df[col] = atom_ids
    return t0


def get_n_found(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    n_exact_found = 0
    n_all_found = 0
    for ix, row in df.iterrows():
        mv_spl = set(row.atoms.split(','))
        col_spl = set(row[col].split(','))
        if mv_spl == col_spl:
            n_exact_found += 1
        if mv_spl <= col_spl:
            n_all_found += 1
    return n_exact_found, n_all_found


def add_counts_to_dict(df: pd.DataFrame, scores_dict: Dict, col: str, key: str) -> None:
    n_exact_found, n_all_found = get_n_found(df, col)
    scores_dict[key]['exact'] = n_exact_found
    scores_dict[key]['less'] = len(df) - n_all_found
    scores_dict[key]['more'] = n_all_found - n_exact_found
    scores_dict[key]['all'] = n_all_found


def main():
    df = PandasTools.LoadSDF(argv[1]).set_index('ID', verify_integrity=True)
    df.ROMol = df.ROMol.apply(Chem.AddHs)
    for mol in df.ROMol:
        mol.Compute2DCoords()

    sma_df = pd.read_csv(argv[2], header=None, names=['SMARTS'])
    sma_df['ROMol'] = [Chem.MolFromSmarts(s) for s in sma_df.SMARTS]  # Atom with ID 0 is titratable atom
    fig = mol_to_grid_image(sma_df.ROMol, int(np.ceil(len(sma_df.ROMol) / 6)), 6)
    fig.savefig(argv[2].replace('csv', 'svg'))

    scores_dict = ddict(dict)
    measured_times = {}

    # radius one without validation
    atom_id_col = 'sma_atom_ids'
    scores_key = 'R1_oV'
    measured_times[scores_key] = get_matches(df, sma_df.ROMol, atom_id_col)
    add_counts_to_dict(df, scores_dict, atom_id_col, scores_key)

    # reproduction of approach above with smarts tree
    tree_file = 'sma_tree_r3.pkl'
    atom_id_col = 'sma_atom_ids2'
    with open(tree_file, 'rb') as f:
        sma_tree_r3 = pkl.load(f)
    get_matches(df, [t.mol for t in sma_tree_r3], atom_id_col)
    n_exact_found2, n_all_found2 = get_n_found(df, atom_id_col)
    assert scores_dict[scores_key]['exact'] == n_exact_found2 and scores_dict[scores_key]['all'] == n_all_found2

    # check with radius 3 smarts from tree
    scores_key = 'R3_oV_SMA'
    atom_id_col = 'sma_atom_ids_r3'
    r3_sma = []
    for t in sma_tree_r3:
        for c in t.children:
            r3_sma.append(c.sma)
    r3_sma = set(r3_sma)
    r3_sma_mols = [Chem.MolFromSmarts(s) for s in r3_sma]

    measured_times[scores_key] = get_matches(df, r3_sma_mols, atom_id_col)
    add_counts_to_dict(df, scores_dict, atom_id_col, scores_key)

    # check with radius 3 smiles from tree
    scores_key = 'R3_oV_SMI'
    atom_id_col = 'smi_atom_ids_r3'
    r3_smi_mols = [Chem.MolFromSmiles(s, sanitize=False) for s in r3_sma]

    measured_times[scores_key] = get_matches(df, r3_smi_mols, atom_id_col)
    add_counts_to_dict(df, scores_dict, atom_id_col, scores_key)

    # TODO Convert to loop
    # check with children radius 3 validation from tree
    atom_id_col = 'sma_atom_ids_vr3'
    trees_to_use = sma_tree_r3
    scores_key = 'R1_V-R3'
    df[atom_id_col], measured_times[scores_key] = get_valid_atom_ids(df, trees_to_use)
    add_counts_to_dict(df, scores_dict, atom_id_col, scores_key)

    # check with children radius 4 validation from tree
    atom_id_col = 'sma_atom_ids_vr4'
    tree_file = 'sma_tree_r4.pkl'
    scores_key = 'R1_V-R4'

    with open(tree_file, 'rb') as f:
        sma_tree_r4 = pkl.load(f)

    trees_to_use = sma_tree_r4
    df[atom_id_col], measured_times[scores_key] = get_valid_atom_ids(df, trees_to_use)
    add_counts_to_dict(df, scores_dict, atom_id_col, scores_key)

    # check with children radius 5 validation from tree
    atom_id_col = 'sma_atom_ids_vr5'
    tree_file = 'sma_tree_r5.pkl'
    scores_key = 'R1_V-R5'

    with open(tree_file, 'rb') as f:
        sma_tree_r5 = pkl.load(f)

    trees_to_use = sma_tree_r5
    df[atom_id_col], measured_times[scores_key] = get_valid_atom_ids(df, trees_to_use)
    add_counts_to_dict(df, scores_dict, atom_id_col, scores_key)

    # check with children radius 6 validation from tree
    atom_id_col = 'sma_atom_ids_vr6'
    tree_file = 'sma_tree_r6.pkl'
    scores_key = 'R1_V-R6'

    with open(tree_file, 'rb') as f:
        sma_tree_r6 = pkl.load(f)

    trees_to_use = sma_tree_r6
    df[atom_id_col], measured_times[scores_key] = get_valid_atom_ids(df, trees_to_use)
    add_counts_to_dict(df, scores_dict, atom_id_col, scores_key)

    # Visualize
    fig = validation_plot(scores_dict, len(df), measured_times)
    fig.savefig('validation_overview.svg')

    with open('dataframe_with_locations.pkl', 'wb') as f:
        pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
