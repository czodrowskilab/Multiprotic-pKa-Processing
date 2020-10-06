"""
This script prepares SDF files which can then be used for machine learning.
This includes sanitizing, filtering molecules with bad functional groups and
unwanted elements, removing salts, filtering by Lipinski's rule of five and
unify different tautomers. OpenEye QUACPAC and ChemAxon Marvin are required.
"""

from argparse import ArgumentParser, Namespace
from io import StringIO
from os.path import basename
from subprocess import PIPE, Popen, DEVNULL, SubprocessError
from sys import argv
from typing import Optional, List

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, SaltRemover, Descriptors, Lipinski, Crippen, Mol

__author__ = 'Marcel Baltruschat'
__copyright__ = 'Copyright © 2020'
__license__ = 'MIT'
__version__ = '1.1.0'

SMILES_COL = 'ISO_SMI'
TEMP_COL = 'temp'
PKA_COL = 'pKa'
DS_COL = 'original_dataset'
OID_COL = 'original_ID'
INDEX_NAME = 'ID'
DEFAULT_COLS = ['ROMol', PKA_COL, TEMP_COL, DS_COL, OID_COL]

TEMP_LOWER_CUT = 20
TEMP_UPPER_CUT = 25

# Selenium, Silicon and Boron
BAD_ELEMENTS = ['Se', 'Si', 'B']
BAD_ELEM_QUERY = Chem.MolFromSmarts(f'[{",".join(BAD_ELEMENTS)}]')

BFG = [
    Chem.MolFromSmarts('[!#8][NX3+](=O)[O-]'),  # "Classical" nitro group
    Chem.MolFromSmarts('[$([NX3+]([O-])O),$([NX3+]([O-])[O-])]=[!#8]'),  # Nitro group in tautomer form
]

ADDITIONAL_SALTS = [
    Chem.MolFromSmarts('[H+]'),
    Chem.MolFromSmarts('[I,N][I,N]'),
    Chem.MolFromSmarts('[Cs+]'),
    Chem.MolFromSmarts('F[As,Sb,P](F)(F)(F)(F)F'),
    Chem.MolFromSmarts('[O-,OH][Cl+3]([O-,OH])([O-,OH])[O-,OH]')
]

PKA_LOWER_CUT = 2
PKA_UPPER_CUT = 12

LIPINSKI_RULES = [
    (Descriptors.MolWt, 500),
    (Lipinski.NumHDonors, 5),
    (Lipinski.NumHAcceptors, 10),
    (Crippen.MolLogP, 5),
]


def count_bad_elements(mol: Mol) -> int:
    """
    Counts occurrences of bad elements
    specified in `BAD_ELEM_QUERY` for a molecule.

    Parameters
    ----------
    mol : Mol
        RDKit mol object

    Returns
    -------
    int
        Bad element count
    """

    return len(mol.GetSubstructMatches(BAD_ELEM_QUERY))


def count_bfg(mol: Mol) -> int:
    """
    Counts occurrences of bad functional groups
    specified in `BFG` for a molecule.

    Parameters
    ----------
    mol : Mol
        RDKit mol object

    Returns
    -------
    int
        Bad functional group count
    """

    n = 0
    for bfg in BFG:
        if mol.HasSubstructMatch(bfg):
            n += 1
    return n


ADDITIONAL_FILTER_RULES = [
    (count_bad_elements, 0),  # Are there any bad elements (more than zero)
    (count_bfg, 0),  # Are there any bad functional groups (more than zero)
]


def parse_args() -> Namespace:
    """
    Parses commandline parameters

    Returns
    -------
    Namespace
        Argparse Namespace object containing parsed commandline
        parameters
    """

    parser = ArgumentParser()
    parser.add_argument('infile', metavar='INFILE')
    parser.add_argument('outfile', metavar='OUTFILE')
    parser.add_argument('--keep-props', '-kp', metavar='PROP1,PROP2,...', default=[], type=lambda x: x.split(','))
    return parser.parse_args()


def check_on_remaining_salts(mol: Mol) -> Optional[Mol]:
    """
    Checks if any salts are remaining in the given molecule.

    Parameters
    ----------
    mol : Mol

    Returns
    -------
    Mol, optional
        Input molecule if no salts were found, None otherwise
    """

    if len(Chem.GetMolFrags(mol)) == 1:
        return mol
    return None


def check_sanitization(mol: Mol) -> Optional[Mol]:
    """
    Checks if molecule is sanitizable.

    Parameters
    ----------
    mol : Mol
        RDKit mol object

    Returns
    -------
    Mol, optional
        Sanitized molecule if possible, None otherwise
    """

    try:
        Chem.SanitizeMol(mol)
        return mol
    except ValueError:
        return None


def cleaning(df: pd.DataFrame, keep_props: List[str]) -> pd.DataFrame:
    """
    Cleans the input DataFrame by removing unwanted columns,
    removing salts from all molecules and sanitize the molecules.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a ROMol column with RDKit molecules
        and all columns specified in "keep_props"
    keep_props : List[str]
        Property names that should be kept through this script

    Returns
    -------
    DataFrame
        Cleaned DataFrame
    """

    df = df.loc[:, DEFAULT_COLS + keep_props]

    salt_rm = SaltRemover.SaltRemover()
    salt_rm.salts.extend(ADDITIONAL_SALTS)
    df.ROMol = df.ROMol.apply(salt_rm.StripMol)
    df.dropna(subset=['ROMol'], inplace=True)

    df.ROMol = df.ROMol.apply(check_on_remaining_salts)
    df.dropna(subset=['ROMol'], inplace=True)

    df.ROMol = df.ROMol.apply(check_sanitization)
    df.dropna(subset=['ROMol'], inplace=True)

    return df


def filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters DataFrame rows by molecules contained in column
    "ROMol" by Lipinski's rule of five, bad functional groups
    and unwanted elements.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing a ROMol column with RDKit molecules

    Returns
    -------
    DataFrame
        Filtered DataFrame
    """

    del_ix = []
    lip = 0
    for ix, row in df.iterrows():
        violations = 0
        for func, thres in LIPINSKI_RULES:
            if func(row.ROMol) > thres:
                violations += 1
            if violations > 1:
                del_ix.append(ix)
                lip += 1
                break
        if lip > 0 and del_ix[-1] == ix:
            continue
        for func, thres in ADDITIONAL_FILTER_RULES:
            if func(row.ROMol) > thres:
                del_ix.append(ix)
                break
    print(f'Dropped {lip} mols because of more than one Lipinski rule violation')
    print(f'Dropped {len(del_ix) - lip} mols through additional filtering')
    return df.drop(index=del_ix)


def mols_to_sdbuffer(df: pd.DataFrame, props: List[str] = None) -> StringIO:
    """
    Writes a DataFrame containing a ROMol column in SD format
    to a StringIO buffer.

    Parameters
    ----------
    df : DataFrame
        DataFrame that should be written to a buffer
    props : List[str]
        List of column names that should also be written
        to the buffer

    Returns
    -------
    StringIO
        StringIO buffer containing data in SD format
    """

    buffer = StringIO()
    PandasTools.WriteSDF(df, buffer, properties=props, idName='RowID')
    return buffer


def run_external(args: List[str], df: pd.DataFrame, reset_ix: bool = False) -> str:
    """
    Calls an external program via subprocess and writes the given
    DataFrame in SD format to stdin of the program. It returns
    the stdout of the external program.

    Parameters
    ----------
    args : List[str]
        List of arguments including the call of the desired program
        that can be directly passed to the subprocess Popen constructor
    df : DataFrame
        DataFrame that should be piped to the external program in SD format
    reset_ix : bool
        If True, the DataFrame index will be reset before passing to the program.
        Additionally the index will be written out as SD tag with the name of INDEX_NAME.

    Returns
    -------
    str
        Stdout of the external program

    Raises
    ------
    SubprocessError
        If the called program exits with a non-zero exit code
    """

    in_df = df.reset_index() if reset_ix else df
    in_prop = [INDEX_NAME] if reset_ix else None
    with mols_to_sdbuffer(in_df, in_prop) as buffer:
        p = Popen(args, text=True, stdin=PIPE, stdout=PIPE, stderr=DEVNULL)
        stdout, _ = p.communicate(buffer.getvalue())
    if p.returncode != 0:
        raise SubprocessError(f'{args[0]} ended with non-zero exit code {p.returncode}')
    return stdout


def run_marvin_pka(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates pKa values at 25°C within the configured pH range with
    ChemAxon Marvin for all molecules contained in `df`. Returns a new
    DataFrame with the Marvin results merged into the input DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing RDKit molecules

    Returns
    -------
    DataFrame
        Merged DataFrame containing the results from ChemAxon Marvin
    """

    cmd_call = ['cxcalc', '--id', INDEX_NAME, 'pka', '-i', str(PKA_LOWER_CUT), '-x', str(PKA_UPPER_CUT),
                '-T', '298.15', '-a', '4', '-b', '4']
    res_df = pd.read_csv(StringIO(run_external(cmd_call, df, True)),
                         sep='\t').set_index(INDEX_NAME, verify_integrity=True)
    res_df.index = res_df.index.astype(str)
    df = df.merge(res_df, right_index=True, left_index=True)
    for ix in df.index:
        try:
            if np.isnan(df.loc[ix, 'atoms']):
                continue
        except TypeError:
            pass
        ci = 0
        for col in ['apKa1', 'apKa2', 'apKa3', 'apKa4', 'bpKa1', 'bpKa2', 'bpKa3', 'bpKa4']:
            val = df.loc[ix, col]
            if np.isnan(val):
                continue
            if val < PKA_LOWER_CUT or val > PKA_UPPER_CUT:
                df.loc[ix, col] = np.nan
                atoms = df.loc[ix, 'atoms'].split(',')
                if len(atoms) == 1:
                    df.loc[ix, 'atoms'] = np.nan
                else:
                    del atoms[ci]
                    df.loc[ix, 'atoms'] = ','.join(atoms)
                    ci -= 1
            ci += 1
    df.dropna(subset=['atoms'], inplace=True)
    return df


def run_oe_tautomers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unifies different tautomers with OpenEye QUACPAC/Tautomers.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing RDKit molecules

    Returns
    -------
    DataFrame
        DataFrame with tautomer canonized structures
    """

    cmd_call = ['tautomers', '-maxtoreturn', '1', '-in', '.sdf', '-warts', 'false']
    mols, ix, ix_to_drop = [], [], []
    for line in run_external(cmd_call, df).split('\n'):
        if not line:
            continue
        smi, idx = line.split(' ')
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            ix_to_drop.append(idx)
            continue
        mols.append(mol)
        ix.append(idx)
    if ix_to_drop:
        df.drop(index=ix_to_drop, inplace=True)
    ixs = set(ix)
    if len(ix) != len(ixs):
        print('WARNING: Duplicates in tautomers result, removing')
    dropped = df.index.difference(ixs)
    df.drop(index=dropped, inplace=True)
    df.ROMol = mols
    return df


def make_dataset_unique(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out duplicated structures and saves all single values
    to a list for pKa, temperature, original dataset and original ID.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing RDKit molecules

    Returns
    -------
    DataFrame
        Aggregated DataFrame without duplicated structures
    """
    df[SMILES_COL] = df.ROMol.apply(Chem.MolToSmiles, isomericSmiles=True, canonical=True)
    grp = df.groupby(SMILES_COL)
    df2 = grp.first()
    list_cols = [PKA_COL, TEMP_COL, DS_COL, OID_COL]
    for col in list_cols:
        df2[col] = grp[col].agg(list)
    df2.index.set_names(INDEX_NAME, inplace=True)
    return df2


def filter_by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out entries outside the temperature range

    Parameters
    ----------
    df : DataFrame
        DataFrame containing RDKit molecules

    Returns
    -------
    DataFrame
        DataFrame without measurements with to high or to low temperatures
    """
    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors='coerce')
    return df.query(f'{TEMP_LOWER_CUT} <= temp <= {TEMP_UPPER_CUT}')


def read_dataset(infile: str) -> pd.DataFrame:
    """
    Reads a SD file from specified path and returns a DataFrame as result.

    Parameters
    ----------
    infile : str
        Path to SD file

    Returns
    -------
    DataFrame
        DataFrame containing the information from the specified SD file
    """
    df = PandasTools.LoadSDF(infile, idName=OID_COL)
    df.index = df.index.astype(str).set_names(INDEX_NAME)
    if DS_COL not in df.columns:
        df[DS_COL] = basename(infile)
    return df


def main(args: Namespace) -> None:
    """
    Main function of this script

    Parameters
    ----------
    args : Namespace
        Namespace object containing the parsed commandline arguments
    """
    df = read_dataset(args.infile)
    print(f'Initial: {len(df)}')

    df = cleaning(df, args.keep_props)
    print(f'After cleaning: {len(df)}')

    df = filtering(df)
    print(f'After filtering: {len(df)}')

    df = filter_by_temperature(df)
    print(f'After temperature control: {len(df)}')

    df = run_oe_tautomers(df)
    print(f'After QuacPac tautomers: {len(df)}')

    df = make_dataset_unique(df)
    print(f'After unifying dataset: {len(df)}')

    df = run_marvin_pka(df)
    print(f'After Marvin pKa: {len(df)}')

    PandasTools.WriteSDF(df, args.outfile, idName='RowID', properties=df.columns)


if __name__ == '__main__':
    main(parse_args())
