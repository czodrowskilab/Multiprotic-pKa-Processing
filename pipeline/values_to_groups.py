import json
import pickle as pkl
from sys import argv

import numpy as np
import seaborn as sns

from .utils import split_tol, plot_corr, plot_corr_joint, get_plot_data

sns.set()


def main():
    atom_col = argv[1]
    exp_tol = float(argv[2])
    pka_upper_cut = 12 + exp_tol
    pka_lower_cut = 2 - exp_tol
    max_err = float(argv[3])

    with open('dataframe_with_locations.pkl', 'rb') as f:
        df = pkl.load(f)
    for col in ['apKa1', 'apKa2', 'apKa3', 'apKa4', 'bpKa1', 'bpKa2', 'bpKa3', 'bpKa4']:
        df[col] = df[col].astype(float)

    skipped = []
    assigned_pkas = []
    preds = []

    for ix, row in df.iterrows():
        exp = sorted(
            [float(x) for x in json.loads(row.pKa.replace("'", '"')) if pka_lower_cut <= float(x) <= pka_upper_cut])
        pred = [x for x in [row.apKa1, row.apKa2, row.apKa3, row.apKa4, row.bpKa1, row.bpKa2, row.bpKa3, row.bpKa4] if
                not np.isnan(x)]
        atoms = [int(x) for x in row[atom_col].split(',') if x != '']

        preds.append(pred)

        if len(atoms) != len(pred) or len(exp) == 0:  # Only exact matches by now and only values in right range
            skipped.append(ix)
            assigned_pkas.append(np.nan)
            continue

        pred = dict(zip(atoms, pred))
        exp = list(split_tol(exp, exp_tol))

        if len(exp) == 1 and len(pred) == 1:
            assigned_pkas.append(exp)
            continue

        assigned_exp = np.full(len(atoms), np.nan)
        for ev in exp:
            last_err = 100
            last_ai = -1
            best_pos = -1
            for i, ai in enumerate(atoms):
                err = abs(pred[ai] - ev)
                if err < last_err:
                    last_err = err
                    best_pos = i
                    last_ai = ai
            if np.isnan(assigned_exp[best_pos]) or last_err < abs(pred[last_ai] - assigned_exp[best_pos]):
                assigned_exp[best_pos] = ev

        assigned_pkas.append(assigned_exp)

    res_col = f'ASSIGNED_{atom_col}'
    df[res_col] = assigned_pkas
    df['predicted'] = preds

    n_skipped = len(skipped)
    n_all = len(df)
    perc = 100 / n_all
    print(f'Skipped: {n_skipped} ({perc * n_skipped:.2f}%)')

    missing = []
    for ix, val in df[res_col].iteritems():
        if isinstance(val, np.ndarray) and np.any(np.isnan(val)):
            missing.append(ix)
    n_missing = len(missing)
    print(f'Experimental values missing: {n_missing} ({perc * n_missing:.2f}%)')
    print(f'Exact matches: {n_all - n_missing - n_skipped} ({perc * (n_all - n_missing - n_skipped):.2f}%)')

    df_sub = df.loc[~df.index.isin(missing + skipped)]
    plot_df, outlier_ix, colors = get_plot_data(df_sub, res_col, min_grp=1, max_grp=5)
    fig = plot_corr_joint(plot_df, reg_line=False, colors=colors)
    fig.savefig('values_corr_G1-5_joined.svg', bbox_inches='tight')

    fig = plot_corr(plot_df, reg_line=False, colors=colors)
    fig.savefig('values_corr_G1-5.svg', bbox_inches='tight')

    plot_df, outlier_ix, colors = get_plot_data(df_sub, res_col, min_grp=2, max_grp=5, min_err=4)
    fig = plot_corr(plot_df, reg_line=False, connect_grps=outlier_ix, colors=colors)
    fig.savefig('values_corr_G2-5_con.svg', bbox_inches='tight')

    plot_df, outlier_ix, colors = get_plot_data(df_sub, res_col, min_grp=3, max_grp=5, min_err=3)
    fig = plot_corr(plot_df, reg_line=False, connect_grps=outlier_ix, colors=colors)
    fig.savefig('values_corr_G3-5_con.svg', bbox_inches='tight')

    outlier_ix = []
    outlier_diff = []
    for ix, row in df_sub.iterrows():
        for i in range(len(row[res_col])):
            diff = abs(row['predicted'][i] - row[res_col][i])
            if diff >= max_err:
                outlier_ix.append(ix)
                outlier_diff.append(diff)
                break

    print(f'After removing outlier (max err: {max_err})...')
    df_sub = df.loc[~df.index.isin(missing + skipped + outlier_ix)]
    plot_df, _, colors = get_plot_data(df_sub, res_col)
    fig = plot_corr_joint(plot_df, reg_line=False, colors=colors)
    fig.savefig(f'values_corr_G1-5_joined_err_cut_{max_err}.svg', bbox_inches='tight')
    fig = plot_corr(plot_df, reg_line=False, colors=colors)
    fig.savefig(f'values_corr_G1-5_err_cut_{max_err}.svg', bbox_inches='tight')

    outlier_df = df.loc[outlier_ix].copy()
    outlier_df['diff'] = outlier_diff
    outlier_df.sort_values('diff', ascending=False, inplace=True)

    with open('outlier_dataframe.pkl', 'wb') as f:
        pkl.dump(outlier_df, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open('final_dataframe.pkl', 'wb') as f:
        pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
