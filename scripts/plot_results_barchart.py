from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


val_file = argv[1]

bar_width = 0.15
gap = 0.00

vals = pd.read_csv(val_file, sep=';', index_col=0)

index = np.arange(len(vals))
cols = vals.columns

sns.set()
font_size = 16
params = {'axes.labelsize': font_size,
          'axes.titlesize': font_size,
          'xtick.labelsize': font_size,
          'ytick.labelsize': font_size,
          'figure.titlesize': font_size,
          'legend.fontsize': font_size}
plt.rcParams.update(params)

fig = plt.figure(figsize=(7, 5))

for ix, col in enumerate(cols):
    plt.bar(index + gap * ix + bar_width * ix, vals[col], bar_width, label=col, antialiased=True)

plt.ylabel('MAE')
plt.xlabel('Testsets', labelpad=10)
offset = bar_width / 2 if len(cols) % 2 == 0 else 0
plt.xticks(index + len(cols) / 2 * bar_width - offset, [x.replace(' ', '\n', 1) for x in vals.index], ha='center')
plt.legend(bbox_to_anchor=(1.0, 1.02), loc='upper left')

fig.savefig('last_plot_V3.svg', bbox_inches='tight')
