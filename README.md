# Multiprotic p*K*<sub>a</sub> Processing

Most data sets of experimental p*K*<sub>a</sub> values of multiprotic molecules lack information about the associated (de-)protonation sites. However, this information is often necessary to train appropriate prediction models which show the corresponding (de-)protonation sites in addition to the predicted p*K*<sub>a</sub> values or whose prediction is based on this information. Additionally, the datasets are often not cleaned or filtered in any way, contain duplicated entries or different tautomers of the same molecule. The tool presented here tries to solve these problems to generate a cleaned, standardized and annotated data set that can be used for different machine learning approaches. 

For cleaning and filtering the tool performs the following steps:
* Removal of salts
* Filtering molecules containing nitro groups
* Filtering molecules containing Boron, Selenium or Silicon
* Filtering by Lipinski's rule of five (one violation allowed)
* Keeping only p*K*<sub>a</sub> values between 2 and 12
* Tautomer standardization
* Protonation at pH 7.4

To get annotations about the (de-)protonation site of every p*K*<sub>a</sub> value, two major 
problems have to be solved: Localization of the titratable groups without licensed software and the once-only assignment of the experimental values to the corresponding groups for all datasets. 

For the localization part the tools *ChemAxon Marvin*<sup>[1]</sup> and *Dimorphite-DL*<sup>[2]</sup> are used to compile a list of SMARTS pattern that catch most of all groups in the given dataset. Finally, the *Marvin* predictions are used to assign the experimental values to the corresponding groups while removing outliers. The resulting data set can be used as a starting point for machine learning in a following step.


## Prerequisites

The Python dependencies are:
* Python = 3.8
* Scikit-Learn
* RDKit >= 2020.03.5
* Matplotlib >= 3.2, < 3.3 
* Seaborn >= 0.11

For the whole pipeline, *ChemAxon Marvin*<sup>[1]</sup>, *Dimorphite-DL*<sup>[2]</sup> and *OpenEye QUACPAC/Tautomers*<sup>[3]</sup> are required.

Of course you also need the code from this repository folder.

### Installing

First of all you need a working Miniconda/Anaconda installation. You can get
Miniconda at https://conda.io/en/latest/miniconda.html.

Now you can create an environment named "MPP" with all needed dependencies and
activate it with:
```bash
conda env create -f environment.yml
conda activate MPP
```

You can also create a new environment by yourself and install all dependencies without the
environment.yml file:
```bash
conda create -n my_env python=3.8
conda activate my_env
conda install -c defaults -c conda-forge scikit-learn "rdkit>=2020.03" matplotlib=3.2 "seaborn>=0.11"
```

## Usage
To use the data preparation pipeline your conda environment has to be activated and the *Marvin* commandline tool `cxcalc` and the *QUACPAC* commandline tool `tautomers` have to be contained in your `PATH` variable.

Also the environment variables `OE_LICENSE` (containing the path to your *OpenEye* license
file) and `JAVA_HOME` (referring to the *Java* installation folder, which is needed for 
`cxcalc`) have to be set.

After preparation you can display a small usage information with `# COMING SOON`.
Example call:
```bash
# COMING SOON
```

NOTE: The pipeline script must be located in the repository folder to use all associated Python scripts. Alternatively, the path to the repository folder can be specified with the environment variable `# COMING SOON`, e.g.
```bash
# COMING SOON
```

## Datasets

1. `Settimo_et_al.sdf` - Manually combined literature p<i>K</i><sub>a</sub> data<sup>[4]</sup>
2. `chembl26.sdf` - Experimental p<i>K</i><sub>a</sub> data extracted from ChEMBL26<sup>[5]</sup>
3. `datawarrior.sdf` - p<i>K</i><sub>a</sub> data shipped with DataWarrior<sup>[6]</sup>
4. TO BE EXTENDED

## Authors

**Marcel Baltruschat** - [GitHub](https://github.com/mrcblt), [E-Mail](mailto:marcel.baltruschat@tu-dortmund.de)<br>
**Paul Czodrowski** - [GitHub](https://github.com/czodrowskilab), [E-Mail](mailto:paul.czodrowski@tu-dortmund.de)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

[1] *Marvin* 20.1.0, 2020, ChemAxon, [http://www.chemaxon.com](http://www.chemaxon.com)<br>
[2] Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
open-source program for enumerating the ionization states of drug-like small
molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.<br>
[3] *QUACPAC* 2.1.0.4: OpenEye Scientific Software, Santa Fe, NM. [http://www.eyesopen.com](http://www.eyesopen.com) <br>
[4] Settimo, L., Bellman, K. & Knegtel, R.M.A. Pharm Res (2014) 31: 1082. 
[https://doi.org/10.1007/s11095-013-1232-z](https://doi.org/10.1007/s11095-013-1232-z) <br>
[5] Gaulton A, Hersey A, Nowotka M, Bento AP, Chambers J, Mendez D, Mutowo P, Atkinson F, 
Bellis LJ, Cibrián-Uhalte E, Davies M, Dedman N, Karlsson A, Magariños MP, Overington JP, 
Papadatos G, Smit I, Leach AR. (2017) 'The ChEMBL database in 2017.' Nucleic Acids Res., 
45(D1) D945-D954.<br>
[6] Thomas Sander, Joel Freyss, Modest von Korff, Christian Rufener. DataWarrior: An Open-Source 
Program For Chemistry Aware Data Visualization And Analysis. J Chem Inf Model 
2015, 55, 460-473, doi 10.1021/ci500588j<br>
