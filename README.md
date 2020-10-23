# Multiprotic p*K*<sub>a</sub> Processing
![GitHub](https://img.shields.io/github/license/czodrowskilab/Multiprotic-pKa-Processing)
![GitHub](https://img.shields.io/badge/Stage-BETA-blue)

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

**Important note**: You should clone this repository with the command 
`git clone --recurse-submodules https://github.com/czodrowskilab/Multiprotic-pKa-Processing.git` to automatically
fetch the integrated submodule *Dimorphite-DL* (https://git.durrantlab.pitt.edu/jdurrant/dimorphite_dl).

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

Additionally, the path to the repository folder needs to be specified with the environment variable `PKA_CODEBASE`, e.g.
```bash
export PKA_CODEBASE="/full/path/to/repository/folder"
```

After preparation you can display a small usage information with `bash pipeline.sh --help`.
Example call:
```bash
bash pipeline.sh --train chembl26.sdf datawarrior.sdf --test sample6.sdf
```

If you want to use different sets of molecules for the generation of the SMARTS patterns you can
use the optional parameter `--grouping <SDF> <SDF> ...`. If not specified all training and test sets will
be used for the SMARTS pattern generation.

If you don't want the full analysis through all different location strategies you can specify a specific
strategy with `--quick-run KEY`. The following keys are available:
```bash
  R1_oV   - Radius 1 only
  R3_oV   - Radius 3 only
  R1_V-R3 - Radius 1 search with radius 3 validation
  R1_V-R4 - Radius 1 search with radius 4 validation
  R1_V-R5 - Radius 1 search with radius 5 validation
  R1_V-R6 - Radius 1 search with radius 6 validation
```

Additional parameters are:
- `--exp-tol`: Experimental value error tolerance as a float value to decide if two experimental values
belong to the same titratable group (default is 0.3).
- `--max-err`: The maximum error between experimental value and the _Marvin_ prediction after assigning the values to the titratable groups at which the value is still reliable
(default is 2.0).

## Datasets

1. `settimo_et_al.sdf` - Manually combined literature p<i>K</i><sub>a</sub> data<sup>[4]</sup>
2. `chembl26.sdf` - Experimental p<i>K</i><sub>a</sub> data extracted from ChEMBL26<sup>[5]</sup>
3. `datawarrior.sdf` - p<i>K</i><sub>a</sub> data shipped with DataWarrior<sup>[6]</sup>
4. `hunt_et_al.sdf` - Extracted from Hunt et al. (2020)<sup>[7]</sup>
5. `literature_compilation.sdf` - Compilation of experimental datapoints from multiple publications 
([list of publications](datasets/literature_compilation_publications.txt))
6. `sampl6.sdf` - Experimental data from SAMPL6 challenge<sup>[8]</sup> ([LICENSE](datasets/sampl6.LICENSE))

## Authors

**Marcel Baltruschat** - [GitHub](https://github.com/mrcblt), [E-Mail](mailto:marcel.baltruschat@tu-dortmund.de)<br>
**Paul Czodrowski** - [GitHub](https://github.com/czodrowskilab), [E-Mail](mailto:paul.czodrowski@tu-dortmund.de)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

[1] *Marvin* 20.1.0, 2020, ChemAxon, [http://www.chemaxon.com](http://www.chemaxon.com) <br>
[2] Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
open-source program for enumerating the ionization states of drug-like small
molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9. <br>
[3] *QUACPAC* 2.1.0.4: OpenEye Scientific Software, Santa Fe, NM. [http://www.eyesopen.com](http://www.eyesopen.com) <br>
[4] Settimo, L., Bellman, K. & Knegtel, R.M.A. Pharm Res (2014) 31: 1082. 
[https://doi.org/10.1007/s11095-013-1232-z](https://doi.org/10.1007/s11095-013-1232-z) <br>
[5] Gaulton A, Hersey A, Nowotka M, Bento AP, Chambers J, Mendez D, Mutowo P, Atkinson F, 
Bellis LJ, Cibrián-Uhalte E, Davies M, Dedman N, Karlsson A, Magariños MP, Overington JP, 
Papadatos G, Smit I, Leach AR. (2017) 'The ChEMBL database in 2017.' Nucleic Acids Res., 
45(D1) D945-D954. <br>
[6] Thomas Sander, Joel Freyss, Modest von Korff, Christian Rufener. DataWarrior: An Open-Source 
Program For Chemistry Aware Data Visualization And Analysis. J Chem Inf Model 
2015, 55, 460-473, doi 10.1021/ci500588j <br>
[7] Hunt, P. et al. Predicting pKa Using a Combination of Semi-Empirical Quantum Mechanics and Radial 
Basis Function Methods. J. Chem. Inf. Model. 60, 2989–2997 (2020). <br>
[8] Mehtap Isik, Andrea Rizzi, David L. Mobley, Michael Shirts, & Danielle Teresa Bergazin. (2019, April 25). 
MobleyLab/SAMPL6: SAMPL6 Part II - Release the evaluation results of log *P* predictions (Version v1.16). Zenodo. 
http://doi.org/10.5281/zenodo.2651393 <br>
