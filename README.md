# Model-agnostic inference using the BAO feature

## Table of contents
* [Introduction](#introduction)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Code organization](#code-organization)
* [Data organization](#data-organization)
* [Examples](#examples)
* [Emulation](#emulation)
* [Citation](#citation)
* [Contact](#contact)

## Introduction
A model-agnostic description of the baryon acoustic oscillation (BAO) feature in redshift space requires a number of ingredients. Physically, one must describe the impact of cosmological bulk flows which progressively and anisotropically smear out the feature over time. One must also model the effects of the scale dependence of tracer bias and the mode coupling between short and long scales. All of these can be incorporated using the Zel'dovich approximation alone, without reference to any particular cosmological model. On the technical front, one needs a robust, complete and cosmology-independent basis to describe the shape of the real space BAO feature in linear theory, which can then be propagated to the nonlinearly evolved, measured feature in redshift space. Finally, as in a traditional analysis, one must account for a possibly incorrect conversion of observed angular and redshift separations to comoving lengths.

These ingredients have been developed by us in a series of recent papers. 

This repository provides an implementation of the final **Zel'dovich smearing** model that brings these pieces together, as described by [Paranjape & Sheth (2026a)](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) (PS26a below) and [Paranjape & Sheth (2026b)](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract) (PS26b below). 

Additionally, we provide measurements of the relevant pairwise correlations and their expected covariance for the following tracer samples: 
* Two $N$-body halo samples drawn from the [publicly available](https://abacussummit.readthedocs.io/en/latest/abacussummit.html) _AbacusSummit_ suite's baseline `c000` cosmology as described by [PS26b](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract):
  * _DESI-LRG2_ ($z=0.8$) mimicking luminous red galaxies (LRGs) being observed by the [DESI](https://ui.adsabs.harvard.edu/abs/2016arXiv161100036D/abstract) survey, 
  * _Euclid-ELG_ ($z=1.1$) mimicking H $\alpha$ emission line galaxies (ELGs) being observed by the [*Euclid*](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A...1E/abstract) mission.
* A toy sample mimicking DESI LRGs as described by [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract).

We also provide the code we used to produce the measurements for the _DESI-LRG2_ and _Euclid-ELG_ halo samples.

The code and data in this repository should be sufficient to reproduce all the main results of [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) and [PS26b](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract). We have provided several *example notebooks* (described below) to implement the MCMC analysis and explore the theoretical model.

We also provide emulators to move back and forth between the cosmological and agnostic parameter spaces, for a variety of cosmological models. These emulators can be incorporated into MCMC pipelines to implement the 'strong priors' discussed in [PS26b](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract).

## Dependencies
* Python 3.9+, NumPy 2.0+
* [Cobaya](https://cobaya.readthedocs.io/en/latest/): Our model is implemented in the $\texttt{Cobaya}$ framework developed by [Torrado & Lewis (2021)](https://ui.adsabs.harvard.edu/abs/2021JCAP...05..057T/abstract). This allows for straightforward integration into Markov Chain Monte Carlo (MCMC) pipelines.
  * `mpi4py`: This is needed if $\texttt{Cobaya}$ is to be installed with MPI support (see [Installation](#installation)).
  * `ipyparallel`: This is needed in `ZelSmear_MCMC_mpi.ipynb` (see [Examples](#examples)) to run MCMC chains with MPI support. 
* [GetDist](https://getdist.readthedocs.io/): This is used for analysing MCMC chains and producing plots of posterior and other distributions.
* [mlfundas](https://github.com/a-paranjape/mlfundas): This machine learning repository provides multiple utilities:
    * It is primarily used for its implementation of the `BiSequential` basis described by [Paranjape & Sheth (2025)](https://ui.adsabs.harvard.edu/abs/2025JCAP...06..009P/abstract), with the source code available as `code/mlalgos.BiSequential` and the specific trained instance stored in `examples/binet/`.
    * When measuring pairwise correlations, it also uses Python's `multiprocessing` package for parallelization using the `code/mllib.MLUtilities.run_processes` method.
    * Finally, it provides the framework for building and utilizing emulators through `code/emulation.py` (usage is demonstrated in example notebooks). 
* [sahyadri-sandbox](https://github.com/a-paranjape/sahyadri-sandbox): This repository is used for our implementation of the core algorithms needed for measuring pairwise correlations (anisotropic 2pcf and power spectrum) in $N$-body tracer samples, although other implementations can also be used.
* [CLASS](https://lesgourg.github.io/class_public/class.html): This is needed for generating training and test samples for emulation (see [Emulation](#emulation)). It can be skipped if new samples are not required.

## Installation
The following steps should be sufficient for using the functionality of this repository:
1. Install $\texttt{Cobaya}$ as described [here](https://cobaya.readthedocs.io/en/latest/installation.html). If you wish to run the MCMC analysis using MPI support, please be sure to install $\texttt{Cobaya}$ with MPI support, as described on its installation page (this will require installing `mpi4py`). Installing $\texttt{Cobaya}$ should also automatically install $\texttt{GetDist}$ if you don't already have it.
2. Clone into the $\texttt{mlfundas}$ repository. E.g., for HTTPS-based transfer, in your chosen install location use
  ```
  git clone https://github.com/a-paranjape/mlfundas.git
  ```
3. Setup $\texttt{sahyadri-sandbox}$. *(Only needed if pairwise correlations need to be measured in N-body samples)*.
    * Clone into the repository. E.g., for HTTPS-based transfer, in your chosen install location use
    ```
    git clone https://github.com/a-paranjape/sahyadri-sandbox.git
    ```
    * In the file `sahyadri-sandbox/scripts/post-process/utilities.py`, edit the `__init__` method of the class `Paths` to set the attribute `self.home_path` to `your/path/to/sahyadri-sandbox/` so this code knows where it is installed.
4. Setup $\texttt{zeldovich-smearing}$ (this repository):
   * Clone into this repository. E.g., for HTTPS-based transfer, in your chosen install location use
     ```
     git clone https://github.com/a-paranjape/zeldovich-smearing.git
     ```
   * Edit the file `code/paths.py`:
       * Set the variable `ML_Path` to `/your/path/to/mlfundas/code/`.
       * Set the variable `Basis_Root` to `/your/path/to/mlfundas/examples/binet/`.
       * Setup halo paths. *(Only needed if pairwise correlations need to be measured in N-body samples)*
         * Set the variable `Sahyadri_Path` to `/your/path/to/sahyadri-sandbox/scripts/post-process/`.
         * Set the variable `Abacus_Path` to `/your/path/to/abacus/halos` (these will need to be separately downloaded, see the _AbacusSummit_ [data access page](https://abacussummit.readthedocs.io/en/latest/data-access.html)).
5. Setup $\texttt{CLASS}$. *(Only needed if new training and/or test samples are desired for building/using emulators; see [Emulation](#emulation))*.
   * Download the latest `class_public***.tar.gz` file from the [CLASS](https://lesgourg.github.io/class_public/class.html) repository and unzip it in the local install folder.
   * Depending on your system, you may need to additionally install `gcc`, `openmpi` and `fftw` libraries. The Makefile should be appropriately edited to reflect their locations.
   * Run the following in the `class_public` folder
     ```
     make clean
     make
     ```
     which should compile the code with Python support. In case of errors, kindly refer to the original documentation.   
7. Test the basic installation:
   * In a terminal in `your/path/to/zeldovich-smearing/code`, type
   ```
   python -c "from theory import TheoryManipulator; tm = TheoryManipulator(sample='DESI-LRG2')"
   ```
   This should produce a bunch of text related to the `BiSequential` basis instance, followed by a list containing two paths to `.txt` files and ending with the phrase `... setup complete`. \
   *\[**Note:** Although this doesn't explicitly check whether MPI capability (if requested) is correctly installed, that should have been accounted for if the Cobaya installation steps for MPI support were followed precisely.\]* \
   If $\texttt{CLASS}$ was installed, test the installation using
   ```
   python -c "from classy import Class"
   ```


## Code organization
The source code is contained in the folder `code/` and is distributed across the following Python scripts.
* `zelsmear.py`:\
  This is the main script containing definitions of the theory class `ZeldovichSmearingTheory` and likelihood class `ZeldovichSmearingLike`:
  * `ZeldovichSmearingLike`: Provides infrastructure to read and manipulate a data vector and covariance matrix from specified locations, so as to calculate a Gaussian (log-)likelihood. Assumes that the theory class will provide a compatible model prediction vector. 
  * `ZeldovichSmearingTheory`: Provides all necessary ingredients to compute a model prediction using a parameter dictionary, so as to be compatible with the data vector used by the `ZeldovichSmearingLike`. In addition to the top-level `calculate` method required by $\texttt{Cobaya}$'s samplers such as `mcmc`, this class includes several auxiliary methods to compute various interesting quantities[^1] such as
    * total prediction for multipoles of the observed tracer 2-point correlation function (2pcf) $\xi^{(\ell)}\_{\rm obs}(s)$ (`calc_xiNL`) and low $k$ power spectrum multipole integrals $\Sigma^{(\ell)2}_{\rm obs}$ (`calc_Sig2obs`) for use in MCMC,
    * individual contributions of the 'propagator' (`calc_xiprop`) and 'mode-coupling' pieces (`calc_xiMC`) in 2pcf multipoles,
    * protohalo 2pcf prediction (`calc_xiprotohalo`),
    * raw basis functions $b_m(r)$ (`calc_basis`) and their first derivatives (`calc_dbdr`),
    * smeared basis functions $\lambda_m(s|\sigma)$ (`calc_lambda`) and their derivatives $\lambda_m^{(n)}(s|\sigma)$ (`calc_der_lambda`) and associated auxiliary functions $\Lambda_m^{(n)}$ (`calc_der_Lambda`),
    * basis function integrals $\bar{\lambda}_m(s|\sigma),\bar{\bar{\lambda}}_m(s|\sigma)$ (`calc_lambda_bars`),
    * interesting length scales such as the peak $r_{\rm peak}$, linear point $r_{\rm LP}$ and zero-crossing $r_{\rm ZC}$ of the linear 2pcf (`calc_linearscales`),
    * first derivative ${\rm d}\xi_{\rm lin}/{\rm d}\ln r$ of the linear 2pcf (`calc_dxidlnr`).
      
* `theory.py`:\
  This script contains the `TheoryManipulator` class that internally sets up a dummy Cobaya-friendly info dictionary and exposes user-friendly routines to compute and manipulate the model prediction. 
  * Upon instantiation of `TheoryManipulator`, an instance of `ZeldovichSmearingTheory` is available as `TheoryManipulator.theory`, containing all the methods described above.
  * Similarly, an instance of `ZeldovichSmearingLike` is available as `TheoryManipulator.like`, which can be used to access stored data and covariance matrix arrays in multiple formats.
  * Additionally, the `TheoryManipulator` instance exposes the methods `calc_model`, `load_data`, `calc_chi2` and `vary_prediction`, among others, which can be used to study the model predictions and compare them with the data sets included in the repository, without explicitly referring to `TheoryManipulator.theory` or `TheoryManipulator.like`.
  * Finally, the `TheoryManipulator` instance has some useful attributes, such as:
    * `fiducial` - a dictionary with an exhaustive list of all varied and derived parameters in the model along with their fiducial values for each data sample,
    * `eval_dict_fid` - a restricted version of `fiducial` that can be passed to `calc_model`,
    * `best_fit` - a dictionary providing the best fit values of the varied model parameters for the _DESI-LRG2_ and _Euclid-ELG_ samples,
    * `params_list` - a list containing the names of all varied parameters,
    * `latex_list` - a list containing Latex descriptions of all varied parameters (useful for labelling plots).
    
  Example usage can be found in the `ZelSmear_Explore_Theory.ipynb` notebook described under [Examples](#examples).

* `pairwise_abacus.py`:\
  This script implements parallelized calculations of the 2pcf and power spectrum of the _DESI-LRG2_ and _Euclid-ELG_ samples of _AbacusSummit_ halos, in real space as well as multipoles in redshift space. It can also be edited to measure these quantities for custom samples from _AbacusSummit_. This script is provided primarily for transparency and is not (yet) very user-friendly, so please [contact](#contact) the authors if you have difficulty using it.\
  **Warning:** The redshift space 2pcf implementation borrowed from $\texttt{sahyadri-sandbox}$ is currently **very** slow. You might be better off with some other publicly available implementation such as [Corrfunc](https://github.com/manodeep/Corrfunc) or [TreeCorr](https://github.com/rmjarvis/TreeCorr).\
  **Note:** To use this script, _AbacusSummit_ halo samples would need to be separately downloaded. The download location should then be provided to the code by editing `paths.py` (see [Installation](#installation)).

* `universe.py`:\
  *[DOCUMENTATION UNDER CONSTRUCTION]*
  
* `emulation.py`:\
  *[DOCUMENTATION UNDER CONSTRUCTION]*
  
* `paths.py`:\
  This is an auxiliary file that must be edited during [installation](#installation) and contains paths to various dependencies.

[^1]:See [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) and [PS26b](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract) for the original definitions of the quantities listed here.

## Data organization

### Primary analysis
We provide several useful data sets in the folder `examples/data/`.

These include the following measurements for the toy model from [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) and the _DESI-LRG2_ and _Euclid-ELG_ samples constructed using _AbacusSummit_ halos from [PS26b](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract):
* Redshift space multipoles $\ell=0,2,4$ of the tracer 2pcf $\xi^{(\ell)}\_{\rm obs}(s)$.
* Redshift space multipoles $\ell=0,2,4$ of the tracer power spectrum $P^{(\ell)}\_{\rm obs}(k)$ for the _DESI-LRG2_ and _Euclid-ELG_ samples.
* Integrals of the power spectrum multipoles over low $k$ bins $\Sigma^{(\ell)2}_{\rm obs}$.
* Gauss-Poisson estimates of the joint covariance matrix of $\xi^{(\ell)}\_{\rm obs}(s)$ and $\Sigma^{(\ell)2}_{\rm obs}$. For the _DESI-LRG2_ and _Euclid-ELG_ samples, these are scaled to match diagonal errors on each quantity as estimated using 25 realizations of the _AbacusSummit_ baseline `c000` boxes.
* Real space 2pcf and power spectrum estimates for the _DESI-LRG2_ and _Euclid-ELG_ samples (for reference only).
    
For each measured quantity, the data set includes separate measurements from each realization (or 'phase') of the 25 available, along with plot-friendly files reporting the measurements and diagonal errors for the reference phase for each sample. The reference phases are `ph000` for _DESI-LRG2_ and `ph009` for _Euclid-ELG_. MCMC-friendly data files for the reference samples are also provided for use with `ZeldovichSmearingLike`.

The data files for different samples are organized as follows.
* _AbacusSummit_ samples: These are in `examples/data/AbacusSummit/base_c000/`, with the files for the two samples respectively placed in the sub-folders `DESI-LRG2/` and `Euclid-ELG/`. For legacy purposes, we have also provided some measurements for _DESI-LRG2_-like samples having a 3 times larger volume, in `examples/data/legacy/`.
* Toy model: These are in `examples/data/SDBMC/`. 

Plots visualizing these data sets can be found in correspondingly organized sub-folders of `examples/plots/`.

### Emulation analysis
To support the emulators for various cosmological models described in [Emulation](#emulation), we provide training and test samples, along with the emulator model weights in each case. These are organized in sub-folders of `emulation/emulators/z<redshift>/<cosmo>/`, where `<redshift>` is the evaluation redshift and `<cosmo>` names the cosmological model:
* `samples/`: training and test samples, each containg two files `cosmological.txt` and `agnostic.txt`.
* `models/`: trained ensembles of dense networks for forward (cosmological $\to$ agnostic) and inverse (agnostic $\to$ cosmological) emulation, with various architectural choices for each (shallow, deep, etc.).
* `plots/`: plots demonstrating emulator performance.
* `xilin/`: for reference, this folder contains the ground truth linear 2pcf at the evaluation redshift.

## Examples
In the folder `examples/` we provide a number of Jupyter notebooks.
* `ZelSmear_Explore_Theory.ipynb`:\
  This notebook demonstrates the use of the theory routines using simple examples. There are step-by-step demonstrations for initializing the `TheoryManipulator` class, accessing file locations and default dictionaries, making a basic model prediction using `calc_model`, loading stored data, comparing data and model in plots and using $\chi^2$, accessing and plotting the ground truth linear 2pcf and (raw and smeared) basis functions and their derivatives, and finally producing 1-parameter variations around the stored best fit parameter vector (or any user-defined parameter vector) to study the impact of each parameter separately.    
* `ZelSmear_MCMC_mpi.ipynb` and `ZelSmear_MCMC.ipynb`:\
  These notebooks show how to use the model in an MCMC analysis. The notebook `ZelSmear_MCMC_mpi.ipynb` is constructed for use with MPI on a multi-core desktop (using 6 cores by default, which can be changed). This notebook was used to produce all the MCMC results and associated plots in [PS26b](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract). The notebook `ZelSmear_MCMC.ipynb` is essentially a copy of `ZelSmear_MCMC_mpi.ipynb`, but is built to be used on a single processor, i.e. it will perform MCMC with a single chain and *can be run without MPI support*.
* `ZelSmear_QuickShow.ipynb`:\
  This is a convenience notebook to track the progress of MCMC chains while they run. The various variables should be set to match the choices made in `ZelSmear_MCMC_mpi.ipynb` or `ZelSmear_MCMC.ipynb`, as the case may be.
* `Pairwise_Visualize_Abacus.ipynb`:\
  This notebook visualizes the stored measurements of pairwise correlations in the _DESI-LRG2_ and _Euclid-ELG_ samples. It was used to generate some of the plots in [PS26b](https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P/abstract).

## Emulation
*[DOCUMENTATION UNDER CONSTRUCTION]* \
*[**Note:** All emulators assume unit bias tracers. The basis coefficients should be interpreted accordingly when used in a full inference analysis.]*

## Citation
If you use any of the code and/or data in this repository, we kindly request that you include the following citations in your publication's .bib file and the URL of this repository in your text or acknowledgments.

```
@ARTICLE{ps26a,
       author = {{Paranjape}, Aseem and {Sheth}, Ravi K.},
        title = "{Zel'dovich smearing approximation of the BAO feature for model-agnostic cosmological inference}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics},
         year = 2026,
        month = feb,
          eid = {arXiv:2602.14533},
        pages = {arXiv:2602.14533},
          doi = {10.48550/arXiv.2602.14533},
archivePrefix = {arXiv},
       eprint = {2602.14533},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

@ARTICLE{ps26b,
       author = {{Paranjape}, Aseem and {Sheth}, Ravi K.},
        title = "{Impact of fiducial cosmology in model-agnostic cosmological inference with the BAO feature}",
      journal = {arXiv e-prints},
     keywords = {Cosmology and Nongalactic Astrophysics},
         year = 2026,
        month = jun,
          eid = {arXiv:2606.06591},
        pages = {arXiv:2606.06591},
archivePrefix = {arXiv},
       eprint = {2606.06591},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv260606591P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contact

Aseem Paranjape | aseem_at_iucaa_dot_in 

Ravi K Sheth    | shethrk_at_physics_dot_upenn_dot_edu 
