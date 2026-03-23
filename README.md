# Model-agnostic inference using the BAO feature

## Table of contents
* [Introduction](#introduction)
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Code organization](#code-organization)
* [Data organization](#data-organization)
* [Examples](#examples)
* [Citation](#citation)
* [Contact](#contact)

## Introduction
A model-agnostic description of the baryon acoustic oscillation (BAO) feature in redshift space requires a number of ingredients. Physically, one must describe the impact of cosmological bulk flows which progressively and anisotropically smear out the feature over time. One must also model the effects of the scale dependence of tracer bias and the mode coupling between short and long scales. All of these can be incorporated using the Zel'dovich approximation alone, without reference to any particular cosmological model. On the technical front, one needs a robust, complete and cosmology-independent basis to describe the shape of the real space BAO feature in linear theory, which can then be propagated to the nonlinearly evolved, measured feature in redshift space. Finally, as in a traditional analysis, one must account for a possibly incorrect conversion of observed angular and redshift separations to comoving lengths.

These ingredients have been developed by us in a series of recent papers. 

This repository provides an implementation of the final **Zel'dovich smearing** model that brings these pieces together, as described by [Paranjape & Sheth (2026a)](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) (PS26a below) and [Paranjape & Sheth (2026b)](??) (PS26b below). 

Additionally, we provide measurements of the relevant pairwise correlations and their expected covariance for the following tracer samples: 
* A toy sample mimicking DESI LRGs as described by [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract).
* Two $N$-body halo samples drawn from the [publicly available](https://abacussummit.readthedocs.io/en/latest/abacussummit.html) $\textsf{AbacusSummit}$ suite's baseline `c000` cosmology as described by [PS26b](??):
  * $\textsf{DESI-LRG2}$ ($z=0.8$) mimicking luminous red galaxies (LRGs) being observed by the [DESI](https://ui.adsabs.harvard.edu/abs/2016arXiv161100036D/abstract) survey, 
  * $\textsf{Euclid-ELG}$ ($z=1.1$) mimicking H $\alpha$ emission line galaxies (ELGs) being observed by the [*Euclid*](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A...1E/abstract) mission.

We also provide the code we used to produce the measurements for the $\textsf{DESI-LRG2}$ and $\textsf{Euclid-ELG}$ halo samples.

The code and data in this repository should be sufficient to reproduce all the main results of [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) and [PS26b](??). We have provided several **example notebooks** (described below) to implement the MCMC analysis and explore the theoretical model.

## Dependencies
* Python 3.9+, NumPy 2.0+
* [Cobaya](https://cobaya.readthedocs.io/en/latest/): Our model is implemented in the $\texttt{Cobaya}$ framework developed by [Torrado & Lewis (2021)](https://ui.adsabs.harvard.edu/abs/2021JCAP...05..057T/abstract). This allows for straightforward integration into Markov Chain Monte Carlo (MCMC) pipelines.
* [GetDist](https://getdist.readthedocs.io/): This is used for analysing MCMC chains and producing plots of posterior and other distributions.
* [mlfundas](https://github.com/a-paranjape/mlfundas): This machine learning repository is primarily used for its implementation of the `BiSequential` basis described by [Paranjape & Sheth (2025)](https://ui.adsabs.harvard.edu/abs/2025JCAP...06..009P/abstract).
* [sahyadri-sandbox](https://github.com/a-paranjape/sahyadri-sandbox): This repository is used for our implementation of the core algorithms needed for measuring pairwise correlations (anisotropic 2pcf and power spectrum) in $N$-body tracer samples, although other implementations can also be used.

## Installation
The following steps should be sufficient for using the functionality of this repository:
1. Install $\texttt{Cobaya}$ as described [here](https://cobaya.readthedocs.io/en/latest/installation.html). If you wish to run the MCMC analysis using MPI support, please be sure to install $\texttt{Cobaya}$ with MPI support, as described on its installation page. Installing $\texttt{Cobaya}$ should also automatically install $\texttt{GetDist}$ if you don't already have it.
2. Clone into the $\texttt{mlfundas}$ repository. E.g., for HTTPS-based transfer, in your chosen install location use
  ```
  git clone https://github.com/a-paranjape/mlfundas.git
  ```
3. Setup $\texttt{sahyadri-sandbox}$. *(Only needed if pairwise correlations need to be measured in N-body samples)*
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
         * Set the variable `Abacus_Path` to `/your/path/to/abacus/halos` (these will need to be separately downloaded, see the $\texttt{AbacusSummit}$ [data access page](https://abacussummit.readthedocs.io/en/latest/data-access.html)).
   * Test the basic installation:
      * In a terminal in `your/path/to/zeldovich-smearing/code`, type
        ```
        python -c "from theory import TheoryManipulator; tm = TheoryManipulator(sample='DESI-LRG2')"
        ```
         This should produce a bunch of text related to the `BiSequential` basis instance, followed by a list containing two paths to `.txt` files and ending with the phrase `... setup complete`. \
        *\[**Note:** Although this doesn't explicitly check whether MPI capability (if requested) is correctly installed, that should have been accounted for if the Cobaya installation steps for MPI support were followed precisely.\]*


## Code organization
* `zelsmear.py`: This is the main script containing definitions of the theory class `ZeldovichSmearingTheory` and likelihood class `ZeldovichSmearingLike`:
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
* `theory.py`: This script contains the `TheoryManipulator` class that internally sets up a dummy Cobaya-friendly info dictionary and exposes user-friendly routines to compute and manipulate the model prediction. 
  * Upon instantiation, the `TheoryManipulator.theory` object contains as methods all the methods of the `ZeldovichSmearingTheory` class, e.g. those described above.
  * Additionally, the `TheoryManipulator` instance exposes the methods `calc_model`, `calc_chi2` and `vary_prediction`, among others, which can be used to study the model predictions and compare them with the data sets included in the repository.
    
  Example usage can be found in the `ZelSmear_Explore_Theory.ipynb` notebook described under [Examples](#examples).
* `pairwise_abacus.py`: \[DESCRIPTION PENDING\]
* `paths.py`: This is an auxiliary file that must be edited during installation and containing paths to various dependencies.

[^1]:See [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) and [PS26b](??) for the original definitions of the quantities listed here.

## Data organization
We provide several useful data sets in the folder `examples/data/`.

These data sets include the following measurements for the toy model from [PS26a](https://ui.adsabs.harvard.edu/abs/2026arXiv260214533P/abstract) and the $\textsf{DESI-LRG2}$ and $\textsf{Euclid-ELG}$ samples constructed using $\textsf{AbacusSummit}$ halos from [PS26b](??):
* Redshift space multipoles $\ell=0,2,4$ of the tracer 2pcf $\xi^{(\ell)}\_{\rm obs}(s)$.
* Redshift space multipoles $\ell=0,2,4$ of the tracer power spectrum $P^{(\ell)}\_{\rm obs}(k)$ for the $\textsf{DESI-LRG2}$ and $\textsf{Euclid-ELG}$ samples.
* Integrals of the power spectrum multipoles over low $k$ bins $\Sigma^{(\ell)2}_{\rm obs}$.
* Gauss-Poisson estimates of the joint covariance matrix of $\xi^{(\ell)}\_{\rm obs}(s)$ and $\Sigma^{(\ell)2}_{\rm obs}$. For the $\textsf{DESI-LRG2}$ and $\textsf{Euclid-ELG}$ samples, these are scaled to match diagonal errors on each quantity as estimated using 25 realizations of the $\textsf{AbacusSummit}$ baseline `c000` boxes.
* Real space 2pcf and power spectrum estimates for the $\textsf{DESI-LRG2}$ and $\textsf{Euclid-ELG}$ samples (for reference only).
    
For each measured quantity, the data set includes separate measurements from each realization (or 'phase') of the 25 available, along with plot-friendly files reporting the measurements and diagonal errors for the reference phase for each sample. The reference phases are `ph000` for $\textsf{DESI-LRG2}$ and `ph009` for $\textsf{Euclid-ELG}$. MCMC-friendly data files for the reference samples are also provided for use with `ZeldovichSmearingLike`.

## Examples
In the folder `examples/` we provide a number of Jupyter notebooks.
* `ZelSmear_Explore_Theory.ipynb`: 
* `ZelSmear_MCMC_mpi.ipynb` and `ZelSmear_MCMC.ipynb`: 
* `ZelSmear_QuickShow.ipynb`:
* `Pairwise_Visualize_Abacus.ipynb`: 

## Citation
If you use any of the code and/or data in this repository, we kindly request that you include the following citations in your publication's .bib file, along with the URL of this repository.

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
```

## Contact

Aseem Paranjape | aseem_at_iucaa_dot_in 

Ravi K Sheth    | shethrk_at_physics_dot_upenn_dot_edu 
