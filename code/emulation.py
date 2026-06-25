import numpy as np
import sys
from pathlib import Path

from paths import *

from universe import Cosmology

sys.path.append(ML_Path)
from mlalgos import HyperOpt,BiSequential
from mllib import Utilities,MLUtilities

import copy,pickle
from time import time
import gc
# import scipy.linalg as linalg
# import scipy.special as sysp
# import scipy.optimize as optimize

#################################################
class AgnosticEmulator(Utilities,MLUtilities):
    """ Emulator routines for mapping cosmological parameters to agnostic parameters and vice-versa. """
    #############################################
    def __init__(self,setup={}):
        """ Emulator routines for mapping cosmological parameters to agnostic parameters and vice-versa.
            setup should be dictionary with a subset of the following keys
            -- out_stem: str (default './'), path/of/folder/ where all outputs [samples and trained models] will be written 
            -- cosmo: str, base cosmology to sample from, one of ['lcdm','wcdm'(default),'w0wacdm','nucdm']
                      Note: 'nucdm' currently will only vary the mass of a single neutrino species.
            -- flat: bool, whether or not to consider only spatially flat cosmologies. 
                     If False (default), Omega_k will be sampled, else will set Omega_k=0.
            -- perc: float in (0,1) (default 0.1), percentage variations around fiducial values for each parameter.
            -- rmin,rmax: floats (default 30.0,150.0), min,max values in Mpc/h_fid for basis evaluation
            -- n_r: int (default 60), number of scales for basis evaluation
            -- mnu_max: float (default 0.3), maximum neutrino mass in eV [only relevant if cosmo=='nucdm']
            -- verbose,logfile: usual I/O control variables
        """
        Utilities.__init__(self)
        MLUtilities.__init__(self)
        start_time = time()
        
        self.neutrino_cosmologies = ['nucdm']
        
        self.out_stem = setup.get('out_stem','./')
        self.cosmo = setup.get('cosmo','wcdm')
        self.flat = setup.get('flat',False)
        self.perc = setup.get('perc',0.1)
        self.mnu_max = setup.get('mnu_max',0.3) if self.cosmo in self.neutrino_cosmologies else None

        self.verbose = setup.get('verbose',True)
        self.logfile = setup.get('logfile',None)

        self.keys_absolute = ['Ok','wa'] # keys for which perc should be treated as absolute variation
        
        if self.verbose:
            self.print_this('Agnostic emulator for BAO inference...',self.logfile)
        
        # BiSequential basis setup
        self.basis_stem = Basis_Stem
        self.load_basis()

        # evaluate basis functions
        self.rmin = setup.get('rmin',30.0)
        self.rmax = setup.get('rmax',150.0)
        self.n_r = setup.get('n_r',60)
        self.rvals = np.linspace(self.rmin,self.rmax,self.n_r)
        self.basis_func = self.evaluate_basis(self.rvals)

        # setup fiducial cosmology and param variation lists
        self.setup_fiducial_cosmology()
        
        if self.verbose:
            self.print_this('... setup complete',self.logfile)
            self.time_this(start_time)    
        
    #############################################

    #############################################
    def load_basis(self):
        """ Simple wrapper to load BiSequential basis. Sets the following class attributes:
            binet,basis,n_basis
        """
        # read and modify setup parameters
        with open(self.basis_stem + '.pkl', 'rb') as f:
            params_setup = pickle.load(f)
        params_setup['file_stem'] = self.basis_stem
        params_setup['verbose'] = False # self.verbose
        # params_setup['logfile'] = self.logfile

        # initialize class
        self.binet = BiSequential(params=params_setup)

        # load network parameters from files
        self.binet.load()
        if self.verbose:
            self.print_this('... extracting basis functions as NN',self.logfile)
        self.basis = self.binet.extract_basis()
        self.n_basis = self.basis.n_layer[-1]+1 # +1 for constant
        if self.verbose:
            self.print_this('... done',self.logfile)
            
        return
    #############################################


    #############################################
    def evaluate_basis(self,rvals):
        """ Simple wrapper to evaluate BiSequential basis functions on specified range. 
            -- rvals: 1-d array of scale values
            Returns array of shape (self.n_basis,rvals.size).
        """
        if self.verbose:
            self.print_this('... evaluating {0:d} basis functions'.format(self.n_basis),self.logfile)

        # evaluate basis functions
        basis_func = self.basis.predict(self.rv(rvals))
        basis_func = np.concatenate((self.rv([1.0]*rvals.size),basis_func),axis=0) # account for constant
        # basis_func has shape (n_basis,n_r)
        # basis_func.T is design matrix M of linear Gaussian problem
        if self.verbose:
            self.print_this('... done',self.logfile)
            
        return basis_func
    #############################################        

    #############################################        
    def setup_fiducial_cosmology(self):
        """ Simple wrapper to setup fiducial cosmology. Sets following class attributes: 
            -- pfid,co_fid,keys_vary,n_params,param_mins,param_maxs
            Notes:
              (i) self.pfid contains fiducial values of *all* params, including those held fixed.
             (ii) self.keys_vary is a subset of self.pfid.keys(), since not all params are varied.
            (iii) self.param_mins,self.param_maxs are ordered by self.keys_vary.
        """
        if self.verbose:
            self.print_this('... setting up fiducial cosmology and sampling ranges',self.logfile)
        # fiducial values from table 2 of Planck18 cosmology paper arXiv:1807.06209
        self.pfid = {'Om':0.3153,'h':0.6737,'As':np.exp(3.045)*1e-10,'ns':0.9649,'Ob':0.02237/0.6737**2,'Ok':0.0,
                     'w0':-1.0,'wa':0.0,
                     'N_ur':3.044,'N_ncdm':0,'m_ncdm':0.0}

        self.co_fid = Cosmology(Om=self.pfid['Om'],hubble=self.pfid['h'],As=self.pfid['As'],ns=self.pfid['ns'],Ob=self.pfid['Ob'],
                                Ok=self.pfid['Ok'],wDE0=self.pfid['w0'],wDEa=self.pfid['wa'],
                                N_ur=self.pfid['N_ur'],N_ncdm=self.pfid['N_ncdm'],m_ncdm=self.pfid['m_ncdm'],verbose=False)
        self.xilin_fid = self.co_fid.calc_xi_lin(self.rvals)

        if self.verbose:
            cosmo_str = self.cosmo + (' (flat)' if self.flat else '')
            self.print_this('... setting up parameter variations for family: '+cosmo_str,self.logfile)
            
        keys_vary = list(self.pfid.keys())
        keys_vary.remove('N_ur')   # never vary N_ur,N_ncdm 
        keys_vary.remove('N_ncdm') # stochastically
        if self.flat:
            keys_vary.remove('Ok')        
        if self.cosmo not in ['wcdm','w0wacdm']:
            keys_vary.remove('w0')
            keys_vary.remove('wa')
        if self.cosmo not in self.neutrino_cosmologies:
            keys_vary.remove('m_ncdm')
        if self.cosmo == 'wcdm':
            keys_vary.remove('wa')
            
        self.keys_vary = keys_vary

        self.n_params = len(self.keys_vary)
        
        if self.verbose:
            self.print_this('... fiducial params:',self.logfile)
            for key in self.keys_vary:
                self.print_this('... ... '+key+': {0:.4e}'.format(self.pfid[key]),self.logfile)

        # ordering of self.param_mins,self.param_maxs is by self.keys_vary
        self.param_mins = []
        self.param_maxs = []
        for key in self.keys_vary:
            value = 1.0*self.pfid[key]
            multiplier = 1.0 if key in self.keys_absolute else value
            self.param_mins.append(value - self.perc*multiplier)
            self.param_maxs.append(value + self.perc*multiplier)
        # although the above also sets some spurious numbers for neutrino params N_ur and N_ncdm, those won't be used in self.gen_samp
        if self.cosmo in self.neutrino_cosmologies:
            ind_mncdm = self.keys_vary.index('m_ncdm')
            self.param_mins[ind_mncdm] = 1e-4
            print('!WARNING!: need to figure out how to set minimum neutrino mass!')
            self.param_maxs[ind_mncdm] = self.mnu_max
            
        return
    #############################################        

    #############################################
    def gen_sample(self,sample_setup={}):
        """ Generate sample for training/testing of downstream emulators.
            sample_setup should be dictionary with a subset of the following keys
            -- n_samp: int (default 1), number of samples to produce. 
                       Arrays for agnostic (self.n_basis,n_samp) and cosmological (self.n_params,n_samp) parameters will be returned.
                       By default, these will also be written into out_dir = self.out_stem + self.cosmo (+'_flat') + '/samples/' + sample_stem
                       in the files out_dir/agnostic.txt and out_dir/cosmological.txt.
            -- seed: int or None (default), random number seed [use None or different values for training and testing samples!]
            -- sample_stem: str (default 'train'), name of sample, e.g., 'train','test','validate', etc.
            -- force: bool (default False), relevant if an earlier sample exists with same sample_stem.
                      If True, then generate and write out a new sample, else read the existing sample.
                      If reading fails for one or more files, a fresh sample is generated and written out.
            -- include_fiducial: bool (default False), whether or not to include fiducial parameter vector in the sample 
                                 [e.g., useful to set to True for test sample]
            -- save_xi: bool (default False), whether or not to store xilin(r) values [can be memory intensive]
                        These will be written into xi_dir = self.out_stem + self.cosmo (+'_flat') + '/xilin/' + sample_stem
                        in the file xi_dir/xilin.txt.
            Returns:
            -- agnostic (self.n_basis,n_samp), cosmological (self.n_params,n_samp)[, xilin (n_samp,self.n_r), only if save_xi=True]
        """
        start_time = time()
        
        n_samp = sample_setup.get('n_samp',1)
        seed = sample_setup.get('seed',None)
        sample_stem = sample_setup.get('sample_stem','train')
        force = sample_setup.get('force',False)
        include_fiducial = sample_setup.get('include_fiducial',False)
        save_xi = sample_setup.get('save_xi',False)
        
        flat_str = '_flat' if self.flat else ''
        out_dir = self.out_stem + self.cosmo + flat_str + '/samples/' + sample_stem # folder to write/read samples to/from
        file_agnostic = out_dir + '/agnostic.txt'
        file_cosmological = out_dir + '/cosmological.txt'

        if save_xi:
            xi_dir = self.out_stem + self.cosmo + flat_str + '/xilin/' + sample_stem # folder to write/read xilin to/from
            file_xi = xi_dir + '/xilin.txt'
        
        if self.verbose:
            self.print_this('Generating/reading sample from :'+out_dir,self.logfile)
            
        if (not force) & Path(out_dir).is_dir():
            if self.verbose:
                self.print_this('... folder exists, attempting to read',self.logfile)
            agnostic = np.loadtxt(file_agnostic).T if Path(file_agnostic).is_file() else None
            cosmological = np.loadtxt(file_cosmological).T if Path(file_cosmological).is_file() else None
            if (agnostic is None) | (cosmological is None):
                if self.verbose:
                    self.print_this('... WARNING: failed to read one or more files. Generating fresh sample.',self.logfile)
            else:
                if self.verbose:
                    self.print_this('... read {0:d} samples'.format(agnostic.shape[1]),self.logfile)
            if save_xi:
                xilin = np.loadtxt(file_xi) if Path(file_xi).is_file() else None
        else:
            agnostic = None
            cosmological = None

        if (agnostic is None) | (cosmological is None):
            if self.verbose:
                self.print_this('... generating and storing fresh sample',self.logfile)
            # either force is True or reading failed for one or more files
            Path(out_dir).mkdir(parents=True,exist_ok=True)
            if save_xi:
                Path(xi_dir).mkdir(parents=True,exist_ok=True)
            rng = np.random.RandomState(seed)

            cosmological = self.gen_latin_hypercube(Nsamp=n_samp,dim=self.n_params,
                                                    param_mins=self.param_mins,param_maxs=self.param_maxs,rng=rng)
                
            if include_fiducial:
                if self.verbose:
                    self.print_this('... ... fiducial param vector will be included in sample',self.logfile)
                values_fid = []
                for key in self.keys_vary:
                    values_fid.append(self.pfid[key])
                cosmological = np.concatenate((cosmological,self.rv(values_fid)),axis=0)
            # LHC sample of shape (n_samp,self.n_params), with axis 1 ordered by self.keys_vary
            n_samp = cosmological.shape[0]

            xilin = np.zeros((n_samp,self.n_r)) # xi(r) values, possibly saved later
            agnostic = np.zeros((n_samp,self.n_basis))
            
            for n in range(n_samp):
                pdict = copy.deepcopy(self.pfid) # copy of full dictionary
                for p in range(self.n_params):
                    pdict[self.keys_vary[p]] = cosmological[n,p] # only varied params modified
                # reset N_ur,N_ncdm for neutrino cosmologies
                if self.cosmo in self.neutrino_cosmologies:
                    pdict['N_ur'] = 2.0328
                    pdict['N_ncdm'] = 1
                # undo reset for fiducial cosmology (in case it does not have massive neutrino)
                if include_fiducial & (n == n_samp-1):
                    pdict['N_ur'] = self.pfid['N_ur']
                    pdict['N_ncdm'] = self.pfid['N_ncdm']

                try:
                    co = Cosmology(Om=pdict['Om'],hubble=pdict['h'],As=pdict['As'],ns=pdict['ns'],Ob=pdict['Ob'],
                                   Ok=pdict['Ok'],wDE0=pdict['w0'],wDEa=pdict['wa'],
                                   N_ur=pdict['N_ur'],N_ncdm=pdict['N_ncdm'],m_ncdm=pdict['m_ncdm'],verbose=False)
                    xilin[n] = co.calc_xi_lin(self.rvals*pdict['h']/self.pfid['h']) # use Mpc/h in varied cosmology
                    
                    Cinv = np.eye(self.n_r)
                    Fisher = np.dot(self.basis_func,np.dot(Cinv,self.basis_func.T)) # since F = M^T C^-1 M and M = basis_func
                    Finv,detF = self.svd_inv(Fisher,hermitian=True)
                    agnostic[n] = np.dot(Finv,np.dot(self.basis_func,np.dot(Cinv,xilin[n]))) # ahat = F^-1 (M^T C^-1 y)            
                except Exception:
                    agnostic[n] += np.nan
                if self.verbose:
                    self.status_bar(n,n_samp)

            agnostic = agnostic.T # (self.n_basis,n_samp)
            cosmological = cosmological.T # (self.n_params,n_samp)

            condfin = np.isfinite(agnostic) 
            indfin = np.where(np.prod(condfin,axis=0).astype(bool))[0]
            agnostic = agnostic[:,indfin]
            cosmological = cosmological[:,indfin]
            xilin = xilin[indfin]
            del condfin,indfin
            gc.collect()
        
            if self.verbose:
                self.print_this('... {0:d} of requested {1:d} samples generated'.format(agnostic.shape[1],n_samp),self.logfile)
                self.print_this('... saving to file: '+file_agnostic,self.logfile)
            np.savetxt(file_agnostic,agnostic.T,fmt='%.8e')
            if self.verbose:
                self.print_this('... saving to file: '+file_cosmological,self.logfile)
            np.savetxt(file_cosmological,cosmological.T,fmt='%.8e')
            
            if save_xi:
                if self.verbose:
                    self.print_this('... saving to file: '+file_xi,self.logfile)
                    np.savetxt(file_xi,xilin,fmt='%.8e')

        if self.verbose:
            self.time_this(start_time)    

        return (agnostic,cosmological) if not save_xi else (agnostic,cosmological,xilin) 
    #############################################        


if __name__ == "__main__":

    # -- out_stem: str (default './'), path/of/folder/ where all outputs [samples and trained models] will be written 
    # -- cosmo: str, base cosmology to sample from, one of ['lcdm','wcdm'(default),'nucdm']
    # -- flat: bool, whether or not to consider only spatially flat cosmologies. 
    #          If False (default), Omega_k will be sampled, else will set Omega_k=0.
    # -- perc: float in (0,1) (default 0.1), percentage variations around fiducial values for each parameter.
    # -- rmin,rmax: floats (default 30.0,150.0), min,max values in Mpc/h_fid for basis evaluation
    # -- n_r: int (default 60), number of scales for basis evaluation
    # -- verbose,logfile: usual I/O control variables
    setup = {'out_stem':'temp/','cosmo':'wcdm','flat':False}
    agem = AgnosticEmulator(setup=setup)
    
    # -- n_samp: int (default 1), number of samples to produce. 
    #            Arrays for agnostic (self.n_basis,n_samp) and cosmological (self.n_params,n_samp) parameters will be returned.
    #            By default, these will also be written into out_dir = self.out_stem + self.cosmo (+'_flat') + '/samples/' + sample_stem
    #            in the files out_dir/agnostic.txt and out_dir/cosmological.txt.
    # -- seed: int or None (default), random number seed [use None or different values for training and testing samples!]
    # -- sample_stem: str (default 'train'), name of sample, e.g., 'train','test','validate', etc.
    # -- force: bool (default False), relevant if an earlier sample exists with same sample_stem.
    #           If True, then generate and write out a new sample, else read the existing sample.
    #           If reading fails for one or more files, a fresh sample is generated and written out.
    # -- include_fiducial: bool (default False), whether or not to include fiducial parameter vector in the sample 
    #                      [e.g., useful to set to True for test sample]
    # -- save_xi: bool (default False), whether or not to store xilin(r) values [can be memory intensive]
    #             These will be written into xi_dir = self.out_stem + self.cosmo (+'_flat') + '/xilin/' + sample_stem
    #             in the files xi_dir/xilin.txt
    sset = {'n_samp':15,'include_fiducial':True,'save_xi':True,'force':False}
    
    out = agem.gen_sample(sample_setup=sset)
    if sset['save_xi']:
        agnostic,cosmological,xilin = out
    else:
        agnostic,cosmological = out
    n_samp_exp = sset['n_samp']+int(sset['include_fiducial'])
    print('agnostic.shape:',agnostic.shape,'; expected: ({0:d},{1:d})'.format(agem.n_basis,n_samp_exp))
    print('cosmological.shape:',cosmological.shape,'; expected: ({0:d},{1:d})'.format(agem.n_params,n_samp_exp))
    if sset['save_xi'] & (xilin is not None):
        print('xilin.shape:',xilin.shape,'; expected: ({0:d},{1:d})'.format(n_samp_exp,agem.n_r))

        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(4,4))
        for n in range(np.min([10,n_samp_exp])):
            plt.plot(agem.rvals,agem.rvals**2*xilin[n],'-',lw=0.5)
        plt.show()
