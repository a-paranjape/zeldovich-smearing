import numpy as np
import sys

from paths import *

from universe import Cosmology

sys.path.append(ML_Path)
from mlalgos import HyperOpt,BiSequential
from mllib import Utilities,MLUtilities

import copy,pickle
from time import time
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
            -- cosmo: str, base cosmology to sample from, one of ['lcdm','wcdm'(default),'nucdm']
            -- flat: bool, whether or not to consider only spatially flat cosmologies. 
                     If False (default), Omega_k will be sampled, else will set Omega_k=0.
            -- perc: float in (0,1) (default 0.1), percentage variations around fiducial values for each parameter.
            -- rmin,rmax: floats (default 30.0,150.0), min,max values in Mpc/h_fid for basis evaluation
            -- n_r: int (default 60), number of scales for basis evaluation
            -- verbose,logfile: usual I/O control variables
        """
        Utilities.__init__(self)
        MLUtilities.__init__(self)

        self.cosmo = setup.get('cosmo','wcdm')
        self.flat = setup.get('flat',False)
        self.perc = setup.get('perc',0.1)

        self.verbose = setup.get('verbose',True)
        self.logfile = setup.get('logfile',None)
        
        # BiSequential basis setup
        self.basis_stem = Basis_Stem
        self.load_basis()

        # evaluate basis functions
        self.rmin = setup.get('rmin',30.0)
        self.rmax = setup.get('rmax',150.0)
        self.n_r = setup.get('n_r',60)
        self.rvals = np.linspace(self.rmin,self.rmin,self.n_r)
        self.basis_func = self.evaluate_basis(self.rvals)
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
        params_setup['verbose'] = self.verbose
        params_setup['logfile'] = self.logfile

        # initialize class
        self.binet = BiSequential(params=params_setup)

        # load network parameters from files
        self.binet.load()
        if self.verbose:
            self.print_this('Extracting basis functions as NN...',self.logfile)
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
            self.print_this('Evaluating {0:d} basis functions...'.format(self.n_basis),self.logfile)

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
            pfa,pfid,co_fid,keys_all,n_params_all,n_params,param_mins,param_maxs
        """
        if self.verbose:
            self.print_this('Setting up fiducial cosmology and sampling ranges',self.logfile)
            start_time = time()
        # fiducial values from table 2 of Planck18 cosmology paper arXiv:1807.06209
        self.pfa = {'Om':0.3153,'h':0.6737,'As':np.exp(3.045)*1e-10,'ns':0.9649,'Ob':0.02237/0.6737**2,'Ok':0.0,'w0':-1.0,
                    'N_ur':3.044,'N_ncdm':0,'m_ncdm':0.0}
        self.keys_all = list(self.pfa.keys())
        self.n_params_all = len(self.pfa.keys())

        self.co_fid = Cosmology(Om=self.pfa['Om'],hubble=self.pfa['h'],As=self.pfa['As'],ns=self.pfa['ns'],Ob=self.pfa['Ob'],
                                Ok=self.pfa['Ok'],wDE0=self.pfa['w0'],
                                N_ur=self.pfa['N_ur'],N_ncdm=self.pfa['N_ncdm'],m_ncdm=self.pfa['m_ncdm'])
        self.xilin_fid = self.co_fid.calc_xi_lin(self.rvals)

        if self.verbose:
            self.print_this('... fiducial params:',pfid,self.logfile)
            self.time_this(start_time)    

# IN PROGRESS HERE
# CHANGE BELOW DEPENDING ON self.cosmo
# param ranges
param_mins = [self.pfa['Om']*(1-self.perc),self.pfa['h']*(1-self.perc),self.pfa['As']*(1-self.perc),self.pfa['ns']*(1-self.perc),
              self.pfa['Ob']*(1-self.perc),self.pfa['Ok']-self.perc,self.pfa['w0']-2*self.perc]
param_maxs = [self.pfa['Om']*(1+self.perc),self.pfa['h']*(1+self.perc),self.pfa['As']*(1+self.perc),self.pfa['ns']*(1+self.perc),
              self.pfa['Ob']*(1+self.perc),self.pfa['Ok']+self.perc,self.pfa['w0']+2*self.perc]
        
#################################################

# # -- seed: int or None (default): random number seed [use None or different values for training and testing samples!]
# self.seed = seed
# self.rng = np.random.RandomState(self.seed)

    # print('... generating data and solving linear Gaussian problem')
    # Nsamp = 3000 # 3000
    # Seed = 1983
    # rng = np.random.RandomState(Seed)
    # Y_all = ut.gen_latin_hypercube(Nsamp=Nsamp,dim=Dim,
    #                                param_mins=param_mins_all,param_maxs=param_maxs_all,rng=rng)
    
    # dummy_Y = np.zeros((Nsamp,n_s_extract)) # xi(r) values, not needed later
    # resid = np.zeros_like(dummy_Y)
    # resid_der = np.zeros_like(dummy_Y)
    # frac_err = 0.001 # expected relative error in linear 2pcf : IRRELEVANT
    # LG_cov = np.zeros((n_basis,n_basis))
    # X = np.zeros((Nsamp,n_basis))
    # Xder = np.zeros((Nsamp,n_params_der+1))
    # Y = np.zeros((Nsamp,n_params_der))
    # keys = list(pfa.keys())
    # for n in range(Nsamp):
    #     pdict = copy.deepcopy(pfa) # note pfa not pfid
    #     for p in range(Dim):
    #         pdict[keys[p]] = Y_all[n,p]
    #     try:
    #         co = Cosmology(Om=pdict['Om'],hubble=pdict['h'],As=pdict['As'],ns=pdict['ns'],Ob=pdict['Ob'],Ok=pdict['Ok'],wDE0=pdict['w0'],
    #                        verbose=False)
    #         Y[n] = calc_Y(co)
    #         dummy_Y[n] = co.calc_xi_lin(svals_extract*pdict['h']/self.pfa['h']) # use Mpc/h in varied cosmology
    #         sig = np.maximum(frac_err*np.fabs(dummy_Y[n]),1e-10*np.ones_like(dummy_Y[n]))
    #         sig = np.median(sig) # irrelevant
    #         Cinv = np.eye(n_s_extract)/sig**2
        
    #         Fisher = np.dot(basis_func,np.dot(Cinv,basis_func.T)) # since F = M^T C^-1 M and M = basis_func
    #         Finv,detF = ut.svd_inv(Fisher,hermitian=True)
    #         if n == 0:
    #             LG_cov += Finv 
    #         X[n] = np.dot(Finv,np.dot(basis_func,np.dot(Cinv,dummy_Y[n]))) # ahat = F^-1 (M^T C^-1 y)
            
    #         Fisher = np.dot(derivative_basis,np.dot(Cinv,derivative_basis.T)) # since F = M^T C^-1 M and M = basis_func
    #         Finv,detF = ut.svd_inv(Fisher,hermitian=True)
    #         Xder[n] = np.dot(Finv,np.dot(derivative_basis,np.dot(Cinv,dummy_Y[n]))) # ahat = F^-1 (M^T C^-1 y)
    #     except Exception:
    #         X[n] += np.nan
    #         Xder[n] += np.nan
        
    #     ut.status_bar(n,Nsamp)

    # X = X.T # (n_basis,nsamp)
    # Y = Y.T # (n_theta,nsamp)
    # Xder = Xder.T # (n_params_der+1,nsamp)
    
    # condfin = np.isfinite(X) 
    # condfin_der = np.isfinite(Xder) 
    # indfin = np.where((np.prod(condfin,axis=0) & np.prod(condfin_der,axis=0)).astype(bool))[0]
    # X = X[:,indfin]
    # Y = Y[:,indfin]
    # Xder = Xder[:,indfin]
    # resid = resid[indfin]
    # resid_der = resid_der[indfin]
    # dummy_Y = dummy_Y[indfin]
    # del condfin,condfin_der,indfin
    # gc.collect()
        
    # Nsamp = X.shape[1]
    
    # print('... calculating residuals')
    # for n in range(Nsamp):
    #     resid[n] = np.dot(basis_func.T,X[:,n])/(dummy_Y[n]+1e-15)-1
    #     resid_der[n] = np.dot(derivative_basis.T,Xder[:,n])/(dummy_Y[n]+1e-15)-1
    
    # print('Saving to file:',file_X)
    # np.savetxt(file_X,X,fmt='%.8e')
    # print('Saving to file:',file_Y)
    # np.savetxt(file_Y,Y,fmt='%.8e')
    # print('Saving to file:',file_resid)
    # np.savetxt(file_resid,resid,fmt='%.8e')
    
    # print('Saving to file:',file_Xder)
    # np.savetxt(file_Xder,Xder,fmt='%.8e')
    # print('Saving to file:',file_resid_der)
    # np.savetxt(file_resid_der,resid_der,fmt='%.8e')    
