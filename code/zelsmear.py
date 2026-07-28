import numpy as np
import sys

from paths import *

sys.path.append(ML_Path)
from mlalgos import BiSequential
from mllib import Utilities

import copy,pickle
import scipy.linalg as linalg
import scipy.special as sysp
import scipy.optimize as optimize

from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

# from emulation import AgnosticEmulator

#########################################
class ZeldovichSmearingLike(Likelihood,Utilities):
    #########################################
    scales_file = None # needed for reading scales
    cov_file = None # needed for reading covariance matrix [expect single string]
    data_file = None # needed for reading data
                     # expect list with [filename_Sig2obs,filename_ell] if include_sig2obs == True else
                     # single string with filename_xiell.
    rescale = 1.0
    L_Max = 3 # 1,2 or 3
    modify_data = False
    include_Sig2obs = False
    #########################################
    def initialize(self):
        Utilities.__init__(self)

        self.offset = self.L_Max if self.include_Sig2obs else 0
        
        if self.scales_file is None:
            raise Exception("scales_file should be valid file path in ZeldovichSmearingLike.")
        
        self.svals = np.loadtxt(self.scales_file)        
        self.N_Data = self.svals.size
        
        expected_size = self.L_Max*self.N_Data
        if self.include_Sig2obs:
            expected_size += self.L_Max

        if self.data_file is None:
            raise Exception("data_file should be valid (list of) file path(s) in ZeldovichSmearingLike.")
        if self.cov_file is None:
            raise Exception("cov_file should be valid file path in ZeldovichSmearingLike.")

        cov_data = np.loadtxt(self.cov_file)
        # check covariance matrix
        if cov_data.shape != (expected_size,expected_size):
            raise Exception("Expecting cov_data matrix of shape ({0:d},{0:d}) in ZeldovichSmearingLike, found shape (".format(expected_size)
                            +','.join([str(d) for d in cov_data.shape])+')')
        data = np.array([])
        if self.include_Sig2obs:
            # expect 2 data files
            print(self.data_file)
            if len(self.data_file)==2:
                Sig2obs_data = np.loadtxt(self.data_file[0])
                if len(Sig2obs_data.shape) > 1:
                    Sig2obs_data = Sig2obs_data.T[0]
                xiell_data = np.loadtxt(self.data_file[1])
                data = np.concatenate((Sig2obs_data,xiell_data))
            else:
                raise Exception("Expecting data_file to be list with [filename_Sig2obs,filename_xiell].")
        else:
            data = np.loadtxt(self.data_file)

        # check data vector
        if data.shape != (expected_size,):
            raise Exception("Expecting data vector of shape ({0:d},) in ZeldovichSmearingLike, found shape (".format(expected_size)
                            +','.join([str(d) for d in data.shape])+')')

        self.data,self.cov_data = self.organize_data(data,cov_data)            
        self.Dim_Data = self.data.size
        
        if np.any(linalg.eigvals(self.cov_data) <= 0.0):
            raise ValueError('non-positive definite covariance matrix detected')
        self.L = linalg.cholesky(self.cov_data,lower=True) # so C = L L^T

        self.derived_list = ['f','peak','LP','ZC'] # EXTEND AS NEEDED
    #########################################

    #########################################
    def get_requirements(self):
        """ Theory code should return model array and derived params ['f','peak','LP','ZC',...]. """
        reqs = {'model': None}
        for par in self.derived_list:
            reqs[par] = None
        return reqs
    #########################################

    #########################################
    def organize_data(self,data,cov_data):
        """ Convenience function to setup data and cov_data. Also useful externally for plotting. """
        cov_data_use = cov_data.copy()
        if cov_data_use.shape != (data.size,data.size):
            raise ValueError('Incorrect covariance shape in ZeldovichSmearingLike().')
            
        data *= self.rescale
        cov_data_use *= self.rescale**2
        if self.include_Sig2obs:
            # undo effect of rescale on Sig2obs.
            data[:self.L_Max] /= self.rescale
            cov_data_use[:self.L_Max,:self.L_Max] /= self.rescale**2
            cov_data_use[:self.L_Max,self.L_Max:] /= self.rescale
            cov_data_use[self.L_Max:,:self.L_Max] /= self.rescale

        if (self.L_Max > 1) & (self.modify_data):
            sminBys3 = (self.svals.min()/self.svals)**3
            sminBys5 = (self.svals.min()/self.svals)**5
            data_mod = data.copy()
            for L in range(1,self.L_Max):
                data_new = data_mod[self.offset+L*self.N_Data:self.offset+(L+1)*self.N_Data].copy()
                if L == 1:
                    data_new -= sminBys3*data_new[0]
                else:
                    data_new -= sminBys5*data_new[0]
                # now data_new[0] should be exactly 0.0
                # print(data_new[0])
                data_mod[self.offset+L*self.N_Data:self.offset+(L+1)*self.N_Data] = data_new
            if self.L_Max == 3:
                data = np.delete(data_mod,[self.offset+self.N_Data,self.offset+2*self.N_Data])
            else:
                data = np.delete(data_mod,[self.offset+self.N_Data])

            cov_mod = cov_data_use.copy()
            g_L = np.array([0.0,1.0,1.0])
            for L in range(self.L_Max):
                S_ell = sminBys3.copy() if L != 2 else sminBys5.copy()
                for i in range(self.N_Data):
                    for Lpr in range(self.L_Max):
                        S_ell_pr = sminBys3.copy() if Lpr != 2 else sminBys5.copy()
                        for j in range(self.N_Data):
                            delta_C = -g_L[L]*S_ell[i]*cov_data_use[self.offset+L*self.N_Data,self.offset+j+Lpr*self.N_Data]
                            delta_C -= g_L[Lpr]*S_ell_pr[j]*cov_data_use[self.offset+i+L*self.N_Data,self.offset+Lpr*self.N_Data]
                            delta_C += (g_L[L]*g_L[Lpr]*S_ell[i]*S_ell_pr[j]
                                        *cov_data_use[self.offset+L*self.N_Data,self.offset+Lpr*self.N_Data])
                            cov_mod[self.offset+i+L*self.N_Data,self.offset+j+Lpr*self.N_Data] += delta_C
                del S_ell,S_ell_pr
                
            if self.include_Sig2obs:
                for L in range(self.L_Max):
                    for i in range(self.N_Data):
                        delta_C = -g_L[L]*sminBys3[i]*cov_data_use[:self.L_Max,self.offset+i+L*self.N_Data]
                        cov_mod[:self.L_Max,self.offset+i+L*self.N_Data] += delta_C
                        delta_C = -g_L[L]*sminBys3[i]*cov_data_use[self.offset+i+L*self.N_Data,:self.L_Max]
                        cov_mod[self.offset+i+L*self.N_Data,:self.L_Max] += delta_C
            if self.L_Max == 3:
                cov_data_use = np.delete(np.delete(cov_mod,[self.offset+self.N_Data,self.offset+2*self.N_Data],axis=0),
                                         [self.offset+self.N_Data,self.offset+2*self.N_Data],axis=1)
            else:
                cov_data_use = np.delete(np.delete(cov_mod,[self.offset+self.N_Data],axis=0),[self.offset+self.N_Data],axis=1)

            del sminBys3,sminBys5,data_mod,cov_mod,g_L

        return data,cov_data_use
    #########################################

    #########################################
    def wrap_model(self,xi):
        """ Convenience function to convert 1-d array of data/model used by likelihood into list of arrays useful for plotting. """
        off = 1 if self.include_Sig2obs else 0
        arr_lens = self.N_Data*np.ones(off+self.L_Max,dtype=int) # initialize array lengths
        if self.include_Sig2obs:
            arr_lens[0] = self.L_Max # Sig2obs vals will be at start of array if they exist; not used
        if self.modify_data & (self.L_Max > 1):
            for L in range(off+1,off+self.L_Max):
                arr_lens[L] -= 1

        if xi.size != arr_lens.sum():
            raise Exception('Mismatched sizes in wrap_model().')
        out = [] # build up list ordered by L
        if self.include_Sig2obs:
            out.append(xi[:self.L_Max])
        imin = self.offset # not off
        for L in range(off,off+self.L_Max):
            imax = imin + arr_lens[L]
            sl = np.s_[imin:imax]
            out.append(xi[sl])
            imin = 1*imax
        return out
    #########################################
    
    #########################################
    def logp(self,**params_values_dict):
        """ Calculate logp = -0.5*chi2 and set derived params. """

        for par in self.derived_list:
            params_values_dict['_derived'][par] = self.provider.get_param(par)
        
        model = self.provider.get_model()*self.rescale
        if self.include_Sig2obs:
            model[:self.L_Max] /= self.rescale
        residual = self.data - model
        z = linalg.cho_solve((self.L,True),residual) # solves (L L^T) z = residual or z = C^-1 residual
        chi2 = np.dot(residual,z)
        return -0.5*chi2
    #########################################


#########################################
class ZeldovichSmearingTheory(Theory,Utilities):
    #########################################
    modify_data = True
    include_Sig2obs = False
    use_basis = np.arange(9) # list of indices of basis vectors to use
    basis_stem = Basis_Stem # imported from paths by default, can be changed at run time
    r_min = 30.0 # Mpc/h
    r_max = 150.0 # Mpc/h
    n_r = 100 # 100 gives LP convergence error ~0.1%, i.e. 3x smaller than DESI expectation
    acc_vals = {'low':8.0,'mid':24.0,'high':48.0}
    accuracy = 'low'
    scales_file = None # needed for reading scales
    # svals = None # should be specified as 1-d array of values in Mpc/h_fid
    Rpiv2 = 2.5**2 # fixed pivot of 2.5Mpc/h_fid
    L_Max = 3 # 1,2 or 3
    sdbmc = True # default True. if False, dynamically set sigma = sqrt(2)*sigv for consistency with 'no sdbmc'.
    model_AP = True # default True. if True, model effects of anisotropy due to wrong fiducial cosmology
    strong_prior = False # default False. if True, assume sampled params are cosmological+sdbmc (requires emulator), else agnostic+sdbmc.
    # emulator_setup = {} # needed when strong_prior=True, to initialize AgnosticEmulator.
    #                     # keys subset of [out_stem,cosmo,flat,z_eval,scale_planck18].
    #                     # other setup keys shouldn't be touched.
    #                     # leave empty to use default setup (flat LCDM at z_eval=0.8, scale 6.0).
    # emulator_model_name = 'shallow' # needed when strong_prior=True, to instantiate emulator ensemble. default 'shallow' (don't change unless others have been trained.)
    #########################################
    def initialize(self):
        Utilities.__init__(self)

        if self.scales_file is None:
            raise Exception("scales_file should be valid file path in ZeldovichSmearingLike.")
        
        self.svals = np.loadtxt(self.scales_file)        

        self.max_deriv = 6 if self.model_AP else 5 # largest derivative needed for smoothed basis
        if self.model_AP:
            # matrices to model AP-type effects due to fiducial cosmology
            self.Cmat = np.array([[0.,1/5.,0.],
                                  [1/5.,2/35.,2/35,],
                                  [0.,2/35.,20/693.]])
            self.Amat = np.array([[0.,2/5.,0.],
                                  [0.,2/7.,20/21.],
                                  [0.,-24/35.,20/77.]])

        ##########################
        # BiSequential basis setup
        ##########################
        # read setup parameters
        with open(self.basis_stem + '.pkl', 'rb') as f:
            params_setup = pickle.load(f)
        params_setup['file_stem'] = self.basis_stem
        # initialize class
        self.binet = BiSequential(params=params_setup)
        # load network parameters from files
        self.binet.load()
        print('Extracting basis functions as NN...')
        self.basis = self.binet.extract_basis()
        self.n_basis_all = self.basis.n_layer[-1]+1 # +1 for constant
        self.n_basis = len(self.use_basis)
        print('... done')
        
        self.rvals = np.linspace(self.r_min,self.r_max,self.n_r)
        self.dr = self.rvals[1]-self.rvals[0]
        self.bfunc = self.calc_basis()
        self.dbdr = self.calc_dbdr()

        # use these to identify unphysical solutions
        self.rvals_max_check = self.rvals.max()-2*self.dr
        self.rvals_min_check = self.rvals.min()+2*self.dr
        ##########################
        
        ##########################
        # lists of strings, useful for indexing sampled params
        self.w_names = ['w_{0:d}'.format(m) for m in self.use_basis]
        # self.param_names_all = ['beta','sigv']
        # self.param_names_all.extend(self.w_names)
        # self.param_names_all += ['b','B1st','Bvst','sigma','AMC']
        # self.param_names_all += ['qbar2','qbar4']
        ##########################

        self.N_Data = self.svals.size
        self.ds = self.svals[1]-self.svals[0]
        self.sminBys3 = (self.svals.min()/self.svals)**3
        self.sminBys5 = (self.svals.min()/self.svals)**5

        self.dim = self.N_Data*self.L_Max
        if self.modify_data & (self.L_Max > 1):
            self.dim -= 1 if self.L_Max==2 else 2
        if self.include_Sig2obs:
            self.dim += self.L_Max

        # fine svals array for integrals \bar\lambda and \bar\bar\lambda
        # quadrupole error (insensitive to n_r):
        # 48.0 --> <~1% except at s~65-71Mpc/h, where <~3% 
        # 24.0 --> <~3% 
        # 16.0 --> <~5% 
        #  8.0 --> <~11%
        if self.accuracy not in self.acc_vals.keys():
            print("accuracy must be one of ["+','.join([key for key in self.acc_vals.keys()])+"]")
            print("assuming accuracy = 'mid'")
            self.accuracy = 'mid'
        self.ds_fine = self.ds/self.acc_vals[self.accuracy] # self.ds/24.0
        self.svals_fine = np.linspace(self.svals.min(),self.svals.max(),int((self.svals.max()-self.svals.min())/self.ds_fine))
        self.ds_fine = self.svals_fine[1]-self.svals_fine[0]

        # ##########################
        # # Emulator setup
        # ##########################
        # if self.strong_prior:
        #     em_setup = {'out_stem':'../emulation/emulators/','cosmo':'lcdm','flat':True,'z_eval':0.8,'scale_planck18':6.0,'verbose':False}
        #     for key in em_setup.keys():
        #         if key in self.emulator_setup.keys():
        #             em_setup[key] = self.emulator_setup[key] # switch to user-defined value if requested
        #     self.agem = AgnosticEmulator(setup=em_setup)
        #     self.FwdEmulator = self.agem.load(invert=False,model_name=self.emulator_model_name)
        
    #########################################

    #########################################
    def calc_basis(self):
        """ Convenience function to evaluate basis functions on self.rvals. 
            Returns array basis_func (self.n_basis,self.rvals.size).
            basis_func.T is design matrix M of linear Gaussian problem.
        """
        basis_func = self.basis.predict(self.binet.rv(self.rvals))
        # apred = np.concatenate((np.ones((1,apred.shape[1])),apred),axis=0) # (n_layer_w[-1],n_samp) # original code from mlfundas
        basis_func = np.concatenate((np.ones((1,basis_func.shape[1])),basis_func),axis=0) # account for constant
        basis_func = basis_func[self.use_basis]
        return basis_func
    #########################################

    #########################################
    def calc_dbdr(self):
        """ Convenience function to evaluate basis function gradients on self.rvals. 
            Returns array dbdr (self.n_basis,self.rvals.size).
        """
        dbdr = self.basis.gradient(self.binet.rv(self.rvals)) # (n_basis,r)
        dbdr = np.concatenate((np.zeros((1,dbdr.shape[2])),dbdr[0]),axis=0) # account for constant
        dbdr = dbdr[self.use_basis]
        return dbdr
    #########################################

    #########################################
    def calc_lambda(self,sigma,fine=False):
        """ Calculate lambda_m(s|sigma) on s=self.svals (fine=False) or self.svals_fine (fine=True). """
        s = self.svals_fine.copy() if fine else self.svals.copy()
        integrand = np.zeros((self.n_basis,s.size,self.rvals.size),dtype=float)
        exp_minus = np.exp(-0.5*((np.outer(np.ones_like(s),self.rvals) - np.outer(s,np.ones_like(self.rvals)))/sigma)**2)
        exp_plus = np.exp(-0.5*((np.outer(np.ones_like(s),self.rvals) + np.outer(s,np.ones_like(self.rvals)))/sigma)**2)
        r_exp_diff = (exp_minus - exp_plus)*self.rvals
        for m in range(self.n_basis):
            integrand[m] = r_exp_diff*self.bfunc[m]
            
        lambda_m = np.trapezoid(integrand,dx=self.dr,axis=-1)/(s*np.sqrt(2*np.pi)*sigma)

        del s,integrand,exp_minus,exp_plus,r_exp_diff
        
        return lambda_m
    #########################################

    #########################################
    def calc_der_Lambda(self,sigma):
        """ Calculate Lambda_m^{(n)}(s|sigma) on s=self.svals for 1 <= n <= self.max_deriv. """
        sig_inv_n = (-1/sigma)**np.arange(1,self.max_deriv+1)
        integrand = np.zeros((self.max_deriv,self.n_basis,self.svals.size,self.rvals.size),dtype=float)
        herm_exp_diff = np.zeros((self.max_deriv,self.svals.size,self.rvals.size),dtype=float)

        r_minus_s_by_sigma = (np.outer(np.ones_like(self.svals),self.rvals) - np.outer(self.svals,np.ones_like(self.rvals)))/sigma
        r_plus_s_by_sigma = (np.outer(np.ones_like(self.svals),self.rvals) + np.outer(self.svals,np.ones_like(self.rvals)))/sigma
        exp_minus = np.exp(-0.5*r_minus_s_by_sigma**2)
        exp_plus = np.exp(-0.5*r_plus_s_by_sigma**2)

        for n in range(self.max_deriv):
            herm_exp_minus = sysp.eval_hermitenorm(n+1,-1.0*r_minus_s_by_sigma)*exp_minus # note -1*(r-s)=(s-r) as argument
            herm_exp_plus = sysp.eval_hermitenorm(n+1,r_plus_s_by_sigma)*exp_plus
            herm_exp_diff[n] = (herm_exp_minus - herm_exp_plus)*sig_inv_n[n]
            
        herm_exp_diff = herm_exp_diff*self.rvals

        for n in range(self.max_deriv):
            for m in range(self.n_basis):
                integrand[n,m] = herm_exp_diff[n]*self.bfunc[m]
            
        Lambda_m_n = np.trapezoid(integrand,dx=self.dr,axis=-1)/(np.sqrt(2*np.pi)*sigma)

        del sig_inv_n,integrand,herm_exp_diff,r_minus_s_by_sigma,r_plus_s_by_sigma,exp_minus,exp_plus
        
        return Lambda_m_n
    #########################################

    #########################################
    def calc_der_lambda(self,sigma):
        """ Calculate lambda_m^{(n)}(s|sigma) on s=self.svals for 1 <= n <= self.max_deriv. 
            Returns lambda_m^{(n)}(s|sigma) and Lambda_m^{(n)}(s|sigma), each array of shape (self.max_deriv,self.n_basis,self.svals.size)
        """
        Lambda_m_n = self.calc_der_Lambda(sigma) # (n,m,s) = (self.max_deriv,n_basis,svals.size)
        lambda_m_n = np.zeros_like(Lambda_m_n)   
        lambda_m_n[0] = (Lambda_m_n[0] - self.calc_lambda(sigma))/self.svals
        for n in range(1,self.max_deriv):
            lambda_m_n[n] = (Lambda_m_n[n] - (n+1)*lambda_m_n[n-1])/self.svals

        return lambda_m_n,Lambda_m_n
    #########################################
    
    #########################################
    def calc_lambda_bars(self,sigma):
        """ Calculate \bar lambda_m(s|sigma) [and \bar\bar lambda_m(s|sigma)] on s=self.svals. """

        # this routine will be called only if self.L_Max > 1
        # MAKE THIS EFFICIENT BY PRE-COMPUTING SLICING!

        # pre-compute lambda_m on fine grid
        lambda_m = self.calc_lambda(sigma,fine=True)

        # \bar lambda setup
        integrand = self.svals_fine**2*lambda_m
        lambda_m_bar = np.zeros((self.n_basis,self.svals.size),dtype=float)        
        # lambda_m_bar[:,0] = 0 by definition
        ind_fine = np.where(self.svals_fine <= self.svals[1])[0]
        lambda_m_bar[:,1] = np.trapezoid(integrand[:,ind_fine],dx=self.ds_fine,axis=1)
        for s in range(2,self.svals.size):
            ind_fine = np.where((self.svals_fine > self.svals[s-1]) & (self.svals_fine <= self.svals[s]))[0]
            lambda_m_bar[:,s] = np.trapezoid(integrand[:,ind_fine],dx=self.ds_fine,axis=1) + lambda_m_bar[:,s-1]
            
        lambda_m_bar = 3*lambda_m_bar/self.svals**3
        
        if self.L_Max == 2:
            lambda_m_barbar = None
        else:
            # \bar\bar lambda setup
            integrand = self.svals_fine**4*lambda_m
            lambda_m_barbar = np.zeros((self.n_basis,self.svals.size),dtype=float)
            # lambda_m_barbar[:,0] = 0 by definition
            ind_fine = np.where(self.svals_fine <= self.svals[1])[0]
            lambda_m_barbar[:,1] = np.trapezoid(integrand[:,ind_fine],dx=self.ds_fine,axis=1)
            for s in range(2,self.svals.size):
                ind_fine = np.where((self.svals_fine > self.svals[s-1]) & (self.svals_fine <= self.svals[s]))[0]
                lambda_m_barbar[:,s] = np.trapezoid(integrand[:,ind_fine],dx=self.ds_fine,axis=1) + lambda_m_barbar[:,s-1]

            lambda_m_barbar = 5*lambda_m_barbar/self.svals**5

        del integrand,ind_fine

        return lambda_m_bar,lambda_m_barbar
    #########################################
    

    #########################################
    def calc_eta_ellJ(self,beta,B1,Bvst,kstsq=None):
        """ Calculate eta_ell,J or psi_ell,J.
            -- beta: sampled value of beta=f/b
            -- B1: value of B1 inferred from B1* and sigma
            -- Bvst: sampled value of Bv* = beta * Bv
            -- kstsq: None or positive float: value of f(f+2)*sigv**2 using sampled value of sigv,beta and b
                      if None, output is psi_{ell,J}*Rpiv^{2J}, else eta_{ell,J}*Rpiv^{2J}.
            Returns array of shape (self.L_Max,3)
        """
        cond_eta = (kstsq is not None)
        eta = np.zeros((self.L_Max,3),dtype=float)
        
        B1p = B1*self.Rpiv2
        Bvstp = Bvst*self.Rpiv2
        
        betasq = beta**2
        kst4 = kstsq**2 if cond_eta else None # won't be used if not cond_eta
        Bvstpsq = Bvstp**2
        
        eta[0,0] = 1 + 2*beta/3.0 + betasq/5.0 # chkd
        eta[0,1] = 2*B1p*(1+beta/3.0) - 2*Bvstp*(1/3.0+beta/5.0) # chkd
        # eta[0,2] = B1p**2 - 2*Bvstp*B1p/5.0 + Bvstpsq/5.0 # chkd 
        eta[0,2] = B1p**2 - 2*Bvstp*B1p/3.0 + Bvstpsq/5.0 # fixed 5-->3 on 24/01/26
        if cond_eta:
            eta[0,1] -= kstsq*(1/3.0+2*beta/5.0+betasq/7.0) # chkd
            eta[0,2] += (-2*kstsq)*(B1p*(1/3.0+beta/5.0)-Bvstp*(1/5.0+beta/7.0)) + 0.5*kst4*(1/5.0+2*beta/7.0+betasq/9.0) # chkd
        if self.L_Max > 1:
            eta[1,0] = 4*beta*(1/3.0+beta/7.0) # chkd
            eta[1,1] = 4*beta*B1p/3.0 - 4*Bvstp*(1/3.0+2*beta/7.0) # chkd
            eta[1,2] = -4*Bvstp*B1p/3.0 + 4*Bvstpsq/7.0 # chkd 
            if cond_eta:
                eta[1,1] -= 2*kstsq*(1/3.0+4*beta/7.0+5*betasq/21.0) # chkd
                eta[1,2] += (-4*kstsq)*(B1p*(1/3.0+2*beta/7.0)-(Bvstp/7)*(2+5*beta/3.0)) + 2*kst4*(1/7.0+5*beta/21.0+10*betasq/99.0) # chkd 
            if self.L_Max == 3:
                eta[2,0] = 8*betasq/35.0 # chkd 
                eta[2,1] = -16*beta*Bvstp/35.0 # chkd
                eta[2,2] = 8*Bvstpsq/35.0 # chkd
                if cond_eta:
                    eta[2,1] -= (8*kstsq*beta/7.0)*(2/5.0+3*beta/11.0) # chkd
                    eta[2,2] += (-16*kstsq/7.0)*(beta*B1p/5.0-Bvstp*(1/5.0+3*beta/11.0)) + 4*kst4*(1/35.0+6*beta/77.0+6*betasq/143.0) # chkd
        
        return eta
    #########################################


    #########################################
    def calc_xiNL(self,params_dict):
        """ Calculate propagator + mode-coupling terms of config space multipoles at self.svals.
            Returns array of shape (self.L_Max,self.svals.size)
            If self.model_AP==True, then also returns dxiNL/dlns as second array of same shape (useful for AP calculations).
        """
        beta = params_dict['beta']
        sigv = params_dict['sigv']
        w_m = np.array([params_dict[key] for key in self.w_names])
        
        b = params_dict['b']
        Bvst = params_dict['Bvst']
        sigma = params_dict['sigma'] # can we exploit very low sampling speed of sigma?
        B1 = params_dict['B1st'] + 0.5*sigma**2/self.Rpiv2

        AMC = params_dict['AMC'] 
        
        f = beta*b
        kstsq = f*(f+2)*sigv**2 

        # storage for propagator and mode-coupling terms
        xi_prop = np.zeros((self.L_Max,self.svals.size),dtype=float)
        xi_MC = np.zeros((self.L_Max,self.svals.size),dtype=float)
        if self.model_AP:
            xi_prop_der = np.zeros((self.L_Max,self.svals.size),dtype=float)
            xi_MC_der = np.zeros((self.L_Max,self.svals.size),dtype=float)
        
        lambda_m = self.calc_lambda(sigma) # (m,s)
        if (self.L_Max == 1) & (not self.model_AP):
            Lambda_m_n = self.calc_der_Lambda(sigma) # (n,m,s)
        else:
            lambda_m_n,Lambda_m_n = self.calc_der_lambda(sigma) # (n,m,s)

        eta_ellJ = self.calc_eta_ellJ(beta,B1,Bvst,kstsq)
        psi_ellJ = self.calc_eta_ellJ(beta,B1,Bvst,kstsq=None)

        # prop
        xi_temp = eta_ellJ[0,0]*lambda_m - (eta_ellJ[0,1]*Lambda_m_n[1] - eta_ellJ[0,2]*Lambda_m_n[3])/self.svals # (m,s)
        xi_prop[0] = np.sum(xi_temp.T*w_m,axis=1) # (s,)

        # MC
        xi_temp = psi_ellJ[0,0]*(Lambda_m_n[0] - lambda_m) # (m,s)
        xi_temp -= psi_ellJ[0,1]*(Lambda_m_n[2] - Lambda_m_n[1]/self.svals)
        xi_temp += psi_ellJ[0,2]*(Lambda_m_n[4] - Lambda_m_n[3]/self.svals) 
        xi_MC[0] = np.sum(xi_temp.T*w_m,axis=1) # (s,)

        if self.model_AP:
            # prop
            xi_temp = eta_ellJ[0,0]*(Lambda_m_n[0] - lambda_m) # (m,s)
            xi_temp -= eta_ellJ[0,1]*(Lambda_m_n[2] - Lambda_m_n[1]/self.svals)
            xi_temp += eta_ellJ[0,2]*(Lambda_m_n[4] - Lambda_m_n[3]/self.svals) 
            xi_prop_der[0] = np.sum(xi_temp.T*w_m,axis=1) # (s,)

            # MC
            xi_temp = psi_ellJ[0,0]*(Lambda_m_n[1] - lambda_m_n[0])*self.svals # (m,s)
            xi_temp -= psi_ellJ[0,1]*(Lambda_m_n[3]*self.svals - Lambda_m_n[2] + Lambda_m_n[1]/self.svals)
            xi_temp += psi_ellJ[0,2]*(Lambda_m_n[5]*self.svals - Lambda_m_n[4] + Lambda_m_n[3]/self.svals) 
            xi_MC_der[0] = np.sum(xi_temp.T*w_m,axis=1) # (s,)
            
        if self.L_Max > 1:
            qbar2 = params_dict['qbar2']
            lambda_m_bar,lambda_m_barbar = self.calc_lambda_bars(sigma)
            s2 = self.svals**2
            s3 = s2*self.svals

            # prop
            xi_temp = eta_ellJ[1,0]*(lambda_m - lambda_m_bar) # (m,s)
            xi_temp -= eta_ellJ[1,1]*(lambda_m_n[1] - lambda_m_n[0]/self.svals)
            xi_temp += eta_ellJ[1,2]*(lambda_m_n[3] + lambda_m_n[2]/self.svals - 6*lambda_m_n[1]/s2 + 6*lambda_m_n[0]/s3) 
            xi_prop[1] = np.sum(xi_temp.T*w_m,axis=1) - eta_ellJ[1,0]*qbar2*self.sminBys3 # (s,)

            # MC
            xi_temp = psi_ellJ[1,0]*(3*(lambda_m_bar - lambda_m) + self.svals*lambda_m_n[0]) # (m,s)
            xi_temp -= psi_ellJ[1,1]*(self.svals*lambda_m_n[2] - lambda_m_n[1] + lambda_m_n[0]/self.svals)
            xi_temp += psi_ellJ[1,2]*(self.svals*lambda_m_n[4] + lambda_m_n[3] - 7*lambda_m_n[2]/self.svals + 18*lambda_m_n[1]/s2 - 18*lambda_m_n[0]/s3) 
            xi_MC[1] = np.sum(xi_temp.T*w_m,axis=1) + 3*psi_ellJ[1,0]*qbar2*self.sminBys3 # (s,)

            if self.model_AP:
                # prop
                xi_temp = eta_ellJ[1,0]*(3*(lambda_m_bar - lambda_m) + self.svals*lambda_m_n[0]) # (m,s)
                xi_temp -= eta_ellJ[1,1]*(self.svals*lambda_m_n[2] - lambda_m_n[1] + lambda_m_n[0]/self.svals)
                xi_temp += eta_ellJ[1,2]*(self.svals*lambda_m_n[4] + lambda_m_n[3] - 7*lambda_m_n[2]/self.svals + 18*lambda_m_n[1]/s2 - 18*lambda_m_n[0]/s3) 
                xi_prop_der[1] = np.sum(xi_temp.T*w_m,axis=1) + 3*eta_ellJ[1,0]*qbar2*self.sminBys3 # (s,)

                # MC
                xi_temp = psi_ellJ[1,0]*(9*(lambda_m - lambda_m_bar) - 2*self.svals*lambda_m_n[0] + self.svals**2*lambda_m_n[1]) # (m,s)
                xi_temp -= psi_ellJ[1,1]*(self.svals**2*lambda_m_n[3] + lambda_m_n[1] - lambda_m_n[0]/self.svals)
                xi_temp += psi_ellJ[1,2]*(self.svals**2*lambda_m_n[5] + 2*self.svals*lambda_m_n[4] - 7*lambda_m_n[3]
                                          + 25*lambda_m_n[2]/self.svals - 54*lambda_m_n[1]/s2 + 54*lambda_m_n[0]/s3) 
                xi_MC_der[1] = np.sum(xi_temp.T*w_m,axis=1) - 9*psi_ellJ[1,0]*qbar2*self.sminBys3 # (s,)
                
            if self.L_Max == 3:
                qbar4 = params_dict['qbar4']

                # prop
                xi_temp = eta_ellJ[2,0]*(lambda_m + 2.5*lambda_m_bar - 3.5*lambda_m_barbar) # (m,s)
                xi_temp -= eta_ellJ[2,1]*(lambda_m_n[1] - 8*lambda_m_n[0]/self.svals + 35*(lambda_m - lambda_m_bar)/s2)
                xi_temp += eta_ellJ[2,2]*(lambda_m_n[3] - 6*lambda_m_n[2]/self.svals + 15*lambda_m_n[1]/s2 - 15*lambda_m_n[0]/s3)
                qbar_contrib = 2.5*eta_ellJ[2,0]*qbar2*self.sminBys3
                qbar_contrib += self.sminBys5*(35*eta_ellJ[2,1]*qbar2/self.svals.min()**2 - 3.5*eta_ellJ[2,0]*qbar4)
                xi_prop[2] = np.sum(xi_temp.T*w_m,axis=1) + qbar_contrib # (s,)

                # MC
                xi_temp = psi_ellJ[2,0]*(self.svals*lambda_m_n[0] - 10*lambda_m - 7.5*lambda_m_bar + 17.5*lambda_m_barbar) # (m,s)
                xi_temp -= psi_ellJ[2,1]*(self.svals*lambda_m_n[2] - 8*lambda_m_n[1] + 43*lambda_m_n[0]/self.svals - 175*(lambda_m - lambda_m_bar)/s2)
                xi_temp += psi_ellJ[2,2]*(self.svals*lambda_m_n[4] - 6*lambda_m_n[3] + 21*lambda_m_n[2]/self.svals - 45*lambda_m_n[1]/s2 + 45*lambda_m_n[0]/s3)
                qbar_contrib = 1.5*psi_ellJ[2,0]*qbar2*self.sminBys3
                qbar_contrib += self.sminBys5*(35*psi_ellJ[2,1]*qbar2/self.svals.min()**2 - 3.5*psi_ellJ[2,0]*qbar4)
                qbar_contrib *= 5.0
                xi_MC[2] = np.sum(xi_temp.T*w_m,axis=1) - qbar_contrib # (s,)

                if self.model_AP:
                    # prop
                    xi_temp = eta_ellJ[2,0]*(self.svals*lambda_m_n[0] - 10*lambda_m - 7.5*lambda_m_bar + 17.5*lambda_m_barbar) # (m,s)
                    xi_temp -= eta_ellJ[2,1]*(self.svals*lambda_m_n[2] - 8*lambda_m_n[1] + 43*lambda_m_n[0]/self.svals - 175*(lambda_m - lambda_m_bar)/s2)
                    xi_temp += eta_ellJ[2,2]*(self.svals*lambda_m_n[4] - 6*lambda_m_n[3] + 21*lambda_m_n[2]/self.svals - 45*lambda_m_n[1]/s2 + 45*lambda_m_n[0]/s3)
                    qbar_contrib = 1.5*eta_ellJ[2,0]*qbar2*self.sminBys3
                    qbar_contrib += self.sminBys5*(35*eta_ellJ[2,1]*qbar2/self.svals.min()**2 - 3.5*eta_ellJ[2,0]*qbar4)
                    qbar_contrib *= 5.0
                    xi_prop_der[2] = np.sum(xi_temp.T*w_m,axis=1) - qbar_contrib # (s,)

                    # MC
                    xi_temp = psi_ellJ[2,0]*(self.svals**2*lambda_m_n[1] - 9*self.svals*lambda_m_n[0]
                                             + 65*lambda_m + 22.5*lambda_m_bar - 87.5*lambda_m_barbar) # (m,s)
                    xi_temp -= psi_ellJ[2,1]*(self.svals**2*lambda_m_n[3] - 7*self.svals*lambda_m_n[2]
                                              + 43*lambda_m_n[1] - 218*lambda_m_n[0]/self.svals
                                              + 875*(lambda_m - lambda_m_bar)/s2)
                    xi_temp += psi_ellJ[2,2]*(self.svals**2*lambda_m_n[5] - 5*self.svals*lambda_m_n[4]
                                              + 21*lambda_m_n[3] - 66*lambda_m_n[2]/self.svals + 135*lambda_m_n[1]/s2 - 135*lambda_m_n[0]/s3)
                    qbar_contrib = -4.5*psi_ellJ[2,0]*qbar2*self.sminBys3
                    qbar_contrib -= 5*self.sminBys5*(35*psi_ellJ[2,1]*qbar2/self.svals.min()**2 - 3.5*psi_ellJ[2,0]*qbar4)
                    qbar_contrib *= 5.0
                    xi_MC_der[2] = np.sum(xi_temp.T*w_m,axis=1) - qbar_contrib # (s,)
                    
        xi_MC *= AMC
        if self.model_AP:
            xi_MC_der *= AMC

        xiNL = xi_prop + xi_MC
        if self.model_AP:
            return xiNL,xi_prop_der + xi_MC_der
        else:
            return xiNL

    #########################################

    ############################################################
    def calc_zerocrossing(self,rvals,func,first=True,down=True):
        """ Simple utility to calculate (first/last) (up/down) zero crossing of some tabulated function func assumed to be evaluated on given rvals.
            Output is scalar in units of rvals.
        """
        r_use = rvals.copy()
        func_use = func.copy()
        if not first:
            # {last up/dn cross by f} = {first up/dn x by -f[::-1]}
            r_use = 1.0*r_use[::-1]
            func_use = -1.0*func_use[::-1]

        ind_zc = np.where(func_use < 0)[0] if down else np.where(func_use > 0)[0] # down/up cross of zero
            
        if len(ind_zc):
            i_zc = ind_zc[0] # first cross of zero
            if i_zc == 0:
                zc = 1.0*r_use[0]
            else:
                zc = (func_use[i_zc-1]*r_use[i_zc]-func_use[i_zc]*r_use[i_zc-1])/(func_use[i_zc-1]-func_use[i_zc])
        else:
            zc = 1.0*r_use[-1]
                
        return zc
    ############################################################
    
    #########################################
    def calc_dxilindr(self,w_m):
        """ Helper function to calculate dxi_lin/dr for given basis coeffs.
            -- w_m: array of shape (self.n_basis,)
            Returns array of shape (self.rvals.size,) 
        """
        return np.sum(self.dbdr.T*w_m,axis=1)
    #########################################
    
    #########################################
    def calc_linearscales(self,w_m):
        """ Calculate peak,dip,LP of xi_lin.
            Returns scalars peak,dip,LP.
        """
        ZC = self.calc_zerocrossing(self.rvals,np.sum(self.bfunc.T*w_m,axis=1))
        i_vals = np.where(self.rvals <= ZC)[0] # search for peak,dip only below ZC.
        rvals = self.rvals[i_vals]
        if rvals.size:
            dxilindr = self.calc_dxilindr(w_m)[i_vals]
            dip =  self.calc_zerocrossing(rvals,dxilindr,first=True,down=False) #  dip is first   up-crossing of dxi/dr=0
            peak = self.calc_zerocrossing(rvals,dxilindr,first=False,down=True) # peak is  last down-crossing of dxi/dr=0
        else:
            dip = 0.0
            peak = 1e6
            
        LP = (peak+dip)/2.0
        
        return peak,dip,LP,ZC
    #########################################

    #########################################
    def calc_Sig2obs(self,params_dict):
        """ Calculate Sig2obs values. Only called if self.include_Sig2obs is True.
            Returns array of shape (self.L_Max,)
        """
        fv = params_dict['fv']
        beta = params_dict['beta']
        sigv = params_dict['sigv']
        b = params_dict['b']
        betasq = beta**2
        out = [1 + 2*beta/3.0 + betasq/5.0]
        if self.L_Max > 1:
            out.append(4*beta*(1/3.0+beta/7.0))
            if self.L_Max == 3:
                out.append(8*betasq/35.0)
        out = np.array(out)*fv*(b*sigv)**2
        return out
    #########################################

    
    #########################################
    # see https://cobaya.readthedocs.io/en/latest/theories_and_dependencies.html
    def calculate(self,state, want_derived=True, **params_values_dict):
        if self.strong_prior:
            raise NotImplementedError('Sorry, strong prior not yet implemented!')
            # # in this case, sampled params contain self.agem.keys_vary and sdbmc
            # sampled_keys = list(params_values_dict.keys())
            # params_dict = {}
            
            # # extract and set values of (subset of) sdbmc and nuisance params
            # for key in ['b','B1st','Bvst','sigma','AMC','qbar2','qbar4']:
            #     if key in sampled_keys:
            #         params_dict[key] = params_values_dict[key]
                    
            # # extract cosmological params [assumes params_values_dict.keys() contains all of self.agem.keys_vary]
            # cosmological = self.agem.cv([params_values_dict[key] for key in self.agem.keys_vary])
            
            # # emulate agnostic params
            # agnostic = self.FwdEmulator.predict(cosmological)

            # # set values of agnostic params in dict
            # for a in range(len(self.agem.n_agnostic)):
            #     params_dict[self.agem.keys_agnostic[a]] = agnostic[a,0]
            # params_dict['beta'] = params_dict['f']/params_dict['b']
            # # note: params_dict['f'] will be ignored
        else:
            # in this case, sampled params contain beta,sigv,{w_m}[,DaAP][,fv] and sdbmc
            params_dict = copy.deepcopy(params_values_dict)

        keys = params_dict.keys()
        f = params_dict['beta']*params_dict['b'] 
        w_m = np.array([params_dict[key] for key in self.w_names])
        peak,dip,LP,ZC = self.calc_linearscales(w_m)
        
        state['derived'] = {'f':f,'peak':peak,'LP':LP,'ZC':ZC}

        if not self.sdbmc:
            params_dict['sigma'] = np.sqrt(2)*params_dict['sigv']
        
        # enforce sigma >= sqrt(2)*sigv since R* >= 0
        # enforce peak,dip in range [rvals.min(),rvals.max()] else unphysical
        out_of_bounds = (params_dict['sigma'] < np.sqrt(2)*params_dict['sigv'])
        out_of_bounds = out_of_bounds | (peak > self.rvals_max_check) | (peak < self.rvals_min_check)
        out_of_bounds = out_of_bounds | (dip > self.rvals_max_check) | (dip < self.rvals_min_check)
        # if self.L_Max < 3:
        out_of_bounds = out_of_bounds | (ZC > self.rvals_max_check) | (ZC < self.rvals_min_check)
        if out_of_bounds:
            state['model'] = np.inf*np.ones(self.dim)
            return

        if self.model_AP:
            xiNL,xiNL_der = self.calc_xiNL(params_dict) # (L_Max,s)
            # Daiso = params_dict['Daiso']
            DaAP = params_dict['DaAP']
            temp = (2/3.)*np.dot(self.Cmat[:self.L_Max,:self.L_Max],xiNL_der) # (L_Max,s)
            temp = (temp.T*np.array([2*(2*L) + 1 for L in range(self.L_Max)])).T
            temp += np.dot(self.Amat[:self.L_Max,:self.L_Max],xiNL)
            xiNL = xiNL + DaAP*temp # - Daiso*xiNL_der # uncomment to model AP as func of s rather than y = s/DV
        else:
            xiNL = self.calc_xiNL(params_dict) # (L_Max,s)

        if self.modify_data & (self.L_Max > 1):
            xiNL[1] -= self.sminBys3*xiNL[1,0]
            if self.L_Max == 3:
                xiNL[2] -= self.sminBys5*xiNL[2,0]

        xiNL = xiNL.flatten() # order xiNL(ell=0),xiNL(ell=2),xiNL(ell=4)

        # NOTE: no indexing offset below since deletion is on array *before* concatenating with Sig2obs 
        if self.modify_data & (self.L_Max > 1):
            if self.L_Max == 3:
                xiNL = np.delete(xiNL,[self.N_Data,2*self.N_Data])
            else:
                xiNL = np.delete(xiNL,[self.N_Data])

        if self.include_Sig2obs:
            Sig2obs = self.calc_Sig2obs(params_dict) # (L_Max,)
            if self.model_AP:
                temp = (4/3.)*np.dot(self.Cmat[:self.L_Max,:self.L_Max],Sig2obs)
                temp = (temp.T*np.array([2*(2*L) + 1 for L in range(self.L_Max)])).T
                temp = np.dot(self.Amat[:self.L_Max,:self.L_Max],Sig2obs) - temp
                # Sig2obs *= (1+2*Daiso) 
                Sig2obs += DaAP*temp
            xiNL = np.concatenate((Sig2obs,xiNL))
        
        state['model'] = xiNL.copy()
            
        return
    #########################################
    
    #########################################
    def calc_xiprotohalo(self,params_values_dict,Rpk2=None):
        # only used in post-processing
        params_dict = copy.deepcopy(params_values_dict)
        Rpk2_use = self.Rpiv2 if Rpk2 is None else 1.0*Rpk2

        keys = params_dict.keys()

        # proto-halo condition
        params_dict['Bvst'] = 0.0
        sigv = 1.0*params_dict['sigv']
        sigma = 1.0*params_dict['sigma']
        sig_proto = np.sqrt(sigma**2 - 2*sigv**2) # this is sqrt(2)*Rst
        params_dict['sigv'] = 0.0
        params_dict['sigma'] = sig_proto
        params_dict['beta'] = 0.0
        B1 = params_dict['B1st'] + 0.5*sigma**2/self.Rpiv2  # this is sampled B1
        B1eff = B1 + (Rpk2_use/self.Rpiv2)/params_dict['b'] # this is B1^eff for protohalo+matter reconstruction
        params_dict['B1st'] = B1eff - 0.5*sig_proto**2/self.Rpiv2 # this is effective value of B1st
        
        xi_proto = self.calc_xiprop(params_dict)
        xi_proto = xi_proto.flatten() # order xi_proto(ell=0),xi_proto(ell=2),xi_proto(ell=4)
        xi_proto = 1.0*xi_proto[:self.svals.size] # only monopole relevant
            
        return xi_proto
    #########################################


    #########################################
    # useful for external plotting
    #########################################
    def calc_xiprop(self,params_dict):
        """ Calculate propagator term of config space multipoles at self.svals.
            Returns array of shape (self.L_Max,self.svals.size)
        """
        beta = params_dict['beta']
        sigv = params_dict['sigv']
        w_m = np.array([params_dict[key] for key in self.w_names])
        
        b = params_dict['b']
        Bvst = params_dict['Bvst']
        sigma = params_dict['sigma'] # can we exploit very low sampling speed of sigma?
        B1 = params_dict['B1st'] + 0.5*sigma**2/self.Rpiv2

        # AMC = params_dict['AMC'] # not needed in xi_prop
        
        f = beta*b
        kstsq = f*(f+2)*sigv**2 
        
        xi_prop = np.zeros((self.L_Max,self.svals.size),dtype=float)
        
        lambda_m = self.calc_lambda(sigma) # (m,s)
        if self.L_Max == 1:
            Lambda_m_n = self.calc_der_Lambda(sigma) # (n,m,s)
        else:
            lambda_m_n,Lambda_m_n = self.calc_der_lambda(sigma) # (n,m,s)

        eta_ellJ = self.calc_eta_ellJ(beta,B1,Bvst,kstsq)

        xi_temp = eta_ellJ[0,0]*lambda_m - (eta_ellJ[0,1]*Lambda_m_n[1] - eta_ellJ[0,2]*Lambda_m_n[3])/self.svals # (m,s)
        xi_prop[0] = np.sum(xi_temp.T*w_m,axis=1) # (s,)

        if self.L_Max > 1:
            qbar2 = params_dict['qbar2']
            lambda_m_bar,lambda_m_barbar = self.calc_lambda_bars(sigma)
            s2 = self.svals**2
            s3 = s2*self.svals
            
            xi_temp = eta_ellJ[1,0]*(lambda_m - lambda_m_bar) # (m,s)
            xi_temp -= eta_ellJ[1,1]*(lambda_m_n[1] - lambda_m_n[0]/self.svals)
            xi_temp += eta_ellJ[1,2]*(lambda_m_n[3] + lambda_m_n[2]/self.svals - 6*lambda_m_n[1]/s2 + 6*lambda_m_n[0]/s3) 
            xi_prop[1] = np.sum(xi_temp.T*w_m,axis=1) - eta_ellJ[1,0]*qbar2*self.sminBys3 # (s,)
            
            if self.L_Max == 3:
                qbar4 = params_dict['qbar4']
                
                xi_temp = eta_ellJ[2,0]*(lambda_m + 2.5*lambda_m_bar - 3.5*lambda_m_barbar) # (m,s)
                xi_temp -= eta_ellJ[2,1]*(lambda_m_n[1] - 8*lambda_m_n[0]/self.svals + 35*(lambda_m - lambda_m_bar)/s2)
                xi_temp += eta_ellJ[2,2]*(lambda_m_n[3] - 6*lambda_m_n[2]/self.svals + 15*lambda_m_n[1]/s2 - 15*lambda_m_n[0]/s3)
                qbar_contrib = 2.5*eta_ellJ[2,0]*qbar2*self.sminBys3
                qbar_contrib += self.sminBys5*(35*eta_ellJ[2,1]*qbar2/self.svals.min()**2 - 3.5*eta_ellJ[2,0]*qbar4)
                xi_prop[2] = np.sum(xi_temp.T*w_m,axis=1) + qbar_contrib # (s,)

        return xi_prop
    #########################################


    #########################################
    # useful for external plotting
    #########################################
    def calc_xiMC(self,params_dict):
        """ Calculate mode-coupling term of config space multipoles at self.svals.
            Returns array of shape (self.L_Max,self.svals.size)
        """
        beta = params_dict['beta']
        # sigv = params_dict['sigv'] # not needed in xi_MC
        w_m = np.array([params_dict[key] for key in self.w_names])
        
        # b = params_dict['b'] # not needed in xi_MC
        Bvst = params_dict['Bvst']
        sigma = params_dict['sigma'] # can we exploit very low sampling speed of sigma?
        B1 = params_dict['B1st'] + 0.5*sigma**2/self.Rpiv2

        AMC = params_dict['AMC'] 
        
        # f = beta*b
        # kstsq = f*(f+2)*sigv**2 # not needed in xi_MC
        
        xi_MC = np.zeros((self.L_Max,self.svals.size),dtype=float)
        
        lambda_m = self.calc_lambda(sigma) # (m,s)
        if self.L_Max == 1:
            Lambda_m_n = self.calc_der_Lambda(sigma) # (n,m,s)
        else:
            lambda_m_n,Lambda_m_n = self.calc_der_lambda(sigma) # (n,m,s)

        psi_ellJ = self.calc_eta_ellJ(beta,B1,Bvst,kstsq=None)

        xi_temp = psi_ellJ[0,0]*(Lambda_m_n[0] - lambda_m) # (m,s)
        xi_temp -= psi_ellJ[0,1]*(Lambda_m_n[2] - Lambda_m_n[1]/self.svals)
        xi_temp += psi_ellJ[0,2]*(Lambda_m_n[4] - Lambda_m_n[3]/self.svals) 
        xi_MC[0] = np.sum(xi_temp.T*w_m,axis=1) # (s,)

        if self.L_Max > 1:
            qbar2 = params_dict['qbar2']
            lambda_m_bar,lambda_m_barbar = self.calc_lambda_bars(sigma)
            s2 = self.svals**2
            s3 = s2*self.svals
            
            xi_temp = psi_ellJ[1,0]*(3*(lambda_m_bar - lambda_m) + self.svals*lambda_m_n[0]) # (m,s)
            xi_temp -= psi_ellJ[1,1]*(self.svals*lambda_m_n[2] - lambda_m_n[1] + lambda_m_n[0]/self.svals)
            xi_temp += psi_ellJ[1,2]*(self.svals*lambda_m_n[4] + lambda_m_n[3] - 7*lambda_m_n[2]/self.svals + 18*lambda_m_n[1]/s2 - 18*lambda_m_n[0]/s3) 
            xi_MC[1] = np.sum(xi_temp.T*w_m,axis=1) + 3*psi_ellJ[1,0]*qbar2*self.sminBys3 # (s,)
            
            if self.L_Max == 3:
                qbar4 = params_dict['qbar4']
                
                xi_temp = psi_ellJ[2,0]*(self.svals*lambda_m_n[0] - 10*lambda_m - 7.5*lambda_m_bar + 17.5*lambda_m_barbar) # (m,s)
                xi_temp -= psi_ellJ[2,1]*(self.svals*lambda_m_n[2] - 8*lambda_m_n[1] + 43*lambda_m_n[0]/self.svals - 175*(lambda_m - lambda_m_bar)/s2)
                xi_temp += psi_ellJ[2,2]*(self.svals*lambda_m_n[4] - 6*lambda_m_n[3] + 21*lambda_m_n[2]/self.svals - 45*lambda_m_n[1]/s2 + 45*lambda_m_n[0]/s3)
                qbar_contrib = 1.5*psi_ellJ[2,0]*qbar2*self.sminBys3
                qbar_contrib += self.sminBys5*(35*psi_ellJ[2,1]*qbar2/self.svals.min()**2 - 3.5*psi_ellJ[2,0]*qbar4)
                qbar_contrib *= 5.0
                xi_MC[2] = np.sum(xi_temp.T*w_m,axis=1) - qbar_contrib # (s,)

        xi_MC *= AMC
        
        return xi_MC
    #########################################
    
    #########################################
    def get_model(self):
        return self.current_state['model']
    #########################################

    #########################################
    def get_allow_agnostic(self):
        return True
    #########################################

    #########################################
    def get_can_provide_params(self):
        return ['f','peak','LP','ZC']
    #########################################
