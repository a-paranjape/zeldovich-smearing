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
            -- out_stem: str (default './'), path/of/folder/ where all outputs [samples and trained models] will be written.
                         This will be internally modified to out_stem + 'scale{0:.1f}/z{1:.2f}/'.format(self.scale_planck18,self.z_eval)
            -- cosmo: str, base cosmology to sample from, one of ['lcdm','wcdm'(default),'w0wacdm','nucdm']
                      Note: 'nucdm' currently will only vary the mass of a single neutrino species.
            -- z_eval: float >= 0.0 (default 0.0), evaluation redshift.
            -- flat: bool, whether or not to consider only spatially flat cosmologies. 
                     If False (default), Omega_k will be sampled, else will set Omega_k=0.
            -- scale_planck18: float > 0 (default 6.0), scale factor to apply to nominal Planck18 errors to define parameter variation ranges.
            -- rmin,rmax: floats (default 30.0,150.0), min,max values in Mpc/h_fid for basis evaluation
            -- n_r: int (default 60), number of scales for basis evaluation
            -- mnu_max: float (default 0.3), maximum neutrino mass in eV [only relevant if cosmo=='nucdm']
            -- kmin,kmax: floats (default 0.02,0.05), min,max values in h_fid/Mpc for fv evaluation
            -- high_acc: bool (default True), control accuracy of k-space integrals
            -- verbose,logfile: usual I/O control variables
            Note: plots can be stored in self.plot_dir which is created at self.out_stem + self.cosmo (+'_flat') + '/plots/'.
        """
        Utilities.__init__(self)
        MLUtilities.__init__(self)
        start_time = time()
        
        self.neutrino_cosmologies = ['nucdm']
        
        self.out_stem = setup.get('out_stem','./')
        self.cosmo = setup.get('cosmo','wcdm')
        self.z_eval = setup.get('z_eval',0.0)
        self.flat = setup.get('flat',False)
        self.scale_planck18 = setup.get('scale_planck18',6.0)
        self.out_stem += 'scale{0:.1f}/z{1:.2f}/'.format(self.scale_planck18,self.z_eval)
        self.mnu_max = setup.get('mnu_max',0.3) if self.cosmo in self.neutrino_cosmologies else None

        # stuff needed for plotting
        self.flat_str = '_flat' if self.flat else ''
        self.plot_dir = self.out_stem + self.cosmo + self.flat_str + '/plots/'
        Path(self.plot_dir).mkdir(parents=True,exist_ok=True)
        
        self.verbose = setup.get('verbose',True)
        self.logfile = setup.get('logfile',None)
        
        if self.verbose:
            self.print_this('Agnostic emulator for BAO inference...',self.logfile)
            self.print_this('... will work in folder: '+self.out_stem,self.logfile)

        if self.cosmo in ['w0wacdm']:
            raise NotImplementedError(self.cosmo+' not yet implemented!')
            
        # BiSequential basis setup
        self.basis_stem = Basis_Stem
        self.load_basis() # sets self.binet,self.basis,self.n_basis

        # evaluate basis functions
        self.rmin = setup.get('rmin',30.0)
        self.rmax = setup.get('rmax',150.0)
        self.n_r = setup.get('n_r',60)
        self.rvals = np.linspace(self.rmin,self.rmax,self.n_r)
        self.basis_func = self.evaluate_basis(self.rvals)

        self.n_agnostic = self.n_basis + 4 # (9) basis coeffs + f,sigv,DaAP,fv ( = 13)
        self.keys_agnostic = ['w{0:d}'.format(n) for n in range(self.n_basis)]
        self.keys_agnostic += ['f','sigv','DaAP','fv']
        
        # setup fiducial cosmology and param variation lists
        self.setup_fiducial_cosmology()

        # fiducial distances for DaAP calculation (nominally in Mpc/h_fid)
        self.d_Hub_fid = self.co_fid.EHub_inv(self.z_eval) # physical Hubble distance
        self.d_Ang_com_fid = self.co_fid.rCom(self.z_eval) # comoving angular diameter distance
        
        # k-space setup
        # .. min/max values for fv evaluation from PS26a,b.
        self.kmin = setup.get('kmin',0.02)
        self.kmax = setup.get('kmax',0.05)

        # .. arrays for sigv,fv evaluation
        self.high_acc = setup.get('high_acc',True)
        self.nk_int = 15000 if self.high_acc else 1500 # 50000 
        self.ktab_int = np.logspace(np.log10(self.co_fid.ktab_lin.min()),np.log10(self.co_fid.ktab_lin.max()),self.nk_int)
        self.dlnk_int = np.log(self.ktab_int[1]/self.ktab_int[0])
        self.k3by2pi2 = self.ktab_int**3/(2*np.pi**2)
        self.cond_k = (self.ktab_int <= self.kmax) & (self.ktab_int >= self.kmin)

        # latex setup
        self.setup_latex()
        
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
            -- pfid,err_fid,co_fid,keys_vary,n_params,param_mins,param_maxs
            Notes:
              (i) self.pfid contains fiducial values of *all* params, including those held fixed.
             (ii) self.keys_vary is a subset of self.pfid.keys(), since not all params are varied.
            (iii) self.param_mins,self.param_maxs are ordered by self.keys_vary.
             (iv) self.err_fid contains approximate Planck18 1sigma errors for all relevant params described in self.pfid.
                  These are scaled by self.scale_planck18 to define self.param_mins,self.param_maxs.
                  (Neutrinos are ignored since the mnu range is set by self.mnu_max; wDEa error is arbitrarily set to 0.5).
        """
        if self.verbose:
            self.print_this('... setting up fiducial cosmology and sampling ranges',self.logfile)
        # fiducial values from table 2 of Planck18 cosmology paper arXiv:1807.06209
        self.pfid = {'Om':0.3153,'h':0.6737,'As':np.exp(3.045)*1e-10,'ns':0.9649,'Ob':0.02237/0.6737**2,'Ok':0.0,
                     'wDE0':-1.0,'wDEa':0.0,
                     'N_ur':3.044,'N_ncdm':0,'m_ncdm':0.0}
        self.err_fid = {'Om':0.0073,'h':0.0054,'As':0.014*np.exp(3.045)*1e-10,'ns':0.0042,
                        'Ob':0.02237/0.6737**2*np.sqrt((0.00015/0.02237)**2 + 4*(0.0054/0.6737)**2),'Ok':0.0125,
                        'wDE0':0.1,'wDEa':0.5,
                        'N_ur':0,'N_ncdm':0,'m_ncdm':0.0}

        self.co_fid = Cosmology(Om=self.pfid['Om'],hubble=self.pfid['h'],As=self.pfid['As'],ns=self.pfid['ns'],Ob=self.pfid['Ob'],
                                Ok=self.pfid['Ok'],wDE0=self.pfid['wDE0'],wDEa=self.pfid['wDEa'],
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
            keys_vary.remove('wDE0')
            keys_vary.remove('wDEa')
        if self.cosmo not in self.neutrino_cosmologies:
            keys_vary.remove('m_ncdm')
        if self.cosmo == 'wcdm':
            keys_vary.remove('wDEa')
            
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
            self.param_mins.append(self.pfid[key] - self.scale_planck18*self.err_fid[key])
            self.param_maxs.append(self.pfid[key] + self.scale_planck18*self.err_fid[key])

        if self.cosmo in self.neutrino_cosmologies:
            ind_mncdm = self.keys_vary.index('m_ncdm')
            self.param_mins[ind_mncdm] = 1e-4
            print('!WARNING!: need to figure out how to set minimum neutrino mass! Currently hard-coded to {0:.2f} meV'.format(1e3*self.param_mins[ind_mncdm]))
            self.param_maxs[ind_mncdm] = self.mnu_max
            
        return
    #############################################        

    #############################################
    def setup_latex(self):
        """ Simple utility to setup lists and dictionaries for generating Latex labels for plotting. """

        # global label defining cosmological model
        self.cosmo_latex_list = {'lcdm':"$\\Lambda$CDM",'lcdm_flat':"flat $\\Lambda$CDM",
                                 'wcdm':"$w$CDM",'wcdm_flat':"flat $w$CDM",
                                 'w0wacdm':"$(w_{{0}},w_{{a}})$CDM",'wcdm_flat':"flat $(w_{{0}},w_{{a}})$CDM",
                                 'nucdm':"$\\nu\\Lambda$CDM",'ncdm_flat':"flat $\\nu\\Lambda$CDM"}
        self.cosmo_latex = self.cosmo_latex_list[self.cosmo+self.flat_str]

        # individual parameter labels (useful for getdist)
        self.latex_keys_all = {'Om':"\\Omega_{\\rm m}",
                               'h':'h',
                               'As':"A_{\\rm s}",
                               'ns':"n_{\\rm s}",
                               'Ob':"\\Omega_{\\rm b}",
                               'Ok':"\\Omega_{\\rm k}",
                               'wDE0':"w^{\\rm (DE)}_0",
                               'wDEa':"w^{\\rm (DE)}_a",
                               'm_ncdm':"m_{\\nu}",
                               'f':'f',
                               'sigv':"\\sigma_{\\rm v}",
                               'DaAP':"\\Delta\\alpha_{\\rm AP}",
                               'fv':"f_{\\rm v}"}
        for b in range(self.n_basis):
            self.latex_keys_all[f"w{b}"] = f"w_{{{b}}}"

        self.latex_agnostic = []
        for key in self.keys_agnostic:
            self.latex_agnostic.append(self.latex_keys_all[key])

        self.latex_cosmological = []
        for key in self.keys_vary:
            self.latex_cosmological.append(self.latex_keys_all[key])
            
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
            -- agnostic (self.n_agnostic,n_samp), cosmological (self.n_params,n_samp)[, xilin (n_samp,self.n_r), only if save_xi=True]
        """
        start_time = time()
        
        n_samp = sample_setup.get('n_samp',1)
        seed = sample_setup.get('seed',None)
        sample_stem = sample_setup.get('sample_stem','train')
        force = sample_setup.get('force',False)
        include_fiducial = sample_setup.get('include_fiducial',False)
        save_xi = sample_setup.get('save_xi',False)
        
        out_dir = self.out_stem + self.cosmo + self.flat_str + '/samples/' + sample_stem # folder to write/read samples to/from
        file_agnostic = out_dir + '/agnostic.txt'
        file_cosmological = out_dir + '/cosmological.txt'

        if save_xi:
            xi_dir = self.out_stem + self.cosmo + self.flat_str + '/xilin/' + sample_stem # folder to write/read xilin to/from
            file_xi = xi_dir + '/xilin.txt'
        
        if self.verbose:
            self.print_this('Generating/reading sample from: '+out_dir,self.logfile)
            
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
            agnostic = np.zeros((n_samp,self.n_agnostic))
            
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
                                   Ok=pdict['Ok'],wDE0=pdict['wDE0'],wDEa=pdict['wDEa'],
                                   N_ur=pdict['N_ur'],N_ncdm=pdict['N_ncdm'],m_ncdm=pdict['m_ncdm'],z_eval=self.z_eval,
                                   verbose=False)
                    
                    others,growth = self.calc_others(co)
                    
                    # for neutrino cosmologies, xilin is directly provided at evaluation redshift.
                    # for others, multiply by growth**2
                    xi_calc = co.calc_xi_lin(self.rvals*pdict['h']/self.pfid['h']) # use Mpc/h in varied cosmology
                    if self.cosmo not in self.neutrino_cosmologies:
                        xi_calc *= growth**2
                    xilin[n] = xi_calc
                    
                    agnostic[n,:self.n_basis] = self.calc_basiscoeffs(xilin[n])
                    agnostic[n,self.n_basis:] = others
                except Exception:
                    agnostic[n] += np.nan
                if self.verbose:
                    self.status_bar(n,n_samp)

            agnostic = agnostic.T # (self.n_agnostic,n_samp)
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

    #############################################        
    def calc_basiscoeffs(self,xilin):
        """ Least squares fit to input function (linear 2pcf) using self.basis_func.
            Expect xilin to be array of shape (self.n_r,).
            Returns array of shape (self.n_basis,)
        """
        Cinv = np.eye(self.n_r)
        Fisher = np.dot(self.basis_func,np.dot(Cinv,self.basis_func.T)) # since F = M^T C^-1 M and M = basis_func
        Finv,detF = self.svd_inv(Fisher,hermitian=True)
        coeffs = np.dot(Finv,np.dot(self.basis_func,np.dot(Cinv,xilin))) # ahat = F^-1 (M^T C^-1 y)
        return coeffs
    #############################################

    #############################################
    def calc_others(self,co):
        """ Calculate cosmological parameters other than basis coeffs, namely f,sigv,DaAP,fv at evaluation redshift.
            -- co: instance of Cosmology
            Returns 
            -- array of shape (4,) containing [f,sigv,DaAP,fv]
            -- scalar value of growth (normalized to unity at z=0)
        """
        growth = co.Growth(self.z_eval)/co.Growth(0.0)
        f = co.fGrowth(z=self.z_eval)

        # growth will be included later in sigv
        Dlin_int = np.interp(self.ktab_int,co.ktab_lin,co.Dlin)
        sigv2 = np.trapezoid(Dlin_int/self.ktab_int**2,dx=self.dlnk_int)/3.0
        sigv2_frac = np.trapezoid(Dlin_int[self.cond_k]/self.ktab_int[self.cond_k]**2,dx=self.dlnk_int)/3.0

        sigv = growth*np.sqrt(sigv2)
        fv = sigv2_frac/(sigv2 + co.TINY)

        # distances (nominally in Mpc/h)
        d_Hub = co.EHub_inv(self.z_eval) # physical Hubble distance
        d_Ang_com = co.rCom(self.z_eval) # comoving angular diameter distance

        alpha_par = self.d_Hub_fid/d_Hub
        alpha_perp = self.d_Ang_com_fid/d_Ang_com
        # nominally, ratios of (Mpc/h_fid) with (Mpc/h), but h/h_fid not accounted for

        alpha_AP = alpha_perp/alpha_par # so h/h_fid dependence cancels
        DaAP = alpha_AP - 1.0
        
        Dlin_int = None

        out = np.array([f,sigv,DaAP,fv])
        
        return out,growth
    #############################################

    #############################################
    def emulate(self,agnostic,cosmological,invert=True,setup_hopt={},optimize=True):
        """ Wrapper around HyperOpt.optimize.
            -- agnostic,cosmological: mutually consistent outputs of self.gen_sample. 
                                      Can be dummy arrays of shape (1,1) if optimize=False (i.e. for loading existing emulator).
            -- invert: bool (default True), whether to construct forward or inverse emulator 
                       True : forward emulator, with (input=cosmological, output=agnostic)
                       False: inverse emulator, with (input=agnostic,output=cosmological)
            -- setup_hopt: setup dictionary to instantiate HyperOpt, with keys being a subset of following
                           [defaults below are same as in HyperOpt source code]
                ------------
                :: mandatory
                ------------
                -- theta_dim: int; dimensionality of parameter space in BiSequential (not needed for other network families)
                ------------
                :: optional
                ------------
                -- family: str [default 'seq']; one of 'seq' (Sequential), 'biseq' (BiSequential), 'gan' (GAN)
                -- model_name: str [defaults to family]; unique name for model (e.g., 'wide','telescopic', or anything else)
                               model will be stored in the folder 
                               self.out_stem + self.cosmo [+'_flat'] + '/models/' + model_name + inv_str
                               where inv_str = '_inverse' if invert else '_forward'
                ------
                :: :: training sample
                ------
                -- train_frac: float (default 0.8); fraction of input samples to use for training+validation, 
                               remaining used for hyperparam/architecture comparison.
                -- val_frac: float (default 0.2); fraction of train_frac to use for early-stopping validation. 
                             Set to zero to switch off validation check. 
                ------
                :: :: training setup
                ------
                -- standardize_X: bool (default True); whether or not to standardize features.
                -- standardize_Y: bool (default True); whether or not to standardize labels.
                -- max_epoch: int (default 1000000); maximum number of training epochs
                -- check_after: int (default 300); epoch after which to activate validation (early stopping) checks. 
                                To swith off early stopping, set >= max_epoch.
                -- decay_norm: int (default 2); value of norm for weight decay, either 1 or 2.
                -- test_type: str (default 'perc'); one of 'perc' (residual percentiles) or 'mse' (mean squared error),
                              relevant for regression (square/hinge loss).
                -- seed: int or None (default); seed for random number generation. 

                -- n_iter: int (default 3); number of iterations for each choice of hyperparams + architecture
                -- max_config: int (default 10); total number of distinct configurations to search over.
                   ** Note: ** Total number of networks trained will be (n_iter * max_config)

                -- ensemble_size: int (default 5); number of top networks to use in ensemble. Should not be larger than max_config. 
                                  (Only used if ensemble is True.)
                -- parallel: bool (default False); whether or not to parallelize analysis of each configuration. (CURRENTLY REDUNDANT.)
                -- nproc: int (default 4); number of concurrent processes to spawn. 
                -- fixed_width: bool or None (default True)
                                True : each layer l has the same width W_l = W sampled from the range
                                False: each layer l has a width W_l sampled independently from the range
                                None: layer widths telescope from data dim to sampled W
                -- fixed_htype: bool (default True)
                                True : each layer l has the same activation A_l = A sampled from the htypes list
                                False: each layer l has an activation A_l sampled independently from the htypes list
                ------
                :: :: sampled parameters
                ------
                -- layers: range for number of layers
                           dict with structure 
                           {'min': int (default 1), 'max': int (default 3)}
                -- widths: range for layer width
                           dict with structure 
                           {'min': int (default 2), 'max': int (default 2)}
                -- lglrates: range for log10(learning rate)
                             dict with structure 
                             {'min': float (default -2.0), 'max': float (default -1.0)}
                -- wt_decays: range for weight decay
                              dict with structure 
                              {'min': float (default 0.0), 'max': float (default 0.0)} [default is no weight decay]
                -- htypes: None (default) or list; hidden activation types (will be randomly sampled). 
                           None will default to ['relu','tanh'].
                           If not None, expect subset of ['tanh','relu','lrelu','splus','sin','requ']. 
                -- lrelu_slopes: None (default) or range for slopes of LReLU
                                 If not None, expect dict with structure 
                                 {'min': float (e.g., -1e-2), 'max': float (e.g., 1e-2)}
                                 None will default to 1e-2.
                -- reg_funs: None (default) or list; regularization function types (will be randomly sampled). 
                             None will default to ['none'].
                             If not None, expect subset of ['bn','drop','none']. 
                -- p_drops: None (default) or range for drop probabilities (only needed if reg_funs contains 'drop')
                            If not None, expect dict with structure 
                            {'min': float (e.g., 0.4), 'max': float (e.g., 0.6)}
                            None will default to 0.5.
                ------
            -- optimize: bool (default True). 
                         If True, run HyperOpt.optimize to train network ensemble.
                         If False, run HyperOpt.load to load existing network ensemble.
            ***
            NOTE: To train a single network rather than an ensemble, do the following:
                  (i)   set min/max of hyperparam ranges to the same value,
                  (ii)  set discrete hyperparam lists to have single elements
                  (iii) set n_iter = 1
                  (iv)  set max_config = 1
            ***
            Returns loaded instance of NetworkEnsembleObject.
        """
        inv_str = '_inverse' if invert else '_forward'
        if self.verbose:
            self.print_this("Emulating "+inv_str[1:]+" mapping...",self.logfile)
            
        setup_dict = copy.deepcopy(setup_hopt)

        # inverse: X=agnostic, Y=cosmological
        # forward: X=cosmological, Y=agnostic
        setup_dict['X'] = agnostic if invert else cosmological
        setup_dict['Y'] = cosmological if invert else agnostic

        family = setup_dict.get('family','seq')
        setup_dict['family'] = family
        model_name = setup_dict.get('model_name',family)
        setup_dict['model_name'] = model_name
        
        setup_dict['file_stem'] = self.out_stem + self.cosmo + self.flat_str + '/models/' + model_name + inv_str # folder to write/read samples to/from

        setup_dict['loss_type'] = 'square'
        setup_dict['ensemble'] = True
        
        setup_dict['verbose'] = self.verbose
        setup_dict['logfile'] = self.logfile
        
        hopt = HyperOpt(setup_dict=setup_dict)

        if optimize:
            model = hopt.optimize()
        else:
            model = hopt.load()
        return model
    #############################################

    #############################################
    def load(self,invert=True,model_name=''):
        """ Simple wrapper around self.emulate to load existing emulator instance."""
        dummy_ag = np.zeros((self.n_agnostic,1))
        dummy_co = np.zeros((self.n_params,1))
        neo = self.emulate(dummy_ag,dummy_co,invert=invert,setup_hopt={'model_name':model_name},optimize=False)
        return neo
    #############################################
    
#################################################


#################################################
if __name__ == "__main__":

    # -- out_stem: str (default './'), path/of/folder/ where all outputs [samples and trained models] will be written 
    # -- cosmo: str, base cosmology to sample from, one of ['lcdm','wcdm'(default),'nucdm']
    # -- z_eval: float >= 0.0 (default 0.0), evaluation redshift.
    # -- flat: bool, whether or not to consider only spatially flat cosmologies. 
    #          If False (default), Omega_k will be sampled, else will set Omega_k=0.
    # -- perc: float in (0,1) (default 0.1), percentage variations around fiducial values for each parameter.
    # -- rmin,rmax: floats (default 30.0,150.0), min,max values in Mpc/h_fid for basis evaluation
    # -- n_r: int (default 60), number of scales for basis evaluation
    # -- mnu_max: float (default 0.3), maximum neutrino mass in eV [only relevant if cosmo=='nucdm']
    # -- kmin,kmax: floats (default 0.02,0.05), min,max values in h_fid/Mpc for fv evaluation
    # -- high_acc: bool (default True), control accuracy of k-space integrals
    # -- verbose,logfile: usual I/O control variables
    setup = {'out_stem':'temp/','z_eval':0.8,'cosmo':'lcdm','flat':True}
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
    sset = {'n_samp':599,'include_fiducial':False,'sample_stem':'train','seed':None,'save_xi':False,'force':False}
    
    out = agem.gen_sample(sample_setup=sset)
    if sset['save_xi']:
        agnostic,cosmological,xilin = out
    else:
        agnostic,cosmological = out
        xilin = None
        
    n_samp_exp = sset['n_samp']+int(sset['include_fiducial'])
    print('agnostic.shape:',agnostic.shape,'; expected: ({0:d},{1:d})'.format(agem.n_agnostic,n_samp_exp))
    print('cosmological.shape:',cosmological.shape,'; expected: ({0:d},{1:d})'.format(agem.n_params,n_samp_exp))
    
    if sset['save_xi'] & (xilin is not None):
        print('xilin.shape:',xilin.shape,'; expected: ({0:d},{1:d})'.format(n_samp_exp,agem.n_r))
        predicted = np.dot(agem.basis_func.T,agnostic[:agem.n_basis]).T
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(4,4))
        for n in range(np.min([10,n_samp_exp])):
            plt.plot(agem.rvals,agem.rvals**2*xilin[n],'k-',lw=0.5)
            plt.plot(agem.rvals,agem.rvals**2*predicted[n],'r--',lw=1)
        plt.show()


    # -- setup_hopt: setup dictionary to instantiate HyperOpt, with keys being a subset of following
    #                [defaults below are same as in HyperOpt source code]
    #     ------------
    #     :: mandatory
    #     ------------
    #     -- theta_dim: int; dimensionality of parameter space in BiSequential (not needed for other network families)
    #     ------------
    #     :: optional
    #     ------------
    #     -- family: str [default 'seq']; one of 'seq' (Sequential), 'biseq' (BiSequential), 'gan' (GAN)
    #     -- model_name: str [defaults to family]; unique name for model (e.g., 'inverse_wide','forward_telescopic', or anything else)
    #                    model will be stored in the folder self.out_stem + self.cosmo [+'_flat'] + '/models/' + model_name
    #     ------
    #     :: :: training sample
    #     ------
    #     -- train_frac: float (default 0.8); fraction of input samples to use for training+validation, 
    #                    remaining used for hyperparam/architecture comparison.
    #     -- val_frac: float (default 0.2); fraction of train_frac to use for early-stopping validation. 
    #                  Set to zero to switch off validation check. 
    #     ------
    #     :: :: training setup
    #     ------
    #     -- standardize_X: bool (default True); whether or not to standardize features.
    #     -- standardize_Y: bool (default True); whether or not to standardize labels.
    #     -- max_epoch: int (default 1000000); maximum number of training epochs
    #     -- check_after: int (default 300); epoch after which to activate validation (early stopping) checks. 
    #                     To swith off early stopping, set >= max_epoch.
    #     -- decay_norm: int (default 2); value of norm for weight decay, either 1 or 2.
    #     -- test_type: str (default 'perc'); one of 'perc' (residual percentiles) or 'mse' (mean squared error),
    #                   relevant for regression (square/hinge loss).
    #     -- seed: int or None (default); seed for random number generation. 
    #     -- n_iter: int (default 3); number of iterations for each choice of hyperparams + architecture
    #     -- max_config: int (default 10); total number of distinct configurations to search over.
    #        ** Note: ** Total number of networks trained will be (n_iter * max_config)
    #     -- ensemble_size: int (default 5); number of top networks to use in ensemble. Should not be larger than max_config. 
    #                       (Only used if ensemble is True.)
    #     -- parallel: bool (default False); whether or not to parallelize analysis of each configuration. (CURRENTLY REDUNDANT.)
    #     -- nproc: int (default 4); number of concurrent processes to spawn. 
    #     -- fixed_width: bool or None (default True)
    #                     True : each layer l has the same width W_l = W sampled from the range
    #                     False: each layer l has a width W_l sampled independently from the range
    #                     None: layer widths telescope from data dim to sampled W
    #     -- fixed_htype: bool (default True)
    #                     True : each layer l has the same activation A_l = A sampled from the htypes list
    #                     False: each layer l has an activation A_l sampled independently from the htypes list
    #     ------
    #     :: :: sampled parameters
    #     ------
    #     -- layers: range for number of layers
    #                dict with structure 
    #                {'min': int (default 1), 'max': int (default 3)}
    #     -- widths: range for layer width
    #                dict with structure 
    #                {'min': int (default 2), 'max': int (default 2)}
    #     -- lglrates: range for log10(learning rate)
    #                  dict with structure 
    #                  {'min': float (default -2.0), 'max': float (default -1.0)}
    #     -- wt_decays: range for weight decay
    #                   dict with structure 
    #                   {'min': float (default 0.0), 'max': float (default 0.0)} [default is no weight decay]
    #     -- htypes: None (default) or list; hidden activation types (will be randomly sampled). 
    #                None will default to ['relu','tanh'].
    #                If not None, expect subset of ['tanh','relu','lrelu','splus','sin','requ']. 
    #     -- lrelu_slopes: None (default) or range for slopes of LReLU
    #                      If not None, expect dict with structure 
    #                      {'min': float (e.g., -1e-2), 'max': float (e.g., 1e-2)}
    #                      None will default to 1e-2.
    #     -- reg_funs: None (default) or list; regularization function types (will be randomly sampled). 
    #                  None will default to ['none'].
    #                  If not None, expect subset of ['bn','drop','none']. 
    #     -- p_drops: None (default) or range for drop probabilities (only needed if reg_funs contains 'drop')
    #                 If not None, expect dict with structure 
    #                 {'min': float (e.g., 0.4), 'max': float (e.g., 0.6)}
    #                 None will default to 0.5.
    #     ------
    # ***
    # NOTE: To train a single network rather than an ensemble, do the following:
    #       (i)   set min/max of hyperparam ranges to the same value,
    #       (ii)  set discrete hyperparam lists to have single elements
    #       (iii) set n_iter = 1
    #       (iv)  set max_config = 1
    # ***
    
    # -- agnostic,cosmological: mutually consistent outputs of self.gen_sample.
    # -- invert: bool (default True), whether to construct forward or inverse emulator 
    #            True : forward emulator, with (input=cosmological, output=agnostic)
    #            False: inverse emulator, with (input=agnostic,output=cosmological)
    # -- optimize: bool (default True). 
    #              If True, run HyperOpt.optimize to train network ensemble.
    #              If False, run HyperOpt.load to load existing network ensemble.
        
    setup_hopt = {'max_epoch':3000,'check_after':1000,'n_iter':3,'max_config':50,
                  'train_frac':0.9,'val_frac':0.1,
                  'nproc':16,
                  'model_name':'dummy',
                  'layers':{'min':2,'max':5},
                  'widths':{'min':5,'max':45},
                  'lglrates':{'min':-3.0,'max':-2.0},
                  'wt_decays':{'min':0.0,'max':0.05},
                  'htypes':['relu','tanh','splus'],
                  'fixed_width':False,'fixed_htype':False}

    start_time = time()
    neo_fwd = agem.emulate(agnostic,cosmological,invert=False,setup_hopt=setup_hopt,optimize=False)
    # neo_fwd.display_summary()
    agem.time_this(start_time)

    start_time = time()
    neo_inv = agem.emulate(agnostic,cosmological,invert=True,setup_hopt=setup_hopt,optimize=False)
    # neo_inv.display_summary()
    agem.time_this(start_time)

    # setup test data
    sset = {'n_samp':199,'include_fiducial':False,'sample_stem':'test','seed':None,'save_xi':False,'force':False}    
    start_time = time()
    print('Test sample...')
    agnostic_test,cosmological_test = agem.gen_sample(sample_setup=sset)
    agem.time_this(start_time)

    start_time = time()
    print('Predictions...')
    print('... forward')
    agnostic_predict = neo_fwd.predict(cosmological_test)
    print('... inverse')
    cosmological_predict = neo_inv.predict(agnostic_test)
    agem.time_this(start_time)

    print('Flattened residuals...')
    err_agnos = (agnostic_predict/(agnostic_test + 1e-15) - 1).flatten()
    err_agnos = err_agnos[np.isfinite(err_agnos)]
    errwidth_agnos = 0.5*(np.percentile(err_agnos,84)-np.percentile(err_agnos,16))

    err_cosmo = (cosmological_predict/(cosmological_test + 1e-15) - 1).flatten()
    err_cosmo = err_cosmo[np.isfinite(err_cosmo)]
    errwidth_cosmo = 0.5*(np.percentile(err_cosmo,84)-np.percentile(err_cosmo,16))

    bins = np.linspace(-1,1,900)
    bin_mid = 0.5*(bins[1:]+bins[:-1])
    dx = bins[1]-bins[0]
    
    hist_agnos,dummy = np.histogram(err_agnos,bins=bins,density=False)
    hist_agnos = hist_agnos/err_agnos.size/dx

    hist_cosmo,dummy = np.histogram(err_cosmo,bins=bins,density=False)
    hist_cosmo = hist_cosmo/err_cosmo.size/dx

    import matplotlib.pyplot as plt
    FS3 = 13
    
    FSize = 5
    plt.figure(figsize=(FSize,FSize))
    plt.yscale('log')
    plt.xlim(-0.25,0.25)
    plt.ylim(4e-1,8e2)
    plt.xlabel('residual')
    plt.ylabel('probability density')
    plt.plot(bin_mid,hist_agnos,'r-',drawstyle='steps',lw=1,label='forward')
    plt.plot(bin_mid,hist_cosmo,'k--',drawstyle='steps',lw=1.2,label='inverse')
    plt.legend(loc='upper left')
    plt.text(-0.2,7e1,'$\\sigma = {0:.4f}$'.format(errwidth_agnos),fontsize=FS3,c='r')
    plt.text(-0.2,4e1,'$\\sigma = {0:.4f}$'.format(errwidth_cosmo),fontsize=FS3)
    plt.text(0.05,3e2,'range: {0:.0f}%'.format(agem.perc*100),fontsize=FS3)
    plt.minorticks_on()
    # if Save_Fig:
    #     outfile = agem.plot_dir + 'residuals_{0:.0f}pc.png'.format(agem.perc*100)
    #     print('Saving to file:',outfile)
    #     plt.savefig(outfile,bbox_inches='tight')
    # else:
    plt.show()

    print('Parameter-wise residuals...')
    err_agnos = (agnostic_predict/(agnostic_test + 1e-15) - 1)
    errwidth_agnos = 0.5*(np.percentile(err_agnos,84,axis=1)-np.percentile(err_agnos,16,axis=1))

    err_cosmo = (cosmological_predict/(cosmological_test + 1e-15) - 1)
    errwidth_cosmo = 0.5*(np.percentile(err_cosmo,84,axis=1)-np.percentile(err_cosmo,16,axis=1))

    print('... forward')
    for p in range(len(agem.keys_agnostic)):
        print(agem.keys_agnostic[p]+': {0:.3e}'.format(errwidth_agnos[p]))

    print('... inverse')
    for p in range(len(agem.keys_vary)):
        print(agem.keys_vary[p]+': {0:.3e}'.format(errwidth_cosmo[p]))
        
    # hists_cosmo = []
    # for p in range(len(list(agem.keys_vary))):
    #     hist,bins = np.histogram(err_cosmo[p],bins=bins,density=False)
    #     hist = hist/err_cosmo.shape[1]/dx
    #     hists_cosmo.append(hist)        
    
    print('... all done!')
    
#################################################
