import numpy as np
import sys,os
import scipy.special as sysp
import scipy.signal as sysig
import scipy.linalg as linalg
from pathlib import Path

from paths import *
sys.path.append(ML_Path)
from mlalgos import BiSequential
from mllib import Utilities

import copy,pickle,json

import gc
from time import time

from cobaya.model import get_model

########################################################
class TheoryManipulator(Utilities):
    """ Manipulate Cobaya-based theory and likelihood modules for Zeldovich smearing analysis. """ 
    ####################################################
    def __init__(self,sample=None,model_AP=False,include_Sig2obs=True,L_Max=3,modify_data=True,accuracy='mid',base_dir='../examples/',verbose=True):
        """ Wrapper for user-friendly routines to manipulate Zel'dovich smearing theoretical model.
            -- sample: str; one of ['','DESI-LRG2','Euclid-ELG'], needed to initialize likelihood (default None).
            -- model_AP: bool (default False), whether or not to include effects of fiducial cosmology in observables.
            -- include_Sig2obs: bool (default True), whether or not to include power spectrum multipole integrals in observable set.
            -- L_Max: int 1..3 (default 3), number of multipoles to include.
            -- modify_data: bool (default True), whether or not to (partially) eliminate nuisance params by modifying xi^{(ell)}(s) data
            -- accuracy: str (default 'mid'), one of ['high','mid','low']. Accuracy for calculating integrals defining smoothed basis and derivatives.
            -- base_dir: str (default '../examples/'), path/in/calling/script/to/zeldovich-smearing/examples/
            -- verbose: bool (default True), verbose output
            Provides routines:
            
        """
        self.verbose = verbose
        if self.verbose:
            print("-------------------------------------------")
            print("Theory manipulation for Zel'dovich smearing")
            print("-------------------------------------------")

        if self.verbose:
            print('Setup...')
        Utilities.__init__(self)
        if sample is None:
            raise Exception("sample must be valid string in TheoryManipulator.")

        if sample not in ['','DESI-LRG2','Euclid-ELG']:
            raise Exception("sample must be one of ['','DESI-LRG2','Euclid-ELG'] in TheoryManipulator.")

        self.base_dir = base_dir
        self.like_dir = self.base_dir + '../code/'
        self.sample = sample
        self.model_AP = model_AP

        self.modify_data = modify_data
        self.accuracy = accuracy

        if (self.sample == '') & self.model_AP:
            if self.verbose:
                print('model_AP not available for toy model. Setting to False.')
            self.model_AP = False

        self.include_Sig2obs = include_Sig2obs
        self.L_Max = L_Max

        if self.verbose:
            print('... config')
        self.setup_config()

        if self.verbose:
            print('... files and folders')
        self.setup_files()

        if self.verbose:
            print('... basis functions (BiSequential instance)')
        self.setup_basis()

        if self.verbose:
            print('... fiducial parameter values')
        self.setup_fiducial()

        if self.verbose:
            print('... info dictionary')
        self.setup_info()

        if self.verbose:
            print('... model, likelihood and theory instances')
        self.setup_model()
        
        if self.verbose:
            print('... setup complete')
            print("-------------------------------------------")
        
    ####################################################

    ####################################################
    def setup_config(self):
        self.config_dict = {'DESI-LRG2': {'redshift':0.80,'Mmin':8e12,'phase':0,
                                          'sample_root':'AbacusSummit/base_c000/'},
                            'Euclid-ELG':{'redshift':1.10,'Mmin':1e12,'phase':1,
                                          'sample_root':'AbacusSummit/base_c000/'},
                            '':{'redshift':0.7,'Mmin':None,'sample_root':'SDBMC'}}

        self.redshift = self.config_dict[self.sample]['redshift'] 
        self.M_min = self.config_dict[self.sample]['Mmin']
        self.sample_root = self.config_dict[self.sample]['sample_root']
        return
    ####################################################

    ####################################################
    def setup_files(self):
        self.file_tail = 'lgMmin{0:.2f}_z{1:.3f}'.format(np.log10(self.M_min),self.redshift) if self.sample != '' else 'sdbmc'
        self.plots_dir = self.base_dir + 'plots/' + self.sample_root + self.sample + '/'
        Path(self.plots_dir).mkdir(parents=True,exist_ok=True) # folder to store plots

        # Basis_Root imported from paths
        self.basis_prior_file = Basis_Root + 'fitcoeffs/biNN2p-LGbs.json'
        self.cosmo_prior_file = Basis_Root + 'fitcoeffs/biNN2p-cosmo-s.json'

        self.file_body = '_LMax{0:d}_'.format(self.L_Max) + self.file_tail
        # self.data_dir = '../examples/data/'
        self.data_dir = self.base_dir + 'data/'
        self.scales_file = self.data_dir + self.sample_root + self.sample + '/xi' + self.file_body + '_scales.txt'

        self.data_file_xilin = self.data_dir + self.sample_root + self.sample + '/xilin.txt'

        self.data_file_xi = self.data_dir + self.sample_root + self.sample + '/xi' + self.file_body + '.txt'
        self.cov_file = self.data_dir + self.sample_root + self.sample + '/covmat' + self.file_body 
        if self.include_Sig2obs:
            self.data_file_Sig2obs = self.data_dir + self.sample_root + self.sample + '/Sig2obs' + self.file_body + '.txt'
            self.data_file = copy.deepcopy([self.data_file_Sig2obs,self.data_file_xi])
            self.cov_file += '_inclSig2obs'
        else:
            self.data_file = self.data_file_xi
        if self.model_AP:
            self.cov_file += '_AP'
        self.cov_file += '.txt'
        
        return        
    ####################################################


    ####################################################
    def setup_basis(self):
        self.use_basis = np.arange(9)
        self.n_basis = len(self.use_basis)
        self.w_names = ['w_{0:d}'.format(m) for m in self.use_basis] 

        if self.verbose:
            print('... ... using {0:d} basis functions: ['.format(self.n_basis) + ','.join([str(b) for b in self.use_basis]) + ']')

        if self.verbose:
            print('... ... loading BiSequential instance')
        # read setup parameters
        # Basis_Stem imported from paths
        with open(Basis_Stem + '.pkl', 'rb') as f:
            params_setup = pickle.load(f)    
        params_setup['file_stem'] = Basis_Stem
        params_setup['verbose'] = self.verbose
        
        # initialize class
        self.binet = BiSequential(params=params_setup)
        # load network parameters from files
        self.binet.load()

        if self.verbose:
            print('... ... extracting basis instance')
        self.basis = self.binet.extract_basis()
        
        return
    ####################################################

    
    ####################################################
    def setup_fiducial(self):
        self.fiducial = {}

        # cosmology
        self.fiducial['f']      = {'':0.81735,'DESI-LRG2':0.83891,'Euclid-ELG':0.88902}
        self.fiducial['growth'] = {'':0.6965,'DESI-LRG2':0.66476,'Euclid-ELG':0.58168}
        self.fiducial['sigv']   = {'':4.13,'DESI-LRG2':3.8696,'Euclid-ELG':3.3860}

        if self.model_AP:
            self.fiducial['Daiso'] = {'':0.0,'DESI-LRG2': 6.910e-4,'Euclid-ELG': 8.734e-4}
            self.fiducial['DaAP'] =  {'':0.0,'DESI-LRG2':-5.939e-4,'Euclid-ELG':-6.822e-4}
            
        self.fiducial['fv'] = 0.26

        self.fiducial['rLP'] = {'':93.1636,'DESI-LRG2':93.0358,'Euclid-ELG':93.0358}
        self.fiducial['rZC'] = {'':120.5840,'DESI-LRG2':120.6466,'Euclid-ELG':120.6466}

        # basis coeffs
        with open(self.basis_prior_file,'rb') as f:
            self.basis_prior = json.load(f)
        for key in self.basis_prior.keys():
            if int(key) in self.use_basis:
                self.fiducial['w_'+key] = self.basis_prior[key][0] # now keys match self.w_names
        
        # with AP
        # ['beta', 'sigv', 'w_0', 'w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6', 'w_7', 'w_8', 
        #  'b', 'B1st', 'Bvst', 'sigma', 'AMC', 'qbar2', 'fv', 'Daiso', 'DaAP', 
        #  'f', 'peak', 'LP', 'ZC']
        
        # sdbmc
        self.fiducial['b']     = {'':2.435,'DESI-LRG2':2.383,'Euclid-ELG':1.784}
        self.fiducial['B1st']  = {'':-3.120,'DESI-LRG2':-1.2290,'Euclid-ELG':None}
        self.fiducial['Bvst']  = {'':-4.7853,'DESI-LRG2':8.7601,'Euclid-ELG':None}
        self.fiducial['AMC']   = {'':0.008,'DESI-LRG2':2.4058e-2,'Euclid-ELG':None}
        self.fiducial['sigma'] = {'':6.8274,'DESI-LRG2':7.3617,'Euclid-ELG':None}

        # cosmo+bias
        self.fiducial['beta'] = {}
        for key in self.fiducial['f'].keys():
            self.fiducial['beta'][key] = self.fiducial['f'][key]/self.fiducial['b'][key]
        
        # nuisance
        self.fiducial['qbar2'] = {'':0.109,'DESI-LRG2':5.6368e-2,'Euclid-ELG':None}
        self.fiducial['qbar4'] = 0.0

        # update basis coeffs to include growth and bias
        bg_fac = (self.fiducial['b'][self.sample]*self.fiducial['growth'][self.sample])**2
        for key in self.w_names:
            self.fiducial[key] *= bg_fac


        # string lists for use in creating info dictionary
        self.params_list = ['beta','sigv']
        self.latex_list = ['\\beta','\\sigma_{\\rm v}']

        self.params_list.extend(self.w_names)
        self.latex_list.extend(self.w_names)

        self.params_list += ['b','B1st','Bvst','sigma','AMC']
        self.latex_list += ['b','B_{1\\ast}','B_{v\\ast}','\\sigma','A_{\\rm MC}']

        if self.L_Max > 1:
            self.params_list += ['qbar2']
            self.latex_list += ['\\bar q^{(2)}']
            if self.L_Max == 3:
                self.params_list += ['qbar4']
                self.latex_list += ['\\bar q^{(4)}']

        if self.include_Sig2obs:
            self.params_list += ['fv']
            self.latex_list += ['f_{\\rm v}']

        if self.model_AP:
            self.params_list += ['Daiso','DaAP']
            self.latex_list += ['\\Delta\\alpha_{\\rm iso}','\\Delta\\alpha_{\\rm AP}']

        self.eval_dict_fid = {}
        for key in self.params_list:
            if np.isscalar(self.fiducial[key]):
                value = self.fiducial[key]
            else:
                value = self.fiducial[key][self.sample]
            self.eval_dict_fid[key] = value

        self.best_fit = {'params':['beta', 'sigv', 
                                   'w_0', 'w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6', 'w_7', 'w_8', 
                                   'b', 'B1st', 'Bvst', 'sigma', 'AMC', 
                                   'qbar2', 
                                   'fv', 'Daiso', 'DaAP'],
                         'DESI-LRG2':[3.3866e-01,4.0746e+00,
                                      1.4302e-02,-5.2372e-03,-1.0899e-02,1.9982e-02,-4.7548e-03,1.8302e-02,-3.7938e-02,8.3095e-03,1.6701e-02,
                                      2.4575e+00,-1.2290e+00,8.7601e+00,7.3617e+00,2.4058e-02,
                                      5.6368e-02,
                                      2.2611e-01,-1.6575e-02,1.8403e-03],
                         'Euclid-ELG':None}            
        
        return        
    ####################################################

    
    ####################################################
    def load_xilin(self,growth):
        xilin_true = np.loadtxt(self.data_file_xilin).T
        xilin_true[1] *= growth**2 # since stored value was at z=0
        return xilin_true
    ####################################################

    
    ####################################################
    def load_basis_func(self,rvals):
        basis_func = self.basis.predict(self.binet.rv(rvals))
        basis_func = np.concatenate((np.ones((1,basis_func.shape[1])),basis_func),axis=0) 
        basis_func = basis_func[self.use_basis]
        return basis_func
    ####################################################

    
    ####################################################
    def setup_info(self):
        """ Create and store info dictionary (as class attribute self.info) for use with Cobaya. """

        info = {}
        info['likelihood'] = {'zelsmear.ZeldovichSmearingLike':
                              {'python_path':self.like_dir,
                               'L_Max':self.L_Max,
                               'rescale':1.0,
                               'modify_data':self.modify_data,
                               'include_Sig2obs':self.include_Sig2obs,
                               'scales_file':self.scales_file,
                               'data_file':self.data_file,
                               'cov_file':self.cov_file
                              }}

        info['theory'] = {'zelsmear.ZeldovichSmearingTheory':
                          {'python_path':self.like_dir,
                           'stop_at_error': True,
                           'use_basis': self.use_basis,
                           'basis_stem':Basis_Stem,
                           'accuracy':self.accuracy,
                           'n_r':100, 
                           # 100 gives LP convergence error ~0.1%, i.e. 3x smaller than DESI expectation
                           # 300 gives ~0.03%, 1000 is converged
                           'scales_file':self.scales_file,
                           'L_Max':self.L_Max,
                           'modify_data':self.modify_data,
                           'model_AP':self.model_AP,
                           'include_Sig2obs':self.include_Sig2obs
                          }}


        info['params'] = {}
        for p in range(len(self.params_list)):
            param = self.params_list[p]
            defval = self.eval_dict_fid[param]

            info['params'][param] = {'latex':self.latex_list[p]}
            if param == 'AMC':
                info['params'][param] = {'latex':self.latex_list[p]}
            if (param == 'qbar2'):
                if self.modify_data & (self.L_Max == 2):
                    info['params'][param] = defval
                else:
                    info['params'][param] = {'latex':self.latex_list[p]}
            if (param == 'qbar4'):
                if self.modify_data:
                    info['params'][param] = defval                
                else:
                    info['params'][param] = {'latex':self.latex_list[p]}


        # priors (values irrelevant)
        info['params']['beta']['prior'] = {'min':-1.0,'max':1.0} 
        info['params']['sigv']['prior'] = {'min':0.0,'max':12.0}

        bg_fac = (self.eval_dict_fid['b']*self.fiducial['growth'][self.sample])**2
        
        for w in range(len(self.w_names)):
            wname = self.w_names[w]
            w_mean,w_std = self.basis_prior[str(w)][0],self.basis_prior[str(w)][1]
            w_mean *= bg_fac
            w_std *= bg_fac
            info['params'][wname]['prior'] = {'dist':'norm','loc':w_mean,'scale':w_std} 

        info['params']['b']['prior'] =  {'dist':'norm','loc':self.eval_dict_fid['b'],'scale':0.01*np.fabs(self.eval_dict_fid['b'])}

        info['params']['B1st']['prior'] = {'min':-100.0,'max':100.0}
        info['params']['Bvst']['prior'] = {'min':-100.0,'max':100.0}
        info['params']['sigma']['prior'] = {'min':0.0,'max':12.0} 
        info['params']['AMC']['prior'] = {'dist':'norm','loc':0.0,'scale':0.05}

        if self.L_Max > 1:
            if (not self.modify_data) | (self.L_Max == 3):
                # note L_Max==3 not 2
                info['params']['qbar2']['prior'] = {'min':-2.0,'max':2.0}
            if (self.L_Max == 3) & (not self.modify_data):
                info['params']['qbar4']['prior'] = {'min':-2.0,'max':2.0}
        if self.include_Sig2obs:
            info['params']['fv']['prior'] = {'dist':'norm','loc':self.eval_dict_fid['fv'],'scale':0.01*np.fabs(self.eval_dict_fid['fv'])}

        if self.model_AP:
            info['params']['Daiso']['prior'] = {'dist':'norm','loc':self.eval_dict_fid['Daiso'],'scale':0.01*np.fabs(self.eval_dict_fid['Daiso'])}
            info['params']['DaAP']['prior'] = {'dist':'norm','loc':self.eval_dict_fid['DaAP'],'scale':0.01*np.fabs(self.eval_dict_fid['DaAP'])}                    

        self.info = info
        return 
    ####################################################


    ####################################################
    def setup_model(self):
        """ Setup Cobaya model instance. Assumes self.setup_info has already been called. """
        self.model = get_model(self.info)
        self.like = self.model.likelihood['zelsmear.ZeldovichSmearingLike']
        self.theory = self.model.theory['zelsmear.ZeldovichSmearingTheory']
        return
    ####################################################


    ####################################################
    def calc_model(self,eval_dict,wrap=True):
        """ Evaluate model. Assumes self.setup_model has already been called.
            -- eval_dict:  dictionary with keys self.params_list and values being chosen parameter values
            -- wrap: bool (default True). 
                     If False, returns flat array. 
                     If True, returns list of arrays
                     -- If self.include_Sig2obs is True: first element is array of shape (self.L_Max,) containing Sig2ell values
                     -- Followed by: arrays of shape (s,) (suitably modified in case modify_data was True when creating info dict) containing xiell(s) values
        """
        point = dict(zip(self.model.parameterization.sampled_params(),
                         self.model.prior.sample(ignore_external=True)[0]))
        point.update({key:eval_dict[key] for key in point.keys()})
        self.model.logposterior(point)

        output = self.model.provider.get_model() # outputs flat array
        # below needed in case self.like.rescale was set different from unity
        output *= self.like.rescale 
        if self.include_Sig2obs:
            output[:self.L_Max] /= self.like.rescale # Sig2's not rescaled

        if wrap:
            output = self.like.wrap_model(output)
            
        return output            
    ####################################################

    
    ####################################################
    def load_data(self,wrap=True):
        """ Convenience function to load stored data. Assumes self.setup_model has already been called.
            -- wrap: bool (default True). 
                     If False, returns flat array. 
                     If True, returns list of arrays
                     -- If self.include_Sig2obs is True: first element is array of shape (self.L_Max,) containing Sig2ell values
                     -- Followed by: arrays of shape (s,) (suitably modified in case modify_data was True when creating info dict) containing xiell(s) values
            Returns (wrapped) data, cov_data.
        """
        if self.include_Sig2obs:
            data = np.concatenate((np.loadtxt(self.data_file[0]).T[0],np.loadtxt(self.data_file[1])))
        else:
            data = np.loadtxt(self.data_file)
        cov_data = np.loadtxt(self.cov_file)
        data,cov_data = self.like.organize_data(data,cov_data)

        if wrap:
            data = self.like.wrap_model(data)

        return data,cov_data
    ####################################################


    ####################################################
    def calc_chi2(self,data,prediction):
        """ Calculate chi2 between data and prediction. Assumes self.setup_model has already been called. """
        residual = data - prediction
        z = linalg.cho_solve((self.like.L,True),residual) # solves (L L^T) z = residual or z = C^-1 residual
        chi2 = np.dot(residual,z)
        return chi2
    ####################################################
    

    ####################################################
    def vary_prediction(self,ev_ref,param,direction,amount,relative=True,wrap=True):
        """ Convenience function to produce model prediction for specified variation of particular parameter around reference model.
            -- ev_ref: dictionary of param values for reference model
            -- param: str, name of parameter to vary
            -- direction: float/int (only sign matters), direction of variation: upwards (downward) for +ve (-ve)
            -- amount: positive float, amount of variation
            -- relative: bool (default True), if True, treat amount as relative variation, else absolute
            -- wrap: bool, fed to self.calc_model
            Returns model prediction
        """
        ev_dict = copy.deepcopy(ev_ref)
        variation = np.sign(direction)*amount
        if relative:
            variation *= np.fabs(ev_ref[param])
        ev_dict[param] += variation

        prediction = self.calc_model(ev_dict,wrap=wrap)

        return prediction
    ####################################################
    
