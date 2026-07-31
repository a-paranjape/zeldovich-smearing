import numpy as np
from scipy import special
import sys

from paths import *

from pathlib import Path

sys.path.append(ML_Path)
from mllib import Utilities

import copy,gc
from time import time

from cobaya.run import run
from cobaya.log import LoggedError
from cobaya.model import get_model
from cobaya.samplers.mcmc import plot_progress

from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt
from getdist.gaussian_mixtures import GaussianND

import matplotlib.pyplot as plt

from emulation import AgnosticEmulator

######################################
class StrongSampler(Utilities):
    """ Class to setup and run MCMC chains using strong prior. """
    ##################################
    def __init__(self,setup={}):
        """ Class to setup and run MCMC chains using strong prior. 
            setup should be dict with keys being a subset of
            -- base_dir: str, path/to/zeldovich-smearing/examples (default assumes '../examples/')
            -- sample: str, sample name. one of ['DESI-LRG2','Euclid-ELG','']. Default 'DESI-LRG2'.
            -- model_AP: bool (default True), whether or not to model AP distortion.
            -- include_Sig2obs: bool (default True), whether or not to include k-space integrals in observables
            -- L_Max: int (default 3), number of multipoles included. 
                      1 --> ell = [0]
                      2 --> ell = [0,2]
                      3 --> ell = [0,2,4]
            -- cosmo: str, cosmological model acronym, one of ['lcdm','wcdm','nucdm','w0wacdm']. Default 'lcdm'.
            -- flat: bool (default True), whether spatial curvature is varied (False) or not (True).
            -- scale_planck18: float (default 6.0), scale defining prior relative to Planck18.
            -- include_emulator_error: bool (default False), whether or not to average likelihood over emulator noise.
            -- n_proc: int (default 1), number of processors to use. >1 will invoke MPI routines (needs mpi4py).
            -- burn_frac: float in (0,1], fraction of samples to discard as burn-in. Used with GetDist.
            -- verbose,logfile: std I/O variables, default True,None.
        """
        self.setup = setup
        self.iteration = 2 # hard-coded to 2, assuming converged covariance matrix exists.

        self.base_dir = setup.get('base_dir','../examples/')
        self.sample = setup.get('sample','DESI-LRG2')
        
        self.model_AP = setup.get('model_AP',True)
        self.include_Sig2obs = setup.get('include_Sig2obs',True)
        self.L_Max = setup.get('L_Max',3)
        self.cosmo = setup.get('cosmo','lcdm')
        self.flat = setup.get('flat',True)
        self.flat_str = '_flat' if self.flat else ''
        self.scale_planck18 = setup.get('scale_planck18',6.0)

        self.include_emulator_error = setup.get('include_emulator_error',False)
        
        self.n_proc = setup.get('n_proc',1)
        self.burn_frac = setup.get('burn_frac',0.3)

        self.verbose = setup.get('verbose',True)
        self.logfile = setup.get('logfile',None)

        self.setup_files_and_folders() # also sets self.z_eval        
        self.setup_reference()
        self.setup_agnostic_emulator() # will instantiate AgnosticEmulator
        self.setup_mcmc() # needs AgnosticEmulator already instantiated
        
    ##################################

    ##################################
    def setup_files_and_folders(self):
        """ Utility to setup names of relevant files and folders. 
            Sets attributes:
            -- z_eval,like_dir,data_dir,plots_dir,scales_file,data_file_xilin,data_file,cov_file,id_str_stem,config_dict
        """
        
        self.config_dict = {'DESI-LRG2': {'redshift':0.80,'Mmin':8e12,'phase':0,
                                          'sample_root':'AbacusSummit/base_c000/'},
                            'Euclid-ELG':{'redshift':1.10,'Mmin':1e12,'phase':9,
                                          'sample_root':'AbacusSummit/base_c000/'},
                            '':{'redshift':0.7,'Mmin':None,'phase':None,
                                'sample_root':'SDBMC'}}        

        self.like_dir = self.base_dir + '../code/'
        self.data_dir = self.base_dir + 'data/'
        
        self.z_eval = self.config_dict[self.sample]['redshift']
        
        m_min = self.config_dict[self.sample]['Mmin']
        sample_root = self.config_dict[self.sample]['sample_root']
        file_tail = 'lgMmin{0:.2f}_z{1:.3f}'.format(np.log10(m_min),self.z_eval) if self.sample != '' else 'sdbmc'
        file_body = '_LMax{0:d}_'.format(self.L_Max) + file_tail

        if file_tail == 'sdbmc':
            self.model_AP = False # no AP distortions for sdbmc toy model
            self.iteration = 2    # no iterations needed for sdbmc toy model
        
        self.plots_dir = self.base_dir + 'plots/'
        self.plots_dir += sample_root + self.sample + '/'
        self.plots_dir += self.cosmo + self.flat_str + '/'
        self.plots_dir += 'withAP/' if self.model_AP else 'withoutAP/'
        self.plots_dir += 'withEmuErr/' if self.include_emulator_error else 'withoutEmuErr/'
        self.plots_dir += file_body[1:] + '/'

        if self.verbose:
            self.print_this('Plots dir: '+self.plots_dir,self.logfile)
        Path(self.plots_dir).mkdir(parents=True,exist_ok=True) # folder to store plots

        self.scales_file = self.data_dir + sample_root + self.sample + '/xi' + file_body + '_scales.txt'

        data_file_xi = self.data_dir + sample_root + self.sample + '/xi' + file_body + '.txt'
        self.data_file_xilin = self.data_dir + sample_root + self.sample + '/xilin.txt'
        self.cov_file = self.data_dir + sample_root + self.sample + '/covmat' + file_body 
        if self.include_Sig2obs:
            data_file_Sig2obs = self.data_dir + sample_root + self.sample + '/Sig2obs' + file_body + '.txt'
            self.data_file = copy.deepcopy([data_file_Sig2obs,data_file_xi])
            self.cov_file += '_inclSig2obs'
        else:
            self.data_file = data_file_xi
        if self.model_AP:
            self.cov_file += '_AP'
        self.cov_file += '.txt' if self.iteration >= 2 else '_iter{0:d}.txt'.format(self.iteration)
        
        self.id_str_stem = sample_root + self.sample + '/'
        self.id_str_stem += self.cosmo + self.flat_str + '/'
        self.id_str_stem += 'withAP/' if self.model_AP else 'withoutAP/'
        self.id_str_stem += 'withEmuErr/' if self.include_emulator_error else 'withoutEmuErr/'
        self.id_str_stem += 'mcmc' + file_body
        if self.include_Sig2obs:
            self.id_str_stem += '_inclSig2obs'
        
        if self.verbose:
            self.print_this('Scales file: '+self.scales_file,self.logfile)
            self.print_this('Cov mat file: '+self.cov_file,self.logfile)
            self.print_this('xi_lin file: '+self.data_file_xilin,self.logfile)
            if isinstance(self.data_file,list):
                self.print_this('Data files: ['+ ','.join([f for f in self.data_file]) +']',self.logfile)
            else:
                self.print_this('Data file: '+self.data_file,self.logfile)
            self.print_this('id string stem: '+self.id_str_stem,self.logfile)
        
        return
    ##################################

    ##################################
    def setup_reference(self):
        """ Simple utility to setup various reference values and arrays:
            -- prior for linear Eulerian bias 
            -- fiducial value for smearing scale
            -- ground truth linear 2pcf at sample redshift
            Sets attributes:
            -- b_dict,b_gt,b_prior
            -- sigv_fid_dict,sigv_fid,sigma_fid
            -- growth_dict,growth_gt,xilin_true
        """
        
        self.b_dict = {'':2.435,'DESI-LRG2':2.383,'Euclid-ELG':1.784}

        self.b_gt = self.b_dict[self.sample] 
        self.b_prior = [self.b_gt,0.05*self.b_gt] # hard-coded 5% prior, maybe give to user control later.

        self.sigv_fid_dict = {'':4.13,'DESI-LRG2':3.9407,'Euclid-ELG':3.4478}
        self.sigv_fid = self.sigv_fid_dict[self.sample] # Mpc/h_fid
        self.sigma_fid = np.sqrt(2)*self.sigv_fid # Mpc/h_fid; sqrt(2) to satisfy physical prior R*^2 >= 0

        self.growth_dict = {'':0.6965,'DESI-LRG2':0.66476,'Euclid-ELG':0.58168}
        self.growth_gt = self.growth_dict[self.sample] # D(z)/D(0) 

        self.xilin_true = np.loadtxt(self.data_file_xilin).T
        self.xilin_true[1] *= self.growth_gt**2 # since stored value was at z=0
        
        return
    ##################################

    
    ##################################
    def setup_agnostic_emulator(self):
        """ Simple utility to setup AgnostiEmulator instance and related dictionary. 
            Sets attributes:
            -- emulator_setup,agem,emulator_model_name,emulator_cov_file
        """
        
        self.emulator_setup = {'out_stem':self.like_dir+'../emulation/emulators/',
                               'cosmo':self.cosmo,'flat':self.flat,
                               'z_eval':self.z_eval,'scale_planck18':self.scale_planck18,
                               'verbose':self.verbose,'logfile':self.logfile}
        self.agem = AgnosticEmulator(setup=self.emulator_setup) # for easy access to latex variables etc.
        self.emulator_setup['verbose'] = False
        self.emulator_model_name = 'shallow' # may need to be adjusted if alternate emulators trained.

        if self.include_emulator_error:
            self.emulator_cov_file = self.agem.out_stem + self.cosmo + self.flat_str + '/models/' + self.emulator_model_name + '_forward/cov.txt'
        else:
            self.emulator_cov_file = None
        
        return
    ##################################
    

    ##################################
    def set_dim(self,modify_data=True,varyMC=True):
        """ Simple utility to calculate and store dimensions of data and parameter vectors (excluding derived params). 
            Called by self.set_mcmc().
            Sets attributes:
            -- dim,n_data,dof
        """
        dim = self.agem.n_params + 5 # cosmological + 5 sdbmc
        if self.L_Max > 1:
            dim += 1 # qbar2
            if self.L_Max == 3:
                dim += 1 # qbar4
        if not varyMC:
            dim -= 1 # don't sample AMC
        if modify_data & (self.L_Max > 1):
                dim -= 1 # don't sample qbar2 (for L_Max==2) or qbar4 (for L_Max==3)
        self.dim = 1*dim

        n_data = self.L_Max*np.loadtxt(self.scales_file).size
        if modify_data & (self.L_Max > 1):
            n_data -= 1 # delete s=s_min for ell=2
            if self.L_Max == 3:
                n_data -= 1 # delete s=s_min for ell=4
        if self.include_Sig2obs:
            n_data += self.L_Max # include k-space integral(s)

        self.n_data = 1*n_data
        self.dof = n_data - dim
        # print(dim,dof)        

        return
    ##################################
    
    ##################################
    def setup_mcmc(self):
        """ Utility to setup MCMC control variables, priors and info dictionary. 
            Calls self.set_dim() [sets dim,n_data,dof]. 
            Sets attributes:
            -- id_str,info,derived_list,all_sampled_params,dim_all
            Requires self.agem to exist as instance of AgnosticEmulator.
        """
        #############
        # consider giving these to user control
        rescale = 1e3
        modify_data = True
        varyMC = True
        accuracy = 'mid'

        max_samples = 10000000 
        Rminus1_stop = 0.01 # 0.01
        Rminus1_cl_stop = 0.05 # 0.05; arXiv:2602.14533 used 0.035/0.04
        Rminus1_cl_level = 0.95 # 95
        #############

        self.id_str = self.id_str_stem + ''
        if not modify_data:
            self.id_str += '_unmod'
        if accuracy != 'mid':
            self.id_str += '_'+accuracy+'acc'

        self.set_dim(modify_data,varyMC)

        self.derived_list = ['f','peak','LP','ZC'] + [f"w_{b}" for b in range(self.agem.n_basis)]

        self.info = {}
        self.info['likelihood'] = {'zelsmear.ZeldovichSmearingLike':
                                   {'python_path':self.like_dir,
                                    'rescale':rescale,
                                    'L_Max':self.L_Max,
                                    'modify_data':modify_data,
                                    'include_Sig2obs':self.include_Sig2obs,
                                    'derived_list':self.derived_list,
                                    'strong_prior':True,
                                    'include_emulator_error':self.include_emulator_error,
                                    'scales_file':self.scales_file,
                                    'data_file':self.data_file,
                                    'cov_file':self.cov_file
                                    }}

        self.info['theory'] = {'zelsmear.ZeldovichSmearingTheory':
                               {'python_path':self.like_dir,
                                'stop_at_error': True,
                                'accuracy':accuracy,
                                'scales_file':self.scales_file,
                                'L_Max':self.L_Max,
                                'modify_data':modify_data,
                                'model_AP':self.model_AP,
                                'include_Sig2obs':self.include_Sig2obs,
                                'strong_prior':True,
                                'include_emulator_error':self.include_emulator_error,
                                'emulator_cov_file':self.emulator_cov_file,
                                'emulator_setup':self.emulator_setup,
                                'emulator_model_name':self.emulator_model_name
                                }}

        self.info['sampler'] = {'mcmc':
                                {'learn_proposal': True,
                                 'measure_speeds': True,
                                 'max_samples': max_samples,
                                 'max_tries': np.inf, # 1000
                                 'Rminus1_stop': Rminus1_stop,
                                 'Rminus1_cl_stop': Rminus1_cl_stop,
                                 'Rminus1_cl_level': Rminus1_cl_level,
                                 'burn_in':0}}


        info_output = 'stats/chains/' 
        info_output += 'varymc/' if varyMC else 'nomc/'
        info_output += self.id_str
        self.info['output'] = info_output
        
        self.info['params'] = {}
        # cosmological params
        for p in range(len(self.agem.keys_vary)):
            key = self.agem.keys_vary[p]
            proposal = 1e-4
            if key == 'As':
                proposal = 1e-12
            if key == 'Ob':
                proposal = 1e-5
            defval = 0.5*(self.agem.param_maxs[p] + self.agem.param_mins[p]) #self.agem.pfid[key]
            dtheta = 0.005*(self.agem.param_maxs[p] - self.agem.param_mins[p])
            self.info['params'][key] = {'ref':{'min':defval - dtheta,'max':defval + dtheta},
                                        'proposal':proposal,
                                        'latex':self.agem.latex_cosmological[p],
                                        'prior':{'min':self.agem.param_mins[p],'max':self.agem.param_maxs[p]}}

        # sdbmc + nuisance params
        Dp = 1e-3
        params_list = ['b','B1st','Bvst','sigma','AMC']
        latex_list = ['b','B_{1\\ast}','B_{v\\ast}','\\sigma','A_{\\rm MC}']
        # defaults_list = [self.b_prior[0],0.0,0.0,self.sigma_fid,0.0] 
        proposal_list = [1e-3,1e-2,3e-3,5e-3,5e-3]
        ref_min_list = [self.b_prior[0]*(1-Dp),-Dp,-Dp,self.sigma_fid*(1-Dp),0.0]
        ref_max_list = [self.b_prior[0]*(1+Dp), Dp, Dp,self.sigma_fid*(1+Dp),Dp]
        
        if self.L_Max > 1:
            params_list += ['qbar2']
            latex_list += ['\\bar q^{(2)}']
            # defaults_list += [0.0] 
            proposal_list += [None] if (modify_data & (self.L_Max == 2)) else [2e-4]
            ref_min_list += [-Dp]
            ref_max_list += [Dp]
            if self.L_Max == 3:
                params_list += ['qbar4']
                latex_list += ['\\bar q^{(4)}']
                # defaults_list += [0.0] 
                proposal_list += [None] if modify_data else [2e-4]
                ref_min_list += [-Dp]
                ref_max_list += [Dp]

        for p in range(len(params_list)):
            param = params_list[p]
            self.info['params'][param] = {'ref':{'min':ref_min_list[p],'max':ref_max_list[p]},
                                          'proposal':proposal_list[p],
                                          'latex':latex_list[p]}
            
            if (param == 'qbar2'):
                if modify_data & (self.L_Max == 2):
                    self.info['params'][param] = 0.0
                    
            if (param == 'qbar4'):
                if modify_data:
                    self.info['params'][param] = 0.0

        self.info['params']['b']['prior'] =  {'dist':'norm','loc':self.b_prior[0],'scale':self.b_prior[1]}
        self.info['params']['B1st']['prior'] = {'min':-100.0,'max':100.0}
        self.info['params']['Bvst']['prior'] = {'min':-100.0,'max':100.0}
        self.info['params']['sigma']['prior'] = {'min':0.0,'max':12.0} 
        if varyMC:
            self.info['params']['AMC']['prior'] = {'dist':'norm','loc':0.0,'scale':0.05}
        else:
            self.info['params']['AMC'] = 0.0
        if self.L_Max > 1:
            if (not modify_data) | (self.L_Max == 3):
                # note L_Max==3 not 2
                self.info['params']['qbar2']['prior'] = {'min':-2.0,'max':2.0}
            if (self.L_Max == 3) & (not modify_data):
                self.info['params']['qbar4']['prior'] = {'min':-2.0,'max':2.0}

        derived_latex = ['f','r_{\\rm peak}','r_{\\rm LP}','r_{\\rm ZC}'] + [f"w_{b}" for b in range(self.agem.n_basis)]
        for d in range(len(self.derived_list)):
            dpar = self.derived_list[d]
            self.info['params'][dpar] = {'latex':derived_latex[d]}

        self.all_sampled_params = []
        for key in self.info['params'].keys():
            if isinstance(self.info['params'][key],dict):
                if 'prior' in self.info['params'][key].keys():
                    self.all_sampled_params.append(key)
        self.dim_all = len(self.all_sampled_params) + len(self.derived_list)
        
        return
    ##################################
    
    ##################################
    def run_mcmc(self,force=True,resume=False):
        """ Run MCMC chains using self.info. 
            -- force:  bool (default True), whether to run chains or only load from file.
            -- resume: bool (default False), whether or not to load and resume existing chain.
                       False -- force overwrite of any existing chain files
                       True  -- restart using existing chain (assuming filenames match).
               Chains will be read/written using stem self.info['output'].
            Returns:
            -- gd_sample: GetDist sample that can be passed on to self.corner_plot and <OTHER METHODS>
        """
        start_time = time()
        if force:
            if resume:
                self.info["resume"] = True
                self.info["force"] = False
            else:
                self.info["force"] = True
                self.info["resume"] = False

            # comm = MPI.COMM_WORLD
            # rank = comm.Get_rank()
            success = False
            try:
                upd_info, sampler = run(self.info,resume=resume)#,allow_changes=True)
                success = True
            except LoggedError as err:
                pass
            # success = all(comm.allgather(success))
            if not success: # and rank == 0:
                print("Sampling failed!")
                
        # gd_sample = loadMCSamples(os.path.abspath(self.info["output"]),settings={'ignore_rows':self.burn_frac})
        gd_sample = loadMCSamples(str(Path(self.info["output"]).resolve()),settings={'ignore_rows':self.burn_frac})

        if self.verbose:
            self.time_this(start_time)
        
        return gd_sample
    ##################################

    ##################################
    def display_stats(self,gd_sample,include_chi2=False,verbose=True):
        """ Display best-fit, MAP and goodness-of-fit stats. 
            -- gd_sample: GetDist sample, output of self.run_mcmc().
            -- include_chi2: bool (default False), whether or not to include chi2 as a derived parameter.
            -- verbose: bool (default True), whether or not to print results.
            Returns:
            -- mcmc_MAP,mcmc_best: arrays of MAP and best-fit parameters (useful for corner plots)
            -- chi2_min,pval: chi2 and p-value of best fit
            -- par_show: list of ints for indexing sample file columns.
        """
        mcmc_covmat = gd_sample.getCovMat().matrix[:self.dim_all,:self.dim_all]
        ibest = gd_sample.samples.T[-2].argmin()
        mcmc_best = gd_sample.samples[ibest]
        chi2_min = mcmc_best[-1]
        i_last = -3 if include_chi2 else self.dim_all
        mcmc_best = mcmc_best[:i_last].copy()
        pval = special.gammainc(chi2_min/2,self.dof/2)
        mcmc_sig = np.sqrt(np.diag(mcmc_covmat))

        offset = 2
        par_show = offset + np.arange(self.dim_all + (1 if include_chi2 else 0))
        if include_chi2:
            par_show[-1] = -2
        # sample_for_MAP = np.loadtxt(os.path.abspath(self.info["output"])+'.'+str(1)+'.txt')
        sample_for_MAP = np.loadtxt(str(Path(self.info["output"]).resolve())+'.'+str(1)+'.txt')
        sample_for_MAP = sample_for_MAP[int(self.burn_frac*sample_for_MAP.shape[0]):]
        for c in range(1,self.n_proc):
            # sample_c = np.loadtxt(os.path.abspath(self.info["output"])+'.'+str(c+1)+'.txt')
            sample_c = np.loadtxt(str(Path(self.info["output"]).resolve())+'.'+str(c+1)+'.txt')
            sample_c = sample_c[int(self.burn_frac*sample_c.shape[0]):]
            sample_for_MAP = np.concatenate((sample_for_MAP,sample_c),axis=0)
        iMAP = sample_for_MAP.T[1].argmin()
        mcmc_MAP = sample_for_MAP[iMAP][par_show]

        del sample_for_MAP
        gc.collect()

        if verbose:
            print("... MAP       = ( "+','.join(['%.3e' % (pval,) for pval in mcmc_MAP])+" )")
            print("... best fit  = ( "+','.join(['%.3e' % (pval,) for pval in mcmc_best])+" )")
            print("... std dev   = ( "+','.join(['%.3e' % (pval,) for pval in mcmc_sig])+" )")
            print("... chi2_best,dof,chi2_red,pval: {0:.3f},{1:d},{2:.3f},{3:.3e}".format(chi2_min,self.dof,chi2_min/self.dof,pval))

        return mcmc_MAP,mcmc_best,chi2_min,pval,par_show
    ##################################

    
    ##################################
    def corner_plot(self,params_list,gd_sample,setup={}):
        """ Produce corner plot for specified parameters, with various options. 
            -- params_list: list of str, should be subset of self.all_sampled_params + self.derived_list
            -- gd_sample: GetDist sample, output of self.run_mcmc().
            -- setup: dict for optional control, keys should be subset of
            .. -- save_fig: bool (default True), whether or not to save to file.
            .. -- out_file: str (default 'fig.pdf'), filename in which to save, will be placed in self.plots_dir.
            .. -- subplot_size: float (default 1.2), size of each subplot in corner plot.
            .. -- marker_MAP : None (default) or array of length len(params_list), giving MAP values
            .. -- marker_best: None (default) or array of length len(params_list), giving best-fit values
            .. -- marker_truth: None (default) or array of length len(params_list), giving ground-truth values
        """
        for key in params_list:
            if key not in (self.all_sampled_params + self.derived_list):
                raise Exception("Unrecognised parameter '"+key+"' supplied to corner_plot().")

        save_fig = setup.get('save_fig',True)
        out_file = setup.get('out_file','fig.pdf')
        subplot_size = setup.get('subplot_size',1.2)
        marker_MAP = setup.get('marker_MAP',None)
        marker_best = setup.get('marker_best',None)
        marker_truth = setup.get('marker_truth',None)

        gdplot = gdplt.get_subplot_plotter(subplot_size=subplot_size)
        gdplot.settings.num_plot_contours = 3
        gdplot.settings.axes_fontsize = 13
        gdplot.settings.axes_labelsize = 15
        gdplot.settings.title_limit_fontsize = 15 

        gdplot.triangle_plot([gd_sample], params_list,
                             filled=[True],contour_colors=['indigo'],legend_loc='upper center',
                             markers=marker_MAP,
                             marker_args={'c': 'indigo','ls': '--','lw': 1.5,'alpha': 0.6},
                             title_limit=1)
        
        if (marker_MAP is not None) | (marker_best is not None):
            for par_y in range(len(params_list)):
                str_y = params_list[par_y]
                ax = gdplot.subplots[par_y,par_y]
                if marker_truth is not None:
                    ax.axvline(marker_truth[par_y],c='goldenrod',ls=':',lw=1.5,alpha=0.6)
                for par_x in range(par_y):
                    str_x = params_list[par_x]
                    ax = gdplot.subplots[par_y,par_x]
                    if marker_MAP is not None:
                        ax.scatter([marker_MAP[par_x]],[marker_MAP[par_y]],
                                   marker='*',s=100,c='white')
                    if marker_best is not None:
                        ax.scatter([marker_best[par_x]],[marker_best[par_y]],
                                   marker='o',s=25,c='peachpuff')
                    if marker_truth is not None:
                        ax.scatter([marker_truth[par_x]],[marker_truth[par_y]],
                                   marker='*',s=80,c='goldenrod')
                        ax.axvline(marker_truth[par_x],c='goldenrod',ls=':',lw=1.5,alpha=0.6)
                        ax.axhline(marker_truth[par_y],c='goldenrod',ls=':',lw=1.5,alpha=0.6)

        if save_fig:
            print('Writing to file: '+self.plots_dir+out_file)
            gdplot.export(fname=out_file,adir=self.plots_dir)
                                    
        return
    ##################################

    ##################################
    def show_bestfit(self,gd_sample,setup={}):
        """ Produce plots (i) comparing data with best fit and (ii) showing linear reconstruction (optionally compared with ground truth). 
            -- gd_sample: GetDist sample, output of self.run_mcmc().
            -- setup: dict for optional control, keys should be subset of
            .. -- save_fig: bool (default True), whether or not to save to file.
            .. -- out_file_bf,out_file_lin: str (defaults 'bf.pdf','lin.pdf'), filenames in which to save, will be placed in self.plots_dir.
            .. -- show_gt: bool (default True), whether or not to show ground truth xi_lin. (self.xilin_true should exist.)
            .. -- cols: list of str of length >= self.L_Max (default ['crimson','indigo','forestgreen']), colours to use for each multipole.
            .. -- FS,FS2,FS3,FSL: ints (defaults 18,15,13,22), various font size specs.
        """
        start_time = time()
        save_fig = setup.get('save_fig',True)
        out_file_bf = setup.get('out_file_bf','bf.pdf')
        out_file_lin = setup.get('out_file_lin','lin.pdf')
        show_gt = setup.get('show_gt',True)
        cols = setup.get('cols',['crimson','indigo','forestgreen'])
        FS = setup.get('FS',18)
        FS2 = setup.get('FS2',15)
        FS3 = setup.get('FS3',13)
        FSL = setup.get('FSL',22)
        
        sample = gd_sample.samples.T
        
        model = get_model(self.info)
        stab = np.loadtxt(self.scales_file)
        like = model.likelihood['zelsmear.ZeldovichSmearingLike']
        theory = model.theory['zelsmear.ZeldovichSmearingTheory']
        off = 1 if self.include_Sig2obs else 0

        # extract best fit parameter vector
        mcmc_MAP,mcmc_best,chi2_min,pval,par_show = self.display_stats(gd_sample,verbose=False)

        # initialize dictionary of sampled and derived params
        dp_best = {p:0.0 for p in self.all_sampled_params + self.derived_list}
        # set MAP values
        cnt = 0
        for key in (self.all_sampled_params + self.derived_list):
            dp_best[key] = mcmc_MAP[cnt] # since mcmc_MAP contains sampled params followed by derived params
            cnt += 1

        point = dict(zip(model.parameterization.sampled_params(),
                         model.prior.sample(ignore_external=True)[0]))
        point.update({key:dp_best[key] for key in point.keys()})
        model.logposterior(point)
        xiNLell_best = like.wrap_model(model.provider.get_model())
        W_best = np.array([dp_best['w_{0:d}'.format(m)] for m in range(self.agem.n_basis)])
        xilin_best = np.dot(theory.bfunc.T,W_best)/(dp_best['b']**2 + 1e-15)

        n_boot = np.min([5000,int(0.2*sample[0].size)])
        indices = gd_sample.random_single_samples_indices(random_state=42,max_samples=n_boot)
        n_boot = indices.size
        if self.verbose:
            self.print_this('n_boot:{0:d}'.format(n_boot),self.logfile)

        xiNLell_boot = np.zeros((n_boot,like.Dim_Data),dtype=float)
        xilin_boot = np.zeros((n_boot,theory.rvals.size),dtype=float)

        if self.verbose:
            self.print_this('... extracting xiNL_ell stats from subsample',self.logfile)
        for b in range(n_boot):
            params_b = sample[:self.dim_all,indices[b]] 
            dp_b = {p:0.0 for p in self.all_sampled_params + self.derived_list}
            cnt = 0
            for key in self.all_sampled_params + self.derived_list:
                dp_b[key] = params_b[cnt]
                cnt += 1

            point = dict(zip(model.parameterization.sampled_params(),
                             model.prior.sample(ignore_external=True)[0]))
            point.update({key:dp_b[key] for key in point.keys()})
            model.logposterior(point)
            xiNLell_b = model.provider.get_model()
            xiNLell_boot[b] = xiNLell_b
            W_b = np.array([dp_b['w_{0:d}'.format(m)] for m in range(self.agem.n_basis)])
            xilin_boot[b] = np.dot(theory.bfunc.T,W_b)/(dp_b['b']**2 + 1e-15)
            if self.verbose:
                self.status_bar(b,n_boot)

        xiNL_16pc = np.percentile(xiNLell_boot,16,axis=0)
        xiNL_50pc = np.percentile(xiNLell_boot,50,axis=0)
        xiNL_84pc = np.percentile(xiNLell_boot,84,axis=0)

        xiNL_16pc = like.wrap_model(xiNL_16pc)
        xiNL_50pc = like.wrap_model(xiNL_50pc)
        xiNL_84pc = like.wrap_model(xiNL_84pc)

        xilin_2p5pc = np.percentile(xilin_boot,2.5,axis=0)
        xilin_16pc = np.percentile(xilin_boot,16,axis=0)
        xilin_50pc = np.percentile(xilin_boot,50,axis=0)
        xilin_84pc = np.percentile(xilin_boot,84,axis=0)
        xilin_97p5pc = np.percentile(xilin_boot,97.5,axis=0)

        del xiNLell_boot,xilin_boot,sample
        gc.collect()

        if self.include_Sig2obs:
            Sig2_temp = np.loadtxt(self.data_file[0])
            if len(Sig2_temp.shape) > 1:
                Sig2_temp = Sig2_temp.T[0]
            data = np.concatenate((Sig2_temp,np.loadtxt(self.data_file[1])))
        else:
            data = np.loadtxt(self.data_file)
        cov_data = np.loadtxt(self.cov_file)
        data,cov_data = like.organize_data(data,cov_data)
        data = like.wrap_model(data)
        err = like.wrap_model(np.sqrt(np.diag(cov_data)))

        theory_name = list(self.info['theory'].keys())[0]
        modify_data = self.info['theory'][theory_name]['modify_data']

        # best fit
        plt.figure(figsize=(7,7))
        plt.xlabel('$s\\,(h^{{-1}}\\,{{\\rm Mpc}})$')
        plt.xlim(64,126)
        YFac = 1.4 if self.sample== 'Euclid-ELG' else 1.0
        if modify_data:
            plt.ylim(-50/YFac,90/YFac)
            ylabel = '$s^{{2}}\\,\\Delta\\xi^{{(\\ell)}}(s)$'
        else:
            plt.ylim(-120/YFac,120/YFac)
            ylabel = '$s^{{2}}\\,\\xi^{{(\\ell)}}(s)$'
        if self.include_Sig2obs:
            off_vert = 50//YFac
            ylabel += '$;\\,\\Sigma^{{(\\ell)2}}+{0:.0f}$'.format(off_vert)
        ylabel += '$\\,\\,(h^{{-1}}{{\\rm Mpc}})^{{2}}$'
        plt.ylabel(ylabel)
        
        for L in range(self.L_Max):
            imin_s = 1 if (modify_data & (L >= 1)) else 0
            stab_show = stab[imin_s:]
            Label = 'best fit (MAP)' if L==0 else None
            plt.plot(stab_show,stab_show**2*xiNLell_best[off+L],'-',lw=1,c=cols[L],label=Label)
            plt.fill_between(stab_show,stab_show**2*xiNL_84pc[off+L],stab_show**2*xiNL_16pc[off+L],color=cols[L],alpha=0.15)

            if L==0:
                data_label = 'toy data' if self.sample == '' else self.sample
            else:
                data_label = None
            plt.errorbar(stab_show,stab_show**2*data[off+L]/like.rescale,yerr=stab_show**2*err[off+L]/like.rescale,
                         c=cols[L],ls='none',capsize=5,marker='o',markersize=4,alpha=0.3,label=data_label)

        if self.include_Sig2obs:
            Sigx = np.linspace(70,70+5*(self.L_Max-1),self.L_Max).astype(int) 
            plt.errorbar(Sigx,data[0]+off_vert,yerr=err[0],
                         marker='o',ls='none',c='teal',capsize=6,markersize=4,lw=1,alpha=0.3)
            plt.errorbar(Sigx+0.5,xiNL_50pc[0]+off_vert,yerr=[xiNL_50pc[0]-xiNL_16pc[0],xiNL_84pc[0]-xiNL_50pc[0]],
                         marker='none',ls='none',c='turquoise',capsize=4,markersize=0,lw=1,alpha=0.9)
            plt.scatter(Sigx+0.5,xiNLell_best[0]+off_vert,
                         marker='*',color='turquoise',s=40,alpha=0.6)

        if self.sample != '':
            plt.text(75,70/YFac,'$\\tt{{AbacusSummit}}$',fontsize=FS)
        plt.text(66,55/YFac,'$\ell = 0$',fontsize=FS2,c=cols[0])
        if self.L_Max > 1:
            plt.text(66,-18/YFac,'$\ell = 2$',fontsize=FS2,c=cols[1])
            if self.L_Max == 3:
                plt.text(96 if self.sample=='DESI-LRG2' else 118,25/YFac,'$\ell = 4$',fontsize=FS2,c=cols[2])
        plt.text(66,-40/YFac,"$\\chi^{{2}}\\,/\\,{{\\rm dof}} = {0:.2f}\\,/\\,{1:d}\\,;\\,p = {2:.3f}$".format(chi2_min,self.dof,pval),
                 fontsize=FS3)

        plt.legend(loc='upper right')
        plt.minorticks_on()
        if save_fig:
            print('Writing to file: '+self.plots_dir + out_file_bf)
            plt.savefig(self.plots_dir + out_file_bf,bbox_inches='tight')
        plt.show()


        # linear reconstruction
        plt.figure(figsize=(7,7))
        plt.xlabel('$r\\,(h^{{-1}}\\,{{\\rm Mpc}})$')
        plt.xlim(25,155)
        plt.ylim(-7,28)
        ylabel = '$r^{{2}}\\,\\xi_{{\\rm lin}}(r)$'
        ylabel += '$\\,\\,(h^{{-1}}{{\\rm Mpc}})^{{2}}$'
        plt.ylabel(ylabel)

        plt.plot(theory.rvals,theory.rvals**2*xilin_best,'-',lw=1.5,c='darkgoldenrod',label='best fit (MAP)')
        plt.plot(theory.rvals,theory.rvals**2*xilin_50pc,'--',lw=1.5,c='darkgoldenrod',label='median')
        plt.fill_between(theory.rvals,theory.rvals**2*xilin_84pc,theory.rvals**2*xilin_16pc,color='goldenrod',alpha=0.2)
        plt.fill_between(theory.rvals,theory.rvals**2*xilin_97p5pc,theory.rvals**2*xilin_2p5pc,color='goldenrod',alpha=0.1)
        if show_gt:
            plt.plot(self.xilin_true[0],self.xilin_true[0]**2*self.xilin_true[1],'-',lw=2,c='crimson',label='ground truth')

        if self.sample != '':
            plt.text(30,-2,'$\\tt{{AbacusSummit}}$',fontsize=FS)
            plt.text(30,-4,self.sample,fontsize=FS3)
        plt.text(34,24,self.agem.cosmo_latex,fontsize=FS)

        plt.axhline(0.0,c='k',ls=':',lw=1)
        ell_vals = '['
        ell_vals += ','.join([str(l) for l in range(0,2*self.L_Max,2)])
        ell_vals += ']'
        plt.text(85,18.5,'linear reconstruction',fontsize=FS2)
        plt.text(85,16.5,'at $z = {0:.1f}$ with $\\ell =$'.format(self.z_eval)+ell_vals,fontsize=FS2)

        plt.legend(loc='upper right')
        plt.minorticks_on()
        if save_fig:
            print('Writing to file: '+self.plots_dir + out_file_lin)
            plt.savefig(self.plots_dir + out_file_lin,bbox_inches='tight')
        plt.show()
        
        if self.verbose:
            self.time_this(start_time)
        
        return
    ##################################
    

    ##################################
    def show_progress(self,show_plots=True):
        """ Utility to display progress while chains are running. 
            To be typically invoked from an independent notebook or terminal. 
            Calls self.display_stats
            -- show_plots: bool (default True), whether or not to display corner and chain plots for each parameter.
        """
        plot_progress(self.info['output'])

        gd_sample = loadMCSamples(str(Path(self.info["output"]).resolve()),settings={'ignore_rows':self.burn_frac})
        mcmc_MAP,mcmc_best,chi2_min,pval,par_show = self.display_stats(gd_sample,include_chi2=True)

        if show_plots:
            plot_param_list = self.all_sampled_params + self.derived_list + ['chi2']
            gdplot = gdplt.get_subplot_plotter(subplot_size=1.2)
            gdplot.settings.num_plot_contours = 3
            gdplot.settings.axes_fontsize = 13
            gdplot.settings.axes_labelsize = 15
            gdplot.settings.title_limit_fontsize = 15
            gdplot.settings.title_limit = 1

            gdplot.triangle_plot([gd_sample], plot_param_list,
                                 filled=[True],contour_colors=['indigo'],legend_loc='upper center',
                                 markers=mcmc_MAP,
                                 marker_args={'c': 'indigo','ls': ':','lw': 1.0,'alpha': 0.6})

            for par_y in range(par_show.size):
                str_y = plot_param_list[par_y]
                for par_x in range(par_y):
                    str_x = plot_param_list[par_x]
                    ax = gdplot.subplots[par_y,par_x]
                    ax.scatter([mcmc_MAP[par_x]],[mcmc_MAP[par_y]],
                               marker='*',s=50,c='white')
                    ax.scatter([mcmc_best[par_x]],[mcmc_best[par_y]],
                               marker='o',s=25,c='peachpuff')

            show_chain = []
            for c in range(self.n_proc):
                # sample = np.loadtxt(os.path.abspath(self.info['output'])+'.'+str(c+1)+'.txt')
                sample = np.loadtxt(str(Path(self.info['output']).resolve())+'.'+str(c+1)+'.txt')
                sample = sample[int(self.burn_frac*sample.shape[0]):]
                sample = sample.T
                sample[-2] /= self.dof
                show_chain.append(sample[par_show])

            for p in range(par_show.size): 
                plt.figure(figsize=(1.5,1.5))
                plt.ylabel(plot_param_list[p])
                for c in range(self.n_proc):
                    plt.plot(show_chain[c][p],'-',lw=0.2)
                mark = mcmc_best[p]/self.dof if p == par_show.size-1 else mcmc_best[p]
                plt.axhline(mark,c='k',ls='--',lw=1)
                if p == par_show.size - 1:
                    plt.axhline(1,c='k',ls=':',lw=1)
                plt.minorticks_on()
                plt.show()                    
        
        return
    ##################################

    ##################################
    def find_indices(self,params_list):
        """ Simple utility to find indices of elements of supplied params_list in self.all_sampled_params + self.derived_list.
            -- params_list: list of str, subset of self.all_sampled_params + self.derived_list.
            Returns list of ints of length len(params_list)
        """
        full_list = np.array(self.all_sampled_params + self.derived_list + [f"w_{b}" for b in range(self.agem.n_basis)])
        ind_list = []
        for param in params_list:
            ind = np.where(full_list == param)[0]
            if ind.size:
                ind_list.append(ind[0])
            else:
                raise Exception("Unrecognised parameter '"+param+"' in find_indices().")
            
        return ind_list        
    ##################################
    
######################################
