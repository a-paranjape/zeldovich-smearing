import numpy as np
import scipy.integrate as syint
import scipy.special as sysp
import sys

from paths import *

sys.path.append(ML_Path)
from mllib import Utilities

# sys.path.append('/home/aseem/simulations_and_mocks/code/N-body/Gadget-Analysis/gadget2-lite/')
# import paths
# sys.path.append('/home/aseem/python_modules/EH_Transfer/')
# from pyEHtransfer import py_TFfit_hmpc as TfEH
from classy import Class
import multiprocessing as mp
from time import time
import gc

#######################################################################

class Constants(object):
    """ Useful constants. """

    # constants of nature
    speed_of_light = 2.99792458e5 # c in km/s
    c_by_H0 = 0.01*speed_of_light # present Hubble radius in Mpc/h
    H0inv = 9.784619421 # 1/H0 in Gyr/h. 
    rhoc = 2.775366e11 # present critical density in (Msun/h)/(Mpc/h)^3; value from PDG2012
    dcsph = 1.686
    
    # useful conversions
    PI = 3.14159265359
    full_sky = 4*PI*(180./PI)**2 # square degrees in the full sky.. ~= 41253
    Mpc_per_km = 3.2408e-20 # Mpc per km
    yr_per_s  = 3.171e-8    # years per second
    
    # utility numerical values
    TINY = 1e-15
    NOTSOTINY = 1e-8

    # no. of physical processors available
    NPROC = np.max([1,mp.cpu_count()//2])

#######################################################################


#######################################################################

# class Utilities(object):
#     """ Useful general-purpose functions. """

#     def __init__(self):
#         self.select_these = np.vectorize(self.select_these_scalar)
#         self.select_not_these = np.vectorize(self.select_not_these_scalar)

#     def heaviside(self,x):
#         return 0.5*(np.sign(x)+1)

#     def wpercentile(self,data,weights=None,percentile=50.0):
#         """ Weighted percentiles of data set (flattened by default).
#              If weights is None, calculates usual percentile using numpy.percentile().
#              If weights is not None, should be of same shape as data containing weights (needn't be normalised).
#              Default percentile is 50 (median), controlled by percentile kwarg.
#              Returns (weighted) percentile of data, without interpolation.
#         """
#         data_flat = data.flatten()
#         if weights is None:
#             out = np.percentile(data_flat,percentile)
#         else:
#             if weights.shape !=  data.shape:
#                 raise TypeError('Incompatible weights and data detected in wpercentiles()')
#             wts = weights.flatten()

#             sorter = np.argsort(data_flat)
#             sorted_data = data_flat[sorter]
#             sorted_wts = wts[sorter]

#             cumsum = np.cumsum(sorted_wts)
#             cutoff = 0.01*percentile*np.sum(sorted_wts)
#             out = sorted_data[cumsum > cutoff][0]
#             del wts,sorter,sorted_data,sorted_wts,cumsum,cutoff

#         del data_flat
#         gc.collect()

#         return out

#     ################################################
#     # Laguerre function utilities
#     ################################################
#     def lag_prefac(self,k,l,x_fid):
#         return sysp.comb(k,l,exact=True)*(-x_fid)**(k-l)
    
#     def lag_nu(self,k,x,x_fid):
#         """ nu_{k}(x) from eqn A7 of Nikakhtar, Sheth & Zehavi (2021). """
#         out = self.lag_prefac(k,0,x_fid)*np.ones_like(x)
#         if k > 0:
#             n = k//2 if (k % 2 == 0) else (k+1)//2
#             for npr in range(1,n):
#                 out += self.lag_prefac(k,2*npr-1,x_fid)*self.lag_mu_odd(npr,x)
#                 out += self.lag_prefac(k,2*npr,x_fid)*self.lag_mu_even(npr,x)
#             out += self.lag_prefac(k,2*n-1,x_fid)*self.lag_mu_odd(n,x)
#             if (k % 2 == 0):
#                 out += self.lag_prefac(k,2*n,x_fid)*self.lag_mu_even(n,x)        
#         return out

#     def lag_mu(self,k,x):
#         """ Wrapper to calculate mu_k(x) of Nikakhtar, Sheth & Zehavi (2021)."""
#         out = np.ones_like(x)
#         if k > 0:
#             if (k % 2 == 0):
#                 n = k//2 
#                 out = self.lag_mu_even(n,x)
#             else:
#                 n = (k+1)//2                
#                 out = self.lag_mu_odd(n,x)
#         return out

#     def lag_mu_bars(self,k,y,ymin,res=1000):
#         """ Volume integrals (3/y^3) int_ymin^y dx x^2 mu_k(x) and (5/y^5) int_ymin^y dx x^4 mu_k(x). 
#              Expect int k, 1-d array y and scalar y_min.
#              res: integer. Controls number of samples between ymin and largest y value.
#         """
#         Dy = y.max() - ymin
#         mubar = np.zeros_like(y)
#         mubarbar = np.zeros_like(y)
#         ipos = np.where(y > ymin)[0]
#         for i in ipos:
#             xvals = np.linspace(ymin,y[i],int(res*(y[i]-ymin)/Dy))
#             xvals_sq = xvals**2
#             dx = xvals[1]-xvals[0]
#             integrand = xvals_sq*self.lag_mu(k,xvals)
#             mubar[i] = np.trapezoid(integrand,dx=dx)
#             integrand *= xvals_sq
#             mubarbar[i] = np.trapezoid(integrand,dx=dx)
#         mubar *= 3/y**3
#         mubarbar *= 5/y**5

#         return mubar,mubarbar
        
#     def lag_mu_bars_explicit(self,k,y,ymin):
#         """ Explicit volume integrals (3/y^3) int_ymin^y dx x^2 mu_k(x) and (5/y^5) int_ymin^y dx x^4 mu_k(x). 
#              Expect int 0 <= k <= 4 , 1-d array y and scalar y_min.
#         """
#         if (k > 4):
#             raise ValueError('Only 0 <= k <= 4 supported in Utilities.lag_mu_bars_explicit().')
#         mubar = np.zeros_like(y)
#         mubarbar = np.zeros_like(y)
#         ymin_by_y3 = (ymin/y)**3
#         ymin_by_y5 = (ymin/y)**5
#         if k == 0:
#             mubar = 1 - ymin_by_y3
#             mubarbar = 1 - ymin_by_y5
#         else:
#             y2 = y**2
#             ymin2 = ymin**2
#             if (k % 2) == 0:
#                 if k == 2:
#                     mubar = 3*y2/5 + 3 - ymin_by_y3*(3*ymin2/5 + 3)
#                     mubarbar = 5*y2/7 + 3 - ymin_by_y5*(5*ymin2/7 + 3)
#                 else:
#                     y4 = y2**2
#                     ymin4 = ymin2**2
#                     mubar = 3*y4/7 + 6*y2 + 15 - ymin_by_y3*(3*ymin4/7 + 6*ymin2 + 15)
#                     mubarbar = 5*y4/9 + 50*y2/7 + 15 - ymin_by_y5*(5*ymin4/9 + 50*ymin2/7 + 15)
#             else:
#                 E1 = sysp.erf(y/np.sqrt(2))
#                 E2 = np.sqrt(2/np.pi)*np.exp(-y2/2)
#                 E1min = sysp.erf(ymin/np.sqrt(2))
#                 E2min = np.sqrt(2/np.pi)*np.exp(-ymin2/2)
#                 y3 = y2*y
#                 y5 = y3*y2
#                 Ebar220 = (3/y3)*(E1 - E1min - (y*E2 - ymin*E2min)) 
#                 Ebar420 = (5/y5)*(3*(E1 - E1min) - (y*E2*(y2+3) - ymin*E2min*(ymin2+3))) 
#                 Ebar221 = (3*y2/5)*Ebar420
#                 Ebar421 = (5/y5)*(15*(E1 - E1min) - (y*E2*(y2**2+5*y2+15) - ymin*E2min*(ymin2**2+5*ymin2+15))) 

#                 Ebar210 = 0.5*((3/y3)*(y2*E1 - ymin2*E1min) - Ebar220)
#                 Ebar211 = 0.25*((3/y3)*(y2**2*E1 - ymin2**2*E1min) - Ebar221)
#                 Ebar410 = 0.25*((5/y5)*(y2**2*E1 - ymin2**2*E1min) - Ebar420)
#                 Ebar411 = (1/6.0)*((5/y5)*(y2**3*E1 - ymin2**3*E1min) - Ebar421)

#                 if k == 1:
#                     mubar = Ebar210 + Ebar211 + Ebar220
#                     mubarbar = Ebar410 + Ebar411 + Ebar420
#                 else:
#                     Ebar222 = (3*y2/5)*Ebar421
#                     Ebar422 = (5/y5)*(105*(E1 - E1min) - (y*E2*(y2**3 + 7*y2**2+35*y2+105) 
#                                                           - ymin*E2min*(ymin2**3 + 7*ymin2**2+35*ymin2+105))) 
                    
#                     Ebar212 = (1/6.0)*((3/y3)*(y2**3*E1 - ymin2**3*E1min) - Ebar222)
#                     Ebar412 = (1/8.0)*((5/y5)*(y2**4*E1 - ymin2**4*E1min) - Ebar422)

#                     mubar = 3*Ebar210 + 6*Ebar211 + Ebar212 + 5*Ebar220 + Ebar221
#                     mubarbar = 3*Ebar410 + 6*Ebar411 + Ebar412 + 5*Ebar420 + Ebar421
            
#         return mubar,mubarbar


#     def lag_mu_even(self,n,x):
#         """ mu_{2n}(x) from eqn 7 of Nikakhtar, Sheth & Zehavi (2021). """
#         f = sysp.factorial2(2*n,exact=True)*sysp.genlaguerre(n,0.5)
#         out = f(-x**2/2)
#         return out
    
#     def lag_mu_odd(self,n,x):
#         """ mu_{2n-1}(x) from eqn 7 of Nikakhtar, Sheth & Zehavi (2021). """
#         out = sysp.factorial2(2*n-1,exact=True)*np.sqrt(np.pi/2)
#         out = out*self.half_lag_recur(n,x) # note x not -x^2/2
#         return out
    
#     def half_lag_recur(self,n,x):
#         """ Calculate L^(1/2)_(n-1/2)(-x^2/2) recursively. """
#         x2by2 = x**2/2
#         if n < 0:
#             raise ValueError('Only n >= 0 allowed in half_lag_recur()')
#         if n==0:
#             return np.sqrt(1/np.pi/x2by2)*sysp.erf(np.sqrt(x2by2))
#         elif n==1:
#             return (2*x2by2 + 1)*self.half_lag_recur(0,x) + (2/np.pi)*np.exp(-x2by2)
#         else:
#             # b L^(a)_(b)(z) = (a+2b-1-z)*L^(a)_(b-1)(z) - (a+b-1)*L^(a)_(b-2)(z)
#             # a=1/2, b=n-1/2, z=-x^2/2
#             # (n-1/2)L^(1/2)_(n-1/2)(-x^2/2) 
#             #  = (1/2+2n-1-1+x^2/2)L^(1/2)_(n-1-1/2)(-x^2/2) - (1/2+n-1/2-1)L^(1/2)_(n-2-1/2)(-x^2/2)
#             #  = (2n-3/2+x^2/2)*L^(1/2)_(n-1-1/2)(-x^2/2) - (n-1)*L^(1/2)_(n-2-1/2)(-x^2/2)
#             out = (2*n-1.5+x2by2)*self.half_lag_recur(n-1,x)
#             out -= (n-1)*self.half_lag_recur(n-2,x)
#             out /= (n-0.5)
#             return out
#     ################################################
        
#     def svd_inv(self,cov,return_eig=False):
#         """ Convenience function to calculate inverse of square matrix using SVD. 
#             Returns inverse and determinant of input matrix and, if requested, array of eigenvalues.
#         """
#         U,s,Vh = linalg.svd(cov)
#         invcov = np.dot(Vh.T,np.dot(np.diag(1.0/s),U.T))
#         if not return_eig:
#             return invcov,np.prod(s)
#         else:
#             return invcov,np.prod(s),s
#     ################################################


#     # def polyfit_custom(self,x,y,deg,sig2=None,start=0):
#     #     """ Polynomial fit of degree deg to data y at locations x.
#     #         Optionally pass squared errors sig2 on y.
#     #         Minimises chi2 = sum_i (y_i - p(x_i))^2 / sig2_i
#     #         with p(x) = sum_alpha a[alpha]*x^alpha
#     #         for alpha=start..deg.
#     #         Returns minimum variance estimator a[alpha] 
#     #         and covariance matrix C[alpha,beta].

#     #         Not very well tested, so use with care. """

#     #     Y = np.zeros(deg+1-start,dtype=float) 

#     #     # Matrix
#     #     F = np.zeros((deg+1-start,deg+1-start),dtype=float)

#     #     if sig2 is None:
#     #         sig2 = np.ones(x.size,dtype=float)

#     #     for alpha in range(start,deg+1):
#     #         Y[alpha-start] = np.sum(y*(x**(alpha))/sig2)
#     #         for beta in range(start,deg+1):
#     #             F[alpha-start,beta-start] = np.sum(x**(alpha+beta)/sig2)
#     #             F[beta-start,alpha-start] = F[alpha-start,beta-start]

#     #     Y = Y.T
#     #     U,s,Vh = linalg.svd(F)
#     #     Cov = np.dot(Vh.T,np.dot(np.diag(1.0/s),U.T))
#     #     a_minVar = Cov.dot(Y)

#     #     return np.squeeze(np.asarray(a_minVar)),np.asarray(Cov)


#     # def gen_latin_hypercube(self,Nsamp=10,dim=2,symmetric=True,param_mins=None,param_maxs=None,
#     #                         rng=None):
#     #     """ Generate Latin hypercube sample (symmetric by default). 
#     #          Either param_mins and param_maxs should both be None or both be array-like of shape (dim,). 
#     #         -- rng: either None or instance of numpy.random.RandomState(). Default None.
#     #          Code from FirefoxMetzger's answer at
#     #          https://codereview.stackexchange.com/questions/223569/generating-latin-hypercube-samples-with-numpy
#     #          See https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.49.7292&rep=rep1&type=pdf
#     #          for some ideas reg utility of symmetric Latin hypercubes.
#     #          Returns array of shape (Nsamp,dim) with values in range (0,1) or respective minimum to maximum values.
#     #     """
#     #     if (param_mins is not None): 
#     #         if (param_maxs is None):
#     #             raise TypeError("param_mins and param_maxs should both be None or array-like of shape (dim,)")
#     #         if len(param_mins) != dim:
#     #             raise TypeError("len(param_mins) should be equal to dim")
#     #     if (param_maxs is not None) :
#     #         if(param_mins is None):
#     #             raise TypeError("param_mins and param_maxs should both be None or array-like of shape (dim,)")
#     #         if len(param_maxs) != dim:
#     #             raise TypeError("len(param_maxs) should be equal to dim")

#     #     rng_use = rng if rng is not None else np.random.RandomState()

#     #     if symmetric:
#     #         available_indices = [set(range(Nsamp)) for _ in range(dim)]
#     #         samples = []

#     #         # if Nsamp is odd, we have to choose the midpoint as a sample
#     #         if Nsamp % 2 != 0:
#     #             k = Nsamp//2
#     #             samples.append([k] * dim)
#     #             for idx in available_indices:
#     #                 idx.remove(k)

#     #         # sample symmetrical pairs
#     #         for _ in range(Nsamp//2):
#     #             sample1 = list()
#     #             sample2 = list()

#     #             for idx in available_indices:
#     #                 k = rng_use.choice(np.array(list(idx)),size=1,replace=False)[0]# random.sample(idx, 1)[0]
#     #                 sample1.append(k)
#     #                 sample2.append(Nsamp-1-k)
#     #                 idx.remove(k)
#     #                 idx.remove(Nsamp-1-k)

#     #             samples.append(sample1)
#     #             samples.append(sample2)

#     #         samples = np.array(samples)/(1.0*Nsamp)
#     #     else:
#     #         samples = np.array([rng_use.permutation(Nsamp) for i in range(dim)])/(1.0*Nsamp)
#     #         samples = samples.T

#     #     if (param_mins is not None):
#     #         for d in range(dim):
#     #             samples[:,d] *= (param_maxs[d] - param_mins[d])
#     #             samples[:,d] += param_mins[d]

#     #     return samples


#     # def model_poly(self,x,params):
#     #     """ Polynomial in x, of degree n = len(params)-1. 
#     #         Expect x to be scalar or numpy array (arbitrary dimensions).
#     #         Expect params to be list ordered as a_0,a_1..a_n
#     #         where y = sum_i a_i x**i.
#     #         Returns array y of shape x.shape.
#     #     """
#     #     if len(params) == 1:
#     #         out = params[0]
#     #     else:
#     #         out = np.sum(np.array([params[i]*x**i for i in range(len(params))]),axis=0)
#     #     return out


#     ############################################################
#     def time_this(self,start_time,logfile=None):
#         totsec = time() - start_time
#         minutes = int(totsec/60)
#         seconds = totsec - 60*minutes
#         self.print_this("{0:d} min {1:.2f} seconds\n".format(minutes,seconds),logfile)
#         return
#     ############################################################


#     def write_to_file(self,filestring,seq):
#         """ Opens filestring for appending and writes tab-separated list seq to it. """
#         with open(filestring,'a') as f:
#             s = "{0:.6e}".format(seq[0])
#             for i in range(1,len(seq)):
#                 s += "\t" + "{0:.6e}".format(seq[i])
#             s += "\n"
#             f.write(s)
#         return


#     def write_structured(self,filestring,recarray,dlmt=' '):
#         """ Opens filestring for appending and writes 
#         rows of structured array recarray to it.
#         """
#         with open(filestring,'a') as f:
#             for row in recarray:
#                 f.write(dlmt.join([str(item) for item in row]))
#                 f.write('\n')
#         return


#     def writelog(self,logfile,strng,overwrite=False):
#         """ Convenience function for pipe-safety. """
#         app_str = 'w' if overwrite else 'a'
#         with open(logfile,app_str) as g:
#             g.write(strng)
#         return

#     # def time_this_log(self,start_time,logfile):
#     #     totsec = time() - start_time
#     #     minutes = int(totsec/60)
#     #     seconds = totsec - 60*minutes
#     #     self.writelog(logfile,"{0:d} min {1:.2f} seconds\n".format(minutes,seconds))
#     #     return


#     ############################################################
#     def print_this(self,print_string,logfile,overwrite=False):
#         """ Convenience function for printing to logfile or stdout."""
#         if logfile is not None:
#             self.writelog(logfile,print_string+'\n',overwrite=overwrite)
#         else:
#             print(print_string)
#         return
#     ############################################################


#     ############################################################
#     def status_bar(self,n,ntot,freq=100,text='done'):
#         """ Print status bar with user-defined text and frequency. """
#         if freq > ntot:
#             freq = ntot
#         if ((n+1) % int(1.0*ntot/freq) == 0):
#             frac = (n+1.)/ntot
#             sys.stdout.write('\r')
#             sys.stdout.write("[%-20s] %.f%% " % ('.'*int(frac*20),100*frac) + text)
#             sys.stdout.flush()
#         if n==ntot-1: self.print_this('',None)
#         return
#     ############################################################

#     def select_these_scalar(self,elmt,wanted_elmt):
#         """ Select elements from elmt which belong to wanted_elmt.
#              For fast evaluation with large arrays, ensure wanted_elmt is
#              a set by applying set() to the array.
#              Returns boolean array of shape elmt.
#         """
#         return elmt in wanted_elmt

#     def select_not_these_scalar(self,elmt,wanted_elmt):
#         """ Select elements from elmt which *do not* belong to wanted_elmt.
#              For fast evaluation with large arrays, ensure wanted_elmt is
#              a set by applying set() to the array.
#              Returns boolean array of shape elmt.
#         """
#         return elmt not in wanted_elmt

#######################################################################

class Cosmology(Constants,Utilities):
    """ Useful functions for cosmology and extra-Galactic astrophysics. """
    ############################################################
    ############################################################
    def __init__(self,Om=0.3063,Ob=0.0484,Ok=0.0,hubble=0.6781,Tcmb=2.7255,
                 wDE0=-1.0,wDEa=0.0,sig8=0.815,As=None,kpivot=0.05,ns=0.9677,
                 N_ur=3.044,N_ncdm=0,m_ncdm=0.0,
                 verbose=True,logfile=None): 
        """ Initialise various constants. 
            -- Om : Total matter density parameter (DM + baryons)
            -- Ob : Baryonic matter density parameter
            -- Ok : Curvature parameter
            -- hubble : little Hubble: H0 = 100h km/s/Mpc
            -- Tcmb : Current CMB temperature
            -- wDE0,wDEa : Dark energy equation of state at z=0 (-1,0 for cosmological constant)
            -- sig8: power spectrum normalisation sigma8
            -- As: power spectrum normalisation A_s. If not None, overrides sig8. 
                      In this case, self.sig8 will be set to actual value of sigma8.
            -- kpivot: pivot scale in 1/Mpc. Only used if As is not None.
            -- ns: scalar spectral index
            -- N_ur: number of ultra-relativistic species
            -- N_ncdm: number of non-cold dark matter species
            -- m_ncdm: mass(es) of non-cold dark matter species in eV.
                       if N_ncdm==1, m_ncdm should be a single float.
                       if N_ncdm > 1, m_ncdm should be a list of N_ncdm floats
        
            Methods:
            -- calc_xi_lin()
            -- calc_xi2_lin()
            -- calc_xi_lin_dot()
            -- scale_to_redshift()
            -- redshift_to_scale()
            -- dHub0()
            -- EHub()
            -- dHubz()
            -- EHub_inv()
            -- pool_it_1d()
            -- chiConf_scalar()
            -- chiConf()
            -- test_redshift_pair()
            -- chiConf_analytic_EdS()
            -- chiConf_analytic_pureRad()
            -- chiConf_analytic_Milne()
            -- chiConf_analytic_dS()
            -- chiConf_analytic_MatRad()
            -- chiConf_analytic_Curv_MatRadLam()
            -- rCom()
            -- dLum()
            -- dAng()
            -- dVdz()
            -- age_integrand()
            -- age_scalar()
            -- age()
            -- lookback()
            -- Wth_scalar()
            -- Wth()
            -- Wthpr_scalar()
            -- Wthpr()
            -- Growth()
            -- dcsph_lcdm()
            -- massfuncbiasTinker()
            -- massfuncbiasTinker_norm()
            -- massfuncbiasST()
            -- massfuncbias_thresh()
            -- massfuncbiasESP()
        """

        Constants.__init__(self)
        Utilities.__init__(self)

        self.verbose = verbose
        self.logfile = logfile
        if self.verbose:
            self.print_this("... initialising Cosmology()",self.logfile)

        self.Om = Om
        self.Ok = Ok
        self.Ob = Ob
        self.hubble = hubble
        self.Tcmb = Tcmb
        self.wDE0 = wDE0
        self.wDEa = wDEa
        self.sig8 = sig8
        self.As = As
        self.kpivot = kpivot
        self.ns = ns
        self.N_ur = N_ur
        self.N_ncdm = N_ncdm
        self.m_ncdm = m_ncdm

        if np.fabs(self.wDEa) > self.NOTSOTINY:
            raise NotImplementedError("wa dependence not yet implemented!")
        
        if self.verbose:
            if self.As is not None:
                self.print_this("... ... detected As normalisation, will override sig8.",self.logfile)

        self.Odm = self.Om - self.Ob # dark matter density parameter
        self.Orad = 4.158e-5*(self.Tcmb/2.7255)**4/self.hubble**2 # radiation density parameter
        self.OLam = 1.0 - self.Om - self.Orad - self.Ok # cosmological constant (or dark energy) parameter

        # flag to switch to flat-space functions. can speed things up in some places.
        self.FLAT = True if np.fabs(self.Ok) < self.NOTSOTINY else False

        self.Wth = np.vectorize(self.Wth_scalar)
        self.Wthpr = np.vectorize(self.Wthpr_scalar)

        if self.verbose:
            self.print_this("... ... calculating linear power spectrum at present epoch",self.logfile)
            self.print_this("... ... cosmology (Om,OLam,Ob,h,sig8,ns) = ({0:.4f},{1:.4f},{2:.5f},{3:.4f},{4:.4f},{5:.4f})"
                            .format(self.Om,self.OLam,self.Ob,self.hubble,self.sig8,self.ns),
                            self.logfile)
        kmin = 1e-5
        kmax = 1e3

        if self.verbose:
            self.print_this("... ... using CLASS",self.logfile)
        z_out = 0.0 # MAY NEED TO ALTER THIS so as not to have radiation at late times            
        params = {'output': 'mPk',
                  'n_s': self.ns, 
                  'h': self.hubble,
                  'Omega_b': self.Ob,
                  'Omega_cdm': self.Odm,
                  'Omega_k': self.Ok,
                  'T_cmb': self.Tcmb,
                  'N_ur': self.N_ur,
                  'N_ncdm': self.N_ncdm,
                  'P_k_max_h/Mpc': kmax,
                  'z_pk': z_out
                  }
        if (np.fabs(self.wDE0 + 1) > self.NOTSOTINY) | (np.fabs(self.wDEa) > self.NOTSOTINY):
            if self.verbose:
                self.print_this("... ... ... dark energy EoS w0 = {0:.3f}".format(self.wDE0),self.logfile)
            params['Omega_Lambda'] = 0 # force Omega_fld to be activated
            params['w0_fld'] = self.wDE0
            params['wa_fld'] = self.wDEa
        if self.As is not None:
            params['A_s'] = self.As
            # include kpivot here
        else:
            params['sigma8'] = self.sig8
        if self.N_ncdm > 0:
            if self.N_ncdm > 1:
                params['m_ncdm'] = ', '.join([str(self.m_ncdm[n]) for n in range(self.N_ncdm)])
            else:
                params['m_ncdm'] = self.m_ncdm
        cosmo = Class()
        cosmo.set(params)            
        cosmo.compute() # bottle-neck step
        self.nk_lin = 1000
        self.ktab_lin = np.logspace(np.log10(kmin),np.log10(kmax),self.nk_lin) # k in h/Mpc
        ktab_Mpc = self.ktab_lin*self.hubble
        self.Dlin = np.zeros_like(self.ktab_lin)
        for k in range(self.nk_lin):
            self.Dlin[k] = cosmo.pk(ktab_Mpc[k],z_out) # i/p k in 1/Mpc; o/p in Mpc^3
            if self.verbose & (self.logfile is None):
                self.status_bar(k,self.nk_lin)
        self.Dlin *= self.hubble**3 # convert to (h-1Mpc)^3
        self.Dlin *= self.ktab_lin**3/(2*np.pi**2)
        self.nk_lin = self.ktab_lin.size
        # Consider re-setting Orad and Olam at this stage
        cosmo.struct_cleanup()
        cosmo.empty()
        del cosmo,ktab_Mpc
        gc.collect()

        self.ln_ktab_lin = np.log(self.ktab_lin)
        if self.As is None:
            if self.verbose:
                self.print_this("... using sig8 normalisation",self.logfile)
            Amp = self.sig8**2/np.trapezoid(self.Dlin*self.Wth(self.ktab_lin*8)**2,x=self.ln_ktab_lin)
            self.Dlin *= Amp
            del Amp
        else:
            if self.verbose:
                self.print_this("... using As normalisation",self.logfile)
            # if self.Pklin != 'class':
            #     # normalise using
            #     # k3Pdelta = D^2(a) (3/(5Om0))^2 (kc/H0)^4 T^2(k) * k3Pcurvprim
            #     # assume T(kpivot) = 1 and set D(a)=1 since this is not included at this stage in above calculations
            #     kph = self.kpivot/self.hubble
            #     Dlin_pivot = np.interp(self.kpivot/self.hubble,self.ktab_lin,self.Dlin)
            #     Dcurv_pivot = Dlin_pivot*(5*self.Om/3)**2/(kph*self.c_by_H0)**4
            #     Amp = self.As/Dcurv_pivot
            #     self.Dlin *= Amp
            #     del Amp,kph,Dlin_pivot,Dcurv_pivot
            self.sig8 = np.sqrt(np.trapezoid(self.Dlin*self.Wth(self.ktab_lin*8)**2,x=self.ln_ktab_lin))
            if self.verbose:
                self.print_this("... ln(1e10*As),sig8 = {0:.3f},{1:.3f}".format(np.log(1e10*self.As),self.sig8),self.logfile)
        if self.verbose:
            self.print_this("... done",self.logfile)

    ############################################################
    ############################################################


    ############################################################
    #////////////////////////////////////////////////////////////////////////
    # Linear WDM power spectrum 
    # (Refs. Bode, Ostriker \& Turok 2001, eqns A8-A9;
    #        Schneider et al. 2012, eqn 5 actually Viel et al. 2005)
    #////////////////////////////////////////////////////////////////////////
    def linearpower_wdm(self,mdm=2.0,bode=0,gdm=1.5):
        """ Output linear matter spectrum per lnk for Warm DM initial conditions (normalised to CDM ICs at large scales).
            mdm: WDM particle mass in keV.
            Returns Dk[],mFS,mHM. """

        # if not eh:
        #     Dlin,ktab,dlnk = linearpower(Om,sig8,ns,cosmo,verbose=verbose)
        # else:
        #     Dlin,ktab,dlnk = linearpower_EH(Om,Ob,hubble,sig8,ns,verbose=verbose,kmin=kmin,dlnk=dlnk,kmax=kmax)
    
        if self.verbose:
            self.print_this("... using Warm DM ICs with m_DM = {0:.2f} keV".format(mdm),self.logfile)
        if bode:
            if self.verbose:
                self.print_this(".. . Ref. Bode et al. 2001",self.logfile)
            norm = 0.048
            nu = 1.2
            # # Raul used norm = 0.05, nu = 1.0
            # norm = 0.05
            # nu = 1.0 
            alpha = (norm*(self.Om/0.4)**0.15*(self.hubble/0.65)**1.3*(1/mdm)**1.15*(1.5/gdm)**0.29)
        else:
            if self.verbose:
                self.print_this(".. . Ref. Viel et al. 2005",self.logfile)
            alpha = (0.049*(self.Om/0.25)**0.11*(self.hubble/0.7)**1.22*(1/mdm)**1.11)
            nu = 1.12

        mFS = (4*np.pi/3.0)*self.Om*self.rhoc*(alpha/2)**3

        mHM = (2*np.pi*(2**(0.2*nu)-1)**(-0.5/nu))**3
        mHM *= mFS

        Tmod = (1 + (alpha*self.ktab_lin)**(2*nu))**(-5/nu)
        Dlin_wdm = self.Dlin*Tmod**2

        return Dlin_wdm,mFS,mHM
    ############################################################
    
    
    ############################################################
    #////////////////////////////////////////////////////////////////////////
    # Linear (generic) Acoustic DM power spectrum 
    # -- playing with modifications of Bode+
    #////////////////////////////////////////////////////////////////////////
    def linearpower_adm(self,rAc=300.0,kD=15.0):
        """ Output linear matter spectrum per lnk for generic Acoustic DM initial conditions. (Normalise to CDM ICs at large scales.)
            rAc: sound horizon (kpc/h); kD: diffusion damping scale (h/Mpc)
            Model: Transfer function P_adm(k) = P_cdm(k)*exp(-(k/kD)^2)*Cos^2[k*rAc/sqrt(3)]
            Returns Dk[]. """

        if self.verbose:
            self.print_this("... using Acoustic DM ICs with rAc = {0:.2f}kpc/h; kD = {1:.2f}h/Mpc".format(rAc,kD),self.logfile)

        T2mod = np.exp(-(self.ktab_lin/kD)**2)*np.cos(self.ktab_lin*rAc*1e-3/np.sqrt(3.))**2
        Dlin_adm = self.Dlin*T2mod

        return Dlin_adm
    ############################################################


    ############################################################
    #////////////////////////////////////////////////////////////////////////
    # Linear (generic) Ballistic DM power spectrum 
    #////////////////////////////////////////////////////////////////////////
    def linearpower_bdm(self,kAc=1.0,kMod=0.75,kpiv=1.0,kD=30.0):
        """ Output linear matter spectrum per lnk for generic Ballistic DM initial conditions (normalised to CDM ICs at large scales).
            kAc: acoustic scale (h/Mpc)
            kMod: modulation scale (h/Mpc) [kMod < kAc]
            kpiv: pivot scale (h/Mpc)
            kD: diffusion damping scale (h/Mpc) [only for numerical stability]
            Model: Transfer function P_bdm(k) = norm*P_cdm(k)*Theta(k-kstar)*(k/kstar)^2*exp(-(k/kD)^2)*Cos^2[k/kstar]/(ln[k/(8keq)])^2
            norm chosen s.t. P_bdm(kstar) = P_cdm(kstar)
            Returns Dk[]. """

        # if self.verbose:
        #     self.print_this("... EXPERIMENTAL!: using Ballistic DM ICs with kstar = {0:.2f}h/Mpc; kD = {1:.2f}h/Mpc".format(kstar,kD),self.logfile)

        # keq = 0.0128*(self.Om/0.25)*(self.hubble/0.7) # h/Mpc
        # T2mod = np.ones(self.nk_lin,dtype=float)
        # ind = np.where(self.ktab_lin > kstar)[0]
        # norm = np.exp((kstar/kD)**2)/np.cos(1)**2*(np.log(kstar/(8*keq)))**2
        # T2mod[ind] *= norm*(self.ktab_lin[ind]/kstar)**2*(np.exp(-(self.ktab_lin[ind]/kD)**2)*
        #                                                   np.cos(self.ktab_lin[ind]/kstar)**2/(np.log(self.ktab_lin[ind]/(8*keq)))**2)
        # Dlin_bdm = self.Dlin*T2mod
        
        if self.verbose:
            print_str = "... using Ballistic DM ICs with"
            print_str += " kAc = {0:.2f} h/Mpc; kMod = {1:.2f} h/Mpc; kpiv = {2:.2f}h/Mpc; kD = {3:.2f}h/Mpc".format(kAc,kMod,kpiv,kD)
            self.print_this(print_str,self.logfile)

        T2cdm = 2*np.pi**2*self.Dlin/self.ktab_lin**(3+self.ns)
        ind = np.where(self.ktab_lin > kpiv)[0]
        kp = ind.min()
        dlnk = np.log(self.ktab_lin[kp]/self.ktab_lin[kp-1])
        slope = (T2cdm[kp]-T2cdm[kp-1])/dlnk
        intercept = (T2cdm[kp-1]*np.log(self.ktab_lin[kp]) - T2cdm[kp]*np.log(self.ktab_lin[kp-1]))/dlnk
        T2_piv = slope*np.log(kpiv) + intercept
        dlnT2dlnk_piv = np.log(T2cdm[kp]/T2cdm[kp-1])/dlnk
        
        kpbykAc = kpiv/kAc
        kpbykMod = kpiv/kMod
        A = T2_piv/(np.cos(kpbykAc)*np.sin(kpbykMod) + self.TINY)**2
        alpha = dlnT2dlnk_piv + 2*(kpbykAc*np.tan(kpbykAc) - kpbykMod/np.tan(kpbykMod))
        
        T2bdm = A*(self.ktab_lin/kpiv)**alpha*(np.cos(self.ktab_lin/kAc)*np.sin(self.ktab_lin/kMod))**2*np.exp(-(self.ktab_lin/kD)**2)
        T2 = T2cdm.copy()
        T2[ind] = T2bdm[ind]
        
        Dlin_bdm = self.ktab_lin**(3+self.ns)*T2/(2*np.pi**2)

        return Dlin_bdm
    ############################################################


    ############################################################
    #////////////////////////////////////////////////////////////////////////
    # NonLinear CDM power spectrum: HALOFIT
    #////////////////////////////////////////////////////////////////////////
    def nonlinearpower(self,z=0.0,model=0):
        """ Output HALOFIT nonlinear matter spectrum per lnk 
            (Dk=k3Pk/2pi2) using Smith et al (2003) 
            or Takahashi et al (2012). """

        if self.verbose:
            self.print_this("Nonlinear power spectrum using HALOFIT: assumes flat LCDM.",self.logfile)
            model_str = 'Smith et al (2003)' if model==0 else 'Takahashi et al (2012)'
            self.print_this("HALOFIT parameters from "+model_str,self.logfile)

        Dlin_z = self.Dlin*(self.Growth(z)/self.Growth())**2
        Om_z = self.Om*(1+z)**3/self.EHub(z)**2

        #kseek = np.arange(0.25,0.5025,0.0025)
        kseek = np.arange(0.125,0.5025,0.0025) if z < 0.3 else np.arange(0.5025,8.5025,0.0025)
        # sig^2(R) = int dlnk Dlin Wgauss(kR)**2
        sig2gauss = np.trapezoid(np.exp(-np.outer(1/kseek,self.ktab_lin)**2)*Dlin_z,x=self.ln_ktab_lin, axis=1)
        # ksig satisfies sig^2(1/ksig) = 1
        index = np.where(sig2gauss >= 1.0)[0][0]

        ksig = ((kseek[index]*(1-sig2gauss[index-1]) - kseek[index-1]*(1-sig2gauss[index]))
                /(sig2gauss[index]-sig2gauss[index-1]))

        y = self.ktab_lin/ksig
        ysq = y**2
        # n = -3 - dlnsig^2/dlnR
        #   = -3 + 2 int dlnk Dlin (k/ksig)^2 Wgauss(k/ksig)**2
        np3 = 2*np.trapezoid(ysq*np.exp(-ysq)*Dlin_z,x=self.ln_ktab_lin)
        n = np3 - 3
        n2 = n**2
        n3 = n2*n
        n4 = n2**2
        # C = - d^2lnsig^2/dlnR^2
        #   = 2(n+3) + (n+3)^2 - 4 int dlnk Dlin (k/ksig)^4 Wgauss(k/ksig)**2
        C = (np3+1)**2 - 1 - 4*np.trapezoid(ysq**2*np.exp(-ysq)*Dlin_z,x=self.ln_ktab_lin)

        # Parameters from Smith etal 03 (model=0) or Takahashi etal 12 (model=1)
        # Eqns C9-C16 of S+03 or A6-A13 of T+12
        (lga,lgb,lgc,alpha,beta,gamma,mu,lgnu) = ((1.4861 + 1.8369*n + 1.6762*n2 + 0.7940*n3 + 0.1670*n4 - 0.6206*C,
                                                   0.9463 + 0.9466*n + 0.3084*n2 - 0.9400*C,
                                                   -0.2807 + 0.6669*n + 0.3214*n2 - 0.0793*C,
                                                   1.3884 + 0.3700*n - 0.1452*n2,
                                                   0.8291 + 0.9854*n + 0.3401*n2,
                                                   0.8649 + 0.2989*n + 0.1631*C,
                                                   sysp.exp10(-3.5442 + 0.1908*n),
                                                   0.9589 + 1.2857*n)
                                                  if model==0 else
                                                  (1.5222 + 2.8553*n + 2.3706*n2 + 0.9903*n3 + 0.2250*n4 - 0.6038*C,
                                                   -0.5642 + 0.5864*n + 0.5716*n2 - 1.5474*C,
                                                   0.3698 + 2.0404*n + 0.8161*n2 + 0.5869*C,
                                                   np.abs(6.0835 + 1.3373*n - 0.1959*n2 - 5.5274*C),
                                                   2.0379 - 0.7354*n + 0.3157*n2 + 1.2490*n3 + 0.3980*n4 - 0.1682*C,
                                                   0.1971 - 0.0843*n + 0.8460*C,
                                                   0.0,
                                                   5.2105 + 3.6902*n)
                                                  )
        a = sysp.exp10(lga)
        b = sysp.exp10(lgb)
        c = sysp.exp10(lgc)
        nu = sysp.exp10(lgnu)

        # Eqn C18 of S+03 (also used by T+12)
        f1 = Om_z**(-0.0307)
        f2 = Om_z**(-0.0585)
        f3 = Om_z**(0.0743)

        # Functional forms are identical in S+03 and T+12
        # Eqn C2 of S+03
        DQ = Dlin_z*np.exp(-0.25*y-0.125*ysq)*(1+Dlin_z)**beta/(1+alpha*Dlin_z)

        # Eqns C3-C4 of S+03
        DH = a*y**(2+3*f1)/(ysq+mu*y+nu)/(1+b*y**f2+(c*f3*y)**(3-gamma))

        return DQ+DH
    ############################################################
    
    
    ############################################################
    def calc_linearpoint(self,rmin=75,rmax=105,nr=1500,return_dipht=False):
        """ Calculate peak, dip and linear point scale from linear 2pcf. Returns values in Mpc."""
        Scales_Fine = np.linspace(rmin,rmax,nr)
        Xi_Linear_Fine = self.calc_xi_lin(Scales_Fine) 
        dr = (rmax-rmin)/nr
        ii = np.argmax(Xi_Linear_Fine)
        jj = np.argmin(Xi_Linear_Fine)
        if ii < jj:
            if (ii < 0.1*nr) & (jj <= 0.9*nr):
                # peak is too far to left
                output = self.calc_linearpoint(rmin=rmin+5,rmax=rmax,nr=nr,return_dipht=return_dipht)
                if return_dipht:
                    r_pk,r_dip,r_LP,dip_ht = output
                else:
                    r_pk,r_dip,r_LP = output
            elif (jj > 0.9*nr) & (ii >= 0.1*nr):
                # dip is too far to right
                output = self.calc_linearpoint(rmin=rmin,rmax=rmax-5,nr=nr,return_dipht=return_dipht)
                if return_dipht:
                    r_pk,r_dip,r_LP,dip_ht = output
                else:
                    r_pk,r_dip,r_LP = output
            else:
                # range is too broad
                output = self.calc_linearpoint(rmin=rmin+5,rmax=rmax-5,nr=nr,return_dipht=return_dipht)
                if return_dipht:
                    r_pk,r_dip,r_LP,dip_ht = output
                else:
                    r_pk,r_dip,r_LP = output
        else:
            # if dr > 0.1:
            #     # needs testing
            #     nfit = 2
            #     coeffs = np.polyfit(Scales_Fine[ii-nfit:ii+nfit+1],Xi_Linear_Fine[ii-nfit:ii+nfit+1],2)
            #     coeffs = coeffs[::-1]
            #     r_pk = -0.5*coeffs[1]/coeffs[2] if coeffs[2] != 0.0 else Scales_Fine[ii]
            #     coeffs = np.polyfit(Scales_Fine[jj-nfit:jj+nfit+1],Xi_Linear_Fine[jj-nfit:jj+nfit+1],2)
            #     coeffs = coeffs[::-1]
            #     r_dip = -0.5*coeffs[1]/coeffs[2] if coeffs[2] != 0.0 else Scales_Fine[jj]
            #     dip_ht = coeffs[0] - 0.25*coeffs[1]**2/coeffs[2] if coeffs[2] != 0.0 else Xi_Linear_Fine[jj]
            # else:
            r_pk = Scales_Fine[ii]
            r_dip = Scales_Fine[jj]
            dip_ht = Xi_Linear_Fine[jj]

            r_pk /= self.hubble
            r_dip /= self.hubble
            r_LP = 0.5*(r_pk+r_dip)

        if not return_dipht:
            return r_pk,r_dip,r_LP
        else:
            return r_pk,r_dip,r_LP,dip_ht

    ############################################################

    ############################################################
    def calc_xi_lin(self,rtab):
        """ Use linear power spectrum to calculate linear 2-pt correlation function."""
        if len(rtab.shape) != 1:
            raise TypeError("Only 1-d arrays supported for rtab in calc_xi_lin().")

        if (rtab.max() > 150.0) & self.verbose:
            self.print_this("WARNING! xi_lin may be inaccurate for argument >~ 150Mpc/h: detected {0:.1f}Mpc/h"
                            .format(rtab.max()),self.logfile)

        #      xiL ~1% converged at r=100Mpc/h for 25k
        nk_int = 150000 # 50000 
        ktab_int = np.logspace(np.log10(self.ktab_lin.min()),np.log10(self.ktab_lin.max()),nk_int)
        Dlin_int = np.interp(ktab_int,self.ktab_lin,self.Dlin)
        dlnk_int = np.log(ktab_int[1]/ktab_int[0])

        xi_lin = np.trapezoid(np.sinc(np.outer(rtab,ktab_int)/np.pi)*Dlin_int,dx=dlnk_int,axis=1)
        del ktab_int,Dlin_int,dlnk_int
        gc.collect()
        return xi_lin
    ############################################################

    ############################################################
    def calc_xi2_lin(self,rtab):
        """ Use linear power spectrum to calculate j2-weighted linear 2-pt correlation function."""
        if len(rtab.shape) != 1:
            raise TypeError("Only 1-d arrays supported for rtab in calc_xi2_lin().")

        if (rtab.max() > 150.0):
            print ("WARNING! xi2_lin may be inaccurate for argument >~ 150Mpc/h: detected {0:.1f}Mpc/h"
                   .format(rtab.max()))

        #      xiL <~1% converged at r=100Mpc/h for 25k
        nk_int = 50000 
        ktab_int = np.logspace(np.log10(self.ktab_lin.min()),np.log10(self.ktab_lin.max()),nk_int)
        Dlin_int = np.interp(ktab_int,self.ktab_lin,self.Dlin)
        dlnk_int = np.log(ktab_int[1]/ktab_int[0])
        kr = np.outer(rtab,ktab_int)

        xi2_lin = np.trapezoid((self.Wth(kr)-np.sinc(kr/np.pi))*Dlin_int,dx=dlnk_int,axis=1)
        # recall j2(y) = (3/y) j1(y) - j0(y) = Wth(y) - j0(y)
        del ktab_int,Dlin_int,dlnk_int,kr
        gc.collect()
        return xi2_lin
    ############################################################

    ############################################################
    def calc_xi_lin_dot(self,rtab):
        """ Use linear power spectrum to calculate negative log-derivative of linear 2-pt correlation function,
             xiL_dot = - d ln xiL(r) / d ln r.
        """
        if len(rtab.shape) != 1:
            raise TypeError("Only 1-d arrays supported for rtab in calc_xi_lin_dot().")
        # if rtab.max() > 20.0:
        #     print ("WARNING! xi_lin_dot may be wildly inaccurate for argument >~ 20Mpc/h: detected {0:.1f}Mpc/h"
        #            .format(rtab.max()))

        # xiLdot ~1% converged at r=10Mpc/h for 300k, at r=20Mpc/h for 600k
        nk_int = 600000 
        ktab_int = np.logspace(np.log10(self.ktab_lin.min()),np.log10(self.ktab_lin.max()),nk_int)
        Dlin_int = np.interp(ktab_int,self.ktab_lin,self.Dlin)
        dlnk_int = np.log(ktab_int[1]/ktab_int[0])

        kr = np.outer(rtab,ktab_int)
        xi_lin = np.trapezoid(np.sinc(kr/np.pi)*Dlin_int,dx=dlnk_int,axis=1)
        m_dxi_lin_dlnr = (1/3.)*np.trapezoid(kr**2*self.Wth(kr)*Dlin_int,dx=dlnk_int,axis=1)
        # recall dj0(x)/dx = -j1(x) = -(x/3) Wth(x), so above is -dxiL/dlnr
        del ktab_int,Dlin_int,dlnk_int,kr
        gc.collect()

        return m_dxi_lin_dlnr/(xi_lin + self.TINY)
    ############################################################

    ############################################################
    def scale_to_redshift(self,scale):
        """ Convert scale factor to redshift. """
        redshift = 1.0/scale - 1.0
        return redshift
    ############################################################

    ############################################################
    def redshift_to_scale(self,redshift):
        """ Convert redshift to scale factor. """
        scale = 1.0/(1+redshift)
        return scale
    ############################################################

    ############################################################
    def dHub0(self,z):
        """ Distance according to Hubble law, in units of Mpc/h. """
        dist = self.c_by_H0*z
        return dist
    ############################################################
        
    ############################################################
    def EHub(self,z):
        """ H(z) / H0. """ 
        Ez = self.Om*(1+z)**3 + self.Orad*(1+z)**4 + self.Ok*(1+z)**2 + self.OLam*(1+z)**(3*(1+self.wDE0))
        Ez = np.sqrt(Ez)
        # can be generalised to w0,wa cosmologies.
        return Ez
    ############################################################

    ############################################################
    def dHubz(self,z,dz):
        """ Distance according to Hubble law at redshift z, for redshift interval dz, in units of Mpc/h. """
        dist = self.c_by_H0*dz/self.EHub(z)
        return dist
    ############################################################

    ############################################################
    def EHub_inv(self,z):
        """ Convenience function. """
        return 1.0/self.EHub(z)
    ############################################################


    ############################################################
    def pool_it_1d(self,func_obj,zipped_args,arg_size):
        """ Convenience function to evaluate generic function of 1-d array arguments using Pool. 
             func_obj: function object
             zipped_args: array([arg1,arg2,...]).T where arg1,arg2,... are 1-d arrays of equal length
             arg_size: common length of argument arrays.

             Returns evaluation of function as array of size arg_size.
        """
        out = np.zeros(arg_size,dtype=float)
        pool = mp.Pool(processes=self.NPROC)
        results = [pool.apply_async(func_obj,args=tuple(zipped_args[a])) for a in range(arg_size)]
        for a in range(arg_size):
            out[a] = results[a].get()
        pool.close()
        return out
    ############################################################


    ############################################################
    def chiConf_scalar(self,zob,z):
        """ Conformal distance chi between source at z and observer at zob using 0 = -dt^2 + a(t)^2 dchi^2
             chiConf_scalar(zob,z) = int_zob^z dz/E(z).
             Convenience function used when z and zob are both scalars.
        """
        chi,err = syint.quad(self.EHub_inv,zob,z)
        return chi
    ############################################################

    ############################################################
    def chiConf(self,zob,z):
        """ Conformal distance chi between source at z and observer at zob using 0 = -dt^2 + a(t)^2 dchi^2. 
             chiConf(zob,z) = c/H0 int_zob^z dz/E(z).
             If both z and zob are arrays, then must have same length. Only 1-d arrays supported.
             Returns chiCom in Mpc/h, size = z.size or zob.size.
        """
        if np.isscalar(zob):
            if np.isscalar(z):
                # both z and zob are scalars
                if zob >= z:
                    raise ValueError("zob should be strictly less than z!")
                chi = self.chiConf_scalar(zob,z)
            else:
                # z is array and zob is scalar
                if len(z.shape) > 1:
                    raise TypeError("Only 1-d arrays supported for z!")
                if zob >= z.min():
                    raise ValueError("zob should be strictly less than smallest z!")
                # chi = self.pool_it_1d(self.chiConf_scalar,zip(zob*np.ones(z.size),z),z.size)
                chi = self.pool_it_1d(self.chiConf_scalar,np.array([zob*np.ones(z.size),z]).T,z.size)
        else:
            if np.isscalar(z):
                # z is scalar and zob is array
                if len(zob.shape) > 1:
                    raise TypeError("Only 1-d arrays supported for zob!")
                if zob.max() >= z:
                    raise ValueError("Largest zob should be strictly less than z!")
                # chi = self.pool_it_1d(self.chiConf_scalar,zip(zob,z*np.ones(zob.size)),zob.size)
                chi = self.pool_it_1d(self.chiConf_scalar,np.array([zob,z*np.ones(zob.size)]).T,zob.size)
            else:
                # z and zob are arrays
                if len(zob.shape) > 1:
                    raise TypeError("Only 1-d arrays supported for zob and z!")
                if zob.shape != z.shape:
                    raise TypeError("zob and z must have same length!")
                if np.any(zob >= z):
                    raise ValueError("Each zob should be strictly less than corresponding z!")
                # chi = self.pool_it_1d(self.chiConf_scalar,zip(zob,z),z.size)
                chi = self.pool_it_1d(self.chiConf_scalar,np.array([zob,z]).T,z.size)
        chi *= self.c_by_H0
        return chi
    ############################################################

    ############################################################
    def test_redshift_pair(self,zob,z):
        """ Convenience function called by chiConf_analytic_***() routines. """
        if (np.isscalar(z) & np.isscalar(zob)):
            if zob >= z:
                raise ValueError("zob should be strictly less than z!")
        else:
            if np.any(zob >= z):
                raise ValueError("zob should be strictly less than corresponding z!")
        return
    ############################################################

    ############################################################
    def chiConf_analytic_EdS(self,zob,z):
        """ Analytical conformal distance in Mpc/h between observer at zob and source at z 
             for EdS universe. 
             z and zob should have compatible shapes. Output has shape z.shape or zob.shape.
        """
        self.test_redshift_pair(zob,z)
        chi = 2*(1.0/np.sqrt(1+zob) - 1.0/np.sqrt(1+z))
        chi *= self.c_by_H0
        return chi
    ############################################################

    ############################################################
    def chiConf_analytic_pureRad(self,zob,z):
        """ Analytical conformal distance in Mpc/h between observer at zob and source at z 
             for radiation dominated universe. 
             z and zob should have compatible shapes. Output has shape z.shape or zob.shape.
        """
        self.test_redshift_pair(zob,z)
        chi = (z - zob)/(1+zob)/(1+z)
        chi *= self.c_by_H0
        return chi
    ############################################################

    ############################################################
    def chiConf_analytic_Milne(self,zob,z):
        """ Analytical conformal distance in Mpc/h between observer at zob and source at z 
             for Milne universe (Ok=1.0). 
             z and zob should have compatible shapes. Output has shape z.shape or zob.shape.
        """
        self.test_redshift_pair(zob,z)
        chi = np.log((1+z)/(1+zob))
        chi *= self.c_by_H0
        return chi
    ############################################################

    ############################################################
    def chiConf_analytic_dS(self,zob,z):
        """ Analytical conformal distance in Mpc/h between observer at zob and source at z 
             for deSitter universe (OLam=1.0). 
             z and zob should have compatible shapes. Output has shape z.shape or zob.shape.
        """
        self.test_redshift_pair(zob,z)
        chi = z - zob
        chi *= self.c_by_H0
        return chi
    ############################################################

    ############################################################
    def chiConf_analytic_MatRad(self,zob,z,Orad=0.1):
        """ Analytical conformal distance in Mpc/h between observer at zob and source at z 
             for matter+radiation dominated universe. 
             z and zob should have compatible shapes. Output has shape z.shape or zob.shape.
        """
        self.test_redshift_pair(zob,z)
        Om = 1.0 - Orad
        aeq = Orad/Om
        chi = 2/np.sqrt(Om)*(np.sqrt(aeq+1.0/(1+zob)) - np.sqrt(aeq+1.0/(1+z)))
        chi *= self.c_by_H0
        return chi
    ############################################################

    ############################################################
    def chiConf_analytic_Curv_MatRadLam(self,zob,z,Ok=0.1,other=0):
        """ Analytical conformal distance in Mpc/h between observer at zob and source at z 
             for matter+Ok (other=0) or radiation+Ok (other=1) or Lambda+Ok (other=2) universe. 
             z and zob should have compatible shapes. Output has shape z.shape or zob.shape.
        """
        self.test_redshift_pair(zob,z)
        absOk = np.fabs(Ok)
        Oother = 1.0 - Ok
        mkratio = np.sqrt(absOk/Oother)
        if ((other == 2) & (Ok < 0.0) & np.any(z > mkratio)):
            raise ValueError("decrease largest redshift to be smaller than sqrt(|Ok|/OLam) = {0:.3e}!".format(mkratio))
        p = -0.5
        if other:
            p = -1.0 if other==1 else 1.0
        arg_ob = mkratio*(1+zob)**p
        arg_src = mkratio*(1+z)**p
        skinvob = np.arcsin(arg_ob) if Ok < 0.0 else np.arcsinh(arg_ob)
        skinv = np.arcsin(arg_src) if Ok < 0.0 else np.arcsinh(arg_src)
        chi = -1.0/(p*np.sqrt(absOk))*(skinvob - skinv) # note -1 out front.
        chi *= self.c_by_H0
        return chi
    ############################################################


    ############################################################
    def rCom(self,z,zob=0.0):
        """ Comoving distance in Mpc/h between source at z and observer at zob. 
             If z and zob are both arrays, they must have same length. Only 1-d arrays supported.
        """
        rc = self.chiConf(zob,z)
        if not self.FLAT:
            sqrtK = np.sqrt(np.fabs(self.Ok))/self.c_by_H0 # curvature inverse length scale
            rc = np.sin(sqrtK*rc) if self.Ok < 0.0 else np.sinh(sqrtK*rc) # recall Ok = -K*(c/H0)^2
            rc /= sqrtK
        return rc
    ############################################################


    ############################################################
    def dLum(self,z,zob=0.0):
        """ Luminosity distance in physical Mpc/h between source at z and observer at zob. 
             If z and zob are both arrays, they must have same length. Only 1-d arrays supported.
        """
        dL = self.rCom(z,zob)*(1+z)/(1+zob)**2
        return dL
    ############################################################


    ############################################################
    def dAng(self,z,zob=0.0):
        """ Angular diameter distance in physical Mpc/h between source at z and observer at zob. 
             If z and zob are both arrays, they must have same length. Only 1-d arrays supported.
        """
        dA = self.rCom(z,zob)/(1+z)
        return dA
    ############################################################

    ############################################################
    def dVol(self,z):
        """ Volume averaged distance in physical Mpc/h between source at z and observer at zob=0. 
             Only 1-d arrays for z supported.
        """
        dA = self.dAng(z)
        dVol = (self.c_by_H0*self.EHub_inv(z)*z*(1+z)**2*dA**2)**(1/3.)
        return dVol
    ############################################################


    ############################################################
    def dVdz(self,z):
        """ Comoving volume element dV/dz = 4pi*(c/H)*rCom**2 at redshift z in (Mpc/h)^3.
             z can be scalar or 1-d array.
        """
        out = 4*np.pi*self.c_by_H0*self.rCom(z)**2/self.EHub(z)
        return out
    ############################################################


    ############################################################
    def age_integrand(self,z):
        """ Convenience function for age integral. """
        return 1.0/(self.EHub(z)*(1+z))
    ############################################################

    ############################################################
    def age_scalar(self,z):
        """ Age of universe at redshift z in units of H0.
             Convenience function for scalar z.
        """
        t,err = syint.quad(self.age_integrand,z,np.inf)
        return t
    ############################################################

    ############################################################
    def age(self,z):
        """ Age of universe at redshift z in Gyr.
             z can be scalar or 1-d array.
        """
        if np.isscalar(z):
            t = self.age_scalar(z)
        else:
            if len(z.shape) > 1:
                raise TypeError("Only 1-d arrays supported for z!")
            # t = self.pool_it_1d(self.age_scalar,zip(z),z.size)
            t = self.pool_it_1d(self.age_scalar,z,z.size)

        t *= self.H0inv/self.hubble
        return t
    ############################################################

    ############################################################
    def lookback(self,z):
        """ Lookback time to redshift z in Gyr.
             z can be scalar or 1-d array.
        """
        return self.age(0.0) - self.age(z)
    ############################################################
        

    ############################################################
    def Wth_scalar(self,x):
        """ TopHat filter W(x) = (3/x) j1(x) """
        xsq = x**2
        if np.fabs(x) < self.NOTSOTINY:
            xq = xsq**2
            return 1.0 - xsq/10.0 + xq/280.0 - xsq*xq/15120.0
        else: 
            return (3.0/x/xsq)*(np.sin(x)-x*np.cos(x))
    ############################################################


    ############################################################
    def Wthpr_scalar(self,x):
        """ Derivative of TopHat filter = (3/x)( j0(x) - (3/x) j1(x) ) """
        xsq = x**2
        if np.fabs(x) < self.NOTSOTINY:
            xqui = x**5
            return -x/5.0 + x*xsq/70.0 - xqui/2520.0 + xqui*xsq/166320.0
        else: 
            return (3.0/xsq**2)*(3*x*np.cos(x)+(xsq-3)*np.sin(x))
    ############################################################

    ############################################################
    def Growth(self,z=0):
        """ LCDM growth function D(z).
             Normalized to be 1/(1+z) in matter domination.
        """
        a = 1.0/(1+z)
        if self.FLAT & (np.fabs(1+self.wDE0) <= self.NOTSOTINY): 
            acube = a**3
            hbyh0 = self.EHub(z)
            g = hbyh0/np.sqrt(self.Om)*a**2.5*sysp.hyp2f1(5.0/6,1.5,11.0/6,-acube*(1.0/self.Om-1))
        else:
            # 2.5*Om0*E(a)*int_0^a da / (a*E(a))^3
            amin = 1e-3 # z ~ 1e3, safely away from LSS but in matter domination
            avals = np.logspace(np.log10(amin),np.log10(a),500) # 500 should be safe for -0.1 <= Ok <= 0.1 for all z
            g = np.trapezoid(1/(avals*self.EHub(1/avals-1))**3,x=avals)
            g *= 2.5*self.Om*self.EHub(z)
    
        return g
    ############################################################

    
    ############################################################
    def fGrowth(self,z=0):
        """ LCDM growth function derivative f(z) = dlnD/dlna.
        """
        a = 1.0/(1+z)
        acube = a**3
        hbyh0 = self.EHub(z)

        if self.FLAT & (np.fabs(1+self.wDE0) <= self.NOTSOTINY): 
            dlnHdlna = -1.5/(1+acube*(1/self.Om-1))
        else:
            OkbyOm = self.Ok/self.Om
            OLambyOm = self.OLam/self.Om
            aw = 1/acube**self.wDE0
            dlnHdlna = -(1.5 + OkbyOm*a + 1.5*OLambyOm*(1+self.wDE0)*aw)
            dlnHdlna /= (1 + OkbyOm*a + OLambyOm*aw)

        f = dlnHdlna + 2.5*self.Om/self.Growth(z)/(a*hbyh0)**2

        return f
    ############################################################

    ############################################################
    def dcsph_lcdm(self,z=0):
        """ Flat LCDM sph coll critical density. Returns dc(z)*D(z)/D(0)
             Fit from Henry, ApJ (2000), 534, 565.
        """
        if np.fabs(self.Ok) > self.NOTSOTINY: 
            self.print_this("WARNING: dcsph_lcdm() needs to be modified!!",self.logfile)
        x3 = (1/self.Om-1.)/(1.0+z)**3

        return self.dcsph*(1-0.0123*np.log10(1+x3))
    ############################################################


    ############################################################
    def calc_sig02(self,mass):
        """ Calculate sigma_0^2(mass) = int dlnk Dlin Wth(kRLag)^2 [lin exp to z=0] 
             where RLag is Lagrangian radius corresponding to given mass.
             Input mass can be scalar or 1d array. Output has same shape.
        """
        RLag = (mass/(4*np.pi*self.Om*self.rhoc/3.0))**(1/3.0)
        if np.isscalar(mass):
            kR = RLag*self.ktab_lin 
            axis = 0 
        else:
            if len(mass.shape) == 1:
                kR = np.outer(RLag,self.ktab_lin)
                axis = 1 
            else:
                raise TypeError("Input mass value must be scalar or 1d array.")
        Wsq = self.Wth(kR)**2
        out = np.trapezoid(Wsq*self.Dlin, x=self.ln_ktab_lin, axis=axis)
        return out
    ############################################################


    ############################################################
    def calc_nu(self,mass,z=0,lcdm=True):
        """ Calculate nu(mass,z) = dc * D(0) / D(z) / sig0
             Input mass and redshift can be scalars or scalar + 1d array or equal length 1d arrays. 
             Output has same shape.
        """
        dc0 = self.dcsph_lcdm(z) if lcdm else self.dcsph
        if (not np.isscalar(z)) & (not np.isscalar(mass)):
            if z.shape != mass.shape:
                raise TypeError("mass and z cannot have different nonscalar shapes")

        sig0 = np.sqrt(self.calc_sig02(mass))
        nu = dc0*self.Growth(z=0)/self.Growth(z=z)/sig0

        return nu
    ############################################################


    ############################################################
    def Dvir(self,Omega_z):
        """ Convenience function for Delta_vir (what multiplies rho_c(z))
            from Bryan & Norman 98. Omega_z should be rho_m(z)/rho_crit(z).
        """
        return 18*np.pi**2 + 82*(Omega_z-1) - 39*(Omega_z-1)**2
    ############################################################


    ############################################################
    def massfuncbiasTinker(self,mtab,type_dict=None,z=0.0,Delta=200.0):
        """ Tinker+2008 mass function and Tinker+2010 linear Eulerian bias at redshift z, evaluated on given mass grid.
            type_dict should be dictionary with keys:
              'type': one of ['cdm','wdm','adm','bdm']
                  -- if 'type' = 'wdm'
                     'mdm': value of WDM particle mass in keV
                     optionally:
                        'bode': 0 (Viel+) or 1 (Bode+)
                        'gdm': default 1.5, used in Bode+                         
                  -- if 'type' = 'adm'
                     'rAc': sound horizon (kpc/h)
                     'kD': diffusion damping scale (h/Mpc)
                  -- if 'type' = 'bdm'
                     'kAc': acoustic scale (h/Mpc)
                     'kMod': modulation scale (h/Mpc)
                     'kpiv': pivot scale (h/Mpc)
                     'kD': diffusion damping scale (h/Mpc) [only for numerical stability]
            Returns dn/dlnm (h/Mpc)^3 and b1.
        """
        if type_dict is None:
            if self.verbose:
                self.print_this('type_dict None selected. Using CDM transfer function.',self.logfile)
            type_dict = {'type':'cdm'}
            
        mftype = type_dict['type']
        if mftype == 'cdm':
            Dlin = self.Dlin
        elif mftype == 'wdm':
            mdm = type_dict['mdm']
            if 'bode' in type_dict.keys():
                bode = type_dict['bode']
            else:
                bode = 0
            if 'gdm' in type_dict.keys():
                gdm = type_dict['gdm']
            else:
                gdm = 1.5
            Dlin,mFS,mHM = self.linearpower_wdm(mdm=mdm,bode=bode,gdm=gdm)
        elif mftype == 'adm':
            rAc = type_dict['rAc']
            kD = type_dict['kD']
            Dlin = self.linearpower_adm(rAc=rAc,kD=kD)
        elif mftype == 'bdm':
            kAc = type_dict['kAc']
            kMod = type_dict['kMod']
            kpiv = type_dict['kpiv']
            kD = type_dict['kD']
            Dlin = self.linearpower_bdm(kAc=kAc,kMod=kMod,kpiv=kpiv,kD=kD)
        else:
            raise ValueError("type_dict['type'] must be one of  ['cdm','wdm','adm','bdm']")

        mfT08,b1T10 = self.massfuncbiasTinker_workhorse(mtab,Dlin,z=z,Delta=Delta)
        return mfT08,b1T10
    ############################################################

    
    ############################################################
    def massfuncbiasTinker_workhorse(self,mtab,Dlin,z=0.0,Delta=200.0):
        """ Tinker+2008 mass function and Tinker+2010 linear Eulerian bias at redshift z,
            evaluated on given mass grid for given linear power spectrum.
            Returns dn/dlnm (h/Mpc)^3 and b1.
        """
        dc0 = self.dcsph
        # Should be 1.686.
        dc = dc0*self.Growth(z=0)/self.Growth(z)

        if Dlin.size != self.nk_lin:
            raise ValueError('Incompatible Dlin detected. Need 1-d array of size {0:d}'.format(self.nk_lin))
        
        lgDelta = np.log10(Delta)
        aA0 = (0.1*lgDelta-0.05) if Delta < 1600.0 else 0.26
        aa0 = 1.43 + (lgDelta-2.3)**1.5
        ab0 = 1.0 + (lgDelta-1.6)**(-1.5)
        ac = (1.2 + (lgDelta-2.35)**1.6) if lgDelta > 2.35 else 1.19
        alpha = 10**(-(0.75/np.log10(Delta/75.))**1.2)
        
        aA = aA0*(1+z)**(-0.14)
        aa = aa0*(1+z)**(-0.06)
        ab = ab0*(1+z)**(-alpha)

        qT08 = 2*ac/dc0**2

        expfac = np.exp(-(4/lgDelta)**4)
        bA = 1.0 + 0.24*lgDelta*expfac
        ba = 0.44*lgDelta - 0.88
        bB = 0.183
        bb = 1.5
        bC = 0.019 + 0.107*lgDelta + 0.19*expfac
        bc = 2.4

        Rtab = (mtab/(4*np.pi*self.Om*self.rhoc/3.0))**(1/3.0)
        # Lagrangian R-values in Mpc/h corresponding to bin centers
        kR = np.outer(Rtab,self.ktab_lin)
        W = self.Wth(kR) 
        Wsq = W**2
        sig02 = np.trapezoid(Wsq*Dlin, x=self.ln_ktab_lin, axis=1)
        mdWdR = -self.Wthpr(kR)*self.ktab_lin
        sig12 = np.trapezoid(W*mdWdR*Dlin, x=self.ln_ktab_lin, axis=1)/Rtab
        jacob = Rtab**2*sig12/sig02/3.0
        # |dlns/dlnm|/2 = jacob
        #               = 1/6 |dlns/dlnR| 
        #               = -1/6 R/s ds/dR

        nu2 = qT08*dc**2/sig02
        # List of nu^2 values multiplied by qT08
        sigtabz = np.sqrt(sig02)*(dc0/dc)
        vfv = aA*(1+(sigtabz/ab)**(-aa))*np.exp(-0.5*nu2)

        mfT08 = vfv*jacob*self.Om*self.rhoc/mtab
        nu = dc0/sigtabz
        b1T10 = 1-bA/(1+sigtabz**ba) + bB*nu**bb + bC*nu**bc
        # Linear Eulerian bias

        return mfT08,b1T10
    ############################################################

    ############################################################
    def prepare_Dlin(self,type_dict):
        """ Utility to set up Dlin for AltDM models. 
            type_dict: dictionary with keys
              'type': one of ['cdm','wdm','adm','bdm']
                  -- if 'type' = 'wdm'
                     'mdm': value of WDM particle mass in keV
                     optionally:
                        'bode': 0 (Viel+) or 1 (Bode+)
                        'gdm': default 1.5, used in Bode+                         
                  -- if 'type' = 'adm'
                     'rAc': sound horizon (kpc/h)
                     'kD': diffusion damping scale (h/Mpc)
                  -- if 'type' = 'bdm'
                     'kAc': acoustic scale (h/Mpc)
                     'kMod': modulation scale (h/Mpc)
                     'kpiv': pivot scale (h/Mpc)
                     'kD': diffusion damping scale (h/Mpc) [only for numerical stability]
            Returns dict containing Dlin, RHM.
        """
        mftype = type_dict['type']
        if mftype == 'cdm':
            Dlin = self.Dlin
            RHM = 0.0
        elif mftype == 'wdm':
            mdm = type_dict['mdm']
            if 'bode' in type_dict.keys():
                bode = type_dict['bode']
            else:
                bode = 0
            if 'gdm' in type_dict.keys():
                gdm = type_dict['gdm']
            else:
                gdm = 1.5
            Dlin,mFS,mHM = self.linearpower_wdm(mdm=mdm,bode=bode,gdm=gdm)
            RHM = (mHM/(4*np.pi*self.Om*self.rhoc/3.0))**(1/3.0)
        elif mftype == 'adm':
            rAc = type_dict['rAc']
            kD = type_dict['kD']
            Dlin = self.linearpower_adm(rAc=rAc,kD=kD)
            RHM = 0.5*rAc*1e-3
        elif mftype == 'bdm':
            kAc = type_dict['kAc']
            kMod = type_dict['kMod']
            kpiv = type_dict['kpiv']
            kD = type_dict['kD']
            Dlin = self.linearpower_bdm(kAc=kAc,kMod=kMod,kpiv=kpiv,kD=kD)
            RHM = np.pi/kpiv
        else:
            raise ValueError("type_dict['type'] must be one of  ['cdm','wdm','adm','bdm']")

        out = {'Dlin':Dlin,'RHM':RHM}
        return out
    ############################################################

    
    ############################################################
    def massfuncbiasESP(self,mtab,type_dict=None,params=None,esptype='ra',z=0.0,n_fine=550,prep_mf=None,prep_Dlin=None):
        """ ESP mass function and bias at redshift z, evaluated on given mass grid.
            ESP mass function using scheme from Hahn & Paranjape 2014 (arXiv:1308.4142)
            Corresponding linear Eulerian bias from Paranjape 2024 (arXiv:24xx.yyyyy)
            type_dict: dictionary with keys
              'type': one of ['cdm','wdm','adm','bdm']
                  -- if 'type' = 'wdm'
                     'mdm': value of WDM particle mass in keV
                     optionally:
                        'bode': 0 (Viel+) or 1 (Bode+)
                        'gdm': default 1.5, used in Bode+                         
                  -- if 'type' = 'adm'
                     'rAc': sound horizon (kpc/h)
                     'kD': diffusion damping scale (h/Mpc)
                  -- if 'type' = 'bdm'
                     'kAc': acoustic scale (h/Mpc)
                     'kMod': modulation scale (h/Mpc)
                     'kpiv': pivot scale (h/Mpc)
                     'kD': diffusion damping scale (h/Mpc) [only for numerical stability]
            esptype: one of ['ra','det','tau','ratau','ab','raab','taush','absh'] (default 'ra'), giving flavour of ESP calculation.
            params: None (default) or dict with keys ['DBmean','sigDB','expo','bSMT','gSMT','jacob_min'] giving values of HP14 RA model parameters.
                    None or missing keys default to HP14 values.
            prep_mf: None (default) or dictionary containing pre-calculated arrays. Useful for MCMC.
            prep_Dlin: None (default) or dictionary containing pre-calculated arrays. Useful for MCMC.
            Returns dn/dlnm (h/Mpc)^3 and b1.
        """
        if self.verbose:
            self.print_this('ESP mass function and linear bias',self.logfile)
        if type_dict is None:
            if self.verbose:
                self.print_this('... type_dict None selected. Using CDM transfer function.',self.logfile)
            type_dict = {'type':'cdm'}

        
        pDl = self.prepare_Dlin(type_dict) if prep_Dlin is None else prep_Dlin
        Dlin = pDl['Dlin']
        RHM = pDl['RHM']
            
        mtab_fine = np.logspace(4.0,np.log10(mtab.max()),n_fine)
        if prep_mf is not None:
            if mtab_fine.size != prep_mf['Rtab'].size:
                raise ValueError('Incompatible mass arrays. Use prepare_massfunction() with np.logspace(4.0,{0:.2f},{1:d})'.format(np.log10(mtab.max()),n_fine))
        if type_dict['type'] == 'cdm':
            RHM = (mtab_fine.min()/(4*np.pi*self.Om*self.rhoc/3.0))**(1/3.0)
        mfESP_fine,b1_fine = self.massfuncbiasESP_workhorse(mtab_fine,Dlin,esptype=esptype,params=params,RHM=RHM,z=z,prep_mf=prep_mf)
        ln_mfESP = np.interp(np.log(mtab),np.log(mtab_fine),np.log(mfESP_fine))
        mfESP = np.exp(ln_mfESP)
        b1 = np.interp(np.log(mtab),np.log(mtab_fine),b1_fine)
        del mtab_fine,mfESP_fine,b1_fine,ln_mfESP
        gc.collect()
        # mfESP = self.massfuncESP_workhorse(mtab,Dlin,esptype=esptype,params=params,RHM=RHM,z=z) # this doesn't give converged answer
        
        return mfESP,b1
    ############################################################

    ############################################################
    def find_RG(self,R):
        """ Return RG value that satisfies
        < dG d > = < d^2 >.
        Can only be called after Dlin,ktab,dlnk are defined.
        Uses Newton-Raphson.
        """
        eps = 1.0e-2
        MAX_COUNT = 100
        Wtop = self.Wth(np.outer(R,self.ktab_lin))
        sig0top2 = np.trapezoid(Wtop**2*self.Dlin, x=self.ln_ktab_lin, axis=1)

        ktab2 = self.ktab_lin**2
        RG = 0.46*R
        # initial guess
        fRG = (np.trapezoid(Wtop*np.exp(-0.5*np.outer(RG,self.ktab_lin)**2)*self.Dlin,x=self.ln_ktab_lin,axis=1)
               -sig0top2) 
        # zeroth iteration
        count = 0
        while np.any(np.abs(fRG/sig0top2)) > eps:
            # control relative error
            dfdRG = (-RG
                     *np.trapezoid(Wtop*np.exp(-0.5*np.outer(RG,self.ktab_lin)**2)
                               *ktab2*self.Dlin,x=self.ln_ktab_lin,axis=1))
            # derivative
            RG -= fRG/dfdRG
            # step
            fRG = (np.trapezoid(Wtop*np.exp(-0.5*np.outer(RG,self.ktab_lin)**2)*self.Dlin,x=self.ln_ktab_lin,axis=1)
                   -sig0top2) 
            count += 1
            if count > MAX_COUNT:
                break
        return RG
    ############################################################

    ############################################################
    def prepare_massfunction(self,mtab,Dlin,z=0.0,esptype='ra'):
        """ Utility to pre-compute various arrays needed for ESP mass function / bias calculations. """

        cond_type = (esptype in ['tau','ratau','ab','raab','taush','absh'])
        
        dc0 = self.dcsph_lcdm(z) 
        # LCDM value
        dc = dc0*self.Growth(z=0)/self.Growth(z)

        root2pi = np.sqrt(2*np.pi)
        ktab2 = self.ktab_lin**2
        
        Rtab = (mtab/(4*np.pi*self.Om*self.rhoc/3.0))**(1/3.0)
        # Lagrangian radius for TopHat filter.
        RGtab = self.find_RG(Rtab)  # Rtab/np.sqrt(5) 

        kRG = np.outer(RGtab,self.ktab_lin)
        WG = np.exp(-0.5*kRG**2)
        WGsq = WG**2
        WTH = self.Wth(np.outer(Rtab,self.ktab_lin)) if cond_type else None
                
        ###########################
        # Gaussian/TopHat and cross spectral moments
        # sig0^2 = int dlnk Dlin WTH^2 (or WG^2)
        # sig1^2 = int dlnk Dlin WG^2 k^2
        # sig2^2 = int dlnk Dlin WG^2 k^4
        # sig1m^2 = int dlnk Dlin WG WT k^2
        ###########################
        
        sig02 = np.trapezoid(WTH**2*Dlin,x=self.ln_ktab_lin,axis=1) if cond_type else np.trapezoid(WGsq*Dlin,x=self.ln_ktab_lin,axis=1)
        sig1m2 = np.trapezoid(WG*WTH*Dlin*ktab2, x=self.ln_ktab_lin, axis=1) if cond_type else None
        sig12 = np.trapezoid(WGsq*Dlin*ktab2, x=self.ln_ktab_lin, axis=1)
        sig22 = np.trapezoid(WGsq*Dlin*ktab2**2, x=self.ln_ktab_lin, axis=1)

        ###########################
        # Other spectral quantities
        # gam   = < x nu > = sig_1m^2 / sig_0 sig_2
        ###########################
        gam = sig1m2/np.sqrt(sig02*sig22) if cond_type else sig12/np.sqrt(sig02*sig22)
        g2 = gam**2
        omg2 = 1-g2

        # V = mtab/self.Om/self.rhoc
        Vst = (6*np.pi*sig12/sig22)**1.5
        
        ###########################
        nu2 = dc**2/sig02
        nu = np.sqrt(nu2)
        
        #################################
        # Mass-to-variance 
        #################################
        jacob = RGtab**2*sig12/sig02/3.0
        # |dlns/dlnm|/2 = jacob
        #               = 1/6 |dlns/dlnR| 
        #               = -1/6 R/s ds/dR
        #               = 1/3 R^2/s sig1^2
        #################################
        
        #################################
        xinf = 6.0 
        nx = 30 # converged with 6,30
        xtab = np.linspace(0.0,xinf,nx)
        dx = xtab[1] - xtab[0]
        nxones = np.ones(nx,float)
        # shape (x,)

        xmgBdot = np.outer(xtab,np.ones(mtab.size,float)) if (esptype in ['det','ra']) else None

        if esptype in ['tau','ratau','ab','raab','taush','absh']:
            tautab = np.linspace(0.0,6.0,60) # converged with 6,60
            dtau = tautab[1] - tautab[0]
            p5tau = np.sqrt(2/np.pi)/3.0*tautab**4*np.exp(-0.5*tautab**2)
            # shape (tau,)
            rtomg2 = np.sqrt(omg2)
            gmat = np.outer(np.ones_like(tautab),gam)
            xmat = np.zeros((xtab.size,tautab.size,mtab.size),dtype=float)
            for t in range(tautab.size):
                for m in range(mtab.size):
                    xmat[:,t,m] = xtab
            xbyg = xmat/gmat
        else:
            tautab = None
            dtau = None
            p5tau = None
            rtomg2 = None
            gmat = None
            xmat = None
            xbyg = None
        
        out = {'dc0':dc0,'dc':dc,'root2pi':root2pi,'ktab2':ktab2,'Rtab':Rtab,'RGtab':RGtab,
               'kRG':kRG,'WG':WG,'WGsq':WGsq,'WTH':WTH,'sig02':sig02,'sig1m2':sig1m2,'sig12':sig12,'sig22':sig22,
               'gam':gam,'g2':g2,'omg2':omg2,'Vst':Vst,'nu2':nu2,'nu':nu,'jacob':jacob,
               'dx':dx,'xtab':xtab,'nxones':nxones,'xmgBdot':xmgBdot,
               'tautab':tautab,'dtau':dtau,'p5tau':p5tau,'rtomg2':rtomg2,'gmat':gmat,'xmat':xmat,'xbyg':xbyg}

        return out
        
    ############################################################

    
    ############################################################
    def massfuncbiasESP_workhorse(self,mtab,Dlin,esptype='ra',params=None,RHM=None,z=0.0,prep_mf=None):
        """ ESP mass function at redshift z, using scheme from 
            either Hahn & Paranjape 2014 (arXiv:1308.4142) or Castorina+ 2016 (arXiv:1611.03619) 
            or their combination from Paranjape 2024 (arXiv:24xx.yyyyy)
            Corresponding linear Eulerian bias from Paranjape 2024 (arXiv:24xx.yyyyy)
            -- (esptype='ra')  re-assigned masses
            -- (esptype='det') deterministic 
            -- (esptype='tau') ESPtau from Castorina+
            -- (esptype='ab') Assembly-biased mf from 2017 unpublished
            -- (esptype='ratau') ESPtau with re-assignment
            -- (esptype='raab') Assembly-biased with re-assignment
            evaluated on given mass grid mtab for given linear power spectrum Dlin.
            RHM is either half-mode mass (WDM/ADM/BDM) or smallest Lagrangian scale (CDM). Needed only if esptype='ra'.
            params: None (default) or dict with keys ['DBmean','sigDB','expo','bSMT','gSMT','jacob_min'] giving values of HP14 RA model parameters.
                    None or missing keys default to HP14 values.
            prep_mf: None (default) or dictionary containing pre-calculated arrays. Useful for MCMC.
            Returns dn/dlnm (h/Mpc)^3 and b1.
        """
        if esptype not in ['ra','det','tau','ratau','ab','raab','taush','absh']:
            raise ValueError("esptype must be one of ['ra','det','tau','ratau','ab','raab','taush','absh'] in massfuncESP_workhorse()")
        if ((esptype == 'ra') | (esptype == 'ratau') | (esptype == 'raab')) & (RHM is None):
            raise ValueError("RHM must be float when esptype = 'ra' or 'ratau' or 'raab'")

        if Dlin.size != self.nk_lin:
            raise ValueError('Incompatible Dlin detected. Need 1-d array of size {0:d}'.format(self.nk_lin))
            
        #################################
        # Parameters from HP14,
        # with (bSMT,gSMT) modified based
        # on MCMC comparison to T08
        #################################
        DBmean = 0.37 # 0.175
        sigDB = 0.42 # 0.035 # DBmean/5
        expo = 25.0 # 3.0
        jacob_min = 0.01 # 0.1
        bSMT = 0.45 # 0.68
        gSMT = 0.33 # 0.53
        btau = 0.55 # 0.6, beta of ESPtau paper
        b_ab = 0.55 # 0.58, beta of AB model
        kappa2_ab = 1.44 # 0.53, kappa^2 of AB model
        lam_pr = 3.3 # potential parent radius Rpr = lam_pr * R, lam_pr >= 1
        lam = None # separation r = lam * R, lam >= 0. If None (default), then effectively infinite.
        lam_dc = 1.0 # fudge factor to multiply spatially separated barrier in '...sh' flavours. lam_dc > 0
        if params is not None:
            if 'DBmean' in params.keys():
                DBmean = params['DBmean']
            if 'sigDB' in params.keys():
                sigDB = params['sigDB']
            if 'expo' in params.keys():
                expo = params['expo']
            if 'jacob_min' in params.keys():
                jacob_min = params['jacob_min']
            if 'bSMT' in params.keys():
                bSMT = params['bSMT']
            if 'gSMT' in params.keys():
                gSMT = params['gSMT']
            if 'btau' in params.keys():
                btau = params['btau']
            if 'b_ab' in params.keys():
                b_ab = params['b_ab']
            if 'kappa2_ab' in params.keys():
                kappa2_ab = params['kappa2_ab']
            if 'lam_dc' in params.keys():
                lam_dc = params['lam_dc']
            if 'lam' in params.keys():
                lam = params['lam']
            if 'lam_pr' in params.keys():
                lam_pr = params['lam_pr']

        kappa_ab = np.sqrt(kappa2_ab)
        
        pmf = self.prepare_massfunction(mtab,Dlin,z=z,esptype=esptype) if prep_mf is None else prep_mf
        
        dc0 = pmf['dc0']
        dc = pmf['dc']
        root2pi = pmf['root2pi']
        ktab2 = pmf['ktab2']
        Rtab = pmf['Rtab']
        RGtab = pmf['RGtab']
        kRG = pmf['kRG']
        WG = pmf['WG']
        WGsq = pmf['WGsq']
        WTH = pmf['WTH']
        sig02 = pmf['sig02']
        sig1m2 = pmf['sig1m2']
        sig12 = pmf['sig12']
        sig22 = pmf['sig22']
        gam = pmf['gam']
        g2 = pmf['g2']
        omg2 = pmf['omg2']
        Vst = pmf['Vst']
        nu2 = pmf['nu2']
        nu = pmf['nu']
        jacob = pmf['jacob']
        dx = pmf['dx']
        xtab = pmf['xtab']
        nxones = pmf['nxones']
        xmgBdot = pmf['xmgBdot']
        tautab = pmf['tautab']
        dtau = pmf['dtau']
        p5tau = pmf['p5tau']
        rtomg2 = pmf['rtomg2']
        gmat = pmf['gmat']
        xmat = pmf['xmat']
        xbyg = pmf['xbyg']
            
        if esptype in ['det','ra']:
            ###########################
            # SMT01 barrier
            ###########################
            Bbysig0 = nu + bSMT*nu**(1-2*gSMT)
            Bdot = bSMT*2*gSMT*nu**(1-2*gSMT)
            gauss = np.exp(-0.5*Bbysig0**2)/(root2pi)
            xmean = np.outer(nxones,gam*Bbysig0)
            xmin = gam*Bdot
            xmat = np.outer(nxones,xmin)+xmgBdot
            # shape (x,mass)
            pGx = (np.exp(-0.5*(xmat-xmean)**2/omg2)/np.sqrt(omg2)/root2pi).T
            # shape (mass,x)
            ##############
            if esptype == 'det':
                if self.verbose:
                    self.print_this('... ESP with deterministic barrier',self.logfile)
                integrand = xmgBdot.T*pGx*self.Fbbks(xmat.T)
                vNvESPdet = np.trapezoid(integrand,dx=dx,axis=1)
                temp = nu*Bbysig0 - nu*gam/omg2*(xmat - xmean)
                dcb10 = np.trapezoid(integrand*temp,dx=dx,axis=1)/vNvESPdet
                del temp,integrand
                vNvESPdet *= gauss/(gam*Vst)
                # shape (mass,)
                mfESP = vNvESPdet*jacob
                del vNvESPdet
                ##########################
            elif esptype == 'ra':
                if self.verbose:
                    self.print_this('... ESP with re-assigned masses',self.logfile)
                ind = np.where(jacob > jacob_min)[0]
                ind0 = ind[0] if ind.size else 0
                Rturn = Rtab[ind0]
                RGturn = Rturn/np.sqrt(5)
                sig2turn = np.trapezoid(np.exp(-ktab2*RGturn**2)*Dlin,x=self.ln_ktab_lin)
                sigturn = np.sqrt(sig2turn)

                sig0m = np.sqrt(sig02)
                DBbar_eval = DBmean*(sig0m/sigturn)**expo    
                sigDB_eval = np.fabs(sigDB*(sig0m/sigturn)**expo)
                xbgmb = (xmat/gam-bSMT)
                num2 = xbgmb*sig0m - DBbar_eval
                # shape (x,mass)

                temp = xmgBdot.T*pGx*self.Fbbks(xmat.T)
                temp_b1 = (nu*Bbysig0 - nu*gam/omg2*(xmat - xmean)).T

                gjbygV = gauss*jacob/gam/Vst
                
                den = np.sqrt(2)*sigDB_eval
                # shape (0)
                erfc_ratio2 = sysp.erfc((num2/den))

                integ_ClnmRA = np.zeros((mtab.size,mtab.size),float)
                integ_b1_ClnmRA = integ_ClnmRA.copy()
                for M in range(mtab.size):
                    # sig0M = np.sqrt(sig02[M])
                    num1 = num2 - xbgmb*sig0m[M]
                    # shape (x,mass)

                    erfc_prefactor = 0.5*(sysp.erfc((num1/den))-erfc_ratio2).T
                    # shape (mass,x)

                    integrand = erfc_prefactor*temp # Fbbks is bottleneck.
                    integrand_b1 = integrand*temp_b1
                    # shape (mass,x)

                    out = np.trapezoid(integrand,dx=dx,axis=1)
                    out_b1 = np.trapezoid(integrand_b1,dx=dx,axis=1)
                    out *= gjbygV
                    out_b1 *= gjbygV # since deriv will be taken before ratio
                    # shape (mass,)
                    integ_ClnmRA[M] = out
                    integ_b1_ClnmRA[M] = out_b1
                    if self.verbose:
                        self.status_bar(M,mtab.size)

                mfESP = np.zeros(mtab.size,float)
                dcb10 = np.zeros(mtab.size,float)

                dlnm = np.log(mtab[1]/mtab[0]) # workhorse always called with log-spaced mtab
                ClnmRA = np.trapezoid(integ_ClnmRA,dx=dlnm,axis=1)
                b1_ClnmRA = np.trapezoid(integ_b1_ClnmRA,dx=dlnm,axis=1)
                mfESP[0] = (ClnmRA[0]-ClnmRA[1])/dlnm
                mfESP[-1] = (ClnmRA[-2]-ClnmRA[-1])/dlnm
                dcb10[0] = (b1_ClnmRA[0]-b1_ClnmRA[1])/dlnm
                dcb10[-1] = (b1_ClnmRA[-2]-b1_ClnmRA[-1])/dlnm
                for M in range(1,mtab.size-1):
                    mfESP[M] = 0.5*(ClnmRA[M-1]-ClnmRA[M+1])/dlnm
                    dcb10[M] = 0.5*(b1_ClnmRA[M-1]-b1_ClnmRA[M+1])/dlnm
                mfESP[mfESP < 0.0] = 1e-20
                dcb10 = dcb10/mfESP

                del ind,sig0m,DBbar_eval,sigDB_eval,num2,den,erfc_ratio2,xbgmb,temp,temp_b1,gjbygV
                del integ_ClnmRA,ClnmRA,integ_b1_ClnmRA,b1_ClnmRA
            del xmean,xmin,xmgBdot,xmat,pGx
            del Bbysig0,Bdot,gauss
            ########################
        elif esptype in ['ab','tau','taush','absh']:
            rt2 = np.sqrt(2.)
            rt5 = np.sqrt(5.)
            beta = rtomg2/rt5
            if esptype in ['tau','taush']:
                if self.verbose:
                    prn_str = '... ESPtau'
                    if esptype == 'taush':
                        prn_str += ' w/o subhalos'
                    self.print_this(prn_str,self.logfile)
                beta *= btau
            else:
                if self.verbose:
                    prn_str = '... Assembly-biased ESP'
                    if esptype == 'absh':
                        prn_str += ' w/o subhalos'
                    self.print_this(prn_str,self.logfile)
                beta *= b_ab
            Gam_beta = gam/beta 
            betatau = np.outer(tautab,beta)
            
            z_beta = xbyg - betatau
            Gz_beta = Gam_beta*z_beta/rt2
            ESpr_beta = ((Gz_beta/rt2)*(1 + sysp.erf(Gz_beta)) + np.exp(-Gz_beta**2)/root2pi)/Gam_beta

            Fbx = self.Fbbks(xmat)*self.heaviside(xmat)
            nu_st = nu+betatau

            pGxnu_beta = np.exp(-0.5*(xmat**2 + nu_st**2 - 2*gam*xmat*nu_st)/omg2)/rtomg2/(2*np.pi)            
            integrand = pGxnu_beta*ESpr_beta*Fbx
            # shape (x,tau,mass)

            SH_modifier = 1.0 # will become shape (x,tau,mass) if needed
            if esptype in ['taush','absh']:
                kRpr = np.outer(lam_pr*Rtab,self.ktab_lin)
                WTH_pr = self.Wth(kRpr)
                sig2pr_0TT = np.trapezoid(WTH_pr**2*Dlin,x=self.ln_ktab_lin,axis=1)
                if lam is not None:
                    kr = np.outer(lam*Rtab,self.ktab_lin)
                    j0kr = np.sinc(kr/np.pi)
                    sig2_0xTT = np.trapezoid(WTH*WTH_pr*j0kr*Dlin,x=self.ln_ktab_lin,axis=1)
                    sig2_1xGT = np.trapezoid(ktab2*WG*WTH_pr*j0kr*Dlin,x=self.ln_ktab_lin,axis=1)
                
                    gam_nuprnu = sig2_0xTT/np.sqrt(sig02*sig2pr_0TT)
                    gam_nuprx = sig2_1xGT/np.sqrt(sig22*sig2pr_0TT)

                    nu_part = nu*(gam_nuprnu - gam*gam_nuprx) # store this for bias
                    avg_nupr_nux = xmat*(gam_nuprx - gam*gam_nuprnu) + nu_part
                    avg_nupr_nux = avg_nupr_nux/omg2
                    # (x,tau,mass)
                    sig_nupr_nux = 1 - (gam_nuprnu**2 + gam_nuprx**2 - 2*gam*gam_nuprnu*gam_nuprx)/omg2
                    sig_nupr_nux = np.sqrt(sig_nupr_nux)
                    # (mass,)
                    del kr,j0kr,sig2_0xTT,sig2_1xGT,gam_nuprnu,gam_nuprx
                else:
                    nu_part = 0.0
                    avg_nupr_nux = 0.0
                    sig_nupr_nux = 1.0
                
                nupr = lam_dc*dc/np.sqrt(sig2pr_0TT) 
                SH_arg = (nupr - avg_nupr_nux)/sig_nupr_nux/rt2

                SH_modifier = 0.5*(1 + sysp.erf(SH_arg))
                # (x,tau,mass) or (mass,)
                
                del kRpr,WTH_pr,sig2pr_0TT,avg_nupr_nux,sig_nupr_nux
                # assigned variables: SH_modifier,SH_arg,nu_part,nupr
            
            if esptype in ['ab','absh']:
                kappa = rt5*kappa_ab/rtomg2
                Gam_kappa = gam*kappa
                taubykappa = np.outer(tautab,1/kappa)
                z_kappa = xbyg - taubykappa
                Gz_kappa = Gam_kappa*z_kappa/rt2
                ESpr_kappa = ((Gz_kappa/rt2)*(1 + sysp.erf(Gz_kappa)) + np.exp(-Gz_kappa**2)/root2pi)/Gam_kappa
                pGxnu_kappa = np.exp(-0.5*(xmat**2 + taubykappa**2 - 2*gam*xmat*taubykappa)/omg2)/rtomg2/(2*np.pi)
                step = (nu*kappa + ((betatau*kappa).T - tautab).T)
                # separate non-tunnelling
                integrand *= self.heaviside(step)
                # ... and collect contribution to bias
                dcb10 = nu/omg2*np.trapezoid(np.trapezoid(integrand*(nu_st - gam*xmat)*SH_modifier,dx=dx,axis=0).T*p5tau,dx=dtau,axis=1)

                # add tunneling contribution
                integrand += pGxnu_kappa*ESpr_kappa*Fbx*self.heaviside(-1.0*step)
                # ... (negative slope tunnelling trajectories automatically excluded due to x > 0)
                ombk = 1-beta*kappa
                tau_st_by_kappa = nu/ombk
                tau_st = kappa*tau_st_by_kappa
                p5tau_st = np.sqrt(2/np.pi)/3.0*tau_st**4*np.exp(-0.5*tau_st**2)
                pGxtau = np.exp(-0.5*(xmat[:,0,:]**2 + (tau_st_by_kappa)**2 - 2*gam*xmat[:,0,:]*tau_st_by_kappa)/omg2)/rtomg2/(2*np.pi)
                z_tau = xbyg[:,0,:] - tau_st_by_kappa
                Gzt_kappa = Gam_kappa*z_tau/rt2
                ESprt_kappa = ((Gzt_kappa/rt2)*(1 + sysp.erf(Gzt_kappa)) + np.exp(-Gzt_kappa**2)/root2pi)/Gam_kappa
                Gzt_beta = Gam_beta*z_tau/rt2
                ESprt_beta = ((Gzt_beta/rt2)*(1 + sysp.erf(Gzt_beta)) + np.exp(-Gzt_beta**2)/root2pi)/Gam_beta
                # ... and collect contribution to bias (partly from natural walks)
                SH_modifier_taust = 1.0 # will become shape (x,mass) or (mass,) if needed
                if esptype == 'absh':
                    SH_modifier_taust = SH_modifier[:,0,:].copy() if lam is not None else SH_modifier.copy()
                    # (x,mass)
                dcb10 -= self.heaviside(ombk)*tau_st*p5tau_st*np.trapezoid(pGxtau*Fbx[:,0,:]*(ESprt_beta - ESprt_kappa)*SH_modifier_taust,dx=dx,axis=0)

                del kappa,Gam_kappa,z_kappa,Gz_kappa,ESpr_kappa,pGxnu_kappa,step
                del ombk,tau_st,p5tau_st,tau_st_by_kappa,pGxtau
                del z_tau,Gzt_kappa,ESprt_kappa,Gzt_beta,ESprt_beta
            else:
                # here integrand is just ESPtau
                dcb10 = nu/omg2*np.trapezoid(np.trapezoid(integrand*(nu_st - gam*xmat)*SH_modifier,dx=dx,axis=0).T*p5tau,dx=dtau,axis=1)
                
            if esptype in ['absh','taush']:
                dcb10 += np.trapezoid(np.trapezoid(integrand*np.exp(-SH_arg**2)*rt2/root2pi*(nu_part - nupr),dx=dx,axis=0).T*p5tau,dx=dtau,axis=1)
                # this is derivative of SH_modifier, so needs full integrand even in ab model.
                del SH_arg,nu_part,nupr

            vNvESP = np.trapezoid(np.trapezoid(integrand*SH_modifier,dx=dx,axis=0).T*p5tau,dx=dtau,axis=1)            
            dcb10 /= vNvESP
            # shape (mass,)
            mfESP = vNvESP*jacob/Vst
            del beta,Gam_beta,betatau,z_beta,Gz_beta,ESpr_beta,pGxnu_beta
            del Fbx,nu_st,integrand,vNvESP
            ########################
        elif esptype in ['ratau','raab']:
            # IN PROGRESS (reuse from 'ab'/'tau' block)
            btil = rtomg2/np.sqrt(5)
            if esptype == 'ratau':
                btil *= btau
            else:
                btil *= b_ab
            Gtau = gam/(btil+self.TINY)
            # shape (mass,)
            btiltau = np.outer(tautab,btil)
            nu_st = btiltau + nu
            xmean = gmat*nu_st
            gauss = np.exp(-0.5*nu_st**2)/root2pi
            # shape (tau,mass)
            wGtaubyrt2 = (xbyg - btiltau)*Gtau/np.sqrt(2)
            ESpr = (wGtaubyrt2/np.sqrt(2)*(1+sysp.erf(wGtaubyrt2)) + np.exp(-wGtaubyrt2**2)/root2pi)/Gtau
            pGx = (np.exp(-0.5*(xmat-xmean)**2/omg2)/rtomg2/root2pi)
            # shape (x,tau,mass)
            ########################
            if esptype == 'raab':
                kappa = np.sqrt(5.)*kappa_ab/rtomg2 # note rt5
                Gkappa = gam*kappa # note no rt5
                # shape (mass,)
                tau_by_kappa = np.outer(tautab,1/kappa) # note no rt5
                step = kappa*(nu + tau_by_kappa*(btil*kappa-1)) # note no rt5 
                gauss_tun = np.exp(-0.5*tau_by_kappa**2)
                # shape (tau,mass)
                wGtaubyrt2 = (xbyg - tau_by_kappa)*Gkappa/np.sqrt(2) # reused array
                ESpr_tun = (wGtaubyrt2/np.sqrt(2)*(1+sysp.erf(wGtaubyrt2)) + np.exp(-wGtaubyrt2**2)/root2pi)/Gkappa # fixed bug!
                pGx_tun = (np.exp(-0.5*(xmat-gmat*tau_by_kappa)**2/omg2)/rtomg2/root2pi)
                # shape (x,tau,mass)
                ########################
            if esptype == 'ratau':
                ind = np.where(jacob > jacob_min)[0]
                ind0 = ind[0] if ind.size else 0
                Rturn = Rtab[ind0]
                RGturn = Rturn/np.sqrt(5)
                sig2turn = np.trapezoid(np.exp(-ktab2*RGturn**2)*Dlin,x=self.ln_ktab_lin)
                sigturn = np.sqrt(sig2turn)

                sig0m = np.sqrt(sig02)
                DBbar_eval = DBmean*(sig0m/sigturn)**expo    
                sigDB_eval = np.fabs(sigDB*(sig0m/sigturn)**expo)
                num2 = (xbyg - btiltau)*sig0m - DBbar_eval
                # shape (x,tau,mass)

                den = np.sqrt(2)*sigDB_eval
                # shape (0)
                erfc_ratio2 = sysp.erfc((num2/den))
                temp = ESpr*pGx*self.Fbbks(xmat)*gauss
                temp_b1 = (nu*(nu + btiltau) - nu*gam/omg2*(xmat - xmean))
                jbyVst = jacob/Vst

                integ_ClnmRA = np.zeros((mtab.size,mtab.size),float)
                integ_b1_ClnmRA = integ_ClnmRA.copy()
                for M in range(mtab.size):
                    num1 = num2 - xbyg*sig0m[M]

                    erfc_prefactor = 0.5*(sysp.erfc((num1/den))-erfc_ratio2)
                    # shape (x,tau,mass)
                    
                    integrand = erfc_prefactor*temp
                    integrand_b1 = integrand*temp_b1
                    # shape (x,tau,mass)

                    out = np.trapezoid(np.trapezoid(integrand,dx=dx,axis=0).T*p5tau,dx=dtau,axis=1)
                    out_b1 = np.trapezoid(np.trapezoid(integrand_b1,dx=dx,axis=0).T*p5tau,dx=dtau,axis=1)
                    out *= jbyVst
                    out_b1 *= jbyVst # since deriv will be taken before ratio
                    # shape (mass,)
                    integ_ClnmRA[M] = out
                    integ_b1_ClnmRA[M] = out_b1
                    if self.verbose:
                        self.status_bar(M,mtab.size)

                # start_time = time()
                # self.time_this(start_time)
                mfESP = np.zeros(mtab.size,float)
                dcb10 = np.zeros(mtab.size,float)

                dlnm = np.log(mtab[1]/mtab[0]) # workhorse always called with log-spaced mtab
                ClnmRA = np.trapezoid(integ_ClnmRA,dx=dlnm,axis=1)
                b1_ClnmRA = np.trapezoid(integ_b1_ClnmRA,dx=dlnm,axis=1)
                mfESP[0] = (ClnmRA[0]-ClnmRA[1])/dlnm
                mfESP[-1] = (ClnmRA[-2]-ClnmRA[-1])/dlnm
                dcb10[0] = (b1_ClnmRA[0]-b1_ClnmRA[1])/dlnm
                dcb10[-1] = (b1_ClnmRA[-2]-b1_ClnmRA[-1])/dlnm
                for M in range(1,mtab.size-1):
                    mfESP[M] = 0.5*(ClnmRA[M-1]-ClnmRA[M+1])/dlnm
                    dcb10[M] = 0.5*(b1_ClnmRA[M-1]-b1_ClnmRA[M+1])/dlnm
                    # if mfESP[M] < -5e-4:
                    #     print ("Negative number density {0:.3e} at lgm = {1:.3f}".format(mfESP[M],np.log10(mtab[M])))
                mfESP[mfESP < 0.0] = 1e-20
                dcb10 = dcb10/mfESP

                del ind,sig0m,DBbar_eval,sigDB_eval,num2,den,erfc_ratio2,temp,temp_b1,jbyVst
                del integ_ClnmRA,ClnmRA,integ_b1_ClnmRA,b1_ClnmRA
                ########################

            del tautab,p5tau,btil,Gtau,ESpr
            del btiltau,nu_st,gmat,xmat,xbyg,pGx,xmean,gauss,wGtaubyrt2            
            ########################

        b1 = 1.0 + dcb10/dc0
        
        del ktab2,Rtab,RGtab,kRG,WG,WGsq
        if esptype in ['tau','ratau','ab','raab']:
            del WTH,rtomg2
        del sig02,sig12,sig22,gam,g2,omg2,Vst#,V
        del nu2,nu,jacob,dcb10
        del xtab,nxones
        del pmf
        gc.collect()

        return mfESP,b1
    ############################################################

    
    ############################################################
    def massfuncbiasTinker_norm(self,mtab,z=0.0,Delta=200.0):
        """ Tinker++ mass function and linear Eulerian bias at redshift z, evaluated on given mass grid.
             Normalised fit from Appendix C of T08 as rewritten in T10. Only available for Delta = 200b for now.
             Returns dn/dlnm and bias.
        """
        dc0 = self.dcsph

        if np.fabs(Delta-200) > self.NOTSOTINY:
            self.print_this("Warning! Only Delta=200b supported. Using this definition for current calculation.")

        aalpha = 0.368
        abeta_0 = 0.589
        agamma_0 = 0.864
        aphi_0 = -0.729
        aeta_0 = -0.243
        
        abeta = abeta_0*(1+z)**0.20
        agamma = agamma_0*(1+z)**(-0.01)
        aphi = aphi_0*(1+z)**(-0.08)
        aeta = aeta_0*(1+z)**0.27

        Rtab = (mtab/(4*np.pi*self.Om*self.rhoc/3.0))**(1/3.0)
        # Lagrangian R-values in Mpc/h corresponding to bin centers
        kR = np.outer(Rtab,self.ktab_lin)
        W = self.Wth(kR) 
        Wsq = W**2
        sig02 = np.trapezoid(Wsq*self.Dlin, x=self.ln_ktab_lin, axis=1)
        mdWdR = -self.Wthpr(kR)*self.ktab_lin
        sig12 = np.trapezoid(W*mdWdR*self.Dlin, x=self.ln_ktab_lin, axis=1)/Rtab
        jacob = Rtab**2*sig12/sig02/3.0
        # |dlns/dlnm|/2 = jacob
        #               = 1/6 |dlns/dlnR| 
        #               = -1/6 R/s ds/dR

        sigz = np.sqrt(sig02)*self.Growth(z)/self.Growth(z=0)
        nu = dc0/sigz
        vfv = aalpha*(1 + (abeta*nu)**(-2*aphi))*nu**(2*aeta)*np.exp(-0.5*agamma*nu**2)

        mfTinker_norm = vfv*jacob*self.Om*self.rhoc/mtab
        # dn/dlnm

        bTinker_norm = 1 + (1.0/dc0)*(agamma*nu**2 - 1 - 2*aeta + 2*aphi/(1 + (abeta*nu)**(2*aphi)))
        # Linear Eulerian bias

        # SOME TYPO IN T08 App C
        # aB = 0.482
        # ad = 1.97
        # ae = 1.00
        # af = 0.51
        # ag = 1.228

        # sigz = np.sqrt(sig02)*self.Growth(z)/self.Growth(z=0)
        # nu2 = ag/sigz**2
        # vfv = aB*( (sigz/ae)**(-ad) + sigz**(-af) )*np.exp(-0.5*nu2)

        # mfTinker_norm = vfv*jacob*self.Om*self.rhoc/mtab
        # # dn/dlnm

        # esigfac = (ae**ad)*sigz*(af-ad)
        # bTinker_norm = 1 + (2*ag/sigz**2 - af + (af-ad)*esigfac/(1 + esigfac))/dc0
        # # Linear Eulerian bias
        
        return mfTinker_norm,bTinker_norm
    ############################################################


    ############################################################
    def massfuncbiasST(self,mtab,z=0.0,cfit=2.7,qST=0.707,pST=0.3,tophat=True):
        """ Sheth-Tormen mass function and linear Eulerian bias 
             at redshift z, evaluated on given mass grid with TOPHAT filter.
             If tophat=False, evaluated with with SHARP-K filter, with R_shk = R/cfit.
             Returns dn/dlnm and bias.
        """
        dc0 = self.dcsph_lcdm(z)
        dc = dc0*self.Growth(z=0)/self.Growth(z)
        sqrtpi = np.sqrt(np.pi)
        
        AST = 1.0/(1 + sysp.gamma(0.5-pST)/sysp.exp2(pST)/np.sqrt(np.pi))
        
        Rtab = (mtab/(4*np.pi*self.Om*self.rhoc/3.0))**(1/3.0)
        
        if tophat:
            kR = np.outer(Rtab,self.ktab_lin)
            W = self.Wth(kR) 
            Wsq = W**2
            sig02 = np.trapezoid(Wsq*self.Dlin, x=self.ln_ktab_lin, axis=1)
            mdWdR = -self.Wthpr(kR)*self.ktab_lin
            sig12 = np.trapezoid(W*mdWdR*self.Dlin, x=self.ln_ktab_lin, axis=1)/Rtab
            jacob = Rtab**2*sig12/sig02/3.0
            # |dlns/dlnm|/2 = jacob
            #               = 1/6 |dlns/dlnR| 
            #               = -1/6 R/s ds/dR
        else:
            R_shk = Rtab/cfit
            # Lagrangian R-values in Mpc/h corresponding to bin centers
            sig02 = np.array([np.trapezoid(self.Dlin[self.ktab_lin*R_shk[m] < 1.0],
                                       x=np.log(self.ktab_lin[self.ktab_lin*R_shk[m] < 1.0])) 
                              for m in range(len(mtab))])
            
            Dlin_R = np.zeros(len(mtab),dtype=float)
            for m in range(len(mtab)):
                ind_lo = np.where(self.ktab_lin*R_shk[m] < 1.0)[0][-1]
                if ind_lo == self.ktab_lin.size-1:
                    ind_lo -= 1
                ind_hi = ind_lo+1
                slope = (self.Dlin[ind_hi]-self.Dlin[ind_lo])/np.log(self.ktab_lin[ind_hi]/self.ktab_lin[ind_lo])
                intercept = self.Dlin[ind_lo] - slope*np.log(self.ktab_lin[ind_lo])
                Dlin_R[m] = slope*np.log(1/R_shk[m]) + intercept
            jacob = Dlin_R/(6*sig02)
        
        nu2 = qST*dc**2/sig02
        # List of nu^2 values multiplied by qST        
        vfv = AST*np.sqrt(2*nu2)/sqrtpi*np.exp(-nu2/2)*(1+nu2**(-pST))
        # nu f(nu)
        mfST = self.Om*self.rhoc/mtab*vfv*jacob
        # dn/dlnm = (rhobar/m_shk) nu f(nu) (1/2)|dlns/dlnm|        
        bST = 1 + (1/dc0)*(nu2 - 1 + 2*pST/(1+nu2**pST) )
        # Linear Eulerian bias
    
        return mfST,bST
    ############################################################


    ############################################################
    def massfuncbias_thresh(self,mthresh,z=0.0,fit='Tinker',Delta=200.0,type_dict=None,cfit=2.7,qST=0.707,pST=0.3,
                            lower_limit=True,m_upp=1e16,m_low=1e1,dlgm=0.05):
        """ Thresholded mass function and linear Eulerian bias using specified fitting function (default Tinker+2008)
             at redshift z, evaluated for given mass threshold. (Only scalar values supported.)
             If lower_limit = True, mthresh is treated as a lower limit, else as an upper limit.
             Optionally provide m_upp,m_low for integration upper/lower limit and dlgm for spacing.
             Returns n(><mthresh), b1(><mthresh)_no.wtd, b1(><mthresh)_masswtd.
             Parameters calibrated for mMean200.
        """
        if not np.isscalar(mthresh):
            raise TypeError("Only scalar argument supported for mthresh in massfuncbiasTinker_thresh()")

        if fit in ['Tinker','Tinker_norm','ST','ST_shk']:
            if self.verbose:
                self.print_this("... thresholded mass function with "+fit+" fit: m_thresh = {0:.3e} Msun/h".format(mthresh),
                                self.logfile)
            mfb = getattr(self,'massfuncbias'+fit)
        else:
            raise NameError("Specified fit "+fit+" not supported by massfuncbias_thresh()")

        nm = int(np.log10(m_upp/mthresh)/dlgm) if lower_limit else int(np.log10(mthresh/m_low)/dlgm)
        mtab = (np.logspace(np.log10(mthresh),np.log10(m_upp),nm)
                if lower_limit else
                np.logspace(np.log10(m_low),np.log10(mthresh),nm))
        dlnm = np.log(mtab[1]/mtab[0])
        if fit in ['Tinker','Tinker_norm']:
            dndlnm,bias = mfb(mtab,type_dict=type_dict,z=z,Delta=Delta)
        elif fit in ['ST','ST_shk']:
            tophat = False if fit=='ST_shk' else True
            qST = 1.0*qST if fit=='ST' else 1.0 # 1.0 is Schneider+13 value
            dndlnm,bias = mfb(mtab,z=z,cfit=cfit,qST=qST,pST=pST,tophat=tophat)
        n_thresh = np.trapezoid(dndlnm,dx=dlnm)
        b1_thresh_num = np.trapezoid(dndlnm*bias,dx=dlnm)/n_thresh
        b1_thresh_mass = np.trapezoid(mtab*dndlnm*bias,dx=dlnm)/np.trapezoid(mtab*dndlnm,dx=dlnm)

        del mtab,dndlnm,bias
        gc.collect()

        return n_thresh,b1_thresh_num,b1_thresh_mass
    ############################################################


    ############################################################
    def cnuz_CDM_Ludlow16(self,nu,z):
        """ Fitting function c(nu,z) from App. C of Ludlow+16 arXiv:1601.02624. 
        """

        ainv = 1+z
        ainv2 = ainv*ainv
        ainv3 = ainv2*ainv
        ainv4 = ainv2*ainv2

        c0 = 3.395*ainv**(-0.215)
        beta = 0.307*ainv**(0.540)
        gamma1 = 0.628*ainv**(-0.047)
        gamma2 = 0.317*ainv**(-0.893)
        
        nu0 = 4.135 - 0.564*ainv - 0.210*ainv2 + 0.0557*ainv3 - 0.00348*ainv4
        nu0 *= (self.Growth(z=0)/self.Growth(z))

        nuBynu0 = nu/nu0
        out = c0*(nuBynu0)**(-gamma1)
        out *= (1 + (nuBynu0)**(1.0/beta))**(-beta*(gamma2-gamma1))

        return out
    ############################################################

    ############################################################
    def cmz_CDM_Ludlow16(self,m,z):
        """ Fitting function c(nu(m),z) from App. C of Ludlow+16 arXiv:1601.02624. 
        """

        nu = self.calc_nu(m,z=z,lcdm=False)
        out = self.cnuz_CDM_Ludlow16(nu,z)

        return out
    ############################################################

    ############################################################
    def cnuz_CDM_DK15(self,nu,z,nspec):
        """ Fitting function c(nu,z) [median] from Diemer&Kravtsov15 arXiv:1407.4730. 
        """

        phi0 = 6.58
        phi1 = 1.37
        eta0 = 6.82
        eta1 = 1.42
        alpha = 1.12
        beta = 1.69

        cmin = phi0 + phi1*nspec
        numin = eta0 + eta1*nspec
        
        out = 0.5*cmin*((nu/numin)**(-alpha) + (nu/numin)**beta)

        return out
    ############################################################

    ############################################################
    def cmz_CDM_DK15(self,m,z):
        """ Fitting function c(nu(m),z) from Diemer&Kravtsov15 arXiv:1407.4730. 
        """
        nspec = self.calc_nspec_DK15(m)
        nu = self.calc_nu(m,z=z,lcdm=False)
        out = self.cnuz_CDM_DK15(nu,z,nspec)

        return out
    ############################################################


    ############################################################
    def siglncmz_CDM_DK15(self):
        """ Measured siglnc (constant!) from Diemer&Kravtsov15 arXiv:1407.4730.
        """
        out = 0.16*np.log(10)
        return out
    ############################################################


    ############################################################
    def siglncmz_CDM_W02(self):
        """ Fitting function siglnc(nu(m),z) (constant!) from Wechsler+2002 (footnote 10) . 
        """
        out = 0.14*np.log(10)
        return out
    ############################################################

        
    ############################################################
    def calc_nspec_DK15(self,m):
        """ Convenience function for calculating local slope of 
            power spectrum, as defined by Diemr&Kravtsov15 arXiv:1407.4730.
        """

        kappa = 0.69
        RLag = (3*m/(4*np.pi*self.Om*self.rhoc))**(1/3.)
        keval = kappa*2*np.pi/RLag

        if np.isscalar(keval):
            ind = np.where(self.ktab_lin >= keval)[0][0]
            dlnk2 = np.log(self.ktab_lin[ind+1]/self.ktab_lin[ind-1])
            nspec = np.log(self.Dlin[ind+1]/self.Dlin[ind-1])/(dlnk2)
            nspec -= 3.
        else:
            nspec = np.ones(keval.size,dtype=float)
            for k in range(keval.size):
                ind = np.where(self.ktab_lin >= keval[k])[0][0]
                dlnk2 = np.log(self.ktab_lin[ind+1]/self.ktab_lin[ind-1])
                nspec[k] = np.log(self.Dlin[ind+1]/self.Dlin[ind-1])/(dlnk2)
                nspec[k] -= 3.

        return nspec
    ############################################################


    ############################################################
    def f_nfw(self,x):
        """ Convenience function for NFW normalisation. """

        out = (np.log(1+x)-x/(1+x))/x**3
        return out
    ############################################################

    ########################################################
    def g_nfw(self,x):
        """ Convenience function for NFW velocity profile. 
            See below eqn A24 of Sheth, Hui, Diaferio & Scoccimarro MNRAS (2001).
        """
        xp1  = 1.0 + x
        lnxp1 = np.log(xp1)

        out  = -1 + 1.0/x + 1.0/xp1**2 + 6.0/xp1 
        out += np.log(x) - lnxp1
        out += (6*x**2 + 3*x - 1.0)/(x**2*xp1)*lnxp1
        out -= 3*lnxp1**2

        out += 6*(-sysp.spence(xp1)) 
        # Dilogarithm: spence(1-z) = Li_2(z) 
        #                          = sum_{k=1}^inf z^k / k^2
        #                          = - int_0^z dt ln(1-t) / t = +int_0^{-z} dt ln(1+t)/t
        # so spence(1+x) = int_0^x dt/t ln(1+t) = Li_2(-z)
        # so -spence(1+x) = -int_0^x dt/t ln(1+t) = -Li_2(-z) which is what is needed
        return out
    ########################################################

    ############################################################
    def c_hk03(self,f):
        """ Inverse of f_nfw(c) from Hu & Kravtsov (2003)."""
        lnf = np.log(f)
        a1 = 0.5116
        a2 = -0.4283
        a3 = -3.13e-3
        a4 = -3.52e-5
        p = a2 + a3*lnf + a4*(lnf)**2
        out = 1.0/(2*f + 1.0/np.sqrt(0.5625 + a1*f**(2*p)))
        return out
    ############################################################

    ############################################################
    def cDelta(self,cref,Deltaref,Delta):
        """ Convert from c_ref to c_Delta using HK03 prescription. Delta is what multiplies rho_b (not rho_crit)."""
        fDelta = Delta/Deltaref*self.f_nfw(cref)
        out = self.c_hk03(fDelta)
        return out
    ############################################################

    ############################################################
    def MDelta(self,Mref,cref,Deltaref,Delta):
        """ Convert from M_ref,c_ref to M_Delta using HK03 prescription. Delta is what multiplies rho_b (not rho_crit)."""
        out = (self.cDelta(cref,Deltaref,Delta)/cref)**3
        out *= (Mref*(Delta/Deltaref))
        return out
    ############################################################


    ############################################################
    def meanlnalphamz_RP20(self,m,z):
        """ Fitting function <ln alpha |(nu(m,z),z) > from Ramakrishnan&Paranjape20 arXiv:2007.03711. 
        """
        nu = self.calc_nu(m,z=z,lcdm=False)
        out = self.meanlnalphanuz_RP20(nu,z)

        return out
    ############################################################


    ############################################################
    def meanlnalphanuz_RP20(self,nu,z):
        """ Fitting function <ln alpha |nu,z) > from Ramakrishnan&Paranjape20 arXiv:2007.03711. 
        """
        y = np.log10(nu/2.05)
        m00 = -1.688
        m10 = -1.547
        m1 = -2.038
        m2 = -0.706
        out = m00*(1-z) + m10*z + m1*y + m2*y**2

        return out
    ############################################################


    ############################################################
    def siglnalphamz_RP20(self,m,z):
        """ Fitting function sigma(ln alpha |(nu(m,z),z) ) from Ramakrishnan&Paranjape20 arXiv:2007.03711. 
        """
        nu = self.calc_nu(m,z=z,lcdm=False)
        out = self.siglnalphanuz_RP20(nu)

        return out
    ############################################################


    ############################################################
    def siglnalphanuz_RP20(self,nu):
        """ Fitting function sigma(ln alpha |nu,z) from Ramakrishnan&Paranjape20 arXiv:2007.03711. 
        """
        y = np.log10(nu/2.05)
        S0 = 0.187
        S1 = -0.359
        S2 = 0.572
        out = S0 + S1*y + S2*y**2
        out[out < 0.0] = 0.0
        out = np.sqrt(out)

        return out
    ############################################################


    ############################################################
    def rholnalphalncmz_RPS21(self,m,z):
        """ Fitting function rho(ln alpha<->lnc |(nu(m,z),z) ) from Ramakrishnan,Paranjape,Sheth21 arXiv:2012.10170. 
        """
        nu = self.calc_nu(m,z=z,lcdm=False)
        out = self.rholnalphalncnuz_RPS21(nu)

        return out
    ############################################################


    ############################################################
    def rholnalphalncnuz_RPS21(self,nu):
        """ Fitting function rho(ln alpha<->lnc |nu,z ) from Ramakrishnan,Paranjape,Sheth21 arXiv:2012.10170. 
        """
        y = np.log(nu)
        rho0 = 0.1386
        rho1 = -0.5483
        rho2 = 0.1734
        rho3 = 0.1103
        out = rho0 + rho1*y + rho2*y**2 + rho3*y**3

        return out
    ############################################################



    ############################################################
    def Fbbks(self,x):
        """ BBKS function f(x) from their eqn (A15).
            f(x) = (x^3-3x)(1/2)(erf(sqrt{5/2}x) + erf(sqrt{5/8}x))
                   + sqrt{2/5pi}((31x^2/4+8/5)exp(-5x^2/8) + (x^2/2-8/5)exp(-5x^2/2))
        """
        xsq = x*x
        # xcu = x*xsq
        xsq5by2 = 2.5*xsq
        xrt5by2 = x*np.sqrt(2.5) #np.sqrt(xsq5by2)
        out = (0.5*x*(xsq-3)*(sysp.erf(xrt5by2)+sysp.erf(0.5*xrt5by2)) 
               + 1/np.sqrt(2.5*np.pi)*((7.75*xsq+1.6)*np.exp(-0.25*xsq5by2)
                                       +(0.5*xsq-1.6)*np.exp(-xsq5by2)))
        return out
    ############################################################



    ############################################################
    def Fbbks_xyz(self,x,y,z):
        """ BBKS function F(x,y,z) times prefactor from their eqn (A13).
            F(x,y,z) = (x-2z)[(x+z)^2-(3y)^2]*y(y^2-z^2)
            prefactor = 15^(3/2)*sqrt(2pi) needed if y~N(0,1/15) and z~N(0,1/5) in walks.
            Assumes x,y,z have same shapes.
        """
        y2 = y**2
        out = (x-2*z)*((x+z)**2-9*y2)*y*(y2-z**2)
        out *= 15**1.5*np.sqrt(2*np.pi)
        return out
    ############################################################

    
##############################################################


if __name__ == "__main__":
    
    ut = Utilities()

    co_f = Cosmology()

    # import matplotlib as mpl
    # mpl.use('tkagg')
    # import matplotlib.pyplot as plt

    # Om = 0.3
    # c_map = plt.cm.jet
    # Okvals = np.linspace(-0.5,0.5,25)
    # LW = np.linspace(1.0,1.5,Okvals.size)
    # zvals = np.linspace(0.1,2.0,20)

    # for distance_type in ['Ang','Lum']:
    #     COLORS = iter(c_map(np.linspace(0.1,0.9,Okvals.size)))
    #     plt.figure(figsize=(10,7))
    #     img = plt.contourf(np.array([[0,1],[0,1]]),Okvals,cmap=c_map)
    #     plt.clf()
    #     # hack for colorbar from: 
    #     # https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    #     plt.xlabel("$z$",fontsize=14)
    #     ylabel_text = "$d_{\\rm A}(z)$" if distance_type=='Ang' else "$d_{\\rm L}(z)$"
    #     ylabel_text += " $(h^{-1}{\\rm Mpc})$"
    #     plt.ylabel(ylabel_text,fontsize=14)
    #     plt.xlim([0,2]) 
    #     YLIM = [150,1500] if distance_type=='Ang' else [120,12000]
    #     plt.ylim(YLIM)
    #     for k in range(Okvals.size):
    #         Ok = Okvals[k]
    #         co = Cosmology(Ok=Ok,Om=Om,Tcmb=0.0,Pklin='eh')
    #         # LABEL = "$\\Omega_{\\rm k0} = $"+"  {0:.2f}".format(Ok)
    #         col_r = next(COLORS)
    #         func = co.dAng if distance_type=='Ang' else co.dLum
    #         plt.plot(zvals,func(zvals),'-',c=col_r,lw=LW[k])#,label=LABEL)
    #     text_str = "$\\Omega_{\\rm m0} = $"+"  {0:.2f}".format(co.Om)
    #     if distance_type=='Ang':
    #         plt.text(1.0,600,text_str,fontsize=14)
    #         plt.text(1.0,530,"$\\Lambda$CDM Angular diameter distance",fontsize=12)
    #     else:
    #         plt.text(1.0,2000,text_str,fontsize=14)
    #         plt.text(1.0,1500,"$\\Lambda$CDM Luminosity distance",fontsize=12)
    #     cb = plt.colorbar(img,orientation='vertical')
    #     cb.set_label(label="$\\Omega_{\\rm k0}$",fontsize=14)
    #     # plt.legend(loc='upper right')
    #     plt.show()



    # for cstr in ['EdS','dS','Milne','rad','MatRad','open','closed','openRad','closedRad','openLam','closedLam']:
    #     start_time = time()
    #     print 'Testing conformal distance integral for '+cstr+'...'
    #     Om = 1.0
    #     Tcmb = 0.0
    #     Ok = 0.0
    #     hubble = 0.7
    #     if cstr in ['dS','openRad','closedRad','openLam','closedLam']:
    #         Om = 0.0
    #         if cstr != 'dS':
    #             Ok = 0.2 if cstr[:4]=='open' else -0.2
    #             if cstr[-3:] == 'Rad':
    #                 Orad = 1.0 - Ok
    #                 Tcmb = 2.7255*(Orad*hubble**2/4.158e-5)**0.25
    #     elif cstr in ['open','closed']:
    #         Ok = 0.2 if cstr=='open' else -0.2
    #         Om = 1.0 - Ok
    #     elif cstr == 'Milne':
    #         Om = 0.0
    #         Ok = 1.0
    #     elif cstr == 'rad':
    #         Om = 0.0
    #         Tcmb = 2.7255*(1.0*hubble**2/4.158e-5)**0.25
    #     elif cstr == 'MatRad':
    #         Om = 0.5
    #         Tcmb = 2.7255*(0.5*hubble**2/4.158e-5)**0.25
    #     Orad = 4.158e-5*(Tcmb/2.7255)**4/hubble**2
    #     OLam = 1.0 - Om - Orad - Ok
    #     print "... using Om = {0:.3f}, Orad = {1:.3f}, Ok = {2:.3f}, OLam = {3:.3f}, hubble = {4:.3f}".format(Om,Orad,Ok,OLam,hubble)

    #     co = Cosmology(Om=Om,Tcmb=Tcmb,Ok=Ok,hubble=hubble)        
    #     nz = 1000
    #     zob = np.linspace(0.0,10.0,nz) if cstr != 'closedLam' else np.linspace(0.0,0.9*np.sqrt(-1.0*Ok/OLam),nz)
    #     z = zob + zob[1]-zob[0]
    #     print "... redshift range {0:.3f}-{1:.3f} with min length {2:.3f}".format(zob[0],z[-1],z[0]-zob[0])

    #     quad_vals = np.ones((nz,nz),dtype=float)
    #     analytic_vals = np.ones((nz,nz),dtype=float)
    #     dchi = np.zeros((nz,nz),dtype=float)
    #     diag_ids = np.diag_indices(nz,ndim=2)

    #     print '... using quad'
    #     quad_vals[diag_ids] = co.chiConf(zob,z)
    #     for iz in range(nz-1):
    #         quad_vals[iz,iz+1:] = co.chiConf(zob[iz],z[iz+1:])
    #         ut.status_bar(iz,nz-1)    

    #     print '... using analytical'
    #     if cstr == 'EdS':
    #         analytic_vals[diag_ids] = co.chiConf_analytic_EdS(zob,z)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_EdS(zob[iz],z[iz+1:])
    #             ut.status_bar(iz,nz-1)    
    #     elif cstr == 'dS':
    #         analytic_vals[diag_ids] = co.chiConf_analytic_dS(zob,z)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_dS(zob[iz],z[iz+1:])
    #             ut.status_bar(iz,nz-1)    
    #     elif cstr == 'Milne':
    #         analytic_vals[diag_ids] = co.chiConf_analytic_Milne(zob,z)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_Milne(zob[iz],z[iz+1:])
    #             ut.status_bar(iz,nz-1)    
    #     elif cstr == 'rad':
    #         analytic_vals[diag_ids] = co.chiConf_analytic_pureRad(zob,z)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_pureRad(zob[iz],z[iz+1:])
    #             ut.status_bar(iz,nz-1)    
    #     elif cstr == 'MatRad':
    #         analytic_vals[diag_ids] = co.chiConf_analytic_MatRad(zob,z,Orad=Orad)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_MatRad(zob[iz],z[iz+1:],Orad=Orad)
    #             ut.status_bar(iz,nz-1)    
    #     elif cstr in ['open','closed']:
    #         analytic_vals[diag_ids] = co.chiConf_analytic_Curv_MatRadLam(zob,z,Ok=Ok,other=0)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_Curv_MatRadLam(zob[iz],z[iz+1:],Ok=Ok,other=0)
    #             ut.status_bar(iz,nz-1)    
    #     elif cstr in ['openRad','closedRad']:
    #         analytic_vals[diag_ids] = co.chiConf_analytic_Curv_MatRadLam(zob,z,Ok=Ok,other=1)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_Curv_MatRadLam(zob[iz],z[iz+1:],Ok=Ok,other=1)
    #             ut.status_bar(iz,nz-1)    
    #     elif cstr in ['openLam','closedLam']:
    #         analytic_vals[diag_ids] = co.chiConf_analytic_Curv_MatRadLam(zob,z,Ok=Ok,other=2)
    #         for iz in range(nz-1):
    #             analytic_vals[iz,iz+1:] = co.chiConf_analytic_Curv_MatRadLam(zob[iz],z[iz+1:],Ok=Ok,other=2)
    #             ut.status_bar(iz,nz-1)    

    #     dchi = quad_vals/analytic_vals - 1.0
    #     dchi_max = np.max(np.abs(dchi))
    #     print "... max relative error wrt analytical answer = {0:.3e}".format(dchi_max)
    #     ut.time_this(start_time)


    
    # scale = 1.0
    # redshift = co_f.scale_to_redshift(scale)
    # print "given scale = {0:.3f} corresponds to redshift = {1:.3f}".format(scale,redshift)

    # redshift = 1.0
    # scale = co_f.redshift_to_scale(redshift)
    # print "given redshift = {0:.3f} corresponds to scale = {1:.3f}".format(redshift,scale)

    # print co_f.TINY

    # co_cn = Cosmology(Ok=-0.01)
    # co_cp = Cosmology(Ok=0.01)

    # dz = 0.02
    # z = 1.0
    # print "Flat cosmology:"
    # # print "Hubble distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(dz,co_f.dHub0(dz))
    # # print "Hubble distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(dz,co_f.dHubz(0.0,dz))
    # # print "Hubble distance from z = {0:.2f} to z = {1:.2f}: {2:.3f} Mpc/h".format(z,z+dz,co_f.dHubz(z,dz))
    # print "Conformal distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_f.chiConf(0.0,z))
    # print "Conformal distance from z = {0:.2f} to z = {1:.2f}: {2:.3f} Mpc/h".format(z,z+dz,co_f.chiConf(z,z+dz))
    
    # print "Ok = {0:.4f}:".format(co_cn.Ok)
    # # print "Hubble distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(dz,co_cn.dHub0(dz))
    # # print "Hubble distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(dz,co_cn.dHubz(0.0,dz))
    # # print "Hubble distance from z = {0:.2f} to z = {1:.2f}: {2:.3f} Mpc/h".format(z,z+dz,co_cn.dHubz(z,dz))
    # print "Conformal distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cn.chiConf(0.0,z))
    # print "Conformal distance from z = {0:.2f} to z = {1:.2f}: {2:.3f} Mpc/h".format(z,z+dz,co_cn.chiConf(z,z+dz))

    # print "Ok = {0:.4f}:".format(co_cp.Ok)
    # # print "Hubble distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(dz,co_cp.dHub0(dz))
    # # print "Hubble distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(dz,co_cp.dHubz(0.0,dz))
    # # print "Hubble distance from z = {0:.2f} to z = {1:.2f}: {2:.3f} Mpc/h".format(z,z+dz,co_cp.dHubz(z,dz))
    # print "Conformal distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cp.chiConf(0.0,z))
    # print "Conformal distance from z = {0:.2f} to z = {1:.2f}: {2:.3f} Mpc/h".format(z,z+dz,co_cp.chiConf(z,z+dz))

    # zvals = np.linspace(0.2,1.0,12)
    # print '... testing z vector, zob=0'
    # print co_f.chiConf(0.0,zvals)

    # zobvals = 1.0*zvals
    # zvals += 0.5
    # print '... testing z scalar, zob vector'
    # print co_f.chiConf(zobvals,zvals[-1])
    # print '... testing z, zob vectors'
    # print co_f.chiConf(zobvals,zvals)

    # for n in np.logspace(3.0,4.0,2):
    #     nz = int(n)
    #     start_time = time()
    #     print "... timing for vector zob,z chiCom() with length 10**{0:.1f}".format(np.log10(nz))
    #     zvals = np.linspace(0.1,2.0,nz)
    #     zobvals = 1.0*zvals
    #     zvals += 0.5
    #     co_f.chiConf(zobvals,zvals)
    #     ut.time_this(start_time)

    # z = 1.0
    # print "Flat cosmology:"
    # print "Comoving distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_f.rCom(z))
    # print "Luminosity distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_f.dLum(z))
    # print "Ang diam   distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_f.dAng(z))
    # print "Lookback time to z = {0:.2f}: {1:.3f} Gyr".format(z,co_f.lookback(z))

    # print "Ok = {0:.4f}:".format(co_cn.Ok)
    # print "Comoving distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cn.rCom(z))
    # print "Luminosity distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cn.dLum(z))
    # print "Ang diam   distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cn.dAng(z))
    # print "Lookback time to z = {0:.2f}: {1:.3f} Gyr".format(z,co_cn.lookback(z))

    # print "Ok = {0:.4f}:".format(co_cp.Ok)
    # print "Comoving distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cp.rCom(z))
    # print "Luminosity distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cp.dLum(z))
    # print "Ang diam   distance from z = 0 to z = {0:.2f}: {1:.3f} Mpc/h".format(z,co_cp.dAng(z))
    # print "Lookback time to z = {0:.2f}: {1:.3f} Gyr".format(z,co_cp.lookback(z))

    # print "Age of universe at z = 0 with Ok = {0:.2f}: {1:.3f} Gyr".format(co_f.Ok,co_f.age(0.0))
    # print "Age of universe at z = 0 with Ok = {0:.2f}: {1:.3f} Gyr".format(co_cn.Ok,co_cn.age(0.0))
    # print "Age of universe at z = 0 with Ok = {0:.2f}: {1:.3f} Gyr".format(co_cp.Ok,co_cp.age(0.0))


    # for n in np.logspace(3.0,4.0,2):
    #     nz = int(n)
    #     start_time = time()
    #     print "... timing for vector z age() with length 10**{0:.1f}".format(np.log10(nz))
    #     zvals = np.linspace(0.0,2.0,nz)
    #     co_f.age(zvals)
    #     ut.time_this(start_time)
