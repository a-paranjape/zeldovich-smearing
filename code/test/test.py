import numpy as np
import sys,gc
from pathlib import Path

sys.path.append('../')
from paths import *

sys.path.append(Sahyadri_Path)
from correlations import TwoPointCorrelationFunctionPeriodic,PowerSpectrum
from utilities import Utilities

sys.path.append(ML_Path)
from mllib import MLUtilities

from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['legend.labelspacing'] = 0.25
FS = 18
FS2 = 15
FS3 = 13
FSL = 22
FSS = 11
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.minor.size'] = 3

ut = Utilities()
rng = np.random.RandomState(1991)

do_2pcf = False
do_Pk = True

implement_AP = False

start_time = time()

Lbox = 2000.0
N = 300000
los = 2
if do_Pk:
    grid = 256
    N *= 10

smin,smax = 65.0,125.0

scale_factors = [0.5]*2
if implement_AP:
    alpha_par = 1.05
    alpha_perp = 1.10
    non_los = np.where(np.arange(3) != los)[0]
    scale_factors[0] *= alpha_par
    scale_factors[1] *= alpha_perp

if do_2pcf:
    tpcf = TwoPointCorrelationFunctionPeriodic(smin=smin,smax=smax,Lbox=Lbox,los=los,aniso=False,
                                               pbc=False,scale_factors=scale_factors,verbose=False)
    tpcf_per = TwoPointCorrelationFunctionPeriodic(smin=smin,smax=smax,Lbox=Lbox,los=los,aniso=False,verbose=False)
    nreal = 100
    DD_all  = np.zeros((nreal,tpcf.n_s))
    DD_rest = np.zeros((nreal,tpcf.n_s))

    for n in range(nreal):
        pos = Lbox*rng.rand(N,3)

        cond_keep = np.ones(pos.shape[0],dtype=bool)
        for i in range(3):
            cond_keep = cond_keep & (pos[:,i] <= 0.5*Lbox)
        pos_rest = pos[cond_keep].copy()
        del cond_keep

        if implement_AP:
            pos_rest[:,los] *= alpha_par
            pos_rest[:,non_los] *= alpha_perp

        pos = pos[:int(N/8),:] # approximately match counts with restricted sample

        DD_n = tpcf_per.DD_split(pos)
        DD_all[n] = DD_n

        DD_n = tpcf.DD_split(pos_rest)
        DD_rest[n] = DD_n

        ut.status_bar(n,nreal)

    DD_mean = np.mean(DD_all,axis=0)
    DD_std = np.std(DD_all,axis=0)

    DD_mean_rest = np.mean(DD_rest,axis=0)
    DD_std_rest = np.std(DD_rest,axis=0)

    RR = tpcf_per.RR_theory()
    RR_rest = tpcf.RR_theory()

    plt.figure(figsize=(5,5))
    plt.xlim(smin/Lbox,smax/Lbox)
    plt.ylim(0,0.001)
    plt.xlabel('$s\\,/\\,L_{\\rm box}$')
    plt.errorbar(tpcf.smid/Lbox,DD_mean,yerr=DD_std,capsize=5,marker='o',c='r',ls='none',markerfacecolor='none')
    plt.plot(tpcf.smid/Lbox,RR,'k--',lw=1)
    plt.errorbar(tpcf.smid/Lbox,DD_mean_rest,yerr=DD_std_rest,capsize=5,marker='o',c='r',ls='none',label='DD')
    plt.plot(tpcf.smid/Lbox,RR_rest,'k-',lw=1,label='RR')
    plt.minorticks_on()
    plt.legend(loc='upper left')

    plt.text(0.045,0.0008,'uniform randoms',fontsize=FS3)
    plt.text(0.052,0.0002,'periodic cube',fontsize=FSS)
    nper_str = 'non-periodic cub' + ('oid' if implement_AP else 'e')
    plt.text(0.035,0.0006,nper_str,fontsize=FSS)
    if implement_AP:
        plt.text(0.035,0.00055,"$\\alpha_{{\\|}} = {0:.2f}$".format(alpha_par),fontsize=FSS)
        plt.text(0.035,0.00050,"$\\alpha_{{\\perp}} = {0:.2f}$".format(alpha_perp),fontsize=FSS)
    filename = 'RRtest-' + ('AP' if implement_AP else 'noAP') + '.pdf'
    print('Writing to file:',filename)
    plt.savefig(filename,bbox_inches='tight')

# ANISOTROPIC 2PCF,PK
nreal = 50
alpha_par = 1.05
alpha_perp = 1.10

scale_factors = [0.5]*2
if implement_AP:
    scale_factors[0] *= alpha_par
    scale_factors[1] *= alpha_perp

non_los = np.where(np.arange(3) != los)[0]

if do_2pcf:
    print('... setting up 2pcf')
    tpcf = TwoPointCorrelationFunctionPeriodic(smin=smin,smax=smax,Lbox=Lbox,los=los,aniso=True,
                                               pbc=False,scale_factors=scale_factors,verbose=False) 
    cf_all = np.zeros((nreal,tpcf.L_Max,tpcf.n_s))

if do_Pk:
    print('... setting up Pk')
    alpha_max = np.max([alpha_par,alpha_perp,1.0]) if implement_AP else 1.0
    K_Min_Pk,K_Max_Pk,N_Kbin = 0.01,0.2,19
    powspec = PowerSpectrum(grid=grid,kmin=K_Min_Pk,kmax=K_Max_Pk,Lbox=Lbox,lgbin=None,nbin=N_Kbin,
                            anisotropic=True,los=los,scale_factors=scale_factors,verbose=False) 
    pk_all = np.zeros((nreal,powspec.L_Max,powspec.nbin))

print('... calculating')
for n in range(nreal):
    pos = Lbox*rng.rand(N,3)
    cond_keep = np.ones(pos.shape[0],dtype=bool)
    for i in range(3):
        cond_keep = cond_keep & (pos[:,i] <= 0.5*Lbox)
    pos = pos[cond_keep].copy()
    del cond_keep

    if implement_AP:
        pos[:,los] *= alpha_par
        pos[:,non_los] *= alpha_perp

    if do_2pcf:
        cf_n = tpcf.auto_CF(pos)
        cf_all[n] = cf_n

    if do_Pk:
        pk_n = powspec.Pk_grid(pos.T,aniso=True)
        pk_all[n] = pk_n
    
    ut.status_bar(n,nreal)

cols = ['crimson','indigo','forestgreen']

if do_2pcf:
    cf_mean = np.mean(cf_all,axis=0)
    cf_std = np.std(cf_all,axis=0)

    Y = 0.25
    plt.figure(figsize=(5,5))
    plt.xlim(smin/Lbox,smax/Lbox)
    plt.ylim(-0.2*Y,0.2*Y)
    plt.xlabel('$s\\,/\\,L_{\\rm box}$')
    plt.ylabel("\\xi_{{\\ell}}(s)")
    for L in range(tpcf.L_Max):
        plt.errorbar(tpcf.smid/Lbox,cf_mean[L],yerr=cf_std[L],capsize=5,marker='o',c=cols[L],ls='none',
                     label="$\\ell = {0:d}$".format(2*L))
    plt.axhline(0.0,c='k',ls=':',lw=1)
    plt.minorticks_on()
    plt.legend(loc='upper right')
    plt.text(0.035,-0.1*Y,'uniform randoms',fontsize=FS3)
    nper_str = 'non-periodic cub' + ('oid' if implement_AP else 'e')
    plt.text(0.035,-0.12*Y,nper_str,fontsize=FSS)
    if implement_AP:
        plt.text(0.035,-0.14*Y,"$\\alpha_{{\\|}} = {0:.2f}$".format(alpha_par),fontsize=FSS)
        plt.text(0.035,-0.155*Y,"$\\alpha_{{\\perp}} = {0:.2f}$".format(alpha_perp),fontsize=FSS)
    filename = 'aniso2pcf-test-' + ('AP' if implement_AP else 'noAP') + '.pdf'
    print('Writing to file:',filename)
    plt.savefig(filename,bbox_inches='tight')
    
if do_Pk:
    pk_mean = np.mean(pk_all,axis=0)
    pk_std = np.std(pk_all,axis=0)

    nbar = pos.shape[0]/powspec.Vbox
    # Claude 4.6 code for estimating std dev of P_ell for randoms
    sigma_sq_weights = (powspec.Pell**2 @ powspec.I_MU_SIZE.T) / (powspec.I_SIZE + powspec.TINY)
    # shape (L_Max, nbin)
    sigma_ell = (np.sqrt(2) * (2*np.arange(powspec.L_Max)*2 + 1)[:,None]
                 / nbar
                 * np.sqrt(sigma_sq_weights / (powspec.I_SIZE + powspec.TINY)))
    # shape (L_Max, nbin)

    plt.figure(figsize=(5,5))
    plt.xlim(powspec.kmin,powspec.kmax)
    Y = 2.5*np.max(powspec.ktab*sigma_ell)
    plt.ylim(-Y,Y)
    plt.xlabel('$k\\,(h\\,/\\,{{\\rm Mpc}})$')
    plt.ylabel("$k\\,P_{{\\ell}}(k) \\,\\,({{\\rm Mpc}}\\,/\\,h)^{{2}}$")
    for L in range(powspec.L_Max):
        plt.errorbar(powspec.ktab,powspec.ktab*pk_mean[L],yerr=powspec.ktab*pk_std[L],capsize=5,marker='o',c=cols[L],ls='none',
                     label="$\\ell = {0:d}$".format(2*L))
        plt.plot(powspec.ktab,powspec.ktab*sigma_ell[L],c=cols[L],ls='--',lw=1)
        plt.plot(powspec.ktab,-powspec.ktab*sigma_ell[L],c=cols[L],ls='--',lw=1)
    plt.axhline(0.0,c='k',ls=':',lw=1)
    plt.minorticks_on()
    plt.legend(loc='upper right')
    plt.text(0.035,-0.5*Y,'uniform randoms',fontsize=FS3)
    nper_str = 'non-periodic cub' + ('oid' if implement_AP else 'e')
    plt.text(0.035,-0.6*Y,nper_str,fontsize=FSS)
    if implement_AP:
        plt.text(0.035,-0.7*Y,"$\\alpha_{{\\|}} = {0:.2f}$".format(alpha_par),fontsize=FSS)
        plt.text(0.035,-0.8*Y,"$\\alpha_{{\\perp}} = {0:.2f}$".format(alpha_perp),fontsize=FSS)
    plt.text(0.035,0.6*Y,"$N = {0:d}\\,;\\,N_{{\\rm grid}} = {1:d}^{{3}}$".format(pos.shape[0],powspec.grid),fontsize=FSS)
    plt.text(0.035,0.5*Y,"$L_{{\\rm box}} = {0:.0f}\\,{{\\rm Mpc}}\\,/\\,h$".format(powspec.Lbox),fontsize=FSS)
    filename = 'anisoPk-test-' + ('AP' if implement_AP else 'noAP') + '.pdf'
    print('Writing to file:',filename)
    plt.savefig(filename,bbox_inches='tight')

ut.time_this(start_time)
