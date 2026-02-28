import numpy as np
import sys,gc#,fitsio
from pathlib import Path
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
SPEED_OF_LIGHT = 3e5*u.km/u.s


from paths import *

sys.path.append(Sahyadri_Path)
from correlations import TwoPointCorrelationFunctionPeriodic,PowerSpectrum
from utilities import Utilities

sys.path.append(ML_Path)
from mllib import MLUtilities

from time import time
import matplotlib.pyplot as plt

ut = Utilities()

def queuer_box(phase,tpcf,powspec,aniso,los,alpha_par,alpha_perp,redshift,M_min,kmin,kmax,h_fid,max_file,ddstem,down_to,rng,do_2pcf,do_Pk,mdict):
    data_dir = ddstem + '{0:03d}/z{1:.3f}/'.format(phase,redshift)
    print(data_dir+'\n')
    # data = fitsio.read(data_dir+'sub_0_nsub_4.fits')
    with fits.open(data_dir+'sub_0_nsub_4.fits') as hdu:
        data = hdu[1].data
    data_size_orig = data['Mabacus'].size
    data = data[data['Mabacus'] >= M_min]
    ut.status_bar(0,max_file)
    for i in range(1,max_file):
        # data_i = fitsio.read(data_dir+'sub_{0:d}_nsub_4.fits'.format(i))
        with fits.open(data_dir+'sub_{0:d}_nsub_4.fits'.format(i)) as hdu_i:
            data_i = hdu_i[1].data
        data_size_orig += data_i['Mabacus'].size
        data_i = data_i[data_i['Mabacus'] >= M_min]
        data = np.concatenate((data,data_i))
        data_i = None
        ut.status_bar(i,max_file)
    print('... ... kept {0:d} of {1:d} objects'.format(data['Mabacus'].size,data_size_orig))
    
    data_size = data['Mabacus'].size
    ind = rng.choice(np.arange(data_size),size=data_size//down_to,replace=False) if down_to > 1 else np.arange(data_size)
    data_use = data['x_L2com'][ind]

    E_z = 0.0
    h = 0.0
    Lbox = 0.0
    with open(data_dir+'header.txt','r') as f:
        lines = f.readlines()
    for line in lines:
        line.strip()
        items = [item.strip() for item in line.split(':')]
        if items[0] == 'HubbleNow':
            E_z += float(items[1])
        if items[0] == 'H0':
            h += 0.01*float(items[1])
        if items[0] == 'BoxSizeHMpc':
            Lbox += float(items[1])
    print('... h = {0:.4f}; h_fid = {1:.4f}'.format(h,h_fid))
            
    if aniso:
        print('... E(z) = {0:.4f}'.format(E_z))

        # from prop vel to comov separation
        vel_to_cMpch = (1+redshift)/(100*E_z) # same as BoxSizeHMpc/VelZSpace_to_kms as suggested on website
        zred = data['v_L2com'][ind,los]*vel_to_cMpch

        # add real-space separation
        zred += data_use[:,los]
        
        # store final los separation
        data_use[:,los] = zred

        # account for true-to-fiducial conversion
        non_los = np.where(np.arange(3) != los)[0]
        data_use[:,los] *= alpha_par
        data_use[:,non_los] *= alpha_perp
        
        # account for pbc
        data_use %= Lbox
        
        del zred
        
    del ind,data
    gc.collect()

    data_use *= (h_fid/h) # convert to Mpc/h_fid [ONLY place that this is needed in this routine]

    temp = {}
    if do_2pcf:
        if aniso & (tpcf.los != los):
            raise Exception('Mismatched los in tpcf')
        start_time = time()
        temp['2pcf'] = tpcf.auto_CF(data_use)
        ut.time_this(start_time)
        
    if do_Pk:
        if aniso & (powspec.los != los):
            raise Exception('Mismatched los in powspec')
        start_time = time()
        Pk = powspec.Pk_grid(data_use.T,aniso=aniso)
        temp['Pk'] = Pk
        ut.time_this(start_time)

        if aniso:
            Sig2obs = np.zeros(powspec.L_Max)
            if powspec.lgbin is not None:
                print("Warning!: non-linear k-bins detected. Not calculating Sig2obs.")
            else:
                ind_k = np.where((powspec.ktab > kmin) & (powspec.ktab < kmax))[0]
                if ind_k.size:
                    # Pk has shape (L,k)
                    Sig2obs += np.sum(Pk.T[ind_k],axis=0)*powspec.dk/(6*np.pi**2)

            # powspec arrays already in Mpc/h_fid units
            # Sig2obs *= hfid_by_h**2 # convert to (Mpc/h_fid)^2
            
            temp['Sig2obs'] = Sig2obs

    mdict[phase] = temp
    return

def check_units(name,quantity,expected_unit):
    if quantity.unit != expected_unit:
        raise Exception('expecting unit',expected_unit,'for',name+', got',quantity.unit)
    return

if __name__ == "__main__":
    start_time = time()

    ##################################
    # unique string to identify sample
    ##################################
    # -- if this ends with '-xN' then stats will be averaged over N phases starting with ph000.
    # -- if not, then only one phase will be used, specified by Ref_Phase below.
    Sample = 'DESI-LRG2' 

    print('AbacusSummit pairwise correlations for sample:',Sample)
    
    data_dir = '../examples/data/AbacusSummit/base_c000/' + Sample + '/'
    Path(data_dir).mkdir(parents=True,exist_ok=True) # folder to store data products
    
    ml = MLUtilities()

    Do_2pcf = False
    Do_Pk = True
    N_Phase = 3   # number of boxes to analyse
    Ref_Phase = 0 # index of box to use as data
    
    Down_To = 1   # default 1
    Grid = 256    # default 256 (better than 1% convergence at k <= 0.2 h/Mpc)
    Max_File = 64 # default 64
    NProc = np.min([N_Phase,6])
    
    Redshift = 0.8 # 0.8
    print('... working at redshift z = {0:.3f}'.format(Redshift))

    Mmin_dict = {'DESI-LRG2':8e12,
                 # below are existing tests with 3x DESI volume
                 'DESI-LRG2-AP-x3':8e12 if Redshift > 0.5 else 1.25e13,
                 'DESI-LRG2-noAP-x3':8e12 if Redshift > 0.5 else 1.25e13}
    
    M_min = Mmin_dict[Sample]
    print('... retaining halos with M >= {0:.2e} Msun/h'.format(M_min))
    print('... downsampling by factor {0:d}'.format(Down_To))

    Aniso = False
    LOS = 2
    L_Max = 3

    # Abacus baseline c000 cosmology
    co_c000 = FlatLambdaCDM(H0=67.36,Om0=0.315192,Ob0=0.049302,Neff=3.04,m_nu=[0.060,0.,0.] * u.eV,Tcmb0=2.7255)
    # co_c000 = FlatLambdaCDM(H0=67.36,Om0=0.315192,Ob0=0.049302,Tcmb0=2.7255)
    print('... abacus cosmo c000:',co_c000)
    # ... distances
    d_Hub = (SPEED_OF_LIGHT/co_c000.H(Redshift))
    d_Ang_com = (1+Redshift)*co_c000.angular_diameter_distance(Redshift)
    # ... check units
    check_units('d_Hub',d_Hub,'Mpc')
    check_units('d_Ang_com',d_Ang_com,'Mpc')

    # fiducial cosmology, from arXiv:2410.21374
    h_fid = 0.6737 
    co_fid = FlatLambdaCDM(H0=100*h_fid,Om0=0.3153,Ob0=0.04929,Tcmb0=2.7255)
    print('...   fiducial cosmo:',co_fid)
    # ... distances
    d_Hub_fid = (SPEED_OF_LIGHT/co_fid.H(Redshift))
    d_Ang_com_fid = (1+Redshift)*co_fid.angular_diameter_distance(Redshift)
    # ... check units
    check_units('d_Hub_fid',d_Hub_fid,'Mpc')
    check_units('d_Ang_com_fid',d_Ang_com_fid,'Mpc')

    print('... h = {0:.4f}; h_fid = {1:.4f}'.format(co_c000.h,h_fid))
    hfid_by_h = h_fid/co_c000.h
    
    alpha_par = d_Hub_fid.value/d_Hub.value
    alpha_perp = d_Ang_com_fid.value/d_Ang_com.value

    alpha_AP = alpha_perp/alpha_par
    alpha_iso = (alpha_par*alpha_perp**2)**(1/3.)

    DaAP = alpha_AP - 1.0
    Daiso = alpha_iso - 1.0

    print('... alpha_par : 1 + {0:.3e}'.format(alpha_par-1))
    print('... alpha_perp: 1 + {0:.3e}'.format(alpha_perp-1))
    print('...  DaAP: {0:+.3e}'.format(DaAP))
    print('... Daiso: {0:+.3e}'.format(Daiso))
    
    Abacus_Stem = Abacus_Path + 'AbacusSummit_base_c000_ph'
    Lbox_AbacusSummit = 2000.0*hfid_by_h # AbacusSummit box size in Mpc/h_fid
    
    start_time_setup = time()
    if Do_2pcf:
        tpcf = TwoPointCorrelationFunctionPeriodic(smin=65.0,smax=125.0,n_s=30,aniso=Aniso,L_Max=L_Max,Lbox=Lbox_AbacusSummit,los=LOS)
        tpcf.verbose = False
    else:
        tpcf = None

    if Do_Pk:
        # interpret all scales in h_fid/Mpc
        K_Min_Pk,K_Max_Pk,N_Kbin = 0.01,0.2,19
        powspec = PowerSpectrum(grid=Grid,kmin=K_Min_Pk,kmax=K_Max_Pk,Lbox=Lbox_AbacusSummit,lgbin=None,nbin=N_Kbin,anisotropic=Aniso,los=LOS)
        K_Min,K_Max = 0.02,0.05
    else:
        powspec = None
        K_Min,K_Max = None,None
    ut.time_this(start_time_setup)
        
    rng = np.random.RandomState(42)

    if Do_2pcf:
        task_tuple = (tpcf,powspec,Aniso,LOS,alpha_par,alpha_perp,Redshift,M_min,K_Min,K_Max,h_fid,Max_File,Abacus_Stem,Down_To,rng,Do_2pcf,Do_Pk)
        tasks = [task_tuple]*N_Phase
        # tasks = []
        # for phase in range(N_Phase):
        #     tasks.append(task_tuple)
        pw_dict = ml.run_processes(tasks,queuer_box,NProc)

    if Do_Pk:
        pw_dict = {}
        for phase in range(N_Phase):
            queuer_box(phase,tpcf,powspec,Aniso,LOS,alpha_par,alpha_perp,Redshift,M_min,K_Min,K_Max,h_fid,Max_File,Abacus_Stem,Down_To,rng,Do_2pcf,Do_Pk,pw_dict)

    if Do_2pcf:
        file_stem_2pcf = 'xi'
        if Aniso:
            file_stem_2pcf += '_LMax{0:d}'.format(L_Max)
        file_stem_2pcf += '_lgMmin{0:.2f}_z{1:.3f}'.format(np.log10(M_min),Redshift)

        xi_file = data_dir + file_stem_2pcf + '.txt'
        scales_file = data_dir + file_stem_2pcf + '_scales.txt'
        xi_file_colwise = data_dir + file_stem_2pcf + '_colwise.txt'
        xi_phase_stem = data_dir + file_stem_2pcf + '_ph'
        
        if Aniso:
            xi_phases = np.zeros((N_Phase,L_Max,tpcf.n_s))
        else:
            xi_phases = np.zeros((N_Phase,tpcf.n_s))
        for phase in range(N_Phase):
            xi_phases[phase] = pw_dict[phase]['2pcf']

        err_xi = np.std(xi_phases,axis=0)
        if Sample[-2-len(str(N_Phase)):] == '-x{0:d}'.format(N_Phase):
            xi = np.mean(xi_phases,axis=0)
            err_xi /= np.sqrt(N_Phase-1 + 1e-15) # placeholder until better errors available
        else:
            xi = xi_phases[Ref_Phase].copy()

        print('Writing to file: ',scales_file)
        np.savetxt(scales_file,tpcf.smid,fmt='%.8e')

        print('Writing to file: ',xi_file)
        np.savetxt(xi_file,xi.flatten(),fmt='%.8e')

        print('Writing to file: ',xi_file_colwise)
        with open(xi_file_colwise,'w') as f:
            if Aniso:
                f.write("# s (Mpc/h_fid) | ( xi|err_xi )_{ell="+','.join([str(2*L) for L in range(tpcf.L_Max)])+"}\n")
            else:
                f.write("# r (Mpc/h_fid) | xi | err_xi\n")
        for s in range(tpcf.n_s):
            if Aniso:
                seq = [tpcf.smid[s]]
                for L in range(L_Max):
                    seq.append(xi[L,s])
                    seq.append(err_xi[L,s])
                ut.write_to_file(xi_file_colwise,seq)
            else:
                ut.write_to_file(xi_file_colwise,[tpcf.smid[s],xi[s],err_xi[s]])

        print('Writing to files: ',xi_phase_stem)
        for phase in range(N_Phase):
            ph_str = '{0:03d}'.format(phase)
            print('... phase '+ph_str)
            xi_file_phase = xi_phase_stem + ph_str + '.txt'
            with open(xi_file_phase,'w') as f:
                if Aniso:
                    f.write("# s (Mpc/h_fid) | xi_{ell="+','.join([str(2*L) for L in range(tpcf.L_Max)])+"}\n")
                else:
                    f.write("# r (Mpc/h_fid) | xi\n")
            for s in range(tpcf.n_s):
                if Aniso:
                    seq = [tpcf.smid[s]]
                    for L in range(tpcf.L_Max):
                        seq.append(xi_phases[phase,L,s])
                    ut.write_to_file(xi_file_phase,seq)
                else:
                    ut.write_to_file(xi_file_phase,[tpcf.smid[s],xi_phases[phase,s]])

    if Do_Pk:
        file_stem_Pk = 'Pk'
        file_stem_Sig2obs = 'Sig2obs'
        if Aniso:
            file_stem_Pk += '_LMax{0:d}'.format(L_Max)
            file_stem_Sig2obs += '_LMax{0:d}'.format(L_Max)
        file_stem_Pk += '_lgMmin{0:.2f}_z{1:.3f}'.format(np.log10(M_min),Redshift)
        file_stem_Sig2obs += '_lgMmin{0:.2f}_z{1:.3f}'.format(np.log10(M_min),Redshift)

        Sig2obs_file = data_dir + file_stem_Sig2obs + '.txt'
        Sig2obs_phase_stem = data_dir + file_stem_Sig2obs + '_ph'
        Pk_file = data_dir + file_stem_Pk + '.txt'
        Pk_phase_stem = data_dir + file_stem_Pk + '_ph'

        if Aniso:
            Pk_phases = np.zeros((N_Phase,L_Max,powspec.nbin))
            Sig2obs_phases = np.zeros((N_Phase,L_Max))
        else:
            Pk_phases = np.zeros((N_Phase,powspec.nbin))
            
        for phase in range(N_Phase):
            Pk_phases[phase] = pw_dict[phase]['Pk']
            if Aniso:
                Sig2obs_phases[phase] = pw_dict[phase]['Sig2obs']

        err_Pk = np.std(Pk_phases,axis=0)
        if Sample[-2-len(str(N_Phase)):] == '-x{0:d}'.format(N_Phase):
            Pk = np.mean(Pk_phases,axis=0)
            err_Pk /= np.sqrt(N_Phase-1 + 1e-15) # placeholder until better errors available
        else:
            Pk = Pk_phases[Ref_Phase].copy()

        if Aniso:
            err_Sig2obs = np.std(Sig2obs_phases,axis=0)
            if Sample[-2-len(str(N_Phase)):] == '-x{0:d}'.format(N_Phase):
                Sig2obs = np.mean(Sig2obs_phases,axis=0)
                err_Sig2obs /= np.sqrt(N_Phase-1 + 1e-15) # placeholder until better errors available
            else:
                Sig2obs = Sig2obs_phases[Ref_Phase].copy()
        
        print('Writing to file: ',Pk_file)
        with open(Pk_file,'w') as f:
            if Aniso:
                f.write("# k (h_fid/Mpc) | ( Pk|err_Pk )_{ell="+','.join([str(2*L) for L in range(L_Max)])+"} (Mpc/h_fid)^3\n")
            else:
                f.write("# k (h_fid/Mpc) | (Pk | err_Pk) (Mpc/h_fid)^3\n")
        for k in range(powspec.nbin):
            if Aniso:
                seq = [powspec.ktab[k]]
                for L in range(L_Max):
                    seq.append(Pk[L,k])
                    seq.append(err_Pk[L,k])
                ut.write_to_file(Pk_file,seq)
            else:
                ut.write_to_file(Pk_file,[powspec.ktab[k],Pk[k],err_Pk[k]])

        print('Writing to files: ',Pk_phase_stem)
        for phase in range(N_Phase):
            ph_str = '{0:03d}'.format(phase)
            print('... phase '+ph_str)
            Pk_file_phase = Pk_phase_stem + ph_str + '.txt'
            with open(Pk_file_phase,'w') as f:
                if Aniso:
                    f.write("# k (h_fid/Mpc) | Pk_{ell="+','.join([str(2*L) for L in range(L_Max)])+"} (Mpc/h_fid)^3\n")
                else:
                    f.write("# k (h_fid/Mpc) | Pk (Mpc/h_fid)^3\n")
            for k in range(powspec.nbin):
                if Aniso:
                    seq = [powspec.ktab[k]]
                    for L in range(L_Max):
                        seq.append(Pk_phases[phase,L,k])
                    ut.write_to_file(Pk_file_phase,seq)
                else:
                    ut.write_to_file(Pk_file_phase,[powspec.ktab[k],Pk_phases[phase,k]])


        if Aniso:
            print('Writing to file: ',Sig2obs_file)
            with open(Sig2obs_file,'w') as f:
                f.write("# (Sig2obs | err) (Mpc/h_fid)^2\n")
            for L in range(L_Max):
                ut.write_to_file(Sig2obs_file,[Sig2obs[L],err_Sig2obs[L]])

            print('Writing to files: ',Sig2obs_phase_stem)
            for phase in range(N_Phase):
                ph_str = '{0:03d}'.format(phase)
                print('... phase '+ph_str)
                Sig2obs_file_phase = Sig2obs_phase_stem + ph_str + '.txt'
                np.savetxt(Sig2obs_file_phase,Sig2obs_phases[phase],fmt='%.8e',header="Sig2obs (Mpc/h_fid)^2")

                
                    
    print('... all done!')
    ut.time_this(start_time)
