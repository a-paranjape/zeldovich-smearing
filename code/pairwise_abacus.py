import numpy as np
import sys,fitsio,gc
from pathlib import Path

from paths import *

sys.path.append(Sahyadri_Path)
from correlations import TwoPointCorrelationFunctionPeriodic,PowerSpectrum
from utilities import Utilities

sys.path.append(ML_Path)
from mllib import MLUtilities

from time import time
import matplotlib.pyplot as plt

ut = Utilities()

def queuer_box(phase,tpcf,powspec,aniso,los,redshift,M_min,kmin,kmax,h_fid,max_file,ddstem,down_to,rng,do_2pcf,do_Pk,mdict):
    data_dir = ddstem + '{0:03d}/z{1:.3f}/'.format(phase,redshift)
    print(data_dir+'\n')        
    data = fitsio.read(data_dir+'sub_0_nsub_4.fits')
    data_size_orig = data['Mabacus'].size
    data = data[data['Mabacus'] >= M_min]
    ut.status_bar(0,max_file)
    for i in range(1,max_file):
        data_i = fitsio.read(data_dir+'sub_{0:d}_nsub_4.fits'.format(i))
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

        # account for pbc
        zred = zred % Lbox

        # store as final separation
        data_use[:,los] = zred
        del zred
        
    del ind,data
    gc.collect()

    hfid_by_h = h_fid/h
    data_use *= hfid_by_h # convert to Mpc/h_fid

    mdict[phase] = {}
    if do_2pcf:
        tpcf.Lbox *= hfid_by_h
        if aniso & (tpcf.los != los):
            raise Exception('Mismatched los in tpcf')
        start_time = time()
        mdict[phase]['2pcf'] = tpcf.auto_CF(data_use)
        ut.time_this(start_time)
        
    if do_Pk:
        if aniso & (powspec.los != los):
            raise Exception('Mismatched los in powspec')
        start_time = time()
        Pk = powspec.Pk_grid(data_use.T,aniso=aniso)
        mdict[phase]['Pk'] = Pk
        ut.time_this(start_time)

        if aniso:
            # below can be made more robust to binning. currently demands linear binning.
            Sig2obs = np.zeros(powspec.L_Max)
            if powspec.lgbin is not None:
                print("Warning!: non-linear k-bins detected. Not calculating Sig2obs.")
            else:
                ind_k = np.where((powspec.ktab > kmin) & (powspec.ktab < kmax))[0]
                if ind_k.size:
                    # Pk has shape (L,k)
                    Sig2obs += np.sum(Pk.T[ind_k],axis=0)*powspec.dk/(6*np.pi**2)
                    
            Sig2obs *= hfid_by_h**2 # convert to (Mpc/h_fid)^2
            
            mdict[phase]['Sig2obs'] = Sig2obs
                
    return

if __name__ == "__main__":
    start_time = time()
    
    data_dir = '../examples/data/AbacusSummit/base_c000/'
    Path(data_dir).mkdir(parents=True,exist_ok=True) # folder to store data products
    
    ml = MLUtilities()

    Do_2pcf = False
    Do_Pk = True
    
    Down_To = 1   # default 1
    Grid = 256    # default 256 (better than 1% convergence at k <= 0.2 h/Mpc)
    Max_File = 64 # default 64
    N_Real = 3
    NProc = np.min([N_Real,6])
    
    Redshift = 0.1 # 0.1 or 0.8

    Aniso = True
    LOS = 2
    L_Max = 3
    h_fid = 0.6737 # h in fiducial cosmology, from arXiv:2410.21374
    
    Abacus_Stem = Abacus_Path + 'AbacusSummit_base_c000_ph'
    Lbox_AbacusSummit = 2000.0 # AbacusSummit box size in Mpc/h (not Mpc/h_fid)

    # 1.25e13, 8e12 Table 2 of arXiv:1607.05383 says lgMmin=13.67, sig_lgM=0.81 at z >~ 0.7, so 10**(13.67-0.81) = 7.2e12
    M_min = 8e12 if Redshift > 0.5 else 1.25e13 
    print('Retaining halos with M >= {0:.2e} Msun/h'.format(M_min))
    print('Downsampling by factor {0:d}...'.format(Down_To))

    start_time_setup = time()
    if Do_2pcf:
        tpcf = TwoPointCorrelationFunctionPeriodic(smin=65.0,smax=125.0,n_s=30,aniso=Aniso,L_Max=L_Max,Lbox=Lbox_AbacusSummit,los=LOS)
        tpcf.verbose = False
    else:
        tpcf = None

    if Do_Pk:
        K_Min_Pk,K_Max_Pk,N_Kbin = 0.01,0.2,19
        powspec = PowerSpectrum(grid=Grid,kmin=K_Min_Pk,kmax=K_Max_Pk,Lbox=Lbox_AbacusSummit,lgbin=None,nbin=N_Kbin,anisotropic=Aniso,los=LOS)
        K_Min,K_Max = 0.02,0.05
    else:
        powspec = None
        K_Min,K_Max = None,None
    ut.time_this(start_time_setup)
        
    rng = np.random.RandomState(42)

    if Do_2pcf:
        task_tuple = (tpcf,powspec,Aniso,LOS,Redshift,M_min,K_Min,K_Max,h_fid,Max_File,Abacus_Stem,Down_To,rng,Do_2pcf,Do_Pk)
        tasks = [task_tuple]*N_Real
        # tasks = []
        # for phase in range(N_Real):
        #     tasks.append(task_tuple)
        pw_dict = ml.run_processes(tasks,queuer_box,NProc)

    if Do_Pk:
        pw_dict = {}
        for phase in range(N_Real):
            queuer_box(phase,tpcf,powspec,Aniso,LOS,Redshift,M_min,K_Min,K_Max,h_fid,Max_File,Abacus_Stem,Down_To,rng,Do_2pcf,Do_Pk,pw_dict)

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
            xi_reals = np.zeros((N_Real,L_Max,tpcf.n_s))
        else:
            xi_reals = np.zeros((N_Real,tpcf.n_s))
        for phase in range(N_Real):
            xi_reals[phase] = pw_dict[phase]['2pcf']

        xi = np.mean(xi_reals,axis=0)
        err_xi = np.std(xi_reals,axis=0)/np.sqrt(N_Real-1 + 1e-15) # placeholder until better errors available

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
        for phase in range(N_Real):
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
                        seq.append(xi_reals[phase,L,s])
                    ut.write_to_file(xi_file_phase,seq)
                else:
                    ut.write_to_file(xi_file_phase,[tpcf.smid[s],xi_reals[phase,s]])

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
            Pk_reals = np.zeros((N_Real,L_Max,powspec.nbin))
            Sig2obs_reals = np.zeros((N_Real,L_Max))
        else:
            Pk_reals = np.zeros((N_Real,powspec.nbin))
            
        for phase in range(N_Real):
            Pk_reals[phase] = pw_dict[phase]['Pk']
            if Aniso:
                Sig2obs_reals[phase] = pw_dict[phase]['Sig2obs']

        Pk = np.mean(Pk_reals,axis=0)
        err_Pk = np.std(Pk_reals,axis=0)/np.sqrt(N_Real-1 + 1e-15) # placeholder until better errors available

        if Aniso:
            Sig2obs = np.mean(Sig2obs_reals,axis=0)
            err_Sig2obs = np.std(Sig2obs_reals,axis=0)/np.sqrt(N_Real-1 + 1e-15) # placeholder until better errors available
        
        print('Writing to file: ',Pk_file)
        with open(Pk_file,'w') as f:
            if Aniso:
                f.write("# k (h/Mpc) | ( Pk|err_Pk )_{ell="+','.join([str(2*L) for L in range(L_Max)])+"} (Mpc/h)^3\n")
            else:
                f.write("# k (h/Mpc) | (Pk | err_Pk) (Mpc/h)^3\n")
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
        for phase in range(N_Real):
            ph_str = '{0:03d}'.format(phase)
            print('... phase '+ph_str)
            Pk_file_phase = Pk_phase_stem + ph_str + '.txt'
            with open(Pk_file_phase,'w') as f:
                if Aniso:
                    f.write("# k (h/Mpc) | ( Pk|err_Pk )_{ell="+','.join([str(2*L) for L in range(L_Max)])+"} (Mpc/h)^3\n")
                else:
                    f.write("# k (h/Mpc) | (Pk | err_Pk) (Mpc/h)^3\n")
            for k in range(powspec.nbin):
                if Aniso:
                    seq = [powspec.ktab[k]]
                    for L in range(L_Max):
                        seq.append(Pk_reals[phase,L,k])
                    ut.write_to_file(Pk_file_phase,seq)
                else:
                    ut.write_to_file(Pk_file_phase,[powspec.ktab[k],Pk_reals[phase,k]])


        if Aniso:
            print('Writing to file: ',Sig2obs_file)
            with open(Sig2obs_file,'w') as f:
                f.write("# (Sig2obs | err) (Mpc/h)^2\n")
            for L in range(L_Max):
                ut.write_to_file(Sig2obs_file,[Sig2obs[L],err_Sig2obs[L]])

            print('Writing to files: ',Sig2obs_phase_stem)
            for phase in range(N_Real):
                ph_str = '{0:03d}'.format(phase)
                print('... phase '+ph_str)
                Sig2obs_file_phase = Sig2obs_phase_stem + ph_str + '.txt'
                np.savetxt(Sig2obs_file_phase,Sig2obs_reals[phase],fmt='%.8e',header="Sig2obs (Mpc/h)^2")

                
                    
    print('... all done!')
    ut.time_this(start_time)
