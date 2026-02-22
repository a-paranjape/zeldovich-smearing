import numpy as np
import sys,fitsio,gc
from pathlib import Path

from paths import *

sys.path.append(Sahyadri_Path)
from correlations import TwoPointCorrelationFunctionPeriodic

sys.path.append(ML_Path)
from mllib import MLUtilities

from time import time
import matplotlib.pyplot as plt

def queuer_box(phase,tpcf,redshift,M_min,h_fid,max_file,ddstem,down_to,rng,mdict):
    data_dir = ddstem + '{0:03d}/z{1:.3f}/'.format(phase,redshift)
    print(data_dir+'\n')        
    data = fitsio.read(data_dir+'sub_0_nsub_4.fits')
    data_size_orig = data['Mabacus'].size
    data = data[data['Mabacus'] >= M_min]
    tpcf.status_bar(0,max_file)
    for i in range(1,max_file):
        data_i = fitsio.read(data_dir+'sub_{0:d}_nsub_4.fits'.format(i))
        data_size_orig += data_i['Mabacus'].size
        data_i = data_i[data_i['Mabacus'] >= M_min]
        data = np.concatenate((data,data_i))
        data_i = None
        tpcf.status_bar(i,max_file)
    print('... ... kept {0:d} of {1:d} objects'.format(data['Mabacus'].size,data_size_orig))
    
    data_size = data['Mabacus'].size
    ind = rng.choice(np.arange(data_size),size=data_size//down_to,replace=False) if down_to > 1 else np.arange(data_size)
    data_use = data['x_L2com'][ind]

    E_z = 0.0
    h = 0.0
    with open(data_dir+'header.txt','r') as f:
        lines = f.readlines()
    for line in lines:
        line.strip()
        items = [item.strip() for item in line.split(':')]
        if items[0] == 'HubbleNow':
            E_z += float(items[1])
        if items[0] == 'H0':
            h += 0.01*float(items[1])
    print('... h = {0:.4f}; h_fid = {1:.4f}'.format(h,h_fid))
            
    if tpcf.aniso:
        print('... E(z) = {0:.4f}'.format(E_z))

        # from prop vel to comov separation
        zred = data['v_L2com'][ind,tpcf.los]*(1+redshift)/(100*E_z)

        # add real-space separation
        zred += data_use[:,tpcf.los]

        # account for pbc
        zred = zred % tpcf.Lbox

        # store as final separation
        data_use[:,tpcf.los] = zred
        del zred
        
    del ind,data
    gc.collect()

    hfid_by_h = h_fid/h
    data_use *= hfid_by_h # convert to Mpc/h_fid
    tpcf.Lbox *= hfid_by_h
    
    start_time = time()
    mdict[phase] = tpcf.auto_CF(data_use)
    tpcf.time_this(start_time)

    return

if __name__ == "__main__":
    start_time = time()
    
    data_dir = '../examples/data/AbacusSummit/base_c000/'
    Path(data_dir).mkdir(parents=True,exist_ok=True) # folder to store data products
    
    ml = MLUtilities()    
    
    Down_To = 1   # default 1
    Max_File = 64 # default 64
    N_Real = 3
    NProc = np.min([N_Real,6])
    Redshift = 0.1 # 0.1 or 0.8

    Aniso = True
    L_Max = 3
    h_fid = 0.6737 # h in fiducial cosmology, from arXiv:2410.21374
    
    Abacus_Stem = Abacus_Path + 'AbacusSummit_base_c000_ph'
    Lbox_AbacusSummit = 2000.0 # AbacusSummit box size in Mpc/h (not Mpc/h_fid)

    # 1e13? 8e12? Table 2 of arXiv:1607.05383 says lgMmin=13.67, sig_lgM=0.81 at z >~ 0.7, so 10**(13.67-0.81) = 7.2e12
    M_min = 8e12 if Redshift > 0.5 else 1.25e13 
    print('Retaining halos with M >= {0:.2e} Msun/h'.format(M_min))

    file_stem = 'xi'
    if Aniso:
        file_stem += '_LMax{0:d}'.format(L_Max)
    file_stem += '_lgMmin{0:.2f}_z{1:.3f}'.format(np.log10(M_min),Redshift)

    xi_file = data_dir + file_stem + '.txt'
    scales_file = data_dir + file_stem + '_scales.txt'
    xi_file_colwise = data_dir + file_stem + '_colwise.txt'
    xi_phase_stem = data_dir + file_stem + '_ph'

    print('Downsampling by factor {0:d}...'.format(Down_To))

    tpcf = TwoPointCorrelationFunctionPeriodic(smin=65.0,smax=125.0,n_s=30,aniso=Aniso,L_Max=L_Max,Lbox=Lbox_AbacusSummit)
    # print(tpcf.sbin)
    # raise Exception
    tpcf.verbose = False
    rng = np.random.RandomState(42)

    task_tuple = (tpcf,Redshift,M_min,h_fid,Max_File,Abacus_Stem,Down_To,rng)
    tasks = [task_tuple]*N_Real
    # tasks = []
    # for phase in range(N_Real):
    #     tasks.append(task_tuple)
    xi_dict = ml.run_processes(tasks,queuer_box,NProc)

    if tpcf.aniso:
        xi_reals = np.zeros((N_Real,tpcf.L_Max,tpcf.n_s))
    else:
        xi_reals = np.zeros((N_Real,tpcf.n_s))
    for phase in range(N_Real):
        xi_reals[phase] = xi_dict[phase]

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
            for L in range(tpcf.L_Max):
                seq.append(xi[L,s])
                seq.append(err_xi[L,s])
            tpcf.write_to_file(xi_file_colwise,seq)
        else:
            tpcf.write_to_file(xi_file_colwise,[tpcf.smid[s],xi[s],err_xi[s]])

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
                tpcf.write_to_file(xi_file_phase,seq)
            else:
                tpcf.write_to_file(xi_file_phase,[tpcf.smid[s],xi_reals[phase,s]])

            
    cols = ['crimson','indigo','forestgreen']
    xi_plot_file = data_dir + file_stem + '.pdf'
    print('Writing to file: ',xi_plot_file)
    plt.figure(figsize=(5,5))
    if tpcf.aniso:
        for L in range(tpcf.L_Max):
            plt.errorbar(tpcf.smid,tpcf.smid**2*xi[L],yerr=tpcf.smid**2*err_xi[L],marker='o',capsize=5,lw=1,c=cols[L])
    else:
        plt.errorbar(tpcf.smid,tpcf.smid**2*xi,yerr=tpcf.smid**2*err_xi,marker='o',capsize=5,lw=1,c=cols[0])
    plt.minorticks_on()
    plt.savefig(xi_plot_file,bbox_inches='tight')

    print('... all done!')
    tpcf.time_this(start_time)
