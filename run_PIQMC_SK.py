import argparse
import os
import numpy as np
from models import SK
from python_interface import QuantumPIAnneal

if __name__ == "__main__":

    if not os.path.exists('./results/'):
        os.mkdir('./results')
    if not os.path.exists('./results/SK'):
        os.mkdir('./results/SK')
    if not os.path.exists('./results/SK/PIQMC'):
        os.mkdir('./results/SK/PIQMC')

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1)
    parser.add_argument('--P', default=3)
    parser.add_argument('--gamma_0', default=0.01, type=float)
    parser.add_argument('--tau_schedule', default=[5000])
    parser.add_argument('--mcsteps', default=5) #Number of sweeps
    parser.add_argument('--numruns', default=10, type=int)
    parser.add_argument('--chain_strength', default=0.5, type=float)

    args = parser.parse_args()
    realization = args.seed
    P = args.P
    numruns = args.numruns

    N = 5

    interactions_fname = './data/SK_N'+str(N)+'/'+str(N)+'_SK_seed'+str(realization)+'.txt'
    model = SK(nspins=N, interactions_fname=interactions_fname)

    Energies = np.zeros((numruns, len(args.tau_schedule),int(P)), np.float64)
    spinConfigs = []
    checkpointfile = './results/SK/PIQMC/SK_N'+str(N)+'_PIQMC_realization'+str(realization)+'_Energies.npy'
    try:
        print("Loading checkpoint!")
        Loaded = np.load(checkpointfile)
        Energies[:Loaded.shape[0]] = Loaded
    except:
        print("Failed! Running from scratch")
        Loaded = []

    for annealingrun in range(len(Loaded)+1, numruns+1):
        Q = QuantumPIAnneal(model, latticetype = "FullyConnected",  **vars(args))
        Energies[annealingrun-1], spinConfig = Q.perform_tau_schedule()
        spinConfigs.append(spinConfig)
        np.save(checkpointfile, Energies[:annealingrun])
    print(np.amin(Energies))