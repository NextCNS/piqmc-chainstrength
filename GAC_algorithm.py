import numpy as np
import argparse
import os
import numpy as np
from models import SK
from python_interface import QuantumPIAnneal
import dimod
import math
from dwave.embedding.chain_strength import uniform_torque_compensation

# Parameters
NUM_GENERATIONS = 10


output = []

def first_and_last_occurrence(target_value, input_dict):
    first_occurrence = None
    last_occurrence = None

    for key, value in input_dict.items():
        if value == target_value:
            if first_occurrence is None:
                first_occurrence = key
            last_occurrence = key

    return first_occurrence, last_occurrence

def fitness_function(chain_strength,model, args):
    P = args.P
    Loaded = []
    args.chain_strength = chain_strength
    Energies = np.zeros((args.numruns, len(args.tau_schedule),int(P)), np.float64)
    for annealingrun in range(len(Loaded)+1, args.numruns + 1):
        Q = QuantumPIAnneal(model, latticetype = "FullyConnected",  **vars(args))
        Energies[annealingrun-1], config = Q.perform_tau_schedule()
    return np.amin(Energies)


def GA(model, args):
    chain_strength_candidates = []
    # Initialization
    chain_strength_population = np.arange(0, args.chain_strength, args.step).tolist()
    # Main loop
    for generation in range(NUM_GENERATIONS):
        # Fitness evaluation
        fitness_scores = {}
        for c in chain_strength_population:
            fitness_scores[c] = fitness_function(c, model, args)
        print(f"Fitness_scores is of P at generation {generation} is {fitness_scores}")
        min_energy = min(fitness_scores.values())
        fist_chain, last_chain = first_and_last_occurrence(min_energy, fitness_scores)
        print(fist_chain, last_chain, args.step,   "hehe")
        chain_strength_candidates.append(last_chain)
        if fist_chain ==  last_chain:
            break
        else:
            chain_strength_population = np.arange(fist_chain, args.chain_strength, args.step).tolist()
    print(chain_strength_candidates, "final")
    return max(chain_strength_candidates)
if __name__ == "__main__":
    if not os.path.exists('./results/'):
        os.mkdir('./results')
    if not os.path.exists('./results/SK'):
        os.mkdir('./results/SK')
    if not os.path.exists('./results/SK/PIQMC'):
        os.mkdir('./results/SK/PIQMC')
    N = 5
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1)
    parser.add_argument('--P', default=int(N/4) + 1)
    parser.add_argument('--gamma_0', default=0.1, type=float)
    parser.add_argument('--tau_schedule', default=[1000])
    parser.add_argument('--mcsteps', default=5) #Number of sweeps
    parser.add_argument('--numruns', default=1, type=int)

    args = parser.parse_args()
    realization = args.seed
    P = args.P
    numruns = args.numruns

    interactions_fname = './data/PIMC/SK_N'+str(N)+'/'+str(N)+'_SK_seed'+str(realization)+'.txt'
    model = SK(nspins=N, interactions_fname=interactions_fname)
    loaded = np.loadtxt(interactions_fname)
    coupling = {}
    sum = 0
    for i, j, val in loaded:
        coupling[(i,j)] = -1 * val
        sum = sum + math.pow(val, 2)

    linear = [0] * N
    bqm = dimod.BQM.from_ising(linear,coupling)
    step = (math.sqrt(sum)) / (N)
    print("step is:" ,step)
    chain_strength_initial = uniform_torque_compensation(bqm)
    args.chain_strength = chain_strength_initial
    args.step = step
    suggest_chain_strength = GA(model, args)
    print(suggest_chain_strength)
