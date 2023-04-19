import numpy as np
import argparse
import os
import numpy as np
from models import SK
from python_interface import QuantumPIAnneal

# Parameters
POPULATION_SIZE = 10
GENE_LENGTH = 10
NUM_GENERATIONS = 1
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.03
ALPHA = 0.0
BETA = 1.0
RANDOM_SEED = 42
TOP_N = 3


def fitness_function(chain_strength_list,model, args):
    P = args.P
    Loaded = []
    fitness = []
    for value in chain_strength_list:
        args.chain_strength = value
        Energies = np.zeros((args.numruns, len(args.tau_schedule),int(P)), np.float64)
        print(f"This is the chain strength being test {value}")
        for annealingrun in range(len(Loaded)+1, args.numruns + 1):
            Q = QuantumPIAnneal(model, latticetype = "FullyConnected",  **vars(args))
            Energies[annealingrun-1], config = Q.perform_tau_schedule()
        fitness.append(np.amin(Energies))
    return fitness


def GA(model, args):
    # Set the random seed
    np.random.seed(RANDOM_SEED)

    # Initialization
    chain_strength_population = np.random.uniform(ALPHA, BETA, size=(POPULATION_SIZE, GENE_LENGTH))
    # print("Initial population is ", chain_strength_population)

    # Main loop
    for generation in range(NUM_GENERATIONS):
        # Fitness evaluation
        fitness_scores = np.mean([fitness_function(chain_strength_list, model, args) for chain_strength_list in chain_strength_population], axis=1)
        print(f"Fitness_scores is of P at generation {generation} is {fitness_scores}")

        # Selection (tournament selection)
        offspring_population = np.zeros_like(chain_strength_population)
        for i in range(POPULATION_SIZE):
            tournament_indices = np.random.choice(POPULATION_SIZE, size=2)
            winner_index = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
            offspring_population[i] = chain_strength_population[winner_index]

        # Crossover
        for i in range(0, POPULATION_SIZE, 2):
            if np.random.rand() < CROSSOVER_RATE:
                parent1 = offspring_population[i]
                parent2 = offspring_population[i + 1]
                crossover_point = np.random.randint(0, GENE_LENGTH)
                offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                offspring_population[i] = offspring1
                offspring_population[i + 1] = offspring2

        # Mutation
        mutation_population = np.copy(offspring_population)
        mutation_indices = np.random.rand(POPULATION_SIZE, GENE_LENGTH) < MUTATION_RATE
        mutation_values = np.random.uniform(ALPHA, BETA, size=(POPULATION_SIZE, GENE_LENGTH))
        mutation_population[mutation_indices] = mutation_values[mutation_indices]

        # Replacement
        chain_strength_population = mutation_population

    # Get the top n optimal λ values
    top_n_indices = np.argsort(fitness_scores)[-TOP_N:]
    optimal_lambdas = np.mean(chain_strength_population[top_n_indices], axis=1)
    print(f"Top {TOP_N} optimal lambda values:")
    for i, optimal_lambda in enumerate(optimal_lambdas, 1):
        print(f"{i}. {optimal_lambda}")


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
    parser.add_argument('--tau_schedule', default=[50])
    parser.add_argument('--mcsteps', default=5) #Number of sweeps
    parser.add_argument('--numruns', default=1, type=int)

    args = parser.parse_args()
    realization = args.seed
    P = args.P
    numruns = args.numruns

    N = 5

    interactions_fname = './data/SK_N'+str(N)+'/'+str(N)+'_SK_seed'+str(realization)+'.txt'
    model = SK(nspins=N, interactions_fname=interactions_fname)

    # Energies = np.zeros((numruns, len(args.tau_schedule),int(P)), np.float64)
    # checkpointfile = './results/SK/PIQMC/SK_N'+str(N)+'_PIQMC_realization'+str(realization)+'_Energies.npy'
    
    GA(model, args)
