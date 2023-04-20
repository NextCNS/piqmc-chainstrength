import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation
import math


sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))


# N = [5,10,15,20,25,30,35,40,45]
N = [45]

for index, value in enumerate(N):
    interactions_fname = './data/SK_N'+str(value)+'/'+str(value)+'_SK_seed1'+'.txt'
    loaded = np.loadtxt(interactions_fname)
    coupling = {}
    for i, j, val in loaded:
        coupling[(i,j)] = val
    linear = [0] * value
    bqm = dimod.BQM.from_ising(linear,coupling)
    chain_strength = uniform_torque_compensation(bqm) / 2
    num_reads = 500
    sampleset = sampler.sample_ising(linear,coupling, num_reads=num_reads, chain_strength=chain_strength, annealing_time=2000)
    # # Open a file to write the results
    with open(f'./results/results_{value}.txt', "a+") as f:
            # f.write(sampleset)
            f.write(f"sampleset info: {str(sampleset.info)}\n")
            f.write(f"sampleset record: {str(sampleset.record.sample)}\n")
            f.write(f"sampleset energy: {str(sampleset.record.energy)}\n")
            ground_energy = sampleset.first.energy
            ground_state_samples = [sample for sample in sampleset.data(fields=['sample', 'energy', 'num_occurrences']) if sample.energy == ground_energy]
            ground_state_probability = sum(s.num_occurrences for s in ground_state_samples) / num_reads
            # Write the results to the file
            f.write(f"Chain strength: {chain_strength}, Ground state probability: {ground_state_probability}\n")
            f.write(f"Ground state is: {sampleset.first}, Ground state energy: {sampleset.first.energy}\n")
