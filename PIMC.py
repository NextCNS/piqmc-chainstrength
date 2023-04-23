import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation
import math
import dwave.inspector



sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
# 5 1.437303932275054 1.7603306196848363 1.4084771918636099
# 10 2.4336862877386793 2.6905414246207027 2.4218530095775836
# 15 2.8855908574674958 3.0848263848716018 2.6936249017097964
# 20 3.0368145502812656 3.1926485859071208 0.9959267202802318
# 25 3.984207866387574 4.146895025132259 3.6573519600935334
# 30 4.20514891434222 4.347736625376632 3.902802411942817
# 35 4.598174944562244 4.731482983380952 3.9418880796710924
# 40 5.0517254158757385 5.179637440963981  4.852756639272158
# 45 5.234951058125942 5.352605097955065 4.340175646442815

# N = [5,10,15,20,25,30,35,40,45]
N = [45]

for index, value in enumerate(N):
    interactions_fname = './data/PIMC/SK_N'+str(value)+'/'+str(value)+'_SK_seed1'+'.txt'
    loaded = np.loadtxt(interactions_fname)
    coupling = {}
    for i, j, val in loaded:
        coupling[(i-1,j-1)] = val
    linear = [0] * value
    bqm = dimod.BQM.from_ising(linear,coupling)
    print(bqm)
    num_reads = 400
    annealing = 2000
    chain_strength = 4.340175646442815
    sampleset = sampler.sample_ising(linear,coupling, num_reads=num_reads, chain_strength=chain_strength, annealing_time=annealing)
    # dwave.inspector.show(sampleset)
    # # Open a file to write the results
    with open(f'./results/PIMC/a2000/b400/results_{value}.txt', "a+") as f:
            # f.write(sampleset)
            f.write(f"------annealing = {annealing}, numread = {num_reads}\n")
            f.write(f"sampleset info: {str(sampleset.info)} \n")
            f.write(f"sampleset record: {str(sampleset.record.sample)} \n")
            f.write(f"sampleset energy: {str(sampleset.record.energy)} \n")
            ground_energy = sampleset.first.energy
            ground_state_samples = [sample for sample in sampleset.data(fields=['sample', 'energy', 'num_occurrences']) if sample.energy == ground_energy]
            ground_state_probability = sum(s.num_occurrences for s in ground_state_samples) / num_reads
            # Write the results to the file
            f.write(f"Chain strength: {chain_strength}, Ground state probability: {ground_state_probability}\n")
            f.write(f"Ground state is: {sampleset.first}, Ground state energy: {sampleset.first.energy}\n")
    
