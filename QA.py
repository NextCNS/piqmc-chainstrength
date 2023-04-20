import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np

bqm = dimod.BinaryQuadraticModel.from_qubo(Q_10)

sampler = EmbeddingComposite(DWaveSampler())

# Define a list of chain strengths to test
chain_strengths = [0.1, 0.5, 0.8 , 1.0, 10.0, 20.0, 30.0, 40.0]

# Open a file to write the results
with open('results_Q10.txt', "a+") as f:
    for chain_strength in chain_strengths:
        num_reads = 500
        sampleset = sampler.sample(bqm, num_reads=num_reads, chain_strength=chain_strength, annealing_time=1000)

        ground_energy = sampleset.first.energy
        ground_state_samples = [sample for sample in sampleset.data(fields=['sample', 'energy', 'num_occurrences']) if sample.energy == ground_energy]

        ground_state_probability = sum(s.num_occurrences for s in ground_state_samples) / num_reads

        # Write the results to the file
        f.write(f"Chain strength: {chain_strength}, Ground state probability: {ground_state_probability}\n")
