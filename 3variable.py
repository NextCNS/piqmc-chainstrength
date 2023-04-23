import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation
import math
import dwave.inspector

sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
bqm = dimod.BQM({}, {('s0', 's1'): -1, ('s0', 's2'): -1, ('s1', 's2'): 1},
                     0, dimod.Vartype.SPIN)
sampleset = sampler.sample(bqm, num_reads=100, chain_strength=1.0, annealing_time=1000)
dwave.inspector.show(sampleset)                    
