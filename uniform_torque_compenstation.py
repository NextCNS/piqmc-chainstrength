import dimod
import numpy as np
from dwave.embedding.chain_strength import uniform_torque_compensation
import math

N = 30
interactions_fname = './data/SK_N'+str(N)+'/'+str(N)+'_SK_seed1'+'.txt'
loaded = np.loadtxt(interactions_fname)
sum = 0
coupling = {}
for i, j, val in loaded:
    coupling[(i,j)] = val
    sum = sum + math.pow(val, 2)

linear = [0] * N
bqm = dimod.BQM.from_ising(linear,coupling)

chain_strength_default = uniform_torque_compensation(bqm)
step = (math.sqrt(sum)) / N
venturi_strength = 1 * math.sqrt((2*sum) / ((N-1)))
print(f"chain_strength uniform is {chain_strength_default}")
print(f"Chain strength venturi is {venturi_strength}")
print(f"step is {step}")