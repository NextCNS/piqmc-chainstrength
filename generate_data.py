import random

N = 15

def generate_data():
    for i in range(1, N):
        for j in range(i+1, N+1):
            coupling = round(random.uniform(-1, 1), 2)
            print(f"{i} {j} {coupling} \n",end='')

if __name__ == "__main__":
    generate_data()