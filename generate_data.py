import random

N = 55

def generate_data():
    with open(f"./data/QA/SK_N{N}/{N}_SK_seed1.txt", "a") as f:
        for i in range(1, N):
            for j in range(i+1, N+1):
                coupling = round(random.uniform(-1, 1), 2)
                f.write(f"{i} {j} {coupling}\n")

if __name__ == "__main__":
    generate_data()
