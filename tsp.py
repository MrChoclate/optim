import itertools
import math
import functools
import time
import random
import copy

def timer(func):
    def with_time(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print("{} took {} sec".format(func.__name__, time.time() - t))
        return res
    return with_time

def read():
    n = int(input())
    return [tuple(float(x) for x in input().split()) for _ in range(n)]

@functools.lru_cache(maxsize=1024)
def distance(src, dest):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(src, dest)))

def cost(sol, cities):
    dst = sum(distance(cities[x], cities[y]) for x, y in zip(sol[:-1], sol[1:]))
    dst += distance(cities[-1], cities[0])
    return dst

def random_sol(cities):
    sol = list(range(1, len(cities)))
    random.shuffle(sol)
    return [0] + sol

def neighboor(sol):
    assert(sol[0] == 0)
    i = random.randint(1, len(sol) - 1)
    j = i
    while j == i:
        j = random.randint(1, len(sol) - 1)
    res = copy.copy(sol)
    res[i], res[j] = res[j], res[i]
    return res

@timer
def random_search(cities):
    res = float('inf')
    best_sol = None

    for _ in range(len(cities)):
        sol = random_sol(cities)
        current_cost = cost(sol, cities)
        if res > current_cost:
            best_sol = sol
            res = current_cost
    return res, best_sol

@timer
def stochastic_hill_climbing(cities, kmax=1000):
    best_sol = random_sol(cities)
    best_cost = cost(best_sol, cities)
    k = 0

    while k < kmax:
        k += 1
        current_sol = neighboor(best_sol)
        current_cost = cost(current_sol, cities)
        if current_cost < best_cost:
            best_sol = current_sol
            best_cost = current_cost
            k = 0

    return best_cost, best_sol

@timer
def simulated_annealing(cities):
    current_sol = best_sol = random_sol(cities)
    current_cost = best_cost = cost(best_sol, cities)
    T = 1000 * best_cost / len(cities)
    T_min = best_cost / len(cities) / 1000.
    k = 0

    while T > T_min:
        k += 1
        new_sol = neighboor(current_sol)
        new_cost = cost(new_sol, cities)
        if new_cost < best_cost:
            best_sol = new_sol
            best_cost = new_cost
            k = 0
        if new_cost < current_cost or random.random() <= math.exp((current_cost - new_cost) / T):
            current_sol = new_sol
            current_cost = new_cost

        if k > 100:
            T *= 0.99999

    return best_cost, best_sol

@timer
def brute_solve(cities):
    best_cost = float('inf')
    best_sol = None
    for sol in itertools.permutations(range(len(cities))):
        current_cost = cost(sol, cities)
        if current_cost < best_cost:
            best_cost = current_cost
            best_sol = sol
    return best_cost, best_sol

@timer
def greedy_solve(cities, fn=min):
    sol = [0]
    i = 0

    while i != len(cities) - 1:
        remaining = set(range(len(cities))) - set(sol)
        _, pick = fn((distance(cities[i], cities[x]), x) for x in remaining)
        sol.append(pick)
        i += 1
    return cost(sol, cities), sol

if __name__ == '__main__':
    cities = read()
    print(greedy_solve(cities, fn=min))
    print(greedy_solve(cities, fn=max))
    print(random_search(cities))
    print(stochastic_hill_climbing(cities))
    print(simulated_annealing(cities))
