import itertools
import copy
import random
import functools
import heapq
import math

def read():
    n, maxweight = (int(x) for x in input().split(' '))
    l = [input().split(' ') for _ in range(n)]
    l = [tuple(map(int, x)) for x in l]
    return maxweight, l

def brute_solve(maxweight, l):
    res = 0
    for n in range(1, len(l) + 1):
        for comb in itertools.combinations(l, n):
            if sum(x[1] for x in comb) <= maxweight:
                res = max(res, sum(x[0] for x in comb))
    return res

def recursive_solve(maxweight, l):
    if maxweight < 0:
        return float('-inf')

    if maxweight == 0 or not l:
        return 0

    return max(
        l[0][0] + recursive_solve(maxweight - l[0][1], l[1:]),
        recursive_solve(maxweight, l[1:])
    )

def dynamic_solve(maxweight, l):
    @functools.lru_cache(maxsize=None)
    def rec(maxweight, n):
        if maxweight < 0:
            return float('-inf'), []

        if maxweight == 0 or n < 0:
            return 0, []

        a, backref = rec(maxweight - l[n][1], n - 1)
        b, backref_ = rec(maxweight, n - 1)
        if a + l[n][0] > b:
            return l[n][0] + a, backref + [n]
        else:
            return b, backref_
    return rec(maxweight, len(l) - 1)

def random_pick(maxweight, l):
    pick = copy.copy(l)
    weight = sum(x[1] for x in pick)

    while weight > maxweight:
        remove = random.choice(pick)
        pick.remove(remove)
        weight -= remove[1]

    return pick

def random_search(maxweight, l):
    res = 0
    for _ in range(len(l)):
        res = max(res, sum(x[0] for x in random_pick(maxweight, l)))
    return res

def stupid_random(maxweight, l):
    sol = [False for _ in l]
    res = weight = value = 0
    for _ in range(len(l) ** 2):
        pick = random.randint(0, len(l) - 1)
        if sol[pick]:
            value -= l[pick][0]
            weight -= l[pick][1]
            sol[pick] = False
        else:
            value += l[pick][0]
            weight += l[pick][1]
            sol[pick] = True
        if weight <= maxweight:
            res = max(res, value)
    return res

def simulated_annealing(maxweight, l):
    sol = [False for _ in l]
    res = weight = value = 0
    value_max = max(x[0] for x in l)
    itermax = len(l) ** 2
    for i in range(itermax):
        T = (itermax - i) / itermax * value_max
        pick = random.randint(0, len(l) - 1)
        if sol[pick]:
            # The future solution is worse
            if weight <= maxweight:
                if not random.random() <= math.exp(- l[pick][0] / T):
                    continue

            value -= l[pick][0]
            weight -= l[pick][1]
            sol[pick] = False
        else:
            # The future solution is worse
            if weight + l[pick][1] > maxweight:
                if not random.random() <= math.exp(- l[pick][0] / T):
                    continue

            value += l[pick][0]
            weight += l[pick][1]
            sol[pick] = True
        if weight <= maxweight:
            res = max(res, value)
    return res

def random_weight_based_pick(maxweight, l):
    mean_weight = sum(x[1] for x in l) / len(l)
    p = maxweight / mean_weight  / len(l)
    sol = []
    for item in l:
        if random.random() < p:
            sol.append(item)
    return sol

def random_search_weight_based(maxweight, l):
    res = 0
    sol = None
    weight = 0
    for _ in range(len(l)):
        current = random_weight_based_pick(maxweight, l)
        weight += sum(x[1] for x in current) / maxweight
        if sum(x[1] for x in current) <= maxweight:
            if res < sum(x[0] for x in current):
                sol = current
                res = sum(x[0] for x in current)
    print(weight / len(l))
    return res

def random_neighbor(current, weight, maxweight, l):
    # Remove one object
    pick = copy.copy(current)
    remove = random.choice(pick)
    pick.remove(remove)
    weight -= remove[1]

    # Add one object
    remaining = list(set(l) - set(pick))
    random.shuffle(remaining)
    for obj in remaining:
        if obj[1] + weight <= maxweight:
            weight += obj[1]
            pick.append(obj)
            break

    return pick, weight

def stochastic_hill_climbing(maxweight, l):
    res = 0
    current = random_pick(maxweight, l)
    weight = sum(x[1] for x in current)
    for _ in range(len(l)):
        new, new_weight = random_neighbor(current, weight, maxweight, l)
        value = sum(x[0] for x in new)
        if value > res:
            res = value
            current, weight = new, new_weight

    return res

def upper_bound_greedy_frac(maxweight, l):
    l = [(- x[0] / x[1], x) for x in l]
    heapq.heapify(l)
    weight, value = 0, 0
    while l:
        ratio, item = heapq.heappop(l)
        if weight + item[1] <= maxweight:
            weight += item[1]
            value += item[0]
        else:
            available_weight = maxweight - weight
            if available_weight == 0:
                break
            weight = maxweight
            value += item[0] * (available_weight / item[1])
            break

    return value

def upper_bound_greedy_frac_(maxweight, sorted_l):
    weight, value = 0, 0
    for item in sorted_l:
        if weight + item[1] <= maxweight:
            weight += item[1]
            value += item[0]
        else:
            available_weight = maxweight - weight
            if available_weight == 0:
                break
            weight = maxweight
            value += item[0] * (available_weight / item[1])
            break

    return value

def solve_greedy(maxweight, sorted_l):
    weight, value = 0, 0
    for item in sorted_l:
        if weight + item[1] <= maxweight:
            weight += item[1]
            value += item[0]

    return value

def greedy_frac(maxweight, l):
    return solve_greedy(maxweight, sorted(l, key=lambda x: - x[0] / x[1]))

def greedy_val(maxweight, l):
    return solve_greedy(maxweight, sorted(l, key=lambda x: - x[0]))

def greedy_weight(maxweight, l):
    return solve_greedy(maxweight, sorted(l, key=lambda x: x[1]))

def fptas(maxweight, l, eps=0.1):
    """Fully polynomial time approximation scheme"""
    max_value = max(x[1] for x in l)
    k = eps * max_value / len(l)
    return dynamic_solve(math.floor(maxweight / k), [(x[0], math.ceil(x[1] / k)) for x in l])[0]

def branch_and_bound(maxweight, l):
    l = sorted(l, key=lambda x: - x[0] / x[1])
    lower_bound = max(
        solve_greedy(maxweight, l),
        greedy_val(maxweight, l),
        greedy_weight(maxweight, l),
        random_search(maxweight, l)
    )
    stack = [(0, 0, 0, None)]

    while stack:
        weight, value, i, upper_bound = stack.pop()

        lower_bound = max(lower_bound, value)

        # There is no more items, we reached a leaf
        if i == len(l) or weight == maxweight:
            continue

        # We need to recalculate the upper bound because we did not take an item
        if upper_bound is None:
            upper_bound = math.floor(
                value + upper_bound_greedy_frac_(maxweight - weight, l[i:])
            )

        if upper_bound <= lower_bound:
            continue

        # We take the item, upper_bound is the same
        if weight + l[i][1] <= maxweight:
            stack.append(
                (weight + l[i][1], value + l[i][0], i + 1, upper_bound)
            )

        # We do not take the item
        stack.append((weight, value, i + 1, None))

    return lower_bound

if __name__ == '__main__':
    weight, l = read()

    print(greedy_frac(weight, l))
    print(greedy_val(weight, l))
    print(greedy_weight(weight, l))
    print(dynamic_solve(weight, l))
    print(stupid_random(weight, l))
    print(simulated_annealing(weight, l))

    # print(brute_solve(weight, l))
