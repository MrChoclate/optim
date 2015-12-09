import time

def timer(func):
    def with_time(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print("{} took {} sec".format(func.__name__, time.time() - t))
        return res
    return with_time

def read():
    n, m = tuple(map(int, input().split()))
    graph = {x: [] for x in range(n)}
    for _ in range(m):
        x, y = tuple(map(int, input().split()))
        graph[x].append(y)
        graph[y].append(x)
    return graph

def N():
    i = 0
    while True:
        yield i
        i += 1

def has_conflict(color, node, graph, colored_graph):
    for neighboor in graph[node]:
        if colored_graph[neighboor] == color:
            return True
    return False

def get_min_color(node, graph, colored_graph):
    for color in N():
        if has_conflict(color, node, graph, colored_graph):
            continue
        else:
            return color

def get_nb_color(colored_graph):
    return len(set(colored_graph))

def get_iterator_from_order(order, graph):
    n = len(graph)
    iterator = range(n)
    if order == 'asc':
        iterator = sorted(graph.keys(), key=lambda x: len(graph[x]))
    if order == 'desc':
        iterator = sorted(graph.keys(), key=lambda x: len(graph[x]), reverse=True)
    return iterator

@timer
def greedy_by_nodes(graph, order=None):
    n = len(graph)
    colored_graph = [None for _ in range(n)]
    iterator = get_iterator_from_order(order, graph)

    for i in iterator:
        colored_graph[i] = get_min_color(i, graph, colored_graph)
    return get_nb_color(colored_graph)


@timer
def greedy_by_color(graph, order=None):
    n = len(graph)
    colored_graph = [None for _ in range(n)]
    for color in N():
        for i in get_iterator_from_order(order, graph):
            if colored_graph[i] is None and not has_conflict(color, i, graph, colored_graph):
                colored_graph[i] = color
        if None not in colored_graph:
            break

    return get_nb_color(colored_graph)

if __name__ == '__main__':
    graph = read()

    print(greedy_by_nodes(graph))
    print(greedy_by_nodes(graph, order='asc'))
    print(greedy_by_nodes(graph, order='desc'))
    print(greedy_by_color(graph))
    print(greedy_by_color(graph, order='asc'))
    print(greedy_by_color(graph, order='desc'))
