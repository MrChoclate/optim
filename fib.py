import sys
import functools

@functools.lru_cache(maxsize=None)
def dynamic_fib(n):
    if n <= 1:
        return n
    else:
        return dynamic_fib(n - 1) + dynamic_fib(n - 2)

def fib(n):
    if n <= 1:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

if __name__ == '__main__':
    print(fib(int(sys.argv[1])))
