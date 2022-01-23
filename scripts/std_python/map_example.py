
def add_numbers(x, y):
    return x + y

if __name__ == '__main__':

    # map(function_be_applied, iterable_object, other_iterable_object, ...) --> return a map (like an iterator)

    a = [1, 2, 3]
    b = [10, 11, 12]
    # c = list(map(lambda x, y: x+y, a, b))
    c = list(map(add_numbers, a, b))

    for i in range(len(c)):
        assert c[i] == a[i] + b[i]


