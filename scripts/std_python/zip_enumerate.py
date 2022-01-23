
def my_fun(a, b):
    return a//b, a%b

if __name__ == '__main__':

    a = [21, 22, 23, 24, 25]
    b = [4, 3, 6, 5, 7]

    c1, c2 = zip(*[my_fun(a, b) for a, b in zip(a, b)])

    for i, (ai, bi, c1i, d1i) in enumerate(zip(a, b, c1, c2)):
        print('%d: %d = %d*%d + %d' % (i, ai, bi, c1i, d1i))

