import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    a = np.array([1, 2, 3])

    b = a
    print('==> a and b point to same reference')
    print('a:', a)
    print('b:', b)

    b[0] = -1
    print('==> Setting b[0] = -1, will also change a[0] to %d' % a[0])
    assert a[0] == -1

    in_place = False

    if in_place:
        b += 100
        assert a[0] == 99
        print('==> <In-place operation> b += 100. b still points to the same reference, '
              'so a[0] changes also to %d' % a[0])
    else:
        b = b + 100
        assert a[0] == -1
        print('==> <Not in-place operation on b> b = b + 100. b+100 creates a new object to which b points now.'
              ' a[0] remains %d' % a[0])


