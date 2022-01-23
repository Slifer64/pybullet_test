import numpy as np
import numpy.linalg as linalg
import numpy.matlib as matlib

def quat2rot(q, shape="wxyz"):
    """
    Transforms a quaternion to a rotation matrix.
    """
    if shape == "wxyz":
        n  = q[0]
        ex = q[1]
        ey = q[2]
        ez = q[3]
    elif shape == "xyzw":
        n  = q[3]
        ex = q[0]
        ey = q[1]
        ez = q[2]
    else:
        raise RuntimeError("The shape of quaternion should be wxyz or xyzw. Given " + shape + " instead")

    R = matlib.eye(3)

    R[0, 0] = 2 * (n * n + ex * ex) - 1
    R[0, 1] = 2 * (ex * ey - n * ez)
    R[0, 2] = 2 * (ex * ez + n * ey)

    R[1, 0] = 2 * (ex * ey + n * ez)
    R[1, 1] = 2 * (n * n + ey * ey) - 1
    R[1, 2] = 2 * (ey * ez - n * ex)

    R[2, 0] = 2 * (ex * ez - n * ey)
    R[2, 1] = 2 * (ey * ez + n * ex)
    R[2, 2] = 2 * (n * n + ez * ez) - 1

    return R

def get_homogeneous_transformation(pose):

    M = np.eye(4,4)
    M[0:3, 3] = pose[0:3]
    M[0:3, 0:3] = quat2rot(pose[3:])
    return M

def negate_fun(a):
    assert isinstance(a, np.ndarray)
    a[:] = -a
    return a


class Affine3:
    def __init__(self):
        self.translation = np.zeros(3)
        self.linear = np.eye(3)

    @classmethod
    def from_matrix(cls, matrix):
        assert isinstance(matrix, np.ndarray)
        result = cls()

        # result.translation = matrix[:3, 3].copy()
        # result.linear = matrix[0:3, 0:3].copy()

        # result.translation = matrix[:3, 3]
        # result.linear = matrix[0:3, 0:3]

        for i in range(3):
            for j in range(3):
                result.linear[i, j] = matrix[i,j]
            result.translation[i] = matrix[i, 3]


        return result

    def matrix(self):

        M = np.eye(4)

        M[0:3, 3] = self.translation
        M[0:3, 0:3] = self.linear

        # M[1, 3] = -9

        return M

    def __str__(self):
        return "Affine3:\n" + str(self.matrix())



if __name__ == '__main__':

    m0 = np.eye(4)
    m0[0:3, 3] = [4,4,4]

    tf = Affine3.from_matrix(m0)

    tf.translation[0] = 6

    m2 = tf.matrix()
    m2[1,3] = -9

    print("m0:\n", m0)
    print(tf)
    print("m2:\n", m2)




    a = np.array([1,2,3])

    b = negate_fun(a)

    print("a: ", a)
    print("b: ", b)

    # np.random.seed(0)

    p = np.random.rand(3)
    Q = np.random.rand(4)
    Q = Q / linalg.norm(Q)

    H = get_homogeneous_transformation( np.concatenate((p, Q)) )

    print(p)
    print(Q)
    print(H)

    q = [None] * 4

    print(q)

    I = np.eye(3,3)


    s = np.sum( np.diag(I) )

    print(s)

    a = np.arange(6).reshape((3,2))

    b = np.array([7, 8])
    b = np.append(b, 9)

    # c = np.concatenate((a,b))
    #
    # c = np.row_stack((a,b))
    #
    # c = np.column_stack((a,b))
    #
    # c = np.vstack((a,b))
    #
    # c = np.hstack((a,b))

    print(a)
    print(b)

    print(a.shape)

    c = np.column_stack((a,b))

    c = matlib.repmat(c, 1, 3)

    print(c)

    # n = 2
    # m = 5
    #
    # A = np.random.rand(m,n)
    #
    # x = np.random.rand(n)
    #
    # y = np.matmul(A, x)
    #
    # y_msr = y + 1e-3*np.random.randn(m)
    #
    # # x_hat = np.matmul( linalg.pinv(A), y_msr )
    #
    # x_hat = linalg.lstsq(A, y_msr)
    #
    # y_hat = np.matmul(A, x_hat)
    #
    # e = linalg.norm(y - y_hat)
    #
    # print("e = ", e)
    #
    # # + 0.1*np.random.randn(m, n)


