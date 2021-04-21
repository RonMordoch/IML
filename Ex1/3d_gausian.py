import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

mean = [0, 0, 0]
cov = np.eye(3)  # I3 - Identity matrix 3x3
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def multi_variate_gausian():
    # q11
    plot_3d(x_y_z)
    plt.title("Q11")
    plt.show()

    # q12
    scaling_matrix = np.array([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 2]])
    scaled_matrix = np.matmul(scaling_matrix, x_y_z)
    print("q12: Covariance matrix numerically:\n", np.cov(scaled_matrix))
    plot_3d(scaled_matrix)
    plt.title("Q12")
    plt.show()

    # q13
    orthogonal_mat = get_orthogonal_matrix(3)
    scaled_mul_orthogonal = np.matmul(orthogonal_mat, scaled_matrix)
    print("q13: Random Orthogonal matrix:\n", orthogonal_mat)
    print("q13: Covariance matrix numerically:\n",
          np.cov(scaled_mul_orthogonal))
    plot_3d(scaled_mul_orthogonal)
    plt.title("Q13")

    # q14
    # Orthogonally projecting onto V = R^2, (e1,e2) is an orthonormal basis
    # data matrix x_y_z dimension is: 3x50000
    # we want to multiply P * data , therefore we need P dimension to be 2x3
    projection_matrix = np.array([[1, 0, 0], [0, 1, 0]])
    x_y_projection = np.matmul(projection_matrix, x_y_z)
    plot_2d(x_y_projection)
    plt.title("Q14")
    plt.show()

    # q15
    z_values = x_y_z[2]
    valid_z_values = np.where((-0.4 < z_values) & (z_values < 0.1))[0]
    # works the same as logical &&
    filtered_data = np.take(x_y_z, valid_z_values, axis=1)
    filtered_x_y_projection = np.matmul(projection_matrix, filtered_data)
    plot_2d(filtered_x_y_projection)
    plt.title("Q15")
    plt.show()


multi_variate_gausian()
