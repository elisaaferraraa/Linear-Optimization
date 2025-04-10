import numpy as np
import matplotlib.pyplot as plt
from simplex_solver import solve_simplex

def create_board(rows, cols, markings):
    board = np.zeros((rows, cols), dtype=int)
    for row, col in markings:
        board[row, col] = 1
    return board

def create_incidence_matrix(rows, cols, markings):
    incidence_matrix = np.zeros((rows + cols, len(markings)), dtype=int)
    for j, (row, col) in enumerate(markings):
        incidence_matrix[row, j] = 1
        incidence_matrix[col + rows, j] = 1
    return incidence_matrix

def build_matching_matrix(rows, cols, markings, matchings):
    matrix = np.zeros((rows, cols), dtype=int)
    for j, (row, col) in enumerate(markings):
        if np.abs(matchings[j]) > 1e-8:
            matrix[row, col] = 1
    return matrix

def build_vertexcover_matrix(rows, cols, vertexcovers):
    matrix = np.zeros((rows, cols), dtype=int)
    for row in range(rows):
        if np.abs(vertexcovers[row]) > 1e-8:
            matrix[row, :] = 1
    for col in range(cols):
        if np.abs(vertexcovers[col + rows]) > 1e-8:
            matrix[:, col] = 1
    return matrix

def make_plots(rows, cols, markings, marker_sizes, pivoting_rule):
    # Plot board with marked tiles (in red)
    board = create_board(rows, cols, markings)
    plt.spy(board, markersize=marker_sizes[0], color='red')

    # Define incidence matrix
    incidence_matrix = create_incidence_matrix(rows, cols, markings)
    A = incidence_matrix
    b = np.ones(A.shape[0], dtype=int)
    c = np.ones(len(markings), dtype=int)

    # Solve dual (maximum matching) and plot the maximum pairing (in yellow)
    A_dual = np.hstack([A, np.eye(A.shape[0])])
    c_dual = np.hstack([-c, np.zeros(A.shape[0])])
    dual_sol, dual_value, _ = solve_simplex(A_dual, b, c_dual, pivoting_rule) ### Change pivoting rule here
    print("Optimal matching value:", -dual_value)
    matching_matrix = build_matching_matrix(rows, cols, markings, dual_sol[:A.shape[1]])
    plt.spy(matching_matrix, markersize=marker_sizes[1], color='yellow')

    # Solve primal (minimum vertex cover) and plot the minimum line-cover (in blue)
    A_primal = np.hstack([A.T, -np.eye(A.shape[1])])
    c_primal = np.hstack([b, np.zeros(A.shape[1])])
    primal_sol, primal_value, _ = solve_simplex(A_primal, c, c_primal, pivoting_rule) ### Change pivoting rule here
    print("Optimal vertex cover value:", primal_value)
    vertexcover_matrix = build_vertexcover_matrix(rows, cols, primal_sol[:A.shape[0]])
    plt.spy(vertexcover_matrix, markersize=marker_sizes[2], color='blue')

def main():
    ### Choose pivoting rule
    pivoting_rule = 'dantzig'
    #pivoting_rule = 'bland'

    # Figure 1 board
    rows, cols = 6, 6
    markings = [(0, 1), (2, 0), (2, 3), (2, 5), (3, 1), (4, 1), (4, 2), (4, 4), (5, 0)]
    marker_sizes = [20, 10, 5]
    plt.figure()
    make_plots(rows, cols, markings, marker_sizes, pivoting_rule)
    plt.show()

    # Large 100 x 100 board
    rows, cols = 100, 100
    markings = np.load('markings.npy')
    marker_sizes = [4, 2, 1]
    plt.figure()
    make_plots(rows, cols, markings, marker_sizes, pivoting_rule)
    plt.show()

if __name__ == '__main__':
    main()
