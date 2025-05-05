import numpy as np


def build_phase_one_tableau(A, b):
    """Build the tableau for phase one of the simplex method.
    Args:
        A (np.ndarray): Coefficient matrix of the constraints.
        b (np.ndarray): Right-hand side vector.
    Returns:
        tuple: Tableau (np.ndarray) and basis (list) for phase one. The tableau includes auxillary variables 
        and the basis is initialized to the auxillary variables.

        Note: In this implementation of the tableau
               - the last row consists of the current reduced cost and cost and 
               - the first m entries of last column is the current basic solution.
        i.e., we have
        tableau =  [[ remaining_tableau,  basic solution]
                    [ reduced_cost     ,    -cost       ]]
        This differs slightly from the tableau in the lecture notes, but simplifies indexing.
        We use this implementation for the rest of the code.
    """
    m, n = A.shape
    I = np.eye(m)
    tableau = np.hstack([A, I, b.reshape(-1, 1)])
    cost = np.hstack([np.zeros(n), np.ones(m), 0])
    tableau = np.vstack([tableau, cost])
    # Subtract artificial variable rows to initialize phase one cost
    for i in range(m):
        tableau[-1, :] -= tableau[i, :]
    basis = list(range(n, n + m))
    return tableau, basis


def solve_phase_one(tableau, basis, m, n, pivoting_rule):
    """Solve the phase one of the simplex method.
    Args:
        tableau (np.ndarray): The tableau matrix.
        basis (list): The current basis variables.
        m (int): Number of constraints.
        n (int): Number of variables.
        pivoting_rule (str): Pivoting rule to use ('dantzig' or 'bland').
    Returns:
        tuple: Final tableau (np.ndarray) and bafzsis (list).
    """
    while True:
        row, col = find_pivot(tableau, m, n + m, pivoting_rule)
        if row is None:
            break
        tableau, basis = pivot(tableau, basis, row, col)

    # positive cost (minus the value in tableau is the cost)
    if -tableau[-1, -1] > 1e-8:
        raise Exception("Infeasible problem.")
    return tableau, basis


def cleanup_artificial_variables(tableau, basis, n):
    """Remove artificial variables from the tableau.
    Args:
        tableau (np.ndarray): The tableau matrix.
        basis (list): The current basis variables.
        n (int): Number of original (i.e., non-auxillary) variables.
    Returns:
        tuple: Updated tableau (np.ndarray), basis (list) and number of constraints m (int) (m may decrease if a row is removed).
    """
    m = len(basis)
    to_remove = []
    for i, var in enumerate(basis):
        if var >= n:
            row_has_only_artificial = np.all(np.abs(tableau[i, :n]) < 1e-8)
            if row_has_only_artificial:
                to_remove.append(i)
            else:
                for j in range(n):
                    if j not in basis and abs(tableau[i, j]) >= 1e-8:
                        tableau, basis = pivot(tableau, basis, i, j)
                        break

    if to_remove:
        tableau = np.delete(tableau, to_remove, axis=0)
        basis = [var for idx, var in enumerate(basis) if idx not in to_remove]

    m = len(basis)
    return tableau, basis, m

# The following functions will be written by you.


def dantzig_rule(tableau):
    """Dantzig's rule for selecting entering variable.
    Args:
        tableau (np.ndarray): The tableau matrix.
    Returns:
        int: Index of the entering variable.
    """
    # YOUR CODE HERE
    # Choose the non-basic variable with the most negative reduced cost
    # (i.e., the most negative value in the last row of the tableau)
    reduced_costs = tableau[-1, :-1]
    # In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
    entering_var = np.argmin(reduced_costs)
    if reduced_costs[entering_var] >= 0:
        return None
    return entering_var


def bland_rule(tableau, n):
    """Bland's rule for selecting entering variable.
    Args:
        tableau (np.ndarray): The tableau matrix.
        n (int): Number of variables.
    Returns:
        int: Index of the entering variable.
    """
    # YOUR CODE HERE
    # Always choose the variable with the smallest index among those eligible. This prevents cycling.
    reduced_costs = tableau[-1, :-1]
    # Option 1: Only consider original variables
    entering_var_eligible = np.where(reduced_costs[:n] < 0)[
        0]  # safest: exclude artificial variables
    if len(entering_var_eligible) == 0:
        return None
    entering_var = entering_var_eligible[0]
    return entering_var


def find_pivot(tableau, m, n, pivoting_rule):
    """Find the pivot element in the tableau.
    Args:
        tableau (np.ndarray): The tableau matrix.
        m (int): Number of constraints.
        n (int): Number of variables.
        pivoting_rule (str): Pivoting rule to use ('dantzig' or 'bland').
    Returns:
        tuple: Row and column indices (int, int) of the pivot element.
        If the reduced cost vector is non-negative, returns (None, None).
    Raises:
        Exception: If the problem is unbounded.
    """
    # YOUR CODE HERE

    # ---- Entering variable section----
    if pivoting_rule == 'dantzig':
        col = dantzig_rule(tableau)
    elif pivoting_rule == 'bland':
        col = bland_rule(tableau, n)
    else:
        raise ValueError("Invalid pivoting rule. Use 'dantzig' or 'bland'.")
    if col is None:
        return None, None
    column = tableau[:-1, col]

    # ----Leaving variable section----
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(column > 1e-10, tableau[:-1, -1] / column, np.inf)
    row = np.argmin(ratios)
    if ratios[row] == np.inf:
        raise Exception("Unbounded problem.")
    return row, col


def pivot(tableau, basis, row, col):
    """Perform pivot operation on tableau at specified row and column.
    Args:
        tableau (np.ndarray): The tableau matrix.
        basis (list): The current basis variables.
        row (int): The pivot row index.
        col (int): The pivot column index.
    Returns:
        tuple: Updated tableau (np.ndarray) and basis (list).
    """
    # YOUR CODE HERE
    pivot_element = tableau[row, col]
    tableau[row, :] /= pivot_element
    for i in range(len(tableau)):
        if i != row:
            tableau[i, :] -= tableau[i, col] * tableau[row, :]
    basis[row] = col
    return (tableau, basis)


def build_phase_two_tableau(tableau, basis, c, n):
    """Build the tableau for phase two of the simplex method.
    Args:
        tableau (np.ndarray): The tableau at the end of phase one.
        basis (list): The current basis variables (no auxillary variables are in the basis).
        c (np.ndarray): Coefficients of the objective function.
        n (int): Number of original variables.
    Returns:
        np.ndarray: The tableau for phase two.
    """
    # YOUR CODE HERE
    tableau1 = tableau.copy()
    m = tableau.shape[0] - 1  # number of constraints

    # Initialize new tableau: same number of rows, n original vars + 1 RHS column
    tableau_phase_two = np.zeros((m + 1, n+1))

    # Copy the B⁻¹A and B⁻¹b parts from Phase I
    tableau_phase_two[:-1, :-1] = tableau1[:-1, :n]
    tableau_phase_two[:-1, -1] = tableau1[:-1, -1]

    # Compute reduced cost row
    c_B = c[basis]  # basic cost vector (c_B)
    A_B_inv = tableau1[:-1, :n]  # rows correspond to B⁻¹A
    b_B_inv = tableau1[:-1, -1]  # B⁻¹b

    # Reduced cost row: c̄ = c^T - c_B^T B⁻¹A
    reduced_costs = c - np.dot(c_B, A_B_inv)

    # Objective value: z = c_B^T B⁻¹b
    cost_value = np.dot(c_B, b_B_inv)

    # Populate last row
    tableau_phase_two[-1, :-1] = reduced_costs
    tableau_phase_two[-1, -1] = -cost_value

    return tableau_phase_two


def solve_phase_two(tableau, basis, m, n, pivoting_rule):
    """Solve the phase two of the simplex method.
    Args:
        tableau (np.ndarray): The tableau matrix.
        basis (list): The current basis variables.
        c (np.ndarray): Coefficients of the objective function.
        n (int): Number of original variables.
        pivoting_rule (str): Pivoting rule to use ('dantzig' or 'bland').
    Returns:
        tuple: Final tableau (np.ndarray) and basis (list).
    """
    # YOUR CODE HERE

    while True:
        row, col = find_pivot(tableau, m, n, pivoting_rule)
        if row is None:
            break
        tableau, basis = pivot(tableau, basis, row, col)
    return tableau, basis


def get_solution(tableau, basis, n):
    """Extract the solution to the standard form LP from the tableau.
    Args:
        tableau (np.ndarray): The final tableau matrix.
        basis (list): The current basis variables.
        n (int): Number of original variables.
    Returns:
        tuple: Solution vector (np.ndarray), optimal value (float), and optimal basis (list).
    """
    # YOUR CODE HERE
    solution = np.zeros(n)
    for i, var in enumerate(basis):
        if var < n:
            solution[var] = tableau[i, -1]
    optimal_value = -tableau[-1, -1]
    return solution, optimal_value, basis


def solve_simplex(A, b, c, pivoting_rule):
    """Solve the linear programming problem using the simplex method.
    Args:
        A (np.ndarray): Coefficient matrix of the constraints.
        b (np.ndarray): Right-hand side vector.
        c (np.ndarray): Coefficients of the objective function.
        pivoting_rule (str): Pivoting rule to use ('dantzig' or 'bland').
    Returns:
        tuple: Solution vector (np.ndarray), optimal value (float), and optimal basis (list).
    """
    m, n = A.shape
    tableau, basis = build_phase_one_tableau(A, b)
    tableau, basis = solve_phase_one(tableau, basis, m, n, pivoting_rule)
    tableau, basis, m = cleanup_artificial_variables(tableau, basis, n)
    tableau = build_phase_two_tableau(tableau, basis, c, n)
    tableau, basis = solve_phase_two(tableau, basis, m, n, pivoting_rule)
    return get_solution(tableau, basis, n)


# Example usage
if __name__ == "__main__":
    A = np.array([[1, 2, 1],
                  [2, 1, 3]])  # update x dimension in cvxpy problem to use this problem
    b = np.array([30, 60])
    c = np.array([2, 1, 3])

    # A = np.array([
    #     [2, 3, 1, 0],
    #     [4, 1, 0, 1]
    # ])
    # b = np.array([8, 7])
    # c = np.array([3, 5, 0, 0]) #update x dimension in cvxpy problem to use this problem

    x, value, basis = solve_simplex(A, b, c, pivoting_rule='bland')
    # print("Optimal solution:", x)
    # print("Optimal value:", value)
    # print("Optimal basis:", basis)
    print("Optimal solution:", np.array2string(
        x, formatter={'float_kind': lambda x: f"{x:.1f}"}))
    print(f"Optimal value: {value:.1f}")
    print("Optimal basis:", basis)

    # Compare with cvxpy
    import cvxpy
    x_cvxpy = cvxpy.Variable(3)
    objective = cvxpy.Minimize(c @ x_cvxpy)
    constraints = [A @ x_cvxpy == b, x_cvxpy >= 0]
    prob = cvxpy.Problem(objective, constraints)
    prob.solve()
    # print("cvxpy optimal solution:", x.value)
    # print("cvxpy optimal value:", prob.value)
    print("cvxpy optimal solution:", np.array2string(
        x_cvxpy.value, formatter={'float_kind': lambda x: f"{x:.1f}"}))
    print(f"cvxpy optimal value: {prob.value:.1f}")

    # Check if the two solutions are the same
    tolerance = 1e-4
    if (value - prob.value) < tolerance and np.allclose(x, x_cvxpy.value, atol=tolerance):
        print("The value and the solution are the same with our implemenmtation and cvxpy.")
    elif (value - prob.value) < tolerance:
        print("The values are the same but the solution is not. Let's verify the correctness of the solution")
        vec = x - x_cvxpy.value
        print("c^T = ", c)
        print("The difference between the two solutions is difference = ", vec)
        correctness = np.dot(c.T, vec)
        print(f"c^T . difference = {correctness:.1f}")

        if correctness < tolerance:
            print("Our solution is correct because the line that connects our solution with the cvxpy one is perpendicular to the gradient of the function to minimize")
        else:
            print("The solution is not correct.")
    else:
        print("The value are not the same so probably there is something wrong !!")
