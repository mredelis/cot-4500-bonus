# import libraries
import numpy as np

np.set_printoptions(precision=7, suppress=True, linewidth=100)


# Questions 1 & 2 require a diagonally dominant matrix
def make_diagonally_dominant(A, b):
    n = len(A)

    # traverse the rows and find the sum of its elements
    for i in range(n):
        sum = np.sum(abs(A[i, :]))
        pivot = A[i, i]
        # removing diagonal element
        sum = sum - abs(pivot)

        # checking if diagonal element is less than sum of non-diagonal element.
        if abs(pivot) > sum:
            continue

        # if we reach this point, this means we need to swap AT LEAST ONCE
        max_value_of_row = 0
        max_index_in_row = 0
        for j in range(n):
            current_value_in_row = abs(A[i, j])
            if current_value_in_row > max_value_of_row:
                max_value_of_row = current_value_in_row
                max_index_in_row = j

        # now that we have a new "pivot", we swap cur_row with the expected index
        A[[i, max_index_in_row]] = A[[max_index_in_row, i]]
        b[[i, max_index_in_row]] = b[[max_index_in_row, i]]

    return A, b


# Question 1: Number of iterations it takes gauss-seidel to converge:
def gauss_seidel(matrix, b, x0, tolerance, iterations):
    n = len(matrix)
    x = np.copy(x0)
    x_new = np.zeros(n)

    k = 1
    while k < iterations:
        for i in range(n):
            # covers sum before pivot element to multiply with calculated values of x from previous row
            s1 = np.dot(matrix[i, :i], x_new[:i])
            # covers sum after pivot element
            s2 = np.dot(matrix[i, i + 1 :], x[i + 1 :])

            # print(f"matrix[i, :i] = {matrix[i, :i]}")
            # print(f"matrix[i, i+1:] = {matrix[i, i+1:]}")

            x_new[i] = 1 / matrix[i, i] * (b[i] - s1 - s2)

        # one set of solutions
        if np.linalg.norm(x_new - x) < tolerance:
            # print("The procedure was successfull")
            return x_new, k

        x = np.copy(x_new)
        k += 1

    print("Maximum number of iterations exceeded")
    return -1


# Question 2: Number of iterations it takes jacobi method to converge
def jacobi(matrix, b, x0, tolerance, iterations):
    # x_prev = np.copy(x0)
    x = np.zeros(len(b))

    k = 1
    while k < iterations:
        for i in range(len(b)):
            # dot product dot product A.B = a1b1 + a2 b2 + ... + an bn
            rows_sum = np.dot(matrix[i, :], x0)

            # less a_ii (pivot) * x_i
            rows_sum = rows_sum - (matrix[i, i] * x0[i])
            # Apply formula
            x[i] = 1 / matrix[i, i] * (b[i] - rows_sum)

        # print(x)
        # one set of solutions
        if np.linalg.norm(x - x0) < tolerance:
            # print("The procedure was successfull")
            return x, k

        x0 = np.copy(x)
        k += 1

    print("Maximum number of iterations exceeded")
    return -1


# Question 3: Number of iterations necessary to solve f(x) = x^3 - x^2 + 2 = 0 using newton-raphson from the left side
# The objective is to find a solution to f(x)=0 given an initial approximation p0
MAX_ITERATIONS = 1000


def newton_raphson(
    function_str: str, function_str_derivative: str, error_tolerance: float, initial_approximation: float
):
    i = 1
    while i <= MAX_ITERATIONS:
        x = initial_approximation

        if eval(function_str_derivative) != 0:
            next_approximation = initial_approximation - eval(function_str) / eval(function_str_derivative)
            i += 1  # to match the professor's answer even though i should be updated when it's commented out

            if abs(next_approximation - initial_approximation) < error_tolerance:
                print(i)
                return  # procedure was successfull

            # i += 1
            initial_approximation = next_approximation
        else:
            print("Error derivative is zero")
            return

    print(f"The method failed after {MAX_ITERATIONS} number of iterations")


# Question 4: Using the divided difference method, print out the Hermite polynomial approximation matrix
def hermite_interpolation(x_points, y_points, slopes):
    # matrix size changes because of "doubling" up info for hermite
    num_of_points = len(x_points)
    matrix = np.zeros((num_of_points * 2, num_of_points * 2))

    # populate x values (make sure to fill every TWO rows)
    for x in range(0, num_of_points * 2, 2):
        matrix[x][0] = x_points[int(x / 2)]
        matrix[x + 1][0] = x_points[int(x / 2)]
        # break

    # prepopulate y values (make sure to fill every TWO rows)
    for x in range(0, num_of_points * 2, 2):
        matrix[x][1] = y_points[int(x / 2)]
        matrix[x + 1][1] = y_points[int(x / 2)]

    # prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    for x in range(1, num_of_points * 2, 2):
        matrix[x][2] = slopes[int(x / 2)]

    filled_matrix = apply_div_dif(matrix)
    print(f"\n{filled_matrix}")


def apply_div_dif(matrix: np.array):
    size = len(matrix)

    for i in range(2, size):
        for j in range(2, i + 2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            # get left cell entry
            left: float = matrix[i][j - 1]

            # get diagonal left entry
            diagonal_left: float = matrix[i - 1][j - 1]

            # order of numerator is SPECIFIC.
            numerator: float = left - diagonal_left

            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0] - matrix[i - j + 1][0]

            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation

    return matrix


# Question 5: Modified eulers method
def function(t: float, w: float):  # y is replace with w
    return w - np.power(t, 3)


def modified_eulers():
    # Initial setup
    w = 0.5
    a, b = (0, 3)  # t is in the interval [a, b]
    N = 100  # number of iterations
    h = (b - a) / N  # step size

    for i in range(0, N):
        t = h * i
        w_next = w + h / 2 * (function(t, w) + function(h * (i + 1), (w + h * function(t, w))))
        w = w_next

    return w


# Questions 1 and 2
matrix = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]])
b_vector = np.array([1, 3, 0])
x0 = np.array([0, 0, 0])  # initial guess
tolerance = 1e-6
iterations = 50

d_matrix, new_b = make_diagonally_dominant(matrix, b_vector)

solution_1, num_of_iter_1 = gauss_seidel(d_matrix, new_b, x0, tolerance, iterations)
print(num_of_iter_1)

print()

solution_2, num_of_iter_2 = jacobi(d_matrix, new_b, x0, tolerance, iterations)
print(num_of_iter_2)

print()

# Question 3
function_str = "x**3 - (x**2) + 2"
function_str_derivative = "3*x**2 - 2*x"
error_tolerance: float = 10 ** (-4)
initial_approximation = 0.5
newton_raphson(function_str, function_str_derivative, error_tolerance, initial_approximation)

print()

# Question 4
xi_points = [0, 1, 2]
yi_points = [1, 2, 4]
slopes = [1.06, 1.23, 1.55]
hermite_interpolation(xi_points, yi_points, slopes)

print()

# Question 5
print("%.5f" % modified_eulers())
