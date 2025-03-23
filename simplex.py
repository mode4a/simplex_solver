import numpy as np


np.set_printoptions(suppress=True, precision=6)

TOLERANCE = 1e-7

def get_pivot_col(tableau):
    last_row = tableau[-1, :-1]
    pivot_col = np.argmin(last_row)
    return pivot_col if last_row[pivot_col] < -TOLERANCE else None

def get_pivot_row(tableau, pivot_col):
    pivot_col_values = tableau[:-1, pivot_col]
    rhs_col = tableau[:-1, -1]
    ratios = np.where(pivot_col_values > TOLERANCE, rhs_col / pivot_col_values, np.inf)
    pivot_row = np.argmin(ratios)
    return pivot_row if ratios[pivot_row] != np.inf else None


# def pivot_tableau(tableau, pivot_row, pivot_col):
#     updated_tableau = np.copy(tableau)
#     pivot_element = updated_tableau[pivot_row, pivot_col]
#     updated_tableau[pivot_row, :] = updated_tableau[pivot_row, :] / pivot_element
#
#     for row in range(tableau.shape[0]):
#         if row != pivot_row:
#             multip = updated_tableau[row, pivot_col]
#             updated_tableau[row, :] -= updated_tableau[pivot_row, :] * multip
#     return updated_tableau

def pivot_tableau(tableau, pivot_row, pivot_col):
    updated_tableau = tableau  # No copy, just a reference
    pivot_element = updated_tableau[pivot_row, pivot_col]
    updated_tableau[pivot_row, :] = updated_tableau[pivot_row, :] / pivot_element
    for row in range(tableau.shape[0]):
        if row != pivot_row:
            multip = updated_tableau[row, pivot_col]
            updated_tableau[row, :] -= updated_tableau[pivot_row, :] * multip
    return updated_tableau


def standard_simplex(tableau, basic, variables):
    iteration = 0
    while True:
        print(f"\nIteration {iteration}:")
        print(variables)
        print(tableau)
        print(f"basic : {basic}")

        pivot_col = get_pivot_col(tableau)
        if pivot_col is None:
            status = "Optimal"
            break

        pivot_row = get_pivot_row(tableau, pivot_col)
        if pivot_row is None:
            status = "Unbounded"
            break

        basic[pivot_row] = variables[int(pivot_col)]

        print(f"Pivot element at row {pivot_row}, column {pivot_col}")
        tableau = pivot_tableau(tableau, pivot_row, pivot_col)
        iteration += 1

    return status
