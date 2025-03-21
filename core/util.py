import numpy as np
import copy

from core.lp_problem import VariableType


def is_close_to_zero(value, epsilon = 1e-10):
    return abs(value) < epsilon

def is_nearly_equal(a, b, epsilon = 1e-10):
    return abs(a - b) < epsilon

def pivot_matrix(matrix, row, col):
    new_matrix = matrix.copy()

    pivot_element = new_matrix[row, col]
    new_matrix[row] = new_matrix[row] / pivot_element

    for i in range(matrix.shape[0]):
        if i != row:
            factor = new_matrix[i, col]
            new_matrix[i] -= factor * new_matrix[row]

    return new_matrix

def handle_unrestricted_variables(obj_coeffs, constraint_coeffs, var_types):
    new_obj_coeffs = []
    new_var_types = []
    var_map = {}
    col_offset = 0
    new_constraint_coeffs = copy.deepcopy(constraint_coeffs)

    for i, var_type in enumerate(var_types):
        if var_type == VariableType.UNRESTRICTED:
            # Replace x_i with x_i+ - x_i-
            # In the objective: c_i * x_i becomes c_i * x_i+ - c_i * x_i-
            new_obj_coeffs.append(obj_coeffs[i])
            new_obj_coeffs.append(-obj_coeffs[i])

            # In constraints: a_ij * x_i becomes a_ij * x_i+ - a_ij * x_i-
            for j in range(len(constraint_coeffs)):
                new_constraint_coeffs[j].insert(i + col_offset + 1, -constraint_coeffs[j][i + col_offset])

            new_var_types.append(VariableType.NON_NEGATIVE)
            new_var_types.append(VariableType.NON_NEGATIVE)

            var_map[i] = [i + col_offset, i + col_offset + 1]
            col_offset += 1
        else:
            new_obj_coeffs.append(obj_coeffs[i])
            new_obj_coeffs.append(var_type)
    return new_obj_coeffs, new_constraint_coeffs, new_var_types, var_map
