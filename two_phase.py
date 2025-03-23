import numpy as np
from simplex import standard_simplex, TOLERANCE


def two_phase(tableau, variables, basic):
    obj = np.copy(tableau[-1])
    for i in range(len(variables)):
        if variables[i][0] == 'A':
            tableau[-1, i] = 1
        else:
            tableau[-1, i] = 0
    for i in range(len(basic)):
        if basic[i][0] == 'A':
            tableau[-1] -= tableau[i]
    status = standard_simplex(tableau, basic, variables)
    if status != "Optimal":
        return status
    if tableau[-1, -1] > TOLERANCE:
        return "in-feasible"
    tableau[-1] = obj
    cols_to_remove = []
    variables_to_remove = []

    for i in range(len(variables)):
        if variables[i][0] == 'A':
            cols_to_remove.append(i)
            variables_to_remove.append(variables[i])
    for i in variables_to_remove:
        variables.remove(i)
    tableau = np.delete(tableau, cols_to_remove, axis=1)

    for i in range(len(variables)):
        if variables[i] in basic:
            basic_index = basic.index(variables[i])
            tableau[-1] -= tableau[-1][i] / tableau[basic_index][i] * tableau[basic_index]

    return tableau, standard_simplex(tableau, basic, variables)