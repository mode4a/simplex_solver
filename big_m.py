import numpy as np
from simplex import standard_simplex

def big_m(tableau, variables, basic):
    M = 100
    for i in range(len(variables)):
        if variables[i][0] == 'A':
            tableau[-1][i] = M
    for i in range(len(basic)):
        if basic[i][0] == 'A':
            tableau[-1] = tableau[i] * -M + tableau[-1]
    status = standard_simplex(tableau, basic, variables)
    if status == "Optimal":
        for i in range(len(basic)):
            if basic[i][0] == 'A' and tableau[i][-1] != 0:
                return "in-feasible"
    return status