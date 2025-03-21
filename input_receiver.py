import numpy as np
from collections import Counter
from simplex import standard_simplex

def count_constraint_types(constraints):
    constraint_counts = Counter(constraints)
    return dict(constraint_counts)


def input_processor(obj, constraints_coeff, constraints_type, unr_vars, problem_type):
    # Convert inputs to numpy arrays for easier manipulation
    obj = np.array(obj, dtype=float)
    constraints_coeff = np.array(constraints_coeff, dtype=float)

    num_of_decision_variables = len(constraints_coeff[0]) - 1  # Exclude RHS
    num_of_constraints = len(constraints_coeff)
    constraint_counts = count_constraint_types(constraints_type)

    # Calculate the number of columns in the tableau
    # Decision variables (including unrestricted variables) + slack/surplus/artificial variables + RHS
    num_unrestricted = len(unr_vars)  # Number of unrestricted variables
    cols = (
            num_of_decision_variables + num_unrestricted +  # Decision variables (unrestricted add an extra column each)
            constraint_counts.get('<=', 0) +  # Slack variables for '<=' constraints
            constraint_counts.get('>=', 0) +  # Surplus variables for '>=' constraints
            constraint_counts.get('>=', 0) +  # Artificial variables for '>=' constraints
            constraint_counts.get('=', 0) +  # Artificial variables for '=' constraints
            1  # RHS column
    )

    # Initialize the tableau with zeros
    tableau = np.zeros((num_of_constraints + 1, cols))  # +1 for the objective row

    # Initialize the variables array
    variables = []

    # Add decision variables to the tableau and variables array
    col_index = 0
    for i in range(num_of_decision_variables):
        if (i + 1) in unr_vars:
            # Unrestricted variable: split into x_i_plus and x_i_minus
            variables.append(f"x{i + 1}_plus")
            variables.append(f"x{i + 1}_minus")
            # Copy coefficients for x_i_plus and x_i_minus
            tableau[:-1, col_index] = constraints_coeff[:, i]  # x_i_plus
            tableau[:-1, col_index + 1] = -constraints_coeff[:, i]  # x_i_minus
            col_index += 2
        else:
            # Regular decision variable
            variables.append(f"x{i + 1}")
            tableau[:-1, col_index] = constraints_coeff[:, i]
            col_index += 1

    # Add slack, surplus, and artificial variables
    basic_variables = []
    slack_index = col_index
    surplus_index = col_index + constraint_counts.get('<=', 0)
    artificial_index = surplus_index + constraint_counts.get('>=', 0)

    for i in range(num_of_constraints):
        if constraints_type[i] == '<=':
            # Add slack variable
            variables.append(f"S{i + 1}")
            tableau[i, slack_index] = 1  # Slack variable coefficient
            basic_variables.append(f"S{i + 1}")
            slack_index += 1
        elif constraints_type[i] == '>=':
            # Add surplus and artificial variables
            variables.append(f"E{i + 1}")  # Surplus variable (denoted as E)
            tableau[i, surplus_index] = -1  # Surplus variable coefficient
            variables.append(f"A{i + 1}")  # Artificial variable
            tableau[i, artificial_index] = 1  # Artificial variable coefficient
            basic_variables.append(f"A{i + 1}")
            surplus_index += 1
            artificial_index += 1
        elif constraints_type[i] == '=':
            # Add artificial variable
            variables.append(f"A{i + 1}")  # Artificial variable
            tableau[i, artificial_index] = 1  # Artificial variable coefficient
            basic_variables.append(f"A{i + 1}")
            artificial_index += 1

    # Add RHS values
    tableau[:-1, -1] = constraints_coeff[:, -1]

    # Add the objective function row
    obj_row = np.zeros(cols)
    col_index = 0
    for i in range(num_of_decision_variables):
        if (i + 1) in unr_vars:
            # Unrestricted variable: split into x_i_plus and x_i_minus
            obj_row[col_index] = -obj[i] if problem_type == 'max' else obj[i]  # x_i_plus
            obj_row[col_index + 1] = obj[i] if problem_type == 'max' else -obj[i]  # x_i_minus
            col_index += 2
        else:
            # Regular decision variable
            obj_row[col_index] = -obj[i] if problem_type == 'max' else obj[i]
            col_index += 1

    # Add the objective function row to the tableau
    tableau[-1, :] = obj_row

    return tableau, basic_variables, variables


# Example usage
obj = [3, 2]  # Objective function coefficients
constraints_coeff = [
    [1, 2, 10],  # 1*x1 + 2*x2 <= 10
    [2, 1, 8],  # 2*x1 + 1*x2 <= 8
    [1, -1, 2]  # 1*x1 - 1*x2 >= 2
]
constraints_type = ['<=', '<=', '>=']
unr_vars = [1, 2]  # x1 is unrestricted
problem_type = 'max'  # Maximization problem

tableau, basic_variables, variables = input_processor(obj, constraints_coeff, constraints_type, unr_vars, problem_type)

print("Tableau:")
print(tableau)
print("Basic Variables:", basic_variables)
print("Variables:", variables)

print(standard_simplex(tableau, basic_variables, variables))