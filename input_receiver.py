import numpy as np
from collections import Counter
from simplex import standard_simplex
from big_m import big_m


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
    variables = ["_" for _ in range(cols)]

    # Add decision variables to the tableau and variables array
    col_index = 0
    for i in range(num_of_decision_variables):
        if (i + 1) in unr_vars:
            # Unrestricted variable: split into x_i_plus and x_i_minus
            variables[col_index] = f"x{i + 1}_plus"
            variables[col_index + 1] = f"x{i + 1}_minus"
            # Copy coefficients for x_i_plus and x_i_minus
            tableau[:-1, col_index] = constraints_coeff[:, i]  # x_i_plus
            tableau[:-1, col_index + 1] = -constraints_coeff[:, i]  # x_i_minus
            col_index += 2
        else:
            # Regular decision variable
            variables[col_index] = f"x{i + 1}"
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
            variables[slack_index] = f"S{i + 1}"
            tableau[i, slack_index] = 1  # Slack variable coefficient
            basic_variables.append(f"S{i + 1}")
            slack_index += 1
        elif constraints_type[i] == '>=':
            # Add surplus and artificial variables
            variables[surplus_index] = f"E{i + 1}"  # Surplus variable (denoted as E)
            tableau[i, surplus_index] = -1  # Surplus variable coefficient
            variables[artificial_index] = f"A{i + 1}"  # Artificial variable
            tableau[i, artificial_index] = 1  # Artificial variable coefficient
            basic_variables.append(f"A{i + 1}")
            surplus_index += 1
            artificial_index += 1
        elif constraints_type[i] == '=':
            # Add artificial variable
            variables[artificial_index] = f"A{i + 1}"  # Artificial variable
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


def get_solution(tableau, basic, variables):
    """
    Extracts the optimal objective value and decision variable values from the final tableau.
    Only returns the original decision variables (e.g., x1, x2) and excludes slack, surplus, and artificial variables.

    Args:
        tableau (numpy.ndarray): The final tableau after solving.
        basic (list): List of basic variables in the final tableau.
        variables (list): List of all variables (including slack, surplus, artificial, x⁺, and x⁻).

    Returns:
        dict: A dictionary containing:
            - 'objective': The optimal objective value.
            - 'decision_vars': A dictionary of decision variable values.
    """
    # Extract the optimal objective value from the last row and last column of the tableau
    optimal_objective = float(tableau[-1, -1])

    # Initialize a dictionary to store decision variable values
    decision_vars = {}

    # Iterate through the basic variables to find their values
    for i in range(len(basic)):
        var_name = basic[i]
        if var_name.startswith('x'):  # Check if the variable is a decision variable
            var_value = tableau[i, -1]  # Value is in the last column of the tableau
            decision_vars[var_name] = float(var_value)

    # Non-basic decision variables have a value of 0
    for var in variables:
        if var.startswith('x') and var not in decision_vars:
            decision_vars[var] = 0.0

    # Handle unrestricted variables (x = x⁺ - x⁻)
    unrestricted_vars = set()  # Track original unrestricted variable names
    for var in variables:
        if var.endswith('_plus') or var.endswith('_minus'):
            original_var = var.rsplit('_', 1)[0]  # Remove '_plus' or '_minus' to get the original variable name
            unrestricted_vars.add(original_var)

    for var in unrestricted_vars:
        x_plus = decision_vars.get(f"{var}_plus", 0.0)  # Get x⁺ value (default to 0 if not in basis)
        x_minus = decision_vars.get(f"{var}_minus", 0.0)  # Get x⁻ value (default to 0 if not in basis)
        decision_vars[var] = x_plus - x_minus  # Compute x = x⁺ - x⁻

        # Remove x⁺ and x⁻ from the decision_vars dictionary
        if f"{var}_plus" in decision_vars:
            del decision_vars[f"{var}_plus"]
        if f"{var}_minus" in decision_vars:
            del decision_vars[f"{var}_minus"]

    # Filter out slack, surplus, and artificial variables
    decision_vars = {var: value for var, value in decision_vars.items() if var.startswith('x')}

    return {
        'objective': optimal_objective,
        'decision_vars': decision_vars
    }

obj = [4, 6, 3]
constraints_coeff = [
    [1, 2, 1, 8],
    [2, 1, 3, 12],
    [3, -1, 2, 7]
]
constraints_type = ['<=', '=', '>=']
unr_vars = [2]
problem_type = 'max'



tableau, basic_variables, variables = input_processor(obj, constraints_coeff, constraints_type, unr_vars, problem_type)

print("Tableau:")
print(tableau)
print("Basic Variables:", basic_variables)
print("Variables:", variables)

tableau, basic_variables, variables, status = big_m(tableau, variables, basic_variables, problem_type)
if(status == "Optimal"):
    print(get_solution(tableau, basic_variables, variables))
else:
    print(status)