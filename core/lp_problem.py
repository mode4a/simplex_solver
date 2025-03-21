import numpy as np
from enum import Enum

class ObjectiveSense(Enum):
    MAXIMIZE = "MAX"
    MINIMIZE = "MIN"

class ConstraintType(Enum):
    LESS_EQUAL = "<="
    GREATER_EQUAL = ">="
    EQUAL = "="

class VariableType(Enum):
    NON_NEGATIVE = "NON-NEGATIVE"
    UNRESTRICTED = "UNRESTRICTED"

class LPProblem:
    def __init__(self, name="LP Problem"):
        self.name = name
        self.objective_coefficients = []
        self.objective_sense = ObjectiveSense.MAXIMIZE
        self.constraint_coefficients = []
        self.constraint_types = []
        self.rhs_values = []
        self.variable_types = []
        self.variable_names = []

        # for goal programming
        self.goal_priorities = []
        self.goal_values = []

        self.unrestricted_var_map = {}
        self.artificial_vars = []
        self.slack_vars = []
        self.surplus_vars = []

    def add_variable(self, name, objective_coeff=0, variable_type=VariableType.NON_NEGATIVE):
        """
        Args:
            name (str): Name of the variable
            objective_coeff (float): Coefficient in the objective function
            variable_type (VariableType): Type of the variable

        Returns:
            int: Index of the added variable
        """
        self.objective_coefficients.append(objective_coeff)
        self.variable_types.append(variable_type)
        self.variable_names.append(name)

        for i in range(len(self.constraint_coefficients)):
            self.constraint_coefficients[i].append(0)

        return len(self.variable_names) - 1

    def add_constraint(self, coefficients, constraint_type, rhs_value):
        """
        Args:
            coefficients (list): Coefficients for each variable in the constraint
            constraint_type (ConstraintType): Type of constraint
            rhs_value (float): Right-hand side value

        Returns:
            int: Index of the added constraint
        """
        if len(coefficients) != len(self.variable_names):
            raise ValueError("Number of coefficients must match number of variables")

        self.constraint_coefficients.append(coefficients)
        self.constraint_types.append(constraint_type)
        self.rhs_values.append(rhs_value)

        return len(self.constraint_types) - 1

    def add_goal(self, target_value, priority=0):
        self.goal_priorities.append(priority)
        self.goal_values.append(target_value)

        return len(self.goal_priorities) - 1

    def set_objective_sense(self, sense):
        self.objective_sense = sense

    def get_vars_count(self):
        return len(self.constraint_types)

    def get_constraints_count(self):
        return len(self.constraint_types)

    def get_constraint_matrix(self):
        return np.array(self.constraint_coefficients)

    def get_rhs_vector(self):
        return np.array(self.rhs_values)

    def get_objective_vector(self):
        return np.array(self.objective_coefficients)