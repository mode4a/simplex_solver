import numpy as np
from enum import Enum
import logging

from core.lp_problem import ConstraintType
from core.util import pivot_matrix

class TableauStatus(Enum):
    OPTIMAL = "OPTIMAL"
    UNBOUNDED = "UNBOUNDED"
    INFEASIBLE = "INFEASIBLE"
    CONTINUE = "CONTINUE"


class SimplexTableau:

    def __init__(self, constraints_count, variables_count):
        # add one row for objective
        # add one col for rhs values
        self.tableau = np.zeros((constraints_count + 1, variables_count + 1))

        self.basic_variables = []
        self.non_basic_variables = []

        self.obj_row = constraints_count
        self.original_variables = variables_count

        self.variables_map = {}
        self.artificial_variables = []

        self.iteration_count = 0
        self.iteration_history = []

    def set_objective_row(self, coefficients):
        for i, coeff in enumerate(coefficients):
            self.tableau[self.obj_row, i] = -coeff

    def set_constraints(self, constraint_matrix, rhs_vector):
        m, n = constraint_matrix.shape

        self.tableau[:m, :n] = constraint_matrix
        self.tableau[:m, -1] = rhs_vector

    def add_slack_variables(self, constraint_types):
        slack_vars = []
        m = self.obj_row
        n = self.tableau.shape[1] - 1

        new_n = n + sum(1 for ct in constraint_types if ct == ConstraintType.LESS_EQUAL)
        new_tableau = np.zeros((m + 1, new_n + 1))

        new_tableau[:, :n] = self.tableau[:, :n]
        new_tableau[:, -1] = self.tableau[:, -1]

        slack_col = n
        for i, ct in enumerate(constraint_types):
            if ct == ConstraintType.LESS_EQUAL:
                new_tableau[i, slack_col] = 1
                slack_vars.append(slack_col)
                slack_col += 1

        self.tableau = new_tableau

        for var in slack_vars:
            self.basic_variables.append(var)

        return slack_vars

    def add_surplus_variables(self, constraint_types):
        surplus_vars = []
        m = self.obj_row
        n = self.tableau.shape[1] - 1

        new_n = n + sum(1 for ct in constraint_types if ct == ConstraintType.GREATER_EQUAL)
        new_tableau = np.zeros((m + 1, new_n + 1))

        new_tableau[:, :n] = self.tableau[:, :n]
        new_tableau[:, -1] = self.tableau[:, -1]

        surplus_col = n
        for i, ct in enumerate(constraint_types):
            if ct == ConstraintType.GREATER_EQUAL:
                new_tableau[i, surplus_col] = -1
                surplus_vars.append(surplus_col)
                surplus_col += 1

        self.tableau = new_tableau

        for var in surplus_vars:
            self.non_basic_variables.append(var)

        return surplus_vars

    def add_artificial_variables(self, constraint_types):
        artificial_vars = []
        m = self.obj_row
        n = self.tableau.shape[1] - 1

        artificial_count = sum(1 for ct in constraint_types if ct in [ConstraintType.GREATER_EQUAL, ConstraintType.EQUAL])

        new_n = n + artificial_count
        new_tableau = np.zeros((m + 1, new_n + 1))

        new_tableau[:, :n] = self.tableau[:, :n]
        new_tableau[:, -1] = self.tableau[:, -1]

        artificial_col = n
        for i, ct in enumerate(constraint_types):
            if ct in [ConstraintType.GREATER_EQUAL, ConstraintType.EQUAL]:
                new_tableau[i, artificial_col] = 1
                artificial_vars.append(artificial_col)
                artificial_col += 1

        self.tableau = new_tableau
        self.artificial_variables = artificial_vars

        for var in artificial_vars:
            self.basic_variables.append(var)

        return artificial_vars

    def apply_big_m_penalty(self, artificial_vars, M=1000):
        """
        Args:
            artificial_vars (list): Indices of artificial variables
            M (float): Big-M value
        """
        for var in artificial_vars:
            self.tableau[self.obj_row, var] = M

        for var in artificial_vars:
            for i in range(self.obj_row):
                if self.tableau[i, var] == 1:
                    self.tableau[self.obj_row] -= M * self.tableau[i]
                    break

    def find_entering_variable(self):
        """
        Returns:
            int: Index of the entering variable, or None if optimal
        """
        obj_row = self.tableau[self.obj_row, :-1]

        min_coeff = min(obj_row)
        if min_coeff >= -1e-10: # optimal solution found
            return None

        return np.argmin(obj_row)

    def find_leaving_variable(self, entering_var):
        """
        Returns:
            int: Index of the basic variable to leave, or None if unbounded
        """
        ratios = []

        for i in range(self.obj_row):
            if self.tableau[i, entering_var] > 1e-10:
                ratio = self.tableau[i, -1] / self.tableau[i, entering_var]
                ratios.append((ratio, i))

        if not ratios: # unbounded
            return None

        min_ratio, min_row =min(ratios)

        return self.basic_variables[min_row]

    def pivot(self, entering_var, leaving_var):
        """
        Args:
            entering_var (int): Index of entering variable
            leaving_var (int): Index of leaving variable
        """
        self.record_iteration(entering_var, leaving_var)

        leaving_row = self.basic_variables.index(leaving_var)

        self.tableau = pivot_matrix(self.tableau, entering_var, leaving_row)

        self.basic_variables[leaving_row] = entering_var
        self.non_basic_variables.remove(entering_var)
        self.non_basic_variables.append(leaving_var)

    def record_iteration(self, entering_var, leaving_var):
        self.iteration_count += 1

        iteration_data = {
            'iteration': self.iteration_count,
            'tableau': self.tableau.copy(),
            'basic_variables': self.basic_variables.copy(),
            'entering_variable': entering_var,
            'leaving_variable': leaving_var,
            'solution': self.get_solution()
        }

        self.iteration_history.append(iteration_data)

    def get_solution(self):
        solution = {var: 0 for var in range(self.tableau.shape[1] - 1)}

        for i, var in enumerate(self.basic_variables):
            solution[var] = self.tableau[i, -1].item()

        objective_value = self.tableau[self.obj_row, -1]

        return {'variables': solution, 'objective': objective_value}

    def get_iteration_history(self):
        return self.iteration_history

    def is_optimal(self):
        return all(coeff >= -1e-10 for coeff in self.tableau[self.obj_row, :-1])

    def is_unbounded(self, entering_var):
        return all(self.tableau[i, entering_var] <= 1e-10 for i in range(self.obj_row))

    def is_infeasible(self):
        if self.artificial_variables:
            solution = self.get_solution()['variables']
            return any(solution.get(var, 0) > 1e-10 for var in self.artificial_variables)

        return False

    def get_status(self):
        if self.is_optimal():
            if self.is_infeasible():
                return TableauStatus.INFEASIBLE
            return TableauStatus.OPTIMAL

        entering_var = self.find_entering_variable()
        if entering_var is not None and self.is_unbounded(entering_var):
            return TableauStatus.UNBOUNDED

        return TableauStatus.CONTINUE

if __name__ == '__main__':
    # Define the LP problem
    constraint_matrix = np.array([[2, 1], [1, 3]])
    rhs_vector = np.array([10, 15])
    objective_coefficients = [3, 5]

    # Initialize the tableau
    simplex = SimplexTableau(constraints_count=2, variables_count=2)

    # Set the objective row and constraints
    simplex.set_objective_row(objective_coefficients)
    simplex.set_constraints(constraint_matrix, rhs_vector)

    # Add slack variables
    slack_vars = simplex.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])

    # Solve the problem
    status = simplex.get_status()
    while status == TableauStatus.CONTINUE:
        entering_var = simplex.find_entering_variable()
        leaving_var = simplex.find_leaving_variable(entering_var)
        simplex.pivot(entering_var, leaving_var)
        status = simplex.get_status()

    # Get the solution
    solution = simplex.get_solution()
    print("Solution:", solution)