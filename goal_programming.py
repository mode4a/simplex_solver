from dataclasses import dataclass
from typing import List
from enum import Enum
import numpy as np
from simplex import get_pivot_row, pivot_tableau

class InequalityType(Enum):
    GREATER_THAN_EQUAL = ">="
    LESS_THAN_EQUAL = "<="
    EQUAL = "=="


@dataclass
class Goal:
    name: str
    coefficients: List[float]
    rhs: float
    inequality_type: InequalityType
    priority: int
    pos_dev_index: int = -1
    neg_dev_index: int = -1


@dataclass
class Constraint:
    name: str
    coefficients: List[float]
    rhs: float
    inequality_type: InequalityType
    slack_variable_index: int = -1
    artificial_variable_index: int = -1


@dataclass
class PreemptiveGoalProgramming:
    def __init__(self, variables_names: List[str]):
        self.variables_names = variables_names
        self.num_variables = len(variables_names)
        self.goals: List[Goal] = []
        self.constraints: List[Constraint] = []

        self.tableau_variables = self.variables_names.copy()
        self.num_dev_variables = 0
        self.num_slack_variables = 0
        self.num_artificial_variables = 0
        # Dictionary to map variable names to their indices
        self.variable_indices = {name: idx for idx, name in enumerate(variables_names)}

    def add_goal(self, name: str, coefficients: List[float], rhs: float, inequality_type: InequalityType,
                 priority: int):
        if len(coefficients) != self.num_variables:
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) must match " f"number of variables ({self.num_variables})")
        goal = Goal(name=name, coefficients=coefficients, rhs=rhs, inequality_type=inequality_type, priority=priority)

        pos_dev_name = f"d+_{name}"
        neg_dev_name = f"d-_{name}"
        goal.pos_dev_index = len(self.tableau_variables)
        self.tableau_variables.append(pos_dev_name)
        goal.neg_dev_index = len(self.tableau_variables)
        self.tableau_variables.append(neg_dev_name)
        self.num_dev_variables += 2

        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority)

    def add_constraint(self, name: str, coefficients: List[float], rhs: float, inequality_type: InequalityType):
        if len(coefficients) != self.num_variables:
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) must match " f"number of variables ({self.num_variables})")

        constraint = Constraint(name=name, coefficients=coefficients, rhs=rhs, inequality_type=inequality_type)

        if inequality_type == InequalityType.LESS_THAN_EQUAL:
            slack_name = f"s_{name}"
            constraint.slack_index = len(self.tableau_variables)
            self.tableau_variables.append(slack_name)
            self.num_slack_variables += 1
        elif inequality_type == InequalityType.GREATER_THAN_EQUAL:
            slack_name = f"s_{name}"
            constraint.slack_index = len(self.tableau_variables)
            self.tableau_variables.append(slack_name)
            self.num_slack_variables += 1

            art_name = f"a_{name}"
            constraint.artificial_index = len(self.tableau_variables)
            self.tableau_variables.append(art_name)
            self.num_artificial_variables += 1

        self.constraints.append(Constraint(name=name, coefficients=coefficients, rhs=rhs, inequality_type=inequality_type))

    def build_tableau(self) -> np.ndarray:
        num_rows = len(self.goals) * 2 + len(self.constraints)
        num_cols = (
                self.num_variables +
                self.num_dev_variables +
                self.num_slack_variables +
                self.num_artificial_variables +
                1 # +1 for RHS
        )
        tableau = np.zeros((num_rows, num_cols))

        # map for variables indices
        var_indices = {}
        for i, var in enumerate(self.variables_names):
            var_indices[var] = i


        row_idx = 0
        # add goal rows
        for goal in self.goals:
            for j, coeff in enumerate(goal.coefficients):
                tableau[row_idx, j] = coeff
                tableau[row_idx + len(self.goals), j] = coeff

            if goal.inequality_type == InequalityType.LESS_THAN_EQUAL:
                tableau[row_idx, goal.neg_dev_index] = 1
            elif goal.inequality_type == InequalityType.GREATER_THAN_EQUAL:
                tableau[row_idx, goal.pos_dev_index] = -1
            else:
                tableau[row_idx, goal.neg_dev_index] = 1
                tableau[row_idx, goal.pos_dev_index] = -1

            tableau[row_idx + len(self.goals), goal.neg_dev_index] = 1
            tableau[row_idx + len(self.goals), goal.pos_dev_index] = -1

            tableau[row_idx, -1] = goal.rhs
            tableau[row_idx + len(self.goals), -1] = goal.rhs
            row_idx += 1

        row_idx = len(self.goals) * 2
        # add constraints rows
        for constraint in self.constraints:
            for j, coeff in enumerate(constraint.coefficients):
                tableau[row_idx, j] = coeff

            if constraint.slack_variable_index != -1:
                tableau[row_idx, constraint.slack_variable_index] = 1

            if constraint.artificial_variable_index != -1:
                tableau[row_idx, constraint.artificial_variable_index] = 1

            tableau[row_idx, -1] = constraint.rhs
            row_idx += 1

        return tableau
    
    def print_tableau(self, tableau):
        print("variable names:\n", self.tableau_variables)
        print(tableau)

    def solve(self):
        tableau = self.build_tableau()
        self.print_tableau(tableau)

        basic_variables = [-1] * (len(self.goals) * 2 + len(self.constraints))
        priority_levels = set(goal.priority for goal in self.goals)

        for priority in sorted(priority_levels):
                print(f"\n==== Solving for Priority Level {priority} ====")
                current_goals = [goal for goal in self.goals if goal.priority == priority]

                for goal in current_goals:
                    print(f"\nOptimizing Goal: {goal.name}")

                    goal_index = self.goals.index(goal)
                    objective_row_index = goal_index

                    deviation_vars_to_minimize = []

                    if goal.inequality_type == InequalityType.LESS_THAN_EQUAL:
                        deviation_vars_to_minimize.append(goal.neg_dev_index)
                    elif goal.inequality_type == InequalityType.GREATER_THAN_EQUAL:
                        deviation_vars_to_minimize.append(goal.pos_dev_index)
                    else:
                        deviation_vars_to_minimize.extend([goal.neg_dev_index, goal.pos_dev_index])

                    for dev_var_idx in deviation_vars_to_minimize:
                        while True:
                            entering_var_idx = self._find_entering_variable(tableau, objective_row_index, higher_priority_goals=self.goals[:goal_index])

                            if entering_var_idx is None:
                                print(f"No further improvement possible for {self.tableau_variables[dev_var_idx]}")
                                break

                            # ratio test
                            # Create a sub-tableau without the first goal rows and add an empty row at the end
                            sub_tableau = tableau[len(self.goals):, :]
                            # Create an empty row (filled with zeros) with the same width as the sub_tableau
                            empty_row = np.zeros((1, sub_tableau.shape[1]))
                            # Append the empty row to the sub_tableau
                            sub_tableau_with_obj = np.vstack((sub_tableau, empty_row))

                            # Now use this modified sub_tableau with the empty objective row
                            leaving_row_idx = get_pivot_row(sub_tableau_with_obj, entering_var_idx)
                            if leaving_row_idx is None:
                                print(f"Problem is unbounded for {self.tableau_variables[dev_var_idx]}")
                                return tableau, "Unbounded"
                            
                            basic_variables[leaving_row_idx] = entering_var_idx
                            print(f"Pivoting: {self.tableau_variables[entering_var_idx]} enters, row {leaving_row_idx} leaves")
                            tableau = pivot_tableau(tableau, leaving_row_idx + len(self.goals), entering_var_idx)

                            if abs(self._get_deviation_value(tableau, dev_var_idx)) < 1e-7:
                                print(f"Goal {goal.name} fully satisfied for {self.tableau_variables[dev_var_idx]}")
                                break

        solution = self._extract_solution(tableau, basic_variables)
    
        return tableau, solution, "Optimal"
    
    def _find_entering_variable(self, tableau, objective_row_idx, higher_priority_goals=None):
        objective_coeffs = tableau[objective_row_idx, :-1]
        candidates = np.argsort(-objective_coeffs)

        for candidate in candidates:
            if objective_coeffs[candidate] <= -1e-7:
                return None
            
            # check if this variable would worsen higher priority_goals
            if higher_priority_goals:
                will_worsen = False

                for goal in higher_priority_goals:
                    if tableau[self.goals.index(goal), candidate] < -1e-7:
                        will_worsen = True
                        break

                if will_worsen:
                    continue

            return candidate
        
        return None

    def _get_deviation_value(self, tableau, dev_var_idx):
        for i in range(tableau.shape[0] - 1):
            if tableau[i, dev_var_idx] == 1.0 and np.sum(tableau[i, :] != 0) == 2:
                return tableau[i, -1]
        return 0.0

    def _fix_deviation_values(self, tableau, dev_var_indices, objective_row_idx):
        for idx in dev_var_indices:
            tableau[objective_row_idx, idx] = 0.0
    
    def _extract_solution(self, tableau, basic_variables):
        solution = {}

        for var_names in self.tableau_variables:
            solution[var_names] = 0.0

        for row_idx, var_idx in enumerate(basic_variables):
            if var_idx >= 0:
                var_name = self.tableau_variables[var_idx]
                solution[var_name] = tableau[row_idx + len(self.goals), -1]

        return solutiong