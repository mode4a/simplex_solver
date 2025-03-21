import sys
import os
import unittest
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simplex_tableau import SimplexTableau, TableauStatus
from core.lp_problem import ConstraintType

class TestSimplexTableau(unittest.TestCase):
    def setUp(self):
        self.tableau = SimplexTableau(constraints_count=2, variables_count=2)

        self.tableau.set_objective_row([3, 5])

        constraint_matrix = np.array([[2, 1], [1, 3]])
        rhs_vector = np.array([10, 15])
        self.tableau.set_constraints(constraint_matrix, rhs_vector)

    def test_initialization(self):
        tableau = SimplexTableau(constraints_count=2, variables_count=2)
        self.assertEqual(tableau.tableau.shape, (3, 3))  # 2 constraints + 1 objective, 2 vars + 1 RHS
        self.assertEqual(tableau.obj_row, 2)
        self.assertEqual(tableau.original_variables, 2)
        self.assertEqual(tableau.iteration_count, 0)

    def test_set_objective_row(self):
        # Check if objective function is correctly set
        np.testing.assert_array_almost_equal(
            self.tableau.tableau[self.tableau.obj_row, :2],
            np.array([-3, -5])
        )

    def test_set_constraints(self):
        # Check if constraints are correctly set
        np.testing.assert_array_almost_equal(
            self.tableau.tableau[:2, :2],
            np.array([[2, 1], [1, 3]])
        )

        # Check if RHS is correctly set
        np.testing.assert_array_almost_equal(
            self.tableau.tableau[:2, -1],
            np.array([10, 15])
        )

    def test_add_slack_variables(self):
        # Add slack variables for <= constraints
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])

        # Check if slack variables are added correctly
        self.assertEqual(slack_vars, [2, 3])
        self.assertEqual(self.tableau.tableau.shape, (3, 5))  # Now with 2 slack variables

        # Check if identity matrix for slack variables is correct
        np.testing.assert_array_almost_equal(
            self.tableau.tableau[:2, 2:4],
            np.array([[1, 0], [0, 1]])
        )

        # Check if slack variables are in basic variables
        self.assertEqual(self.tableau.basic_variables, [2, 3])

    def test_add_surplus_variables(self):
        # Add surplus variables for >= constraints
        surplus_vars = self.tableau.add_surplus_variables([ConstraintType.GREATER_EQUAL, ConstraintType.GREATER_EQUAL])

        # Check if surplus variables are added correctly
        self.assertEqual(surplus_vars, [2, 3])
        self.assertEqual(self.tableau.tableau.shape, (3, 5))  # Now with 2 surplus variables

        # Check if -identity matrix for surplus variables is correct
        np.testing.assert_array_almost_equal(
            self.tableau.tableau[:2, 2:4],
            np.array([[-1, 0], [0, -1]])
        )

        # Check if surplus variables are in non-basic variables
        self.assertEqual(self.tableau.non_basic_variables, [2, 3])

    def test_add_artificial_variables(self):
        # Add artificial variables for >= and = constraints
        artificial_vars = self.tableau.add_artificial_variables([ConstraintType.GREATER_EQUAL, ConstraintType.EQUAL])

        # Check if artificial variables are added correctly
        self.assertEqual(artificial_vars, [2, 3])
        self.assertEqual(self.tableau.tableau.shape, (3, 5))  # Now with 2 artificial variables

        # Check if identity matrix for artificial variables is correct
        np.testing.assert_array_almost_equal(
            self.tableau.tableau[:2, 2:4],
            np.array([[1, 0], [0, 1]])
        )

        # Check if artificial variables are tracked correctly
        self.assertEqual(self.tableau.artificial_variables, [2, 3])
        self.assertEqual(self.tableau.basic_variables, [2, 3])

    def test_apply_big_m_penalty(self):
        # Add artificial variables
        artificial_vars = self.tableau.add_artificial_variables([ConstraintType.GREATER_EQUAL, ConstraintType.EQUAL])

        # Apply Big-M penalty
        self.tableau.apply_big_m_penalty(artificial_vars, M=1000)

        # Check if objective row has been updated for constraints with artificial variables
        self.assertLess(self.tableau.tableau[self.tableau.obj_row, 0], 0)  # Should be -3 - 1000*1
        self.assertLess(self.tableau.tableau[self.tableau.obj_row, 1], 0)  # Should be -5 - 1000*3

    def test_find_entering_variable(self):
        # Setup a tableau with slack variables
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])

        # Find entering variable
        entering_var = self.tableau.find_entering_variable()

        # Should be variable 1 (index 1) because it has coefficient -5 in objective row
        self.assertEqual(entering_var, 1)

        # Modify tableau to have all non-negative coefficients in objective row
        self.tableau.tableau[self.tableau.obj_row, :] = np.array([0.1, 0.2, 0, 0, 0])

        # Find entering variable again
        entering_var = self.tableau.find_entering_variable()

        # Should be None because all coefficients are non-negative
        self.assertIsNone(entering_var)

    def test_find_leaving_variable(self):
        # Setup a tableau with slack variables
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])
        self.tableau.non_basic_variables = [0, 1]  # x1, x2

        # Find entering variable
        entering_var = 1  # x2

        # Find leaving variable
        leaving_var = self.tableau.find_leaving_variable(entering_var)

        # Calculate expected leaving variable by hand
        # For entering_var = 1 (x2), we have:
        # constraint 1: 10/1 = 10
        # constraint 2: 15/3 = 5  <- minimum ratio
        # So leaving_var should be the basic variable at row 1 (index 3)
        self.assertEqual(leaving_var, 3)

        # Test for unbounded case
        # Set the column for entering variable to all negative or zero
        self.tableau.tableau[:self.tableau.obj_row, entering_var] = np.array([-1, -2])

        # Find leaving variable
        leaving_var = self.tableau.find_leaving_variable(entering_var)

        # Should be None because problem is unbounded
        self.assertIsNone(leaving_var)

    def test_pivot(self):
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])
        self.tableau.non_basic_variables = [0, 1]  # x1, x2

        # Save the original tableau
        original_tableau = self.tableau.tableau.copy()

        # Perform a pivot operation
        self.tableau.pivot(entering_var=1, leaving_var=3)

        # Check if the pivot element is now 1
        self.assertEqual(self.tableau.tableau[1, 1], 1)

        # Check if the rest of the column is now 0
        self.assertEqual(self.tableau.tableau[0, 1], 0)
        self.assertEqual(self.tableau.tableau[2, 1], 0)

        # Check if basic and non-basic variables are updated
        self.assertEqual(self.tableau.basic_variables, [2, 1])
        self.assertEqual(self.tableau.non_basic_variables, [0, 3])

        # Check if iteration count and history are updated
        self.assertEqual(self.tableau.iteration_count, 1)
        self.assertEqual(len(self.tableau.iteration_history), 1)

    def test_get_solution(self):
        # Setup a tableau with slack variables
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])
        self.tableau.non_basic_variables = [0, 1]  # x1, x2

        # Get initial solution (before any pivoting)
        solution = self.tableau.get_solution()

        # Should have x1=0, x2=0, slack1=10, slack2=15
        self.assertEqual(solution['variables'][0], 0)
        self.assertEqual(solution['variables'][1], 0)
        self.assertEqual(solution['variables'][2], 10)
        self.assertEqual(solution['variables'][3], 15)

        # Objective value should be 0
        self.assertEqual(solution['objective'], 0)

        # Now pivot and get new solution
        self.tableau.pivot(entering_var=1, leaving_var=3)
        solution = self.tableau.get_solution()

        # Should have x1=0, x2=5, slack1=5, slack2=0
        self.assertEqual(solution['variables'][0], 0)
        self.assertEqual(solution['variables'][1], 5)
        self.assertEqual(solution['variables'][2], 5)
        self.assertEqual(solution['variables'][3], 0)

        # Objective value should be 25 (5 * 5)
        self.assertAlmostEqual(solution['objective'], 25)

    def test_is_optimal(self):
        # Setup a tableau with slack variables
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])

        # Initially, the tableau is not optimal
        self.assertFalse(self.tableau.is_optimal())

        # Make all coefficients in objective row positive
        self.tableau.tableau[self.tableau.obj_row, :] = np.array([0.1, 0.2, 0, 0, 0])

        # Now the tableau should be optimal
        self.assertTrue(self.tableau.is_optimal())

    def test_is_unbounded(self):
        # Setup a tableau with slack variables
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])

        # Initially, the problem is not unbounded for variable 1
        self.assertFalse(self.tableau.is_unbounded(1))

        # Set the column for variable 1 to all negative or zero
        self.tableau.tableau[:self.tableau.obj_row, 1] = np.array([-1, -2])

        # Now the problem should be unbounded for variable 1
        self.assertTrue(self.tableau.is_unbounded(1))

    def test_is_infeasible(self):
        # Add artificial variables
        artificial_vars = self.tableau.add_artificial_variables([ConstraintType.GREATER_EQUAL, ConstraintType.EQUAL])
        self.tableau.non_basic_variables = [0, 1]  # x1, x2

        # Initially, the problem appears feasible (artificial vars in basic solution)
        # However, this is not a valid test for infeasibility until after solving
        solution = self.tableau.get_solution()
        self.assertTrue(self.tableau.is_infeasible())

        # To truly test infeasibility, we would need to solve the problem first
        # This is just testing the mechanism of checking if artificial variables are positive

    def test_get_status(self):
        # Setup a tableau with slack variables
        slack_vars = self.tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])
        self.tableau.non_basic_variables = [0, 1]  # x1, x2

        # Initially, the status should be CONTINUE
        self.assertEqual(self.tableau.get_status(), TableauStatus.CONTINUE)

        # Make all coefficients in objective row non-negative
        self.tableau.tableau[self.tableau.obj_row, :] = np.array([0.1, 0.2, 0, 0, 0])

        # Now the status should be OPTIMAL
        self.assertEqual(self.tableau.get_status(), TableauStatus.OPTIMAL)

        # Set up an unbounded case
        self.tableau.tableau[self.tableau.obj_row, 1] = -1  # Make x2 entering variable
        self.tableau.tableau[:self.tableau.obj_row, 1] = np.array([-1, -2])  # Make problem unbounded

        # Now the status should be UNBOUNDED
        self.assertEqual(self.tableau.get_status(), TableauStatus.UNBOUNDED)

    def test_full_solution_cycle(self):
        # This test demonstrates a full solution cycle for a simple LP problem

        # Initialize the tableau
        tableau = SimplexTableau(constraints_count=2, variables_count=2)

        # Set the objective function: maximize 3x1 + 5x2
        tableau.set_objective_row([3, 5])

        # Set the constraints:
        # 2x1 + x2 <= 10
        # x1 + 3x2 <= 15
        constraint_matrix = np.array([[2, 1], [1, 3]])
        rhs_vector = np.array([10, 15])
        tableau.set_constraints(constraint_matrix, rhs_vector)

        # Add slack variables
        slack_vars = tableau.add_slack_variables([ConstraintType.LESS_EQUAL, ConstraintType.LESS_EQUAL])
        tableau.non_basic_variables = [0, 1]  # x1, x2

        # Initial status should be CONTINUE
        status = tableau.get_status()
        self.assertEqual(status, TableauStatus.CONTINUE)

        # First iteration
        entering_var = tableau.find_entering_variable()
        self.assertEqual(entering_var, 1)  # x2 should enter

        leaving_var = tableau.find_leaving_variable(entering_var)
        self.assertEqual(leaving_var, 3)  # slack2 should leave

        tableau.pivot(entering_var, leaving_var)

        # Second iteration
        entering_var = tableau.find_entering_variable()
        self.assertEqual(entering_var, 0)  # x1 should enter

        leaving_var = tableau.find_leaving_variable(entering_var)
        self.assertEqual(leaving_var, 2)  # slack1 should leave

        tableau.pivot(entering_var, leaving_var)

        # Now should be optimal
        status = tableau.get_status()
        self