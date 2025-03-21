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

        # Check if objective row has M for artificial variables
        self.assertEqual(self.tableau.tableau[self.tableau.obj_row, 2], 1000)
        self.assertEqual(self.tableau.tableau[self.tableau.obj_row, 3], 1000)

        # Check if objective row has been updated for constraints with artificial variables
        self.assertLess(self.tableau.tableau[self.tableau.obj_row, 0], 0)  # Should be -3 - 1000*1
        self.assertLess(self.tableau.tableau[self.tableau.obj_row, 1], 0)  # Should be -5 - 1000*3
