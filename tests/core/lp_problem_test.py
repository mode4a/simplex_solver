import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lp_problem import (
    LPProblem,
    VariableType,
    ConstraintType,
    ObjectiveSense
)

class TestLPProblem(unittest.TestCase):
    def setUp(self):
        self.problem = LPProblem("Test Problem")

    def test_initialize_problem(self):
        self.assertEqual(self.problem.name, "Test Problem")
        self.assertEqual(self.problem.objective_sense, ObjectiveSense.MAXIMIZE)
        self.assertEqual(len(self.problem.objective_coefficients), 0)
        self.assertEqual(len(self.problem.constraint_coefficients), 0)
        self.assertEqual(len(self.problem.constraint_types), 0)
        self.assertEqual(len(self.problem.rhs_values), 0)

    def test_add_variable(self):
        # Add a variable and check if it's properly added
        var_idx = self.problem.add_variable("x1", 5.0)
        self.assertEqual(var_idx, 0)
        self.assertEqual(len(self.problem.variable_names), 1)
        self.assertEqual(self.problem.variable_names[0], "x1")
        self.assertEqual(self.problem.objective_coefficients[0], 5.0)
        self.assertEqual(self.problem.variable_types[0], VariableType.NON_NEGATIVE)

        # Add a second variable with different type
        var_idx = self.problem.add_variable("x2", 3.0, VariableType.UNRESTRICTED)
        self.assertEqual(var_idx, 1)
        self.assertEqual(len(self.problem.variable_names), 2)
        self.assertEqual(self.problem.variable_names[1], "x2")
        self.assertEqual(self.problem.objective_coefficients[1], 3.0)
        self.assertEqual(self.problem.variable_types[1], VariableType.UNRESTRICTED)

    def test_add_constraint(self):
        # First add some variables
        self.problem.add_variable("x1", 5.0)
        self.problem.add_variable("x2", 3.0)

        # Now add a constraint
        constraint_idx = self.problem.add_constraint(
            [2.0, 1.0],
            ConstraintType.LESS_EQUAL,
            10.0
        )

        self.assertEqual(constraint_idx, 0)
        self.assertEqual(len(self.problem.constraint_coefficients), 1)
        self.assertEqual(self.problem.constraint_coefficients[0], [2.0, 1.0])
        self.assertEqual(self.problem.constraint_types[0], ConstraintType.LESS_EQUAL)
        self.assertEqual(self.problem.rhs_values[0], 10.0)

        # Add a second constraint
        constraint_idx = self.problem.add_constraint(
            [1.0, 3.0],
            ConstraintType.GREATER_EQUAL,
            15.0
        )

        self.assertEqual(constraint_idx, 1)
        self.assertEqual(len(self.problem.constraint_coefficients), 2)
        self.assertEqual(self.problem.constraint_coefficients[1], [1.0, 3.0])
        self.assertEqual(self.problem.constraint_types[1], ConstraintType.GREATER_EQUAL)
        self.assertEqual(self.problem.rhs_values[1], 15.0)

    def test_constraint_variable_mismatch(self):
        # Add one variable
        self.problem.add_variable("x1", 5.0)

        # Attempt to add a constraint with two coefficients
        with self.assertRaises(ValueError):
            self.problem.add_constraint(
                [2.0, 1.0],
                ConstraintType.LESS_EQUAL,
                10.0
            )

    def test_set_objective_sense(self):
        self.assertEqual(self.problem.objective_sense, ObjectiveSense.MAXIMIZE)

        self.problem.set_objective_sense(ObjectiveSense.MINIMIZE)
        self.assertEqual(self.problem.objective_sense, ObjectiveSense.MINIMIZE)

    def test_get_matrix_and_vectors(self):
        # Setup a simple problem
        self.problem.add_variable("x1", 5.0)
        self.problem.add_variable("x2", 3.0)

        self.problem.add_constraint(
            [2.0, 1.0],
            ConstraintType.LESS_EQUAL,
            10.0
        )

        self.problem.add_constraint(
            [1.0, 3.0],
            ConstraintType.GREATER_EQUAL,
            15.0
        )

        # Test get_constraint_matrix
        constraint_matrix = self.problem.get_constraint_matrix()
        self.assertIsInstance(constraint_matrix, np.ndarray)
        np.testing.assert_array_equal(constraint_matrix, np.array([[2.0, 1.0], [1.0, 3.0]]))

        # Test get_rhs_vector
        rhs_vector = self.problem.get_rhs_vector()
        self.assertIsInstance(rhs_vector, np.ndarray)
        np.testing.assert_array_equal(rhs_vector, np.array([10.0, 15.0]))

        # Test get_objective_vector
        objective_vector = self.problem.get_objective_vector()
        self.assertIsInstance(objective_vector, np.ndarray)
        np.testing.assert_array_equal(objective_vector, np.array([5.0, 3.0]))

    def test_add_goal(self):
        goal_idx = self.problem.add_goal(100.0, 1)
        self.assertEqual(goal_idx, 0)
        self.assertEqual(len(self.problem.goal_priorities), 1)
        self.assertEqual(len(self.problem.goal_values), 1)
        self.assertEqual(self.problem.goal_priorities[0], 1)
        self.assertEqual(self.problem.goal_values[0], 100.0)

        # Add another goal with different priority
        goal_idx = self.problem.add_goal(50.0, 2)
        self.assertEqual(goal_idx, 1)
        self.assertEqual(len(self.problem.goal_priorities), 2)
        self.assertEqual(len(self.problem.goal_values), 2)
        self.assertEqual(self.problem.goal_priorities[1], 2)
        self.assertEqual(self.problem.goal_values[1], 50.0)


if __name__ == '__main__':
    unittest.main()