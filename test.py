import numpy as np
from goal_programming import PreemptiveGoalProgramming, InequalityType

def test_goal_programming():
    # Create a goal programming instance with two variables
    gp = PreemptiveGoalProgramming(['x1', 'x2'])

    # Add goals with priorities
    # Goal 1 (highest priority): 7x1 + 3x2 >= 40
    gp.add_goal('1', [7, 3], 40, InequalityType.GREATER_THAN_EQUAL, priority=1)

    # Goal 2: 10x1 + 5x2 >= 60
    gp.add_goal('2', [10, 5], 60, InequalityType.GREATER_THAN_EQUAL, priority=2)

    # Goal 3 (lowest priority): 5x1 + 4x2 >= 35
    gp.add_goal('3', [5, 4], 35, InequalityType.GREATER_THAN_EQUAL, priority=3)

    # Add constraint: 100x1 + 60x2 <= 600
    gp.add_constraint('4', [100, 60], 600, InequalityType.LESS_THAN_EQUAL)

    # Build and print the tableau
    tableau = gp.build_tableau()
    tableau, solution, status = gp.solve()
    print(status)
    print(solution)
    print(tableau)
    # Optional: Return the tableau for further analysis
    return tableau

# Run the test
if __name__ == '__main__':
    test_goal_programming()