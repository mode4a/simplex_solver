import numpy as np
from goal_programming import PreemptiveGoalProgramming, InequalityType

def test_goal_programming():
    # Create a goal programming instance with two variables
    gp = PreemptiveGoalProgramming(['x1', 'x2'])

    gp.add_goal('1', [200, 0], 1000, InequalityType.GREATER_THAN_EQUAL, priority=1)
    gp.add_goal('2', [100, 400], 1200, InequalityType.GREATER_THAN_EQUAL, priority=2)
    gp.add_goal('3', [0, 250], 800, InequalityType.GREATER_THAN_EQUAL, priority=3)
    gp.add_constraint('4', [1500, 3000], 15000, InequalityType.LESS_THAN_EQUAL)

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