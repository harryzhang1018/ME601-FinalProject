import numpy as np
from scipy.optimize import minimize
import pychrono as chrono

class RobotArmInverseKinematicsSolver:
    def __init__(self):
        """
        Initialize the robot arm inverse kinematics solver.
        """
        self.a1, self.a2, self.a3, self.a4 = 0.32516, 1.27, 1.143, 0.3052+0.2  # Link lengths
        self.orientation_weight = 0.1  # Weight for orientation error in the objective function

    # Define the forward kinematics equations for the robot arm
    def forward_kinematics(self, theta):
        theta1, theta2, theta3, theta4 = theta
        x = self.f1(theta1, theta2, theta3, theta4)
        y = self.f2(theta1, theta2, theta3, theta4)
        z = self.f3(theta1, theta2, theta3, theta4)
        position = np.array([x, y, z])
        orientation_matrix = self.calculate_orientation_matrix(theta)  # Placeholder function
        return position, orientation_matrix

    def calculate_orientation_matrix(self, theta):
        """
        Placeholder function for calculating the orientation matrix based on joint angles (theta).
        Replace this with the actual calculation based on the robot's kinematics.
        """
        theta1, theta2, theta3, theta4 = theta
        s1, s2, s3, s4 = np.sin(theta1), np.sin(theta2), np.sin(theta3), np.sin(theta4)
        c1, c2, c3, c4 = np.cos(theta1), np.cos(theta2), np.cos(theta3), np.cos(theta4)
        sigma1 = c2 * c3 * s1 - s1 * s2 * s3
        sigma2 = c2 * s1 * s3 + c3 * s1 * s2
        sigma3 = c1 * c2 * c3 - c1 * s2 * s3
        sigma4 = c1 * c2 * s3 + c1 * c3 * s2
        sigma5 = c2 * c3 - s2 * s3
        sigma6 = c2 * s3 + c3 * s2

        R = np.array([
            [c4*sigma3-s4*sigma4, -c4*sigma4-s4*sigma3, s1],
            [c4*sigma1-s4*sigma2, -c4*sigma2-s4*sigma1, -c1],
            [c4*sigma6+s4*sigma5, c4*sigma5-s4*sigma6, 0]
        ])

        return R

    # Objective function: minimize the error between current and desired positions and orientations
    def objective_function(self, theta, target_position, target_orientation):
        current_position, current_orientation = self.forward_kinematics(theta)

        # Position error
        position_error = np.linalg.norm(current_position - target_position)

        # Orientation error (Frobenius norm of the difference between rotation matrices)
        orientation_error = np.linalg.norm(current_orientation - target_orientation, ord='fro')

        # Total error
        total_error = position_error + self.orientation_weight * orientation_error
        return total_error

    # Get target orientation matrix based on 'top' or 'side' approach
    def get_target_orientation(self, approach):
        if approach == 'top':
            # Gripper Z-axis pointing down (negative base Z-axis)
            return np.array([
                [0, 0,  .1],
                [0, .1, 0],
                [-1, 0, 0]
            ])
        elif approach == 'side':
            # Gripper fingers parallel to the ground (X-axis pointing towards the object)
            return np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
        else:
            raise ValueError("Unknown target orientation: should be 'top' or 'side'.")

    # Inverse kinematics solver
    def inverse_kinematics_solver(self, target_position, target_orientation='top', tolerance=0.5):
        # Get the desired orientation matrix
        target_orientation_matrix = self.get_target_orientation(target_orientation)

        # Initial guess
        initial_guess = np.array([
            np.arctan2(target_position[1], target_position[0]),  # theta1
            np.pi / 2,                                           # theta2
            -np.pi/2,                                           # theta3
            0                                                    # theta4
        ])

        # Optimization
        result = minimize(
            self.objective_function,
            initial_guess,
            args=(target_position, target_orientation_matrix),
            method='BFGS',
            options={'gtol': tolerance, 'maxiter': 1000}
        )

        final_position, final_orientation = self.forward_kinematics(result.x)
        position_error = np.linalg.norm(final_position - target_position)
        orientation_error = np.linalg.norm(final_orientation - target_orientation_matrix, ord='fro')

        if position_error <= tolerance:
            print(f"Solution found with position error: {position_error}, orientation error: {orientation_error}")
            return result.x
        else:
            print(f"Optimization failed. Message: {result.message}")
            print(f"Final position error: {position_error}")
            print(f"Final orientation error: {orientation_error}")
            raise ValueError("Inverse kinematics solver did not converge")

    # Example kinematic equations (replace with actual equations)
    def f1(self, theta1, theta2, theta3, theta4):
        a1, a2, a3, a4 = self.a1, self.a2, self.a3, self.a4
        s1, s2, s3, s4 = np.sin(theta1), np.sin(theta2), np.sin(theta3), np.sin(theta4)
        c1, c2, c3, c4 = np.cos(theta1), np.cos(theta2), np.cos(theta3), np.cos(theta4)
        sigma3 = c1 * c2 * c3 - c1 * s2 * s3
        sigma4 = c1 * c2 * s3 + c1 * c3 * s2
        return a2 * c1 * c2 + a4 * c4 * sigma3 - a4 * s4 * sigma4 - a3 * c1 * s2 * s3 + a3 * c1 * c2 * c3

    def f2(self, theta1, theta2, theta3, theta4):
        a1, a2, a3, a4 = self.a1, self.a2, self.a3, self.a4
        s1, s2, s3, s4 = np.sin(theta1), np.sin(theta2), np.sin(theta3), np.sin(theta4)
        c1, c2, c3, c4 = np.cos(theta1), np.cos(theta2), np.cos(theta3), np.cos(theta4)
        sigma1 = c2 * c3 * s1 - s1 * s2 * s3
        sigma2 = c2 * s1 * s3 + c3 * s1 * s2
        return a2 * c2 * s1 + a4 * c4 * sigma1 - a4 * s4 * sigma2 - a3 * s1 * s2 * s3 + a3 * c2 * c3 * s1

    def f3(self, theta1, theta2, theta3, theta4):
        a1, a2, a3, a4 = self.a1, self.a2, self.a3, self.a4
        s1, s2, s3, s4 = np.sin(theta1), np.sin(theta2), np.sin(theta3), np.sin(theta4)
        c1, c2, c3, c4 = np.cos(theta1), np.cos(theta2), np.cos(theta3), np.cos(theta4)
        sigma5 = c2 * c3 - s2 * s3
        sigma6 = c2 * s3 + c3 * s2
        return a1 + a2 * s2 + a3 * c2 * s3 + a3 * c3 * s2 + a4 * c4 * sigma6 + a4 * s4 * sigma5


if __name__ == '__main__':
    # Desired position
    desired_position = np.array([-0.53156, 1.208745, 0.25])

    # Solver instance
    solver = RobotArmInverseKinematicsSolver()

    # Approach can be 'top' or 'side'
    approach = 'top'

    # Use solver to figure out control angles
    import time
    start_time = time.time()
    ball_pos = chrono.ChVector3d(-0.53156, 1.208745, 1.19)
    offset = chrono.ChVector3d(0, 0, 1.06)
    desired_pos = np.array([ball_pos.x - offset.x, ball_pos.y - offset.y, ball_pos.z - offset.z + 0.15])

    try:
        final_theta = solver.inverse_kinematics_solver(desired_pos, target_orientation=approach)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        # Output the control angles
        print(f"Control angles (in radians): {final_theta}")

        # Compute the final end-effector position and orientation using the optimized angles
        final_position, final_orientation = solver.forward_kinematics(final_theta)
        print(f"Final end-effector position: x = {final_position[0]}, y = {final_position[1]}, z = {final_position[2]}")
        print(f"Final end-effector orientation matrix:\n{final_orientation}")
    except ValueError as e:
        print(str(e))
