import numpy as np
from scipy.optimize import minimize
import pychrono as chrono

class RobotArmInverseKinematicsSolver:
    def __init__(self):
        """
        Initialize the robot arm inverse kinematics solver.

        """
        self.a1, self.a2, self.a3 , self.a4 = 0.32516, 1.27, 1.143, 0.3052
    # Define the forward kinematics equations for the robot arm
    def forward_kinematics(self, theta):
        theta1, theta2, theta3, theta4 = theta
        # Replace these with the actual kinematic equations
        x = self.f1(theta1, theta2, theta3, theta4)
        y = self.f2(theta1, theta2, theta3, theta4)
        z = self.f3(theta1, theta2, theta3, theta4)
        return np.array([x, y, z])

    # Objective function: minimize the error between current and desired positions
    def objective_function(self, theta, target_position):
        # print('theta:', theta)
        current_position = self.forward_kinematics(theta)
        # print('current_position:', current_position)
        error = np.linalg.norm(current_position - target_position)  # Euclidean distance
        return error

    # Example kinematic equations (replace with actual equations)
    def f1(self, theta1, theta2, theta3,theta4):
        a1, a2, a3, a4 = self.a1, self.a2, self.a3, self.a4
        s1,s2,s3,s4 = np.sin(theta1),np.sin(theta2),np.sin(theta3),np.sin(theta4)
        c1,c2,c3,c4 = np.cos(theta1),np.cos(theta2),np.cos(theta3),np.cos(theta4)
        # define helper parameters
        sigma3 = c1*c2*c3-c1*s2*s3
        sigma4 = c1*c2*s3+c1*c3*s2
        return a2*c1*c2+a4*c4*sigma3-a4*s4*sigma4-a3*c1*s2*s3+a3*c1*c2*c3

    def f2(self, theta1, theta2, theta3,theta4):
        a1, a2, a3, a4 = self.a1, self.a2, self.a3, self.a4
        s1,s2,s3,s4 = np.sin(theta1),np.sin(theta2),np.sin(theta3),np.sin(theta4)
        c1,c2,c3,c4 = np.cos(theta1),np.cos(theta2),np.cos(theta3),np.cos(theta4)
        sigma1 = c2*c3*s1-s1*s2*s3
        sigma2 = c2*s1*s3+c3*s1*s2
        return a2*c2*s1+a4*c4*sigma1-a4*s4*sigma2-a3*s1*s2*s3+a3*c2*c3*s1

    def f3(self, theta1, theta2, theta3, theta4):
        a1, a2, a3, a4 = self.a1, self.a2, self.a3, self.a4
        s1,s2,s3,s4 = np.sin(theta1),np.sin(theta2),np.sin(theta3),np.sin(theta4)
        c1,c2,c3,c4 = np.cos(theta1),np.cos(theta2),np.cos(theta3),np.cos(theta4)
        sigma5 = c2*c3-s2*s3
        sigma6 = c2*s3+c3*s2
        return a1+a2*s2+a3*c2*s3+a3*c3*s2+a4*c4*sigma6+a4*s4*sigma5

    # Inverse kinematics solver
    def inverse_kinematics_solver(self, target_position, tolerance=1e-3):
        initial_guess = np.array([np.arctan2(target_position[1],target_position[0]), np.pi/2, -np.pi/2, -np.pi/2])
        result = minimize(self.objective_function, initial_guess, args=(target_position,),
                        method='BFGS', options={'gtol': 1e-3, 'maxiter': 1000})
        
        final_position = self.forward_kinematics(result.x)  # Assuming you have this function
        error = np.linalg.norm(final_position - target_position)
        
        if error <= tolerance:
            print(f"Solution found with error: {error}")
            return result.x
        else:
            print(f"Optimization failed. Message: {result.message}")
            print(f"Final position error: {error}")
            print(f"Number of iterations: {result.nit}")
            print(f"Final position: {final_position}")
            print(f"Target position: {target_position}")
            raise ValueError("Inverse kinematics solver did not converge")

if __name__ == '__main__':
    # Desired position (replace with actual values)
    desired_position = np.array( [ 1, 1, 0.5 ])

    # Initial guess for theta1, theta2, theta3
    initial_guess = np.array([np.arctan2(desired_position[1],desired_position[0]), np.pi/2, 0, 0])

    solver = RobotArmInverseKinematicsSolver()
    # Use solver to figure out control angles
    import time
    start_time = time.time()
    ball_pos = chrono.ChVector3d(0.03, 2, 3)
    offset = chrono.ChVector3d(0, 0, 1.06)
    desired_pos = [ball_pos.x-offset.x, ball_pos.y-offset.y, ball_pos.z-offset.z+0.15]
    final_theta = solver.inverse_kinematics_solver(desired_pos)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    # Output the control angles
    print(f"Control angles (in radians): {final_theta}")

    # Compute the final end-effector position using the optimized angles
    final_position = solver.forward_kinematics(final_theta)
    print(f"Final end-effector position: x = {final_position[0]}, y = {final_position[1]}, z = {final_position[2]}")