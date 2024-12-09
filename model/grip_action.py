import pychrono as chrono
import pychrono.irrlicht as chronoirr
import sys
import numpy as np
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.ik_update import RobotArmInverseKinematicsSolver

class Gripper_Action_Functions():
    def __init__(self, gripper):
        self.gripper = gripper
        self.start_time = 0
        self.final_theta = None

        # Initialize the gripper into a reasonable configuration for the task
        self.gripper.rotate_motor(self.gripper.motor_base_shoulder, -np.pi)
        self.gripper.rotate_motor(self.gripper.motor_shoulder_biceps, np.pi / 4)
        self.gripper.rotate_motor(self.gripper.motor_biceps_elbow, -np.pi / 4)
        self.gripper.rotate_motor(self.gripper.motor_elbow_eef, 0)

    def trajectory_generation(self, desired_position, desired_orientation='top'):
        """
        Calculate the trajectory for the gripper to follow based on the desired position and orientation.
        """
        offset = chrono.ChVector3d(0, 0, 0.0)  # Adjusted offset
        desired_pos = [
            desired_position.x - offset.x,
            desired_position.y - offset.y,
            desired_position.z - offset.z,
        ]
        iksolver = RobotArmInverseKinematicsSolver()
        theta = iksolver.inverse_kinematics_solver(desired_pos, desired_orientation)
        return theta

    def gripper_pick(self, sim_time, target, approach_orientation='top', time=None):
        """
        Perform picking actions within a specified time horizon.
        Parameters:
        - sim_time: Current simulation time.
        - target: Target object to pick.
        - approach_orientation: The orientation for approaching the object ('top' or 'side').
        - time: A list [start_time, end_time] specifying the time horizon for this action.
        """
        if time is None or len(time) != 2:
            raise ValueError("The 'time' parameter must be a list of [start_time, end_time].")

        start_time, end_time = time

        # Perform actions only within the specified time horizon
        if start_time <= sim_time < end_time:
            action_duration = end_time - start_time
            normalized_time = (sim_time - start_time) / action_duration

            if normalized_time < 0.1:
                print('step 1')
                # Step 1: Open the gripper and Calculate trajectory and
                self.gripper.open()
                ball_pos = target.GetPos()
                desired_pos = chrono.ChVector3d(ball_pos.x, ball_pos.y, ball_pos.z)
                if self.final_theta is None:
                    self.final_theta = self.trajectory_generation(desired_pos, approach_orientation)
            elif normalized_time < 0.3:
                print('step 2')
                # Step 2:  move base and shoulder
                self.gripper.rotate_motor(self.gripper.motor_base_shoulder, self.final_theta[0])
                self.gripper.rotate_motor(self.gripper.motor_shoulder_biceps, self.final_theta[1])
                if approach_orientation == 'side':
                    self.gripper.rotate_motor(self.gripper.motor_biceps_elbow, self.final_theta[2])
                    self.gripper.rotate_motor(self.gripper.motor_elbow_eef, self.final_theta[3])
            elif normalized_time < 0.5:
                print('step 3')
                # Step 3: Move biceps and elbow to position
                self.gripper.rotate_motor(self.gripper.motor_biceps_elbow, self.final_theta[2])
                self.gripper.rotate_motor(self.gripper.motor_elbow_eef, self.final_theta[3])
            elif normalized_time < 0.7:
                print('step 4')
                # Step 4: Grab the object
                self.gripper.grab_object()
            elif normalized_time < 0.95:
                print('step 5')
                # Step 5: Reposition the gripper
                self.gripper.rotate_motor(self.gripper.motor_base_shoulder, -np.pi)
                self.gripper.rotate_motor(self.gripper.motor_shoulder_biceps, np.pi / 4)
                self.gripper.rotate_motor(self.gripper.motor_biceps_elbow, -np.pi / 4)
                self.gripper.rotate_motor(self.gripper.motor_elbow_eef, 0)
            else:
                print('step 6 -- reset')
                # Beyond the time horizon, ensure gripper is open and reset for next action
                self.gripper.open()
                self.final_theta = None  # Reset the final_theta for next action

