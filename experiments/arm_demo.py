import pychrono as chrono
import pychrono.irrlicht as chronoirr
import sys
import numpy as np
import os,csv, argparse

parser = argparse.ArgumentParser(description="A script to demonstrate inline arguments.")
parser.add_argument("arg", type=int, help="An integer argument passed from the command line.")
args = parser.parse_args()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.arm_model import LRV_Arm
from model.ik_update import RobotArmInverseKinematicsSolver
from model.grip_action import Gripper_Action_Functions
import math

system = chrono.ChSystemNSC()
system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

offset = chrono.ChVector3d(0, 0, 0)
gripper = LRV_Arm(system, offset)
ik_solver = RobotArmInverseKinematicsSolver()
### Create environment
# Create a floor
floor_material = chrono.ChContactMaterialNSC()
floor = chrono.ChBodyEasyBox(100, 100, 0.01, 1000, True, True, floor_material)
floor.SetPos(chrono.ChVector3d(0, 0, 0))
floor.SetFixed(True)
floor.GetVisualShape(0).SetColor(chrono.ChColor(0.1, 0.9, .9))
system.Add(floor)



# Create ball
contact_material = chrono.ChContactMaterialNSC()
contact_material.SetRollingFriction(0.95)
contact_material.SetFriction(0.95)
ball1 = chrono.ChBodyEasySphere(0.15, 1, True, True, contact_material)
ball1.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/blue.png"))
ball1.SetName("blue_ball")

# create random position for the ball
rand_r = np.random.uniform(1.0, 2.5)
print(rand_r)
rand_theta = np.random.uniform(0, 2 * np.pi)
rand_phi = np.random.uniform(0, np.pi/2)
# convert to cartesian
rand_x = rand_r * np.sin(rand_phi) * np.cos(rand_theta)
rand_y = rand_r * np.sin(rand_phi) * np.sin(rand_theta)
rand_z = rand_r * np.cos(rand_phi)

ball1.SetPos(chrono.ChVector3d(rand_x, rand_y, rand_z))
ball1.SetRot(chrono.Q_ROTATE_Y_TO_Z)
ball1.SetFixed(True)
ball1.EnableCollision(True)
system.Add(ball1)
gripper.add_object("blue_ball")

# initialize the gripper helper functions for task
gripper_action = Gripper_Action_Functions(gripper)


# vis = chronoirr.ChVisualSystemIrrlicht(system)
# vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
# vis.SetWindowSize(1024, 768)
# vis.SetWindowTitle("robot arm gripper")
# vis.Initialize()
# vis.AddSkyBox()
# vis.AddCamera(chrono.ChVector3d(-4.5, 0, 4.5), chrono.ChVector3d(0, 0, 0))
# Reduce the light magnitude
# vis.AddLightWithShadow(chrono.ChVector3d(10, 10, 100), chrono.ChVector3d(0, 0, -0.5), 100, 1, 9, 90, 512)

timestep = 0.001
rt_timer = chrono.ChRealtimeStepTimer()

solver = chrono.ChSolverPSOR()
# solver.SetMaxIterations(300)
# solver.SetTolerance(1e-6)
# solver.EnableWarmStart(True)
# solver.EnableDiagonalPreconditioner(True)
system.SetSolver(solver)
step_number = 0
render_step_size = 1.0 / 20  # FPS = 25
render_steps = math.ceil(render_step_size / timestep)
render_frame = 0
save_frames = False
theta = [np.random.uniform(-np.pi,np.pi),np.random.uniform(0,np.pi/2),np.random.uniform(-np.pi/2,np.pi/2),np.random.uniform(-np.pi/2,np.pi/2)]
print('random theta:', theta)
exp_ind = args.arg # experiment index

while True:
    sim_time = system.GetChTime()
    system.DoStepDynamics(timestep)

    if step_number % render_steps == 0:
        # vis.BeginScene()
        # vis.Render()
        # vis.EndScene()
        # set arm control:
        # if 0.5<sim_time<0.7:
        #     des_pos = [rand_x, rand_y, rand_z]
        #     des_orientation = 'top' if rand_z < 1.0 else 'side'
        #     iksolver = RobotArmInverseKinematicsSolver()
        #     theta = iksolver.inverse_kinematics_solver(des_pos, des_orientation)
        
        # theta = [np.random.uniform(-np.pi,np.pi),]
        if 0.7<sim_time<2.0:
            gripper.rotate_motor(gripper.motor_base_shoulder, theta[0])
            gripper.rotate_motor(gripper.motor_shoulder_biceps, theta[1])
            gripper.rotate_motor(gripper.motor_biceps_elbow, theta[2])
            gripper.rotate_motor(gripper.motor_elbow_eef, theta[3])
        if sim_time > 4.0:
            local_pt = gripper.endoffactor.TransformPointLocalToParent(chrono.ChVector3d(0, 0.0, 0.2))
            ball1.SetPos(local_pt)
            # record data
            training_data = np.savetxt(project_root+'/data/eff_pos/exp_'+str(exp_ind)+'.csv', np.array([rand_x, rand_y, rand_z,theta[0],theta[1],theta[2],theta[3]]), delimiter=',',fmt='%f')
            break

        # if save_frames:
        #     filename = project_root+'/scenarios/IMG/img_' + str(render_frame) + '.jpg'
        #     print('save frame:', filename)
        #     vis.WriteImageToFile(filename)
        #     render_frame += 1
        
    step_number += 1