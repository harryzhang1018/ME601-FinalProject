import pychrono as chrono
import pychrono.irrlicht as chronoirr
import sys
import numpy as np
import os,time

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

offset = chrono.ChVector3d(0, 0, 1.5)
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
# ball1 = chrono.ChBodyEasySphere(0.15, 1, True, True, contact_material)
ball1 = chrono.ChBodyEasyBox(0.3, 0.5, 0.3, 100, True, True, contact_material)
ball1.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/blue.png"))
ball1.SetName("blue_ball")
rand_x = np.random.uniform(-1, 1)
seed = np.random.random()
if seed>0.5:
    rand_y = np.random.uniform(-2.5, -1.5)
else:
    rand_y = np.random.uniform(1.5, 2.5)
print('!!random position:', rand_x, rand_y)
ball1.SetPos(chrono.ChVector3d(rand_x, rand_y, 3))
ball1.SetRot(chrono.Q_ROTATE_Y_TO_Z)
ball1.SetFixed(False)
ball1.EnableCollision(True)
system.Add(ball1)
gripper.add_object("blue_ball")

# initialize the gripper helper functions for task
gripper_action = Gripper_Action_Functions(gripper)


vis = chronoirr.ChVisualSystemIrrlicht(system)
vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
vis.SetWindowSize(1024, 768)
vis.SetWindowTitle("robot arm gripper")
vis.Initialize()
vis.AddSkyBox()
vis.AddCamera(chrono.ChVector3d(-4.5, 0, 4.5), chrono.ChVector3d(0, 0, 0))
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
while vis.Run():
    sim_time = system.GetChTime()
    system.DoStepDynamics(timestep)

    
    # gripper.rotate_motor(gripper.motor_shoulder_biceps, 1.57)
    if step_number % render_steps == 0:
        vis.BeginScene()
        vis.Render()
        vis.EndScene()

        # First gripper action from time 3 to 13  
        gripper_action.gripper_pick(sim_time, ball1, approach_orientation='side', time=[3, 13])

        # # Second gripper action from time 12 to 19
        gripper_action.gripper_pick(sim_time, ball1, approach_orientation='top', time=[17, 27])


        if save_frames:
            filename = project_root+'/scenarios/IMG/img_' + str(render_frame) + '.jpg'
            print('save frame:', filename)
            vis.WriteImageToFile(filename)
            render_frame += 1
            

        
    rt_timer.Spin(timestep)
    step_number += 1