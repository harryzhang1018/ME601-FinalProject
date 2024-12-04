import pychrono as chrono
import pychrono.irrlicht as chronoirr
import math
import os
import time


class LRV_Arm:
    def __init__(self, system, pos,attached_vehicle=None):
        self.system = system
        self._set_data_dir()
        self._initialize(pos,attached_vehicle)
        
        # self.rotate_motor(self.motor_base_shoulder, 0)
        # self.rotate_motor(self.motor_shoulder_biceps, 0)
        # self.rotate_motor(self.motor_biceps_elbow, 0)
        # self.rotate_motor(self.motor_elbow_eef, 0)
        self.move_linear_motor(self.motor_endoffactor_finger_1, -0.15)
        self.move_linear_motor(self.motor_endoffactor_finger_2, 0.15)
        
        # self._setup_locks()

    def _set_data_dir(self):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the data folder
        self.data_dir = os.path.join(script_dir, '..', 'data/lrv_robotarm')
        # Normalize the path
        self.data_dir = os.path.normpath(self.data_dir)

    def _initialize(self, pos, attached_vehicle):
        filepath = os.path.join(self.data_dir, 'lrv_arm.py')
        imported_items = chrono.ImportSolidWorksSystem(filepath)
        for ii in imported_items:
            self.system.Add(ii)

        # name each of assembly items
        self.base = self.system.SearchBody("base-1")
        self.biceps = self.system.SearchBody("bicep-1")
        self.elbow = self.system.SearchBody("elbow-1")
        self.wrist = self.system.SearchBody("wrist-1")
        self.shoulder = self.system.SearchBody("shoulder-1")
        self.endoffactor = self.system.SearchBody("endeffector-1")  # ----
        self.finger_1 = self.system.SearchBody("finger-1")
        self.finger_2 = self.system.SearchBody("finger-2")

        # create name for each marker
        self.joint_base_shoulder = self.system.SearchMarker("joint_shoulder_base")
        self.joint_shoulder_biceps = self.system.SearchMarker("joint_bicep_shoulder")
        self.joint_biceps_elbow = self.system.SearchMarker("joint_elbow_bicep")
        self.joint_elbow_eff = self.system.SearchMarker("joint_wrist_elbow")
        self.joint_endoffactor = self.system.SearchMarker("joint_eff")

        # Adding lock link between endoffactor and wrist
        self.lock = chrono.ChLinkLockLock()
        self.lock.Initialize(self.endoffactor, self.wrist, chrono.ChFramed(self.joint_endoffactor.GetPos(), self.joint_endoffactor.GetRot()))
        self.system.Add(self.lock)
        # Adding motors to the marker place
        self.motor_base_shoulder = chrono.ChLinkMotorRotationAngle()
        self.motor_base_shoulder.Initialize(self.base, self.shoulder, chrono.ChFramed(self.joint_base_shoulder.GetPos(), self.joint_base_shoulder.GetRot()))
        self.system.Add(self.motor_base_shoulder)

        self.motor_shoulder_biceps = chrono.ChLinkMotorRotationAngle()
        frame = chrono.ChFramed(self.joint_shoulder_biceps.GetPos(), self.joint_shoulder_biceps.GetRot())
        self.motor_shoulder_biceps.Initialize(self.shoulder, self.biceps, frame)
        self.system.Add(self.motor_shoulder_biceps)

        self.motor_biceps_elbow = chrono.ChLinkMotorRotationAngle()
        self.motor_biceps_elbow.Initialize(self.biceps, self.elbow, chrono.ChFramed(self.joint_biceps_elbow.GetPos(), self.joint_biceps_elbow.GetRot()))
        self.system.Add(self.motor_biceps_elbow)

        self.motor_elbow_eef = chrono.ChLinkMotorRotationAngle()
        self.motor_elbow_eef.Initialize(self.elbow, self.endoffactor, chrono.ChFramed(self.joint_elbow_eff.GetPos(), self.joint_elbow_eff.GetRot()))
        self.system.Add(self.motor_elbow_eef)

        # self.motor_wrist_endoffactor = chrono.ChLinkMotorRotationAngle()
        # self.motor_wrist_endoffactor.Initialize(self.wrist, self.endoffactor, chrono.ChFramed(self.joint_wrist_endoffactor.GetPos(), self.joint_wrist_endoffactor.GetRot()))
        # self.system.Add(self.motor_wrist_endoffactor)
        
        self.motor_endoffactor_finger_1 = chrono.ChLinkMotorLinearPosition()
        self.motor_endoffactor_finger_1.Initialize(self.endoffactor, self.finger_1, chrono.ChFramed(self.joint_endoffactor.GetPos(), self.joint_endoffactor.GetRot()))
        self.system.Add(self.motor_endoffactor_finger_1)

        self.motor_endoffactor_finger_2 = chrono.ChLinkMotorLinearPosition()
        self.motor_endoffactor_finger_2.Initialize(self.endoffactor, self.finger_2, chrono.ChFramed(self.joint_endoffactor.GetPos(), self.joint_endoffactor.GetRot()))
        self.system.Add(self.motor_endoffactor_finger_2)

        # # # Set the position of the robot
        # offset = chrono.ChVector3d(0.0,0.8869/2,0.0)
        # pos = pos + offset
        # print(pos)
        self.base.SetFixed(False)
        self.base.SetPos(pos+self.base.GetPos())
        self.shoulder.SetPos(pos+self.shoulder.GetPos())
        self.biceps.SetPos(pos+self.biceps.GetPos())
        self.wrist.SetPos(pos+self.wrist.GetPos())
        self.elbow.SetPos(pos+self.elbow.GetPos())
        self.endoffactor.SetPos(pos+self.endoffactor.GetPos())
        self.finger_1.SetPos(pos+self.finger_1.GetPos())
        self.finger_2.SetPos(pos+self.finger_2.GetPos())
        self.finger_1.EnableCollision(False)
        self.finger_2.EnableCollision(False)
        
        if attached_vehicle:
            # add linklock to the base of the arm and the object
            self.base.SetPos(pos)
            lock = chrono.ChLinkLockLock()
            lock.Initialize( attached_vehicle.GetChassisBody(),self.base, chrono.ChFramed(pos, chrono.QUNIT))
            self.system.Add(lock)
            print("!!!!!!!!!!!!!added lock!!!!!!!!!!!!")
        else:
            self.base.SetFixed(True)
            print("!!!!!!!!!!!!!base fixed!!!!!!!!!!!!")
        # self.finger_1.EnableCollision(True)
        # self.finger_2.EnableCollision(True)

        self.objects = list()
        self.gripper_on = False
        self.cur_lock = None
        self.cur_object = None
        self.object_contact_count = None
        self.motor_val = 0.058

        self.gripper_left_or_right = True
        self.flag = False
        self.lock_flag = False

        
    def rotate_motor(self, motor, angle):
        if motor==self.motor_base_shoulder:
            motor.SetAngleFunction(chrono.ChFunctionConst(-angle-math.pi))
        elif motor==self.motor_shoulder_biceps:
            motor.SetAngleFunction(chrono.ChFunctionConst( angle))
        elif motor==self.motor_biceps_elbow:
            motor.SetAngleFunction(chrono.ChFunctionConst( -angle))
        elif motor==self.motor_elbow_eef:
            motor.SetAngleFunction(chrono.ChFunctionConst( -angle))
     

    def move_linear_motor(self, motor, pos):
        motor.SetMotionFunction(chrono.ChFunctionConst(pos))

    
    def add_object(self, object_name):
        self.objects.append(object_name)

    def add_lock(self):
        if not self.cur_lock:
            for object_name in self.objects:
                object = self.system.SearchBody(object_name)
                dist_1 = (object.GetPos() - self.finger_1.GetPos()).Length()
                dist_2 = (object.GetPos() - self.finger_2.GetPos()).Length()
                if dist_1 < 0.27 and dist_2 < 0.27:
                    print("here")
                    lock = chrono.ChLinkLockLock()
                    lock.SetName('lock' + object.GetName())
                    mid_point = (self.finger_1.GetPos() + self.finger_2.GetPos()) / 2
                    lock.Initialize(self.endoffactor, object, chrono.ChFramed(mid_point, chrono.QUNIT))
                    print("lock added")
                    self.system.Add(lock)
                    self.cur_lock = lock.GetName()
                    print("curlock: ", self.cur_lock)
                    self.cur_object = object.GetName()

    def remove_lock(self):
        self.system.RemoveLink(self.system.SearchLink(self.cur_lock))
        self.cur_lock = None
        self.system.SearchBody(self.cur_object).EnableCollision(True)
        self.cur_object = None
        self.gripper_on = False
        self.lock_flag = False
        print("lock removed")
    
    def open(self):
        self.left_motor_val = 0.0
        self.right_motor_val = 0.0
        self.move_linear_motor(self.motor_endoffactor_finger_1, -self.left_motor_val)
        self.move_linear_motor(self.motor_endoffactor_finger_2, self.right_motor_val)
        self.gripper_on = False
        if self.cur_lock:
            self.remove_lock()

    def grab_object(self):
        self.finger_1.EnableCollision(True)
        self.finger_2.EnableCollision(True)
        if not self.object_contact_count:
                self.object_contact_count = self.system.GetNumContacts()
        if self.flag == False and self.object_contact_count + 1 <= self.system.GetNumContacts():
            self.object_contact_count = self.system.GetNumContacts()
            self.flag = True
        if not self.gripper_on:
            if not self.object_contact_count:
                self.object_contact_count = self.system.GetNumContacts()

            if not self.flag and self.object_contact_count + 1 > self.system.GetNumContacts():
                if self.gripper_left_or_right:
                    self.left_motor_val -= 0.02
                    self.move_linear_motor(self.motor_endoffactor_finger_1, (self.left_motor_val))
                    self.gripper_left_or_right = False
                else:
                    self.right_motor_val -= 0.02
                    self.move_linear_motor(self.motor_endoffactor_finger_2, -(self.right_motor_val))
                    self.gripper_left_or_right = True
            elif self.flag and self.object_contact_count + 1 > self.system.GetNumContacts():
                if self.lock_flag == False:          
                    self.add_lock()
                    if self.cur_object:
                        self.lock_flag = True
                if self.gripper_left_or_right:
                    self.left_motor_val -= 0.02
                    self.move_linear_motor(self.motor_endoffactor_finger_1, (self.left_motor_val))
                else:
                    self.right_motor_val -= 0.02
                    self.move_linear_motor(self.motor_endoffactor_finger_2, -(self.right_motor_val))
            else:    
                if self.cur_object:        
                    self.gripper_on = True
                    self.system.SearchBody(self.cur_object).EnableCollision(False)
                self.object_contact_count = None