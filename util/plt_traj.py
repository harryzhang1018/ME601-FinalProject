import numpy as np
import matplotlib.pyplot as plt

# load data
nn_traj = np.loadtxt('/home/harry/ME601-FinalProject/data_nn.csv', delimiter=',')
nn_traj_time = nn_traj[:, 0]
ik_traj = np.loadtxt('/home/harry/ME601-FinalProject/data_ik.csv', delimiter=',')
ik_traj_time = ik_traj[:, 0]

desired_coords = np.array([[1.85115481, -0.95066582,  0.1112629]])
# match the same dimension
desired_coords = np.repeat(desired_coords, nn_traj.shape[0], axis=0)
plt.figure(figsize=(15, 9))
plt.subplot(3, 1, 1)
plt.plot(nn_traj_time, nn_traj[:, 1], label='end effector pos x--NN')
plt.plot(ik_traj_time, ik_traj[:, 1], label='end effector pos x--IK')
plt.plot(nn_traj_time, desired_coords[:, 0],'--', label='desired pos x')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend(loc='lower right')
plt.subplot(3, 1, 2)
plt.plot(nn_traj_time, nn_traj[:, 2], label='end effector pos y--NN')
plt.plot(ik_traj_time, ik_traj[:, 2], label='end effector pos y--IK')
plt.plot(nn_traj_time, desired_coords[:, 1], '--',label='desired pos y')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend(loc='lower right')
plt.subplot(3, 1, 3)
plt.plot(nn_traj_time, nn_traj[:, 3], label='end effector pos z--NN')
plt.plot(ik_traj_time, ik_traj[:, 3], label='end effector pos z--IK')
plt.plot(nn_traj_time, desired_coords[:, 2], '--',label='desired pos z')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# plot circle trajectory
ik_traj_circle = np.loadtxt('/home/harry/ME601-FinalProject/data_ik_circle.csv', delimiter=',')
# take every 2 points
ik_traj_circle = ik_traj_circle[::6]

nn_traj_circle = np.loadtxt('/home/harry/ME601-FinalProject/data_nn_circle.csv', delimiter=',')
# take every 2 points
nn_traj_circle = nn_traj_circle[::6]

# create circle reference trajectory
alpha = 0.1 * nn_traj_circle[:, 0]
ref_y = 1.5 * np.cos(alpha + np.pi / 2)
ref_z = 1.5 * np.sin(alpha + np.pi / 2)


plt.figure(figsize=(6, 6))
plt.plot(ik_traj_circle[10:, 2], ik_traj_circle[10:, 3],'r', label='end effector pos--IK')
plt.plot(ref_y, ref_z, 'k--', label='reference pos')
plt.plot(nn_traj_circle[10:, 2], nn_traj_circle[10:, 3], label='end effector pos--NN')
plt.xlim(-2.0, 2.0)
plt.ylim(-2.0, 2.0)
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.legend(loc='center')
plt.show()