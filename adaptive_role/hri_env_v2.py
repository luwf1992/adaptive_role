import pybullet as p
import pybullet_data as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from kinematics import Kinematics


class HriEnv_v2(gym.Env):

    def __init__(self):
        self.dt = 0.05
        self.g = 0.
        self.task_num = 3
        self.x_g1 = np.array([0.36, 0.62, 0.06561195563238487])
        self.x_g2 = np.array([-0.1, 0.42, 0.06561195563238487])
        self.x_g3 = np.array([0.4, 0.58, 0.06561195563238487])
        self.x_g = np.concatenate((self.x_g1, self.x_g2, self.x_g3), axis=0).reshape(self.task_num, 3)
        self.m_r = 10.
        self.d_r = 20.
        self.m_r_matrix = self.m_r * np.eye(3)
        self.d_r_matrix = self.d_r * np.eye(3)
        self.d_h = 7.
        self.k_h = 30.
        self.d_h_matrix = self.d_h * np.eye(3)
        self.k_h_matrix = self.k_h * np.eye(3)
        # panda max joint torque and max grasp force
        self.force_list = [87., 87., 87., 87., 12., 12., 12., 140., 140.]
        # panda max joint velocity and max grasp velocity
        self.max_vel = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.05, 0.05]).reshape(9, 1)
        self.jointPositions = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        self.pandaEndEffectorIndex = 11
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.panda_joint_num = 9
        self.joint_list = range(self.panda_joint_num)
        p.connect(p.DIRECT)
        self.kinematics = Kinematics()

    def reset(self):
        #pybullet
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, self.g)
        self.t = 0.

        self.panda = p.loadURDF("panda/panda.urdf", [0,0,0], [0,0,0,1],
                                useFixedBase=True, flags=self.flags)
        index = 0
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.panda, j, self.jointPositions[index])
                index=index+1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.panda, j, self.jointPositions[index])
                index=index+1
        self.ee_pos, ee_orn, _, _, _, _, self.ee_vel, _ = p.getLinkState(
            self.panda, self.pandaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)
        # print('ee_pos')
        # print(self.ee_pos)
        self.ee_pos_array = np.array(self.ee_pos).reshape(3, 1)
        self.ee_vel_array = np.array(self.ee_vel).reshape(3, 1)

        # set random human intention x_d
        self.task_index = np.random.choice(range(self.task_num))
        # self.task_index = 2
        self.x_d = self.x_g[self.task_index, :]
        state = [self.ee_pos[0], self.ee_pos[1], self.ee_pos[2],
                 self.ee_vel[0], self.ee_vel[1], self.ee_vel[2]]

        return state

    def step(self, b_list):
        self.f_h = self.human_dynamics(self.ee_pos_array, self.ee_vel_array)
        # self.f_h = np.array([0., 0., 0.]).reshape(3, 1)

        dx_d = self.task_combination(b_list, self.ee_pos_array.reshape(-1)).reshape(3, 1)
        self.dx_d_output = dx_d
        # dx_d = np.array([0., 0., 0.]).reshape(3, 1)

        acc = self.robot_controller(self.f_h, dx_d, self.ee_vel_array)
        newd_x = self.ee_vel_array + acc * self.dt

        pos_vel = newd_x.reshape(-1)
        orn_vel = [0., 0., 0.]
        joint_vel = self.kinematics.solve_vIK(self.panda, self.pandaEndEffectorIndex, pos_vel, orn_vel)
        joint_vel = np.clip(joint_vel, -self.max_vel, self.max_vel)
        self.target_vel = joint_vel[:, 0]
        p.setJointMotorControlArray(self.panda, self.joint_list, p.VELOCITY_CONTROL,
                                    targetVelocities=self.target_vel, forces=self.force_list)

        p.stepSimulation()
        time.sleep(self.dt)
        self.t += self.dt

        self.ee_pos, ee_orn, _, _, _, _, self.ee_vel, _ = p.getLinkState(
            self.panda, self.pandaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)
        self.ee_pos_array = np.array(self.ee_pos).reshape(3, 1)
        self.ee_vel_array = np.array(self.ee_vel).reshape(3, 1)

        cov_b = 0.
        for i in range(self.task_num):
            cov_b += b_list[i] * (1. - b_list[i]) * \
                     np.sum(self.ds(self.x_g[i, :], self.ee_pos_array.reshape(-1)) ** 2)

        self.theta = dx_d.reshape(-1).dot(self.ee_vel_array.reshape(-1)) \
                     / (np.linalg.norm(dx_d) * np.linalg.norm(self.ee_vel_array))

        self.r_1 = -np.sum((dx_d - self.ee_vel_array) ** 2)
        # self.r_2 = -0.05 * cov_b
        self.r_2 = 0.0

        r = self.r_1 + self.r_2

        state = [self.ee_pos[0], self.ee_pos[1], self.ee_pos[2],
                 self.ee_vel[0], self.ee_vel[1], self.ee_vel[2]]

        return state, r

    def human_dynamics(self, x, d_x):
        h_h = -np.matmul(self.d_h_matrix, d_x) + np.matmul(self.k_h_matrix, self.x_d.reshape(3, 1) - x)
        return h_h

    def robot_controller(self, h_h, dx_d, dx):
        self.f_imp = np.matmul(self.d_r_matrix, dx_d - dx)
        dd_x = np.matmul(np.linalg.inv(self.m_r_matrix), h_h + self.f_imp)
        return dd_x

    def ds(self, x_g_array, x_array):
        dx_array = x_g_array - x_array
        return dx_array

    def task_combination(self, b_list, x_array):
        dx_d = 0
        for i in range(self.task_num):
            dx_d += b_list[i] * self.ds(self.x_g[i, :], x_array)
        return dx_d


if __name__ == "__main__":
    env = HriEnv_v2()
    env.reset()
    print('task_index')
    print(env.task_index)
    print('x_d')
    print(env.x_d)
    b_list = [0.] * env.task_num
    b_list[env.task_index] = 1
    print('b_list')
    print(b_list)
    p_x_list = []
    p_y_list = []
    v_x_list = []
    v_y_list = []
    fh_x_list = []
    fh_y_list = []
    f_imp_x_list = []
    f_imp_y_list = []
    for i in range(200):
        env.step(b_list)
        p_x_list.append(env.ee_pos[0])
        p_y_list.append(env.ee_pos[1])
        v_x_list.append(env.ee_vel[0])
        v_y_list.append(env.ee_vel[1])
        fh_x_list.append(env.f_h[0])
        fh_y_list.append(env.f_h[1])
        f_imp_x_list.append(env.f_imp[0])
        f_imp_y_list.append(env.f_imp[1])

    plt.figure(1)
    plt.plot(p_x_list, label='p_x')
    plt.plot(p_y_list, label='p_y')
    plt.title('Actual position')
    plt.legend()

    plt.figure(2)
    plt.plot(v_x_list, label='v_x')
    plt.plot(v_y_list, label='v_y')
    plt.title('Actual velocity')
    plt.legend()

    plt.figure(3)
    plt.plot(fh_x_list, label='fh_x')
    plt.plot(fh_y_list, label='fh_y')
    plt.plot(f_imp_x_list, label='f_imp_x')
    plt.plot(f_imp_y_list, label='f_imp_y')
    plt.title('Human force')
    plt.legend()

    plt.show()

















