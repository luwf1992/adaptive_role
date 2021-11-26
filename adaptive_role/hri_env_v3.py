import pybullet as p
import pybullet_data as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from kinematics import Kinematics
from ddpg import DDPG

model_path = './model'


class HriEnv_v3(gym.Env):

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
        self.ddpg = DDPG(self.task_num, 6)
        self.ddpg.load_weights(model_path)
        max_b = [0.] * self.task_num
        max_b[0] = 1.
        self.std_max = np.std(max_b)
        self.max_f_imp = np.array([10., 10., 10.])
        self.action_space = spaces.Box(low=-self.max_f_imp, high=self.max_f_imp, dtype=np.float32)
        high_obs = np.array([1.7, 1.7, 1.7, 1.7, 1.7, 1.7])
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

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

        s = [self.ee_pos[0], self.ee_pos[1], self.ee_pos[2],
             self.ee_vel[0], self.ee_vel[1], self.ee_vel[2]]
        a = self.ddpg.choose_action(s, 0.)
        b_list = a.numpy().reshape(-1)
        dx_d = self.task_combination(b_list, self.ee_pos_array.reshape(-1)).reshape(3, 1)

        state = np.array([dx_d[0,0], dx_d[1,0], dx_d[2,0], self.ee_vel[0], self.ee_vel[1], self.ee_vel[2]])
        return state

    def step(self, f_imp):
        self.f_h = self.human_dynamics(self.ee_pos_array, self.ee_vel_array)
        # self.f_h = np.array([0., 0., 0.]).reshape(3, 1)

        acc = self.robot_controller(self.f_h, f_imp.reshape(3, 1))
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

        s = [self.ee_pos[0], self.ee_pos[1], self.ee_pos[2],
             self.ee_vel[0], self.ee_vel[1], self.ee_vel[2]]
        a = self.ddpg.choose_action(s, 0.)
        b_list = a.numpy().reshape(-1)
        dx_d = self.task_combination(b_list, self.ee_pos_array.reshape(-1)).reshape(3, 1)

        state = np.array([dx_d[0,0], dx_d[1,0], dx_d[2,0], self.ee_vel[0], self.ee_vel[1], self.ee_vel[2]])

        alpha = np.std(b_list) / self.std_max
        r = -alpha * np.sum(self.f_h**2) - (1 - alpha) * np.sum(f_imp**2)

        return state, r, False, {}

    def human_dynamics(self, x, d_x):
        h_h = -np.matmul(self.d_h_matrix, d_x) + np.matmul(self.k_h_matrix, self.x_d.reshape(3, 1) - x)
        return h_h

    def robot_controller(self, h_h, f_imp):
        dd_x = np.matmul(np.linalg.inv(self.m_r_matrix), h_h + f_imp)
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
    env = HriEnv_v3()
    env.reset()
    # print('task_index')
    # print(env.task_index)
    # print('x_d')
    # print(env.x_d)
    # # b_list = [0.] * env.task_num
    # # b_list[env.task_index] = 1
    # # print('b_list')
    # # print(b_list)
    # f_imp = np.array([0., 0., 0.])
    # p_x_list = []
    # p_y_list = []
    # v_x_list = []
    # v_y_list = []
    # fh_x_list = []
    # fh_y_list = []
    # f_imp_x_list = []
    # f_imp_y_list = []
    # for i in range(200):
    #     env.step(f_imp)
    #     p_x_list.append(env.ee_pos[0])
    #     p_y_list.append(env.ee_pos[1])
    #     v_x_list.append(env.ee_vel[0])
    #     v_y_list.append(env.ee_vel[1])
    #     fh_x_list.append(env.f_h[0])
    #     fh_y_list.append(env.f_h[1])
    #     # f_imp_x_list.append(env.f_imp[0])
    #     # f_imp_y_list.append(env.f_imp[1])
    #
    # plt.figure(1)
    # plt.plot(p_x_list, label='p_x')
    # plt.plot(p_y_list, label='p_y')
    # plt.title('Actual position')
    # plt.legend()
    #
    # plt.figure(2)
    # plt.plot(v_x_list, label='v_x')
    # plt.plot(v_y_list, label='v_y')
    # plt.title('Actual velocity')
    # plt.legend()
    #
    # plt.figure(3)
    # plt.plot(fh_x_list, label='fh_x')
    # plt.plot(fh_y_list, label='fh_y')
    # # plt.plot(f_imp_x_list, label='f_imp_x')
    # # plt.plot(f_imp_y_list, label='f_imp_y')
    # plt.title('Human force')
    # plt.legend()
    #
    # plt.show()


















