import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy.matlib as matlib
import numpy.linalg as linalg
import yaml
import pybullet as p
import pybullet_data
import time
import enum


# =============================================
# ============ Virtual Env class ==============
# =============================================

class VirtualEnv:

    def __init__(self):
        self.physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)

        self.setTimeStep(1.0 / 240)
        p.setTimeStep(self.Ts)

    def __del__(self):
        p.disconnect()

    def setTimeStep(self, Ts):
        self.Ts = Ts
        p.setTimeStep(self.Ts)

    def step(self):
        p.stepSimulation()


# =======================================
# ============ Robot class ==============
# =======================================

class CtrlMode(enum.Enum):
    IDLE = -1
    JOINT_POSITION = 1
    JOINT_VELOCITY = 2
    JOINT_TORQUE = 3
    CART_POSITION = 4
    CART_VELOCITY = 5

class Robot:

    # --------------------------------
    # -----------  Public  -----------
    # --------------------------------

    # ============= Constructor ===============
    def __init__(self, urdf_filename, joint_names, ee_link):

        self.__robot_id = p.loadURDF(urdf_filename)  # , pos, quat)

        self.__joint_names = joint_names.copy()
        self.__ee_link = ee_link
        self.__N_JOINTS = len(self.__joint_names)

        self.__joint_id = [None] * self.__N_JOINTS
        self.__ee_id = None

        joints_info = [p.getJointInfo(self.__robot_id, i) for i in range(p.getNumJoints(self.__robot_id))]
        for i in range(self.__N_JOINTS):
            for j in range(len(joints_info)):
                if (joint_names[i] == joints_info[j][1].decode('utf8')):
                    self.__joint_id[i] = joints_info[j][0]
                    break
            else:
                raise RuntimeError("Failed to find joint name '" + joint_names[i] + "'...")

        for j in range(len(joints_info)):
            if (joints_info[j][12].decode('utf8') == ee_link):
                self.__ee_id = joints_info[j][0]
                break
        else:
            raise RuntimeError("Failed to find ee_link '" + ee_link + "'...")

        self.__joints_pos = np.zeros(self.__N_JOINTS)
        self.__joints_vel = np.zeros(self.__N_JOINTS)
        self.__cart_vel_cmd = np.zeros(6)
        self.__readState()
        self.__joints_pos_cmd = self.__joints_pos.copy()
        self.__joints_vel_cmd = np.zeros(self.getNumJoints())
        self.__ctrl_mode = None
        self.setCtrlMode(CtrlMode.IDLE)

        self.__ee_offset = [-v for v in p.getLinkState(self.__robot_id, self.__ee_id)[2]]


    # =======================================
    def update(self):

        # read new values from Bullet
        self.__readState()

        # send cmd
        if self.__ctrl_mode is CtrlMode.JOINT_POSITION:
            p.setJointMotorControlArray(self.__robot_id, self.__joint_id, p.POSITION_CONTROL,
                                        targetPositions=self.__joints_pos_cmd)
        elif self.__ctrl_mode is CtrlMode.JOINT_VELOCITY:
            p.setJointMotorControlArray(self.__robot_id, self.__joint_id, p.VELOCITY_CONTROL,
                                        targetVelocity=self.__joints_vel_cmd)
        elif self.__ctrl_mode is CtrlMode.JOINT_TORQUE:
            raise RuntimeError("Not implemented yet...")
        elif self.__ctrl_mode is CtrlMode.CART_POSITION:
            raise RuntimeError("Not implemented yet...")
        elif self.__ctrl_mode is CtrlMode.CART_VELOCITY:
            jvel_cmd = np.matmul( linalg.pinv(self.getEEJacobian()), self.__cart_vel_cmd)
            p.setJointMotorControlArray(self.__robot_id, self.__joint_id, p.VELOCITY_CONTROL, targetVelocity=jvel_cmd)
        else: # CtrlMode.IDLE
            pass

    # =======================================
    def setCtrlMode(self, ctrl_mode):

        assert isinstance(ctrl_mode, CtrlMode)

        # if ctrl_mode not in set(item.value for item in CtrlMode):
        #     raise RuntimeError("Unsupported ctrl mode '" + str(ctrl_mode) + "'...")

        self.__readState()

        self.__joints_pos_cmd = self.getJointsPosition()
        self.__joints_vel_cmd = np.zeros(self.getNumJoints())

        self.__ctrl_mode = ctrl_mode

    # =======================================
    def reset(self, j_pos):

        for id, pos in zip(self.__joint_id, j_pos):
            p.resetJointState(self.__robot_id, id, pos)

        self.__readState()

    # =======================================
    def getNumJoints(self):

        return self.__N_JOINTS

    # =======================================
    def setJointsPosition(self, j_pos_cmd):

        if self.__ctrl_mode is not CtrlMode.JOINT_POSITION:
            raise RuntimeError("The control mode must be 'JOINT_POSITION' to set joint positions!")

        self.__joints_pos_cmd = j_pos_cmd.copy()

    def setJointsVelocity(self, j_pos_cmd):

        if self.__ctrl_mode is not CtrlMode.JOINT_VELOCITY:
            raise RuntimeError("The control mode must be 'JOINT_VELOCITY' to set joint velocities!")

        self.__joints_pos_cmd = j_pos_cmd.copy()

    def setCartPose(self, pos_cmd, quat_cmd):

        if self.__ctrl_mode is not CtrlMode.CART_POSITION:
            raise RuntimeError("The control mode must be 'CART_POSITION' to set cartesian pose!")

        raise NotImplemented

    def setCartVelocity(self, vel_cmd):

        if self.__ctrl_mode is not CtrlMode.CART_VELOCITY:
            raise RuntimeError("The control mode must be 'CART_VELOCITY' to set cartesian velocity!")

        self.__cart_vel_cmd = vel_cmd

    # =====================================
    def getJointsPosition(self):

        return self.__joints_pos

    def getJointsVelocity(self):

        return self.__joints_vel

    def getTaskPosition(self):

        return [*p.getLinkState(self.__robot_id, self.__ee_id, computeLinkVelocity=0)[4]]

    def getTaskQuat(self):

        quat = p.getLinkState(self.__robot_id, self.__ee_id, computeLinkVelocity=0)[5]
        return [quat[3], *quat[0:3]]

    def getTaskPose(self):

        link_state = p.getLinkState(self.__robot_id, self.__ee_id, computeLinkVelocity=0)
        pos = link_state[4]
        quat = link_state[5]
        return [*pos, quat[3], *quat[0:3]]

    def getTaskVelocity(self):

        link_state = p.getLinkState(self.__robot_id, self.__ee_id, computeLinkVelocity=1)
        return [*link_state[6], *link_state[7]]

    def getEEJacobian(self):

        n_joints = self.getNumJoints()
        O_n = [0 for i in range(n_joints)]
        J_t, J_r = p.calculateJacobian(self.__robot_id, self.__ee_id, self.__ee_offset, list(self.getJointsPosition()), O_n, O_n)
        jacob = np.zeros((6, n_joints))
        for i in range(3):
            jacob[i,:] = [*J_t[i]]
            jacob[i+3, :] = [*J_r[i]]

        return jacob

    # =======================================
    def str(self):

        return self.__str__()

    # ---------------------------------
    # -----------  Private  -----------
    # ---------------------------------

    def __readState(self):

        j_states = p.getJointStates(self.__robot_id, self.__joint_id)
        for i in range(self.__N_JOINTS):
            self.__joints_pos[i] = j_states[i][0]
            self.__joints_vel[i] = j_states[i][1]

    def __str__(self):

        return "========= Robot =======" + \
               "\n+ joints:" + str(self.__joint_names) + \
               "\n+ ee_link:" + self.__ee_link


# ====================================
# =============  UTILS ===============
# ====================================

def get5thOrderTraj(t: float, p0: np.array, pf: np.array, total_time: float) -> (np.array, np.array, np.array):
    n_dof = len(p0)

    pos = np.zeros(n_dof)
    vel = np.zeros(n_dof)
    accel = np.zeros(n_dof)

    if t < 0:
        pos = p0
    elif t > total_time:
        pos = pf
    else:
        pos = p0 + (pf - p0) * (10 * pow(t / total_time, 3) -
                                15 * pow(t / total_time, 4) + 6 * pow(t / total_time, 5))
        vel = (pf - p0) * (30 * pow(t, 2) / pow(total_time, 3) -
                           60 * pow(t, 3) / pow(total_time, 4) + 30 * pow(t, 4) / pow(total_time, 5))
        accel = (pf - p0) * (60 * t / pow(total_time, 3) -
                             180 * pow(t, 2) / pow(total_time, 4) + 120 * pow(t, 3) / pow(total_time, 5))

    return pos, vel, accel


# ====================================
# =============  MAIN ================
# ====================================

if __name__ == '__main__':

    print("========= Robot joints trajectory example =========")

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    Ts = 0.004
    env = VirtualEnv()
    env.setTimeStep(Ts)

    robot_parmas = params["robot"]
    robot = Robot(robot_parmas["urdf"], robot_parmas["joints"], robot_parmas["ee_link"])
    print(robot)

    t = 0.
    n_joints = robot.getNumJoints()

    np.random.seed(0)

    joint_limits = np.array([6.28, 6.28, 3.14, 6.28, 6.28, 6.28])
    j_init = np.multiply(2 * np.random.rand(n_joints) - 1, joint_limits)
    j_final = np.multiply(2 * np.random.rand(n_joints) - 1, joint_limits)

    T = max(linalg.norm(j_init[0:n_joints] - j_final[0:n_joints], np.inf) * 4.0 / np.pi, 2.0)

    robot.reset(j_init)
    robot.update()

    p0 = robot.getTaskPosition()
    p_target = p0 + np.array([0.15, 0.2, 0.3])

    Time = np.array([t])
    jpos_data = robot.getJointsPosition()
    pos_data = p0
    jd_pos_data = j_init

    # robot.setCtrlMode(CtrlMode.JOINT_POSITION)
    robot.setCtrlMode(CtrlMode.JOINT_VELOCITY)
    # robot.setCtrlMode(CtrlMode.CART_VELOCITY)

    # simulation loop
    while t < T:
        env.step()
        robot.update()

        jd_pos, jd_vel, _ = get5thOrderTraj(t, j_init, j_final, T)
        # robot.setJointsPosition(jd_pos)
        robot.setJointsPosition(jd_vel)

        # _, vd, _ = get5thOrderTraj(t, p0, p_target, T)
        # robot.setCartVelocity(vd)

        t += Ts
        Time = np.append(Time, t)
        jpos_data = np.column_stack((jpos_data, robot.getJointsPosition()))
        pos_data = np.column_stack((pos_data, robot.getTaskPosition()))
        jd_pos_data = np.column_stack((jd_pos_data, jd_pos))

    # ========= Plot ===========

    # Joint trajectories
    fig, axs = plt.subplots(n_joints, 1)
    for i, ax in enumerate(axs):
        ax.plot(Time, jpos_data[i, :], linewidth=2, color='blue', label='robot')
        ax.plot(Time, jd_pos_data[i, :], linewidth=2, linestyle='--', color='magenta', label='ref')
        ax.set_ylabel("j" + str(i))
        if i == 0:
            ax.legend(loc='upper left', fontsize=14)
    axs[0].set_title("Robot joints trajectories")
    axs[n_joints - 1].set_xlabel("time [s]", fontsize=14)

    # Cartesian Position trajectories
    fig, axs = plt.subplots(3,1)
    y_labels = ["x", "y", "z"]
    for i in range(0,3):
        ax.plot(Time, pos_data[i,:], linewidth=2, color='blue')
        ax.set_ylabel(y_labels[i])
    axs[2].set_xlabel("time [s]", fontsize=14)

    # Cartesian Position path
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(pos_data[0,:], pos_data[1,:], pos_data[2,:], color='blue')
    ax.set_title('Cartesian Position path')

    plt.show()

    dummy = input('Press enter to continue...')
