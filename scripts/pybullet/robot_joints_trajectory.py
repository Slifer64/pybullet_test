import sys

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# import numpy.matlib as matlib
import numpy.linalg as linalg
import yaml
import pybullet as p
import pybullet_data
import time
import enum
# import inspect
import warnings
import math

from utils.orientation.quaternion import Quaternion
from utils.orientation.affine3 import Affine3


# =====================================
# ============ GUI utils ==============
# =====================================

class Button:
    def __init__(self, title):
        self.id = p.addUserDebugParameter(title, 1, 0, 1)
        self.counter = p.readUserDebugParameter(self.id)
        self.counter_prev = self.counter

    def on(self):
        self.counter = p.readUserDebugParameter(self.id)
        if self.counter % 2 == 0:
            return True
        return False


class Gui:
    def __init__(self):
        self.pause_id = p.addUserDebugParameter("Pause", 1, 0, 1)
        self.counter = p.readUserDebugParameter(self.pause_id)
        self.counter_prev = self.counter


        self.id_2 = p.addUserDebugParameter("simulationSpeed", 0, 0.02, 0)

        self.exit_button = Button('Exit')

    def paused(self):
        self.counter = p.readUserDebugParameter(self.pause_id)
        if self.counter % 2 == 0:
            return True
        return False

    def speed(self):
        self.wait = p.readUserDebugParameter(self.id_2)


def plotFrame(pos, quat, name=None, length=0.1, linewidth=3):

    quat = Quaternion(*quat)
    x_id = p.addUserDebugLine(pos, pos + length * quat.rotation_matrix()[:, 0], lineColorRGB=[1, 0, 0], lineWidth=linewidth)
    y_id = p.addUserDebugLine(pos, pos + length * quat.rotation_matrix()[:, 1], lineColorRGB=[0, 1, 0], lineWidth=linewidth)
    z_id = p.addUserDebugLine(pos, pos + length * quat.rotation_matrix()[:, 2], lineColorRGB=[0, 0, 1], lineWidth=linewidth)
    if name is not None:
        p.addUserDebugText(text=name, textPosition=pos)

    return [x_id, y_id, z_id]


def removeFrame(frame_id):

    for id in frame_id:
        p.removeUserDebugItem(id)

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

class BulletRobot:

    # --------------------------------
    # -----------  Public  -----------
    # --------------------------------

    # ============= Constructor ===============
    def __init__(self, urdf_filename, joint_names, ee_link, 
            joint_lower_limits=None, joint_upper_limits=None,
            base_pos=[0, 0, 0], base_quat=[1, 0, 0, 0]):

        self.__robot_id = p.loadURDF(urdf_filename,
                                     basePosition=base_pos,
                                     baseOrientation=[*base_quat[1:], base_quat[0]],
                                     useFixedBase=True)

        self.__joint_names = joint_names.copy()
        self.__ee_link = ee_link
        self.__N_JOINTS = len(self.__joint_names)

        self.__joint_id = [None] * self.__N_JOINTS
        self.__ee_id = None

        self.__joints_lower_lim = joint_lower_limits.copy()
        self.__joints_upper_lim = joint_upper_limits.copy()

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
                                        targetVelocities=self.__joints_vel_cmd)
        elif self.__ctrl_mode is CtrlMode.JOINT_TORQUE:
            raise RuntimeError("Not implemented yet...")
        elif self.__ctrl_mode is CtrlMode.CART_POSITION:
            p.setJointMotorControlArray(self.__robot_id, self.__joint_id, p.POSITION_CONTROL,
                                        targetPositions=self.__joints_pos_cmd)
        elif self.__ctrl_mode is CtrlMode.CART_VELOCITY:
            jvel_cmd = np.matmul(linalg.pinv(self.getEEJacobian()), self.__cart_vel_cmd)
            p.setJointMotorControlArray(self.__robot_id, self.__joint_id, p.VELOCITY_CONTROL, targetVelocities=jvel_cmd)
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

    def getCtrlMode(self):
        return self.__ctrl_mode

    # =======================================
    def resetJoints(self, j_pos):

        for id, pos in zip(self.__joint_id, j_pos):
            p.resetJointState(self.__robot_id, id, pos)

        self.__readState()

    def resetPose(self, pos, quat):
        _, found_solution = self.calculateInverseKinematics(pos, quat, threshold=1e-4, maxIter=100)
        if not found_solution:
            warnings.warn("Failed to find IK solution within defined error tolerance...")

    def calculateInverseKinematics(self, targetPos, targetQuat, threshold=1e-5, maxIter=100):
        found_solution = False
        iter = 0
        dist = float('inf')
        targetQuat = [*targetQuat[1:], targetQuat[0]]
        while (not found_solution and iter < maxIter):
            iter = iter + 1
            jointPoses = p.calculateInverseKinematics(self.__robot_id, self.__ee_id, targetPos, targetQuat,
                                                      lowerLimits=self.getJointsLowerLimit(),
                                                      upperLimits=self.getJointsUpperLimit())
            jointPoses = [*jointPoses]
            self.resetJoints(jointPoses)

            ls = p.getLinkState(self.__robot_id, self.__ee_id, computeForwardKinematics=1)
            newPos, newQuat = ls[4], ls[5]
            pos_dist = math.sqrt(sum([(targetPos[i] - newPos[i]) ** 2 for i in range(3)]))

            qdiff_w = sum([a*b for a, b in zip(targetQuat, newQuat)])
            if qdiff_w < 0:
                newQuat = [-a for a in newQuat]
                qdiff_w = sum([a * b for a, b in zip(targetQuat, newQuat)])
            quat_dist = 1 - qdiff_w

            dist = pos_dist + quat_dist
            found_solution = (dist < threshold)

        return jointPoses, found_solution

    # =======================================
    def getNumJoints(self):

        return self.__N_JOINTS

    # =======================================
    def setJointsPosition(self, jpos_cmd):

        if self.__ctrl_mode is not CtrlMode.JOINT_POSITION:
            raise RuntimeError("The control mode must be 'JOINT_POSITION' to set joint positions!")

        self.__joints_pos_cmd = jpos_cmd.copy()

    def setJointsVelocity(self, jvel_cmd):

        if self.__ctrl_mode is not CtrlMode.JOINT_VELOCITY:
            raise RuntimeError("The control mode must be 'JOINT_VELOCITY' to set joint velocities!")

        self.__joints_vel_cmd = jvel_cmd.copy()

    def setCartPose(self, pos_cmd, quat_cmd):

        if self.__ctrl_mode is not CtrlMode.CART_POSITION:
            raise RuntimeError("The control mode must be 'CART_POSITION' to set cartesian pose!")

        jpos_cmd, _ = self.calculateInverseKinematics(pos_cmd, quat_cmd, threshold=1e-4, maxIter=5)
        self.__joints_pos_cmd = jpos_cmd.copy()

    def setCartVelocity(self, vel_cmd):

        if self.__ctrl_mode is not CtrlMode.CART_VELOCITY:
            raise RuntimeError("The control mode must be 'CART_VELOCITY' to set cartesian velocity!")

        self.__cart_vel_cmd = vel_cmd

    # =====================================
    def getJointsPosition(self):

        return self.__joints_pos.copy()

    def getJointsVelocity(self):

        return self.__joints_vel.copy()

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

    def getJointsLowerLimit(self):
        return self.__joints_lower_lim.copy()

    def getJointsUpperLimit(self):
        return self.__joints_upper_lim.copy()

    # =======================================

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


def get5thOrderQuatTraj(t: float, Q0: Quaternion, Qf: Quaternion, total_time: float):

    assert isinstance(Q0, Quaternion), "'Q0' must be a Quaternion"
    assert isinstance(Qf, Quaternion), "'Qf' must be a Quaternion"

    qLog_0 = np.array([0, 0, 0])
    qLog_f = Quaternion.log(Qf.mul(Q0.inv()))

    logQ1, logQ1_dot, logQ1_ddot = get5thOrderTraj(t, qLog_0, qLog_f, total_time)

    Q1 = Quaternion.exp(logQ1)
    Q = Q1.mul(Q0)
    rot_vel = Quaternion.logDot_to_rotVel(logQ1_dot, Q1)
    rot_accel = Quaternion.logDDot_to_rotAccel(logQ1_ddot, rot_vel, Q1)

    return Q, rot_vel, rot_accel


# ====================================
# =============  MAIN ================
# ====================================


def jointSpaceControl(env, robot, Ts, ctrl_mode="position"):
    
    t = 0.
    n_joints = robot.getNumJoints()

    j_init = np.array([-1.6, -1.73, -2.2, -0.808, 1.6, -0.031])
    j_final = np.array([-0.1, -1.8, -0.8, -0.1, 0.5, -1])

    T = max(linalg.norm(j_init[0:n_joints] - j_final[0:n_joints], np.inf) * 4.0 / np.pi, 2.0)

    robot.resetJoints(j_init)
    robot.update()

    Time = np.array([t])
    jpos_data = robot.getJointsPosition()
    jvel_data = np.zeros_like(j_init)
    pos_data = robot.getTaskPosition()

    jpos_ref_data = j_init
    jvel_ref_data = np.zeros_like(j_init)

    if ctrl_mode == "position":
        robot.setCtrlMode(CtrlMode.JOINT_POSITION)
        setRobotReference = lambda jpos_ref, jvel_ref: robot.setJointsPosition(jpos_ref)
    elif ctrl_mode == "velocity":
        robot.setCtrlMode(CtrlMode.JOINT_VELOCITY)
        setRobotReference = lambda jpos_ref, jvel_ref: robot.setJointsVelocity(jvel_ref)
    else:
        raise ValueError('Unsupported control mode "' + ctrl_mode + '"...')

    # simulation loop
    while t < T:
        env.step()
        robot.update()

        jpos_ref, jvel_vel, _ = get5thOrderTraj(t, j_init, j_final, T)
        setRobotReference(jpos_ref, jvel_vel)

        t += Ts
        Time = np.append(Time, t)
        jpos_data = np.column_stack((jpos_data, robot.getJointsPosition()))
        jvel_data = np.column_stack((jvel_data, robot.getJointsVelocity()))
        pos_data = np.column_stack((pos_data, robot.getTaskPosition()))
        jpos_ref_data = np.column_stack((jpos_ref_data, jpos_ref))
        jvel_ref_data = np.column_stack((jvel_ref_data, jvel_vel))

    # jvel_data = np.column_stack((np.diff(jpos_data, axis=1)/Ts, np.zeros_like(j_init)))

    # print('=========> final pose:', [ '%.4f' % a for a in robot.getTaskPose()])

    # final_pose = np.array([0.426, -0.31, 0.899, 0.0964, 0.2931, -0.5584, 0.7701])
    # plotFrame(robot.getTaskPosition(), robot.getTaskQuat(), name='ee', length=0.15, linewidth=2)
    # plotFrame(final_pose[0:3], final_pose[3:], name='final', length=0.1, linewidth=6)

    # ========= Plot ===========

    # plt.ion()

    # Joint Position trajectories
    fig, axs = plt.subplots(n_joints, 1)
    for i, ax in enumerate(axs):
        ax.plot(Time, jpos_data[i, :], linewidth=2, color='blue', label='robot')
        ax.plot(Time, jpos_ref_data[i, :], linewidth=2, linestyle='--', color='magenta', label='ref')
        ax.set_ylabel("j" + str(i))
        if i == 0:
            ax.legend(loc='upper left', fontsize=14)
    axs[0].set_title("Joints Position")
    axs[n_joints - 1].set_xlabel("time [s]", fontsize=14)

    # Joint Velocity trajectories
    fig, axs = plt.subplots(n_joints, 1)
    for i, ax in enumerate(axs):
        ax.plot(Time, jvel_data[i, :], linewidth=2, color='blue', label='robot')
        ax.plot(Time, jvel_ref_data[i, :], linewidth=2, linestyle='--', color='magenta', label='ref')
        ax.set_ylabel("j" + str(i))
        if i == 0:
            ax.legend(loc='upper left', fontsize=14)
    axs[0].set_title("Joints Velocity")
    axs[n_joints - 1].set_xlabel("time [s]", fontsize=14)

    # Cartesian Position trajectories
    fig, axs = plt.subplots(3, 1)
    y_labels = ["x", "y", "z"]
    for i in range(0, 3):
        axs[i].plot(Time, pos_data[i,:], linewidth=2, color='blue')
        axs[i].set_ylabel(y_labels[i])
    axs[0].set_title("Cartesian position", fontsize=14)
    axs[2].set_xlabel("time [s]", fontsize=14)

    # Cartesian Position path
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot3D(pos_data[0,:], pos_data[1, :], pos_data[2, :], color='blue')
    ax.set_title('Cartesian Position path')

    plt.show()


def CartSpaceControl(env, robot, Ts, ctrl_mode="position"):
    
    t = 0.

    init_pose = np.array([-0.142, -0.436, 0.152, 0, 0, 1, 0])
    final_pose = np.array([0.426, -0.31, 0.75, 0.0964, 0.2931, -0.5584, 0.7701])

    T = max(linalg.norm(init_pose[0:3] - final_pose[0:3]) * 4.0, 1.5)

    # robot.resetPose(init_pose[:3], init_pose[3:]) # produces the same pose with different joint config...
    robot.resetJoints(np.array([-1.6, -1.73, -2.2, -0.808, 1.6, -0.031]))

    # fmt = lambda x: ' , '.join(['%.3f' % a for a in x])
    # print('pose:', fmt(robot.getTaskPose()))
    # print('init_pose:', fmt(init_pose))
    # print('joint:', fmt(robot.getJointsPosition()))
    # print('-------------------------')

    # Jee = robot.getEEJacobian()
    # for row in Jee:
    #     print(' , '.join(['%6.3f' % a for a in row]))
    #
    # sys.exit()

    plotFrame(init_pose[0:3], init_pose[3:], name='init', length=0.1, linewidth=6)
    plotFrame(final_pose[0:3], final_pose[3:], name='final', length=0.1, linewidth=6)

    Time = np.array([t])
    pose_data = robot.getTaskPose()
    vel_data = np.zeros(6)
    # accel_data = np.zeros((6, 1))

    pose_ref_data = init_pose
    vel_ref_data = np.zeros(6)

    if ctrl_mode == "position":
        robot.setCtrlMode(CtrlMode.CART_POSITION)
        setRobotReference = lambda pose_ref, vel_ref: robot.setCartPose(pose_ref[:3], pose_ref[3:])
    elif ctrl_mode == "velocity":
        robot.setCtrlMode(CtrlMode.CART_VELOCITY)
        setRobotReference = lambda pose_ref, vel_ref: robot.setCartVelocity(vel_ref)
    else:
        raise ValueError('Unsupported control mode "' + ctrl_mode + '"...')

    # simulation loop
    while t < T:
        env.step()
        robot.update()

        pos_ref, pdot_ref, _ = get5thOrderTraj(t, init_pose[:3], final_pose[:3], T)
        quat_ref, rotVel_ref, _ = get5thOrderQuatTraj(t, Quaternion(*init_pose[3:]), Quaternion(*final_pose[3:]), T)

        pose_ref = np.concatenate((pos_ref, quat_ref()))
        vel_ref = np.concatenate((pdot_ref, rotVel_ref))

        use_click = False
        if use_click:
            pos_click = 2.0 * (pose_ref[:3] - robot.getTaskPosition())
            Q_ref = Quaternion(*pose_ref[3:])
            Q_robot = Quaternion(*robot.getTaskQuat())
            orient_click = 2.0 * Quaternion.log(Q_ref.diff(Q_robot))
            vel_ref += np.concatenate((pos_click, orient_click))

        setRobotReference(pose_ref, vel_ref)

        t += Ts
        Time = np.append(Time, t)
        pose_data = np.column_stack((pose_data, robot.getTaskPose()))
        pose_ref_data = np.column_stack((pose_ref_data, pose_ref))
        vel_ref_data = np.column_stack((vel_ref_data, vel_ref))

    # ========= Plot ===========

    # Cartesian Position trajectories
    fig, axs = plt.subplots(3, 1)
    y_labels = ["x", "y", "z"]
    for i in range(0, 3):
        axs[i].plot(Time, pose_data[i, :], linewidth=2, color='blue', label='robot')
        axs[i].plot(Time, pose_ref_data[i, :], linewidth=2, linestyle='--', color='magenta', label='ref')
        axs[i].set_ylabel(y_labels[i])
    axs[0].legend(loc='upper left', fontsize=14)
    axs[0].set_title("Cartesian position", fontsize=14)
    axs[2].set_xlabel("time [s]", fontsize=14)

    # Cartesian Position path
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(pose_data[0, :], pose_data[1, :], pose_data[2, :], color='blue', label='robot')
    ax.plot3D(pose_ref_data[0, :], pose_ref_data[1, :], pose_ref_data[2, :], linestyle='--', color='magenta', label='ref')
    ax.legend(loc='upper left', fontsize=14)
    ax.set_title('Cartesian Position path')

    plt.show()

    # input('Press [enter] to continue...')


if __name__ == '__main__':

    print("========= Robot trajectory example =========")

    with open('../yaml/params.yml', 'r') as stream:
        params = yaml.safe_load(stream)

    Ts = 0.004
    env = VirtualEnv()
    env.setTimeStep(Ts)

    robot_parmas = params["robot"]
    robot = BulletRobot(robot_parmas["urdf"], robot_parmas["joints"], robot_parmas["ee_link"],
                    joint_lower_limits=robot_parmas['joint_limits']['lower'],
                    joint_upper_limits=robot_parmas['joint_limits']['upper'],
                    base_pos=robot_parmas['base']['pos'],
                    base_quat=robot_parmas['base']['quat'])
    print(robot)

    # jointSpaceControl(env, robot, Ts, ctrl_mode="position")
    
    CartSpaceControl(env, robot, Ts, ctrl_mode="velocity")

