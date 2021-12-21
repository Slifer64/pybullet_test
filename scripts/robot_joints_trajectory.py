import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import numpy.linalg as linalg
import yaml
import pybullet as p
import pybullet_data
import time

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
class Robot:

    # --------------------------------
    # -----------  Public  -----------
    # --------------------------------

    # ============= Constructor ===============
    def __init__(self, urdf_filename, joint_names, ee_link):

        self.robot_id = p.loadURDF(urdf_filename) #, pos, quat)

        self.joint_names = joint_names.copy()
        self.ee_link = ee_link
        self.N_JOINTS = len(self.joint_names)

        self.joint_id = [None] * self.N_JOINTS
        self.ee_id = None

        joints_info = [p.getJointInfo(self.robot_id, i) for i in range(p.getNumJoints(self.robot_id))]
        for i in range(self.N_JOINTS):
            for j in range(len(joints_info)):
                if ( joint_names[i] == joints_info[j][1].decode('utf8') ):
                    self.joint_id[i] = joints_info[j][0]
                    break
            else:
                raise RuntimeError("Failed to find joint name '" + joint_names[i] + "'...")

        for j in range(len(joints_info)):
            if (joints_info[j][12].decode('utf8') == ee_link):
                self.ee_id = joints_info[j][0]
                break
        else:
            raise RuntimeError("Failed to find ee_link '" + ee_link + "'...")

        self.joints_pos = np.zeros(self.N_JOINTS)
        self.joints_vel = np.zeros(self.N_JOINTS)
        self.__readState()
        self.joints_pos_cmd = self.joints_pos.copy()

        self.setCtrlMode(p.POSITION_CONTROL)


    # =======================================
    def update(self):

        # read new values from Bullet
        self.__readState()

        # send cmd
        if (self.ctrl_mode == p.POSITION_CONTROL):
            print(self.joints_pos_cmd)
            p.setJointMotorControlArray(self.robot_id, self.joint_id, p.POSITION_CONTROL, targetPositions=self.joints_pos_cmd)
        elif (self.ctrl_mode == p.VELOCITY_CONTROL):
            raise RuntimeError("Not implemented yet...")
        elif (self.ctrl_mode == p.TORQUE_CONTROL):
            raise RuntimeError("Not implemented yet...")


    # =======================================
    def setCtrlMode(self, ctrl_mode):

        if ctrl_mode not in (p.POSITION_CONTROL, p.VELOCITY_CONTROL, p.TORQUE_CONTROL):
            raise RuntimeError("Unsupported ctrl mode '" + ctrl_mode + "'...")

        self.ctrl_mode = ctrl_mode

    # =======================================
    def reset(self, j_pos):

        for id, pos in zip(self.joint_id, j_pos):
            p.resetJointState(self.robot_id, id, pos)

        self.__readState()

    # =======================================
    def getNumJoints(self):

        return self.N_JOINTS

    # =======================================
    def setJointsPosition(self, j_pos_cmd):

        self.joints_pos_cmd = j_pos_cmd.copy()

    # =====================================
    def getJointsPosition(self):

        return self.joints_pos

    def getJointsVelocity(self):

        return self.joints_vel

    # =======================================
    def str(self):

        return self.__str__()

    # ---------------------------------
    # -----------  Private  -----------
    # ---------------------------------

    def __readState(self):

        j_states = p.getJointStates(self.robot_id, self.joint_id)
        for i in range(self.N_JOINTS):
            self.joints_pos[i] = j_states[i][0]
            self.joints_vel[i] = j_states[i][1]

    def __str__(self):

        return "========= Robot =======" + \
               "\n+ joints:" + str(self.joint_names) + \
               "\n+ ee_link:" + self.ee_link


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
    j_pos = robot.getJointsPosition()

    Time = np.array([t])
    jpos_data = j_pos

    for i in range(10000):
        env.step()
        robot.update()
        j_pos += 0.001*np.ones(n_joints)
        robot.setJointsPosition(j_pos)

        t += Ts
        Time = np.append(Time, t)
        jpos_data = np.column_stack( (jpos_data, robot.getJointsPosition()) )

    # time.sleep(1. / 240.)

    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos, cubeOrn)

    fig, axs = plt.subplots(n_joints,1)
    for i, ax in enumerate(axs):
        ax.plot(Time, jpos_data[i,:], linewidth=2)
        ax.set_ylabel("j" + str(i))
    axs[0].set_title("Robot joints trajectories")
    axs[n_joints-1].set_xlabel("time [s]")

    plt.show()

    dummy = input('Press enter to continue...')


