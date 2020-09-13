#############################################################
# Title: Custom gym environment for the four legged robot
# Author: Kez Smithson Whitehead
# Last updated: 2nd September 2018
#############################################################

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces
import math
import os

rad2deg = 180./(np.pi)
deg2rad = (np.pi)/180.




# ---------------------------------------------------------
# Converts rotation matrix to euler angles
# output [x y z]
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0] * R[0] + R[3] * R[3])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[7], R[8])
        y = np.arctan2(-R[6], sy)
        z = np.arctan2(R[3], R[0])
    else:
        x = np.arctan2(-R[5], R[4])
        y = np.arctan2(-R[6], sy)
        z = 0

    return np.array([x, y, z]) * 180 / 3.142
# ----------------------------------------------------------

# ----------------------------------------------------------
# Motor Direction Function
# ==========================================================
# Determines the direction from the velocity
def motorDirect(velocity):
    tol = 0.05
    if velocity < -tol:
        direction = -1
    elif velocity > tol:
        direction = 1
    else:
        direction = 0

    return direction
# ----------------------------------------------------------








class RealEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        #mujoco_env.MujocoEnv.__init__(self, '/home/brl/.mujoco/mujoco200/model/real.xml', 5)
        dirpath = os.path.dirname(os.path.realpath(__file__))
        fullpath = os.path.join(dirpath, "assets/real.xml")
        mujoco_env.MujocoEnv.__init__(self, fullpath, 5)
        utils.EzPickle.__init__(self)
        #metadata = {'render.modes': 'rgb_array'}

        # self.obsevation_space = spaces.Box(low=np.array([-90.*deg2rad, 0.*deg2rad]),
        #                                high=np.array([90.*deg2rad, 90.*deg2rad]))

        self.action_space = spaces.Box(low=np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]),
                                       high=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        quat_Matrix_Before = self.data.sensordata
        # euler_Matrix_Before = rotationMatrixToEulerAngles(quat_Matrix_Before)

        self.do_simulation(a, self.frame_skip)

        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        quat_Matrix_After = self.data.sensordata
        # euler_Matrix_After = rotationMatrixToEulerAngles(quat_Matrix_After)

        forwardReward = (xposafter - xposbefore)/self.dt
        sidePunishment = (yposafter - yposbefore)/self.dt
        # pitchReward = (euler_Matrix_After[0] - euler_Matrix_Before[0])/self.dt
        # rollReward = (euler_Matrix_After[1] - euler_Matrix_Before[1])/self.dt
        # yawReward = (euler_Matrix_After[2] - euler_Matrix_Before[2])/self.dt
        reward = forwardReward #- sidePunishment#- pitchReward - rollReward - yawReward


        # notdone = np.isfinite(ob).all()
        # done = not notdone
        # qpos = self.sim.data.qpos
        # bob, fred, ang = self.sim.data.qpos[0:3]
        # print('position x: ', xposafter, '  y: ', yposafter, '  reward: ', reward)
        # print ("Euler orietation torso: ", quat_Matrix_After)

        # termination criteria
        orientation = self.data.sensordata
        orientation = np.round(orientation,2)
        done = bool( (orientation[1] < -0.3) or (orientation[1] > 0.3) )

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forwardReward)
        # return ob, reward, done, done

    # Everything the robot observes-------------------------------------------------------------------------------------
    def _get_obs(self):

        # Joint Angles speed up by removing rad to deg
        a0 = self.sim.data.qpos[7] * rad2deg  # right hip out
        a1 = self.sim.data.qpos[8] * rad2deg
        a2 = self.sim.data.qpos[9] * rad2deg  # left hip out self.get_body_com("torso")
        a3 = self.sim.data.qpos[10] * rad2deg
        a4 = self.sim.data.qpos[11] * rad2deg  # right hip out
        a5 = self.sim.data.qpos[12] * rad2deg
        a6 = self.sim.data.qpos[13] * rad2deg  # left hip out self.get_body_com("torso")
        a7 = self.sim.data.qpos[14] * rad2deg
        a8 = self.sim.data.qpos[15] * rad2deg  # right hip out
        a9 = self.sim.data.qpos[16] * rad2deg
        a10 = self.sim.data.qpos[17] * rad2deg  # left hip out self.get_body_com("torso")
        a11 = self.sim.data.qpos[18] * rad2deg

        # Orientation sensor
        orientation = self.data.sensordata
        orientation = np.round(orientation,2)
        quatr = orientation[0]
        quat1 = orientation[1]
        quat2 = orientation[2]
        quat3 = orientation[3]
        OBS = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, quatr, quat1, quat2, quat3]
        return np.array(OBS)

    # Reset by setting velocity = 0 and position to xml model values----------------------------------------------------

    def reset_model(self):
        qpos = self.init_qpos   # Initial + randomness

        qpos = qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)  # add randomness

        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    # Camera point--------------------------------------------------------------------------------------------------------
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
