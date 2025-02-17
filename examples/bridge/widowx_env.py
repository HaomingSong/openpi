import time

import cv2
import gym
import numpy as np
from pyquaternion import Quaternion
from widowx_envs.widowx_env_service import WidowXClient


def listdict2dictlist(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}


class RHCWrapper(gym.Wrapper):
    """
    Performs receding horizon control. The policy returns `pred_horizon` actions and
    we execute `exec_horizon` of them.
    """

    def __init__(self, env: gym.Env, exec_horizon: int):
        super().__init__(env)
        self.exec_horizon = exec_horizon

    def step(self, actions):
        if self.exec_horizon == 1 and len(actions.shape) == 1:
            actions = actions[None]
        assert len(actions) >= self.exec_horizon
        rewards = []
        observations = []
        infos = []

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            infos.append(info)

            if done or trunc:
                break

        infos = listdict2dictlist(infos)
        infos["rewards"] = rewards
        infos["observations"] = observations

        return obs, np.sum(rewards), done, trunc, infos


def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    default_rotation = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(matrix=default_rotation)
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose


def wait_for_obs(widowx_client):
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs


def convert_obs(obs, im_size, *, flip=False, bgr2rgb=False):
    image_obs = (obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255).astype(np.uint8)

    if flip:
        image_obs = np.flip(image_obs, axis=0)
        full_image = np.flip(obs["full_image"], axis=0)
    if bgr2rgb:
        image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2RGB)
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
    # add padding to proprio to match training
    proprio = np.concatenate([obs["state"][:6], [0], obs["state"][-1:]])

    return {
        "image_primary": image_obs,
        "proprio": proprio,
        "full_image": full_image,
    }


def null_obs(img_size):
    return {
        "image_primary": np.zeros((img_size, img_size, 3), dtype=np.uint8),
    }


class WidowXGym(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    Needed to use Gym wrappers.
    """

    def __init__(
        self,
        env_params: dict,
        host: str = "localhost",
        port: int = 5556,
        im_size: int = 256,
        *,
        blocking: bool = True,
        sticky_gripper_num_steps: int = 1,
    ):
        self.widowx_client = WidowXClient(host, port)
        self.widowx_client.init(env_params, image_size=im_size)
        self.env_params = env_params
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(low=np.ones((8,)) * -1, high=np.ones((8,)), dtype=np.float64),
            }
        )
        self.action_space = gym.spaces.Box(low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64)
        self.sticky_gripper_num_steps = sticky_gripper_num_steps
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

    def step(self, action):
        # sticky gripper logic
        if (action[-1] < 0.5) != self.is_gripper_closed:
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.is_gripper_closed = not self.is_gripper_closed
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.is_gripper_closed else 1.0

        self.widowx_client.step_action(action, blocking=self.blocking)

        raw_obs = self.widowx_client.get_observation()

        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True
            obs = null_obs(self.im_size)  # obs with all zeros
        else:
            obs = convert_obs(
                raw_obs,
                self.im_size,
                flip=self.env_params["camera_topics"][0]["name"] == "/D435/color/image_raw",
                bgr2rgb=self.env_params["camera_topics"][0]["name"] == "/D435/color/image_raw",
            )

        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.widowx_client.reset()

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        raw_obs = wait_for_obs(self.widowx_client)
        obs = convert_obs(
            raw_obs,
            self.im_size,
            flip=self.env_params["camera_topics"][0]["name"] == "/D435/color/image_raw",
            bgr2rgb=self.env_params["camera_topics"][0]["name"] == "/D435/color/image_raw",
        )

        return obs, {}
