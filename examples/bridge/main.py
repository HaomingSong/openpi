"""
This script shows how we evaluated a finetuned Octo model on a real WidowX robot. While the exact specifics may not
be applicable to your use case, this script serves as a didactic example of how to use Octo in a real-world setting.

If you wish, you may reproduce these results by [reproducing the robot setup](https://rail-berkeley.github.io/bridgedata/)
and installing [the robot controller](https://github.com/rail-berkeley/bridge_data_robot)
"""

import contextlib
import dataclasses
from datetime import datetime
import pathlib
import signal
import time

import cv2
import imageio
import numpy as np
import pandas as pd
import tqdm
import tyro
from widowx_env import RHCWrapper
from widowx_env import WidowXGym
from widowx_envs.widowx_env_service import WidowXConfigs


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    im_size: int = 224
    action_horizon: int = 5

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    robot_ip: str = "localhost"  # IP address of the robot
    robot_port: int = 5556  # Port of the robot
    initial_eep: tuple[float, float, float] = (0.3, 0.0, 0.15)  # Initial position
    blocking: bool = False  # Use the blocking controller
    max_timesteps: int = 120  # Number of timesteps to run
    default_instruction: str = "Lift the carrot in the plate"  # Default instruction

    #################################################################################################################
    # Utils
    #################################################################################################################
    show_image: bool = False  # Show image
    video_save_path: pathlib.Path = pathlib.Path("data/bridge/videos")  # Path to save videos
    results_save_path: pathlib.Path = pathlib.Path("data/bridge/results")  # Path to save results


##############################################################################
STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
Be sure to use a step duration of 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.2
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/D435/color/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def eval_bridge(args: Args) -> None:
    # set up the widowx client
    start_state = np.concatenate([args.initial_eep, (0, 0, 0, 1)])
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["start_state"] = list(start_state)

    env = WidowXGym(
        env_params,
        host=args.robot_ip,
        port=args.robot_port,
        im_size=args.im_size,
        blocking=args.blocking,
        sticky_gripper_num_steps=STICKY_GRIPPER_NUM_STEPS,
    )
    if not args.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE
    results_df = pd.DataFrame(columns=["success", "duration", "video_filename"])
    # policy_client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # switch TemporalEnsembleWrapper with RHCWrapper for receding horizon control
    env = RHCWrapper(env, args.action_horizon)

    while True:
        # reset env
        obs, _ = env.reset()
        time.sleep(2.0)

        if input("Use default instruction? (default y) [y/n]").lower() == "n":
            args.default_instruction = input("Enter instruction: ")

        # do rollout
        images = []
        images.append(obs["full_image"])
        last_tstep = time.time()
        bar = tqdm.tqdm(
            range(args.max_timesteps),
            position=0,
            leave=True,
            ncols=80,
            desc="Rollout steps",
        )

        for t_step in bar:
            try:
                bar.set_description(f"Step {t_step}/{args.max_timesteps}")
                if args.show_image:
                    cv2.imshow("img_view", obs["full_image"])
                    cv2.waitKey(1)

                # # Send websocket request to policy server
                # # TODO: implement the request_data
                # request_data = {
                # }
                # with prevent_keyboard_interrupt():
                #     # this returns action chunk [10, 7] of 10 end effector pose (6) + gripper position (1)
                #     forward_pass_time = time.time()
                #     pred_action_chunk = policy_client.infer(request_data)["actions"]
                #     assert pred_action_chunk.shape == (10, 7)
                #     print("request action time: ", time.time() - forward_pass_time)

                # # clip all dimensions of action to [-1, 1]
                # pred_action_chunk = np.clip(pred_action_chunk, -1, 1)
                # # TODO: whether unnorm the action

                pred_action_chunk = np.zeros((10, 7))  # dummy action
                # perform environment step
                start_time = time.time()
                obs, _, _, truncated, infos = env.step(pred_action_chunk)
                print("step time: ", time.time() - start_time)

                # recording history images
                for history_obs in infos["observations"]:
                    image = history_obs["full_image"]
                    images.append(image)
                if truncated:
                    break

                # match the step duration
                elapsed_time = time.time() - last_tstep
                if elapsed_time < STEP_DURATION:
                    time.sleep(STEP_DURATION - elapsed_time)
            except KeyboardInterrupt:
                break

        # saving video
        args.video_save_path.mkdir(parents=True, exist_ok=True)
        curr_time = datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d_%H:%M:%S")
        save_path = args.video_save_path / f"video_{curr_time}.mp4"
        video = np.stack(images)
        imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)

        # logging rollouts
        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        [
                            {
                                "instruction": args.default_instruction,
                                "success": success,
                                "duration": t_step,
                                "video_filename": save_path,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

        if input("Do one more eval (default y)? [y/n]").lower() == "n":
            break

    # save results
    args.results_save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=datetime.UTC).strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = args.results_save_path / f"eval_{timestamp}.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    eval_bridge(args)
