
# Real-world Bridge with WidowX-250

This example runs the real-world [BridgeData v2](https://rail-berkeley.github.io/bridgedata/)

First, setup the robot with the [official instruction](https://github.com/rail-berkeley/bridge_data_robot)

Follow the [Server-client method](https://github.com/rail-berkeley/bridge_data_robot?tab=readme-ov-file#server-client-method) run the robot and verify that everything is set up correctly.

NOTE:
Before running `python widowx_envs/widowx_env_service.py --client` make sure the [camera topic](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/widowx_env_service.py#L36) is correct
In our seeting, we use realsense as the static camra, change the [camera topic](https://github.com/rail-berkeley/bridge_data_robot/blob/main/widowx_envs/widowx_envs/widowx_env_service.py#L36) to `"camera_topics": [{"name": '/D435/color/image_raw'}]`


Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/bridge/.venv
source examples/bridge/.venv/bin/activate

# install deps
uv pip sync examples/bridge/requirements.txt third_party/bridge_data_robot/widowx_envs/requirements.txt --index-strategy=unsafe-best-match
uv pip install -e  third_party/edgeml
uv pip install -e third_party/bridge_data_robot/widowx_envs
uv pip install -e packages/openpi-client

```

```bash
# check installation

```
# Run the simulation
python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```