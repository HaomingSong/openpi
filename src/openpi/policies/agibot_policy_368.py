import dataclasses

import einops
import numpy as np
import torch

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/primary_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wirst_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_yellow_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AgibotInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # NOTE: for bridge dataset at IPEC-COMMUNITY/bridge_orig_lerobot, the state is 8-dim.
        # Get the state. We are padding from 8 to the model action dim.
        state = torch.concat([data["observation/state/joint"], data["observation/state/effector"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        head_image = _parse_image(data["observation/images/head"])
        hand_left_image = _parse_image(data["observation/images/hand_left"])
        hand_right_image = _parse_image(data["observation/images/hand_right"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": hand_left_image,
                "right_wrist_0_rgb": hand_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions/joint" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            # Scale effector actions from [0, 1] to [35, 120]
            scaled_effector = data["actions/effector"] * (120 - 35) + 35
            actions = torch.concat(
                [
                    data["actions/joint"],
                    scaled_effector,
                ],
                dim=-1,
            )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AgibotOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :16])}
