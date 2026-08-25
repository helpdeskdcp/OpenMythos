from pathlib import Path
from typing import Union

import torch

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.variants import mythos_100m


def load_mythos_100m(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> OpenMythos:
    """
    Build the 100M GQA variant and load a checkpoint into it with strict=True.

    Args:
        checkpoint_path -- path to a .pt file containing a flat state_dict
            (tensor name -> tensor), as produced by ``torch.save(model.state_dict(), ...)``
        device -- device to map the checkpoint tensors and model onto

    Returns:
        OpenMythos model in eval() mode with the checkpoint weights loaded

    Raises:
        RuntimeError -- if any key is missing or unexpected (strict=True)
    """
    cfg = mythos_100m()
    model = OpenMythos(cfg)

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(
            f"Expected a state_dict (mapping of tensor name -> tensor) at "
            f"{checkpoint_path}, got {type(state)}"
        )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
