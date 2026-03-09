import torch


MODEL_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def _unwrap_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        return state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict and isinstance(state_dict["model"], dict):
        return state_dict["model"]
    return state_dict


def load_vggt_weights(model, pt_path=None):
    """Load weights from a local checkpoint when provided, otherwise download them."""
    if pt_path:
        print(f"Loading model weights from local file: {pt_path}")
        state_dict = torch.load(pt_path, map_location="cpu")
    else:
        print(f"Loading model weights from URL: {MODEL_URL}")
        state_dict = torch.hub.load_state_dict_from_url(MODEL_URL, map_location="cpu")

    model.load_state_dict(_unwrap_state_dict(state_dict))
    return model