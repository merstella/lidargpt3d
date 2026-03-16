# import timm
# import torch
# from point_encoder import PointcloudEncoder

# # pretrained_pc = '/media/ivsr/IVSR/Trung/Lidar-LLM/MiniGPT-3D/minigpt4/models/uni3d/weights/uni3dtiny/model.pt'

# # point_transformer = timm.create_model('eva02_tiny_patch14_224', checkpoint_path=pretrained_pc, drop_path_rate=0.2)
# ckpt = torch.load(
#     '/media/ivsr/IVSR/Trung/Lidar-LLM/MiniGPT-3D/minigpt4/models/uni3d/weights/uni3dtiny/model.pt',
#     map_location='cpu'
# )


#     # create whole point cloud encoder
# # point_encoder = PointcloudEncoder(point_transformer)
# point_encoder = PointcloudEncoder(ckpt)


import torch
import timm
from point_encoder import PointcloudEncoder


def load_uni3d_pc_encoder(
    ckpt_path,
    eva_model_name="eva02_tiny_patch14_224",
    drop_path_rate=0.2,
    device="cpu"
):
    """
    Load ONLY pretrained PointcloudEncoder from Uni3D checkpoint.
    """

    # -------------------------------------------------
    # 1. Create EVA backbone (NO checkpoint here)
    # -------------------------------------------------
    point_transformer = timm.create_model(
        eva_model_name,
        pretrained=False,
        drop_path_rate=drop_path_rate
    )

    # -------------------------------------------------
    # 2. Create PointcloudEncoder (architecture first!)
    # -------------------------------------------------
    point_encoder = PointcloudEncoder(point_transformer)

    # -------------------------------------------------
    # 3. Load Uni3D checkpoint
    # -------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["module"] if "module" in ckpt else ckpt

    # -------------------------------------------------
    # 4. Filter ONLY point_encoder weights
    # -------------------------------------------------
    pc_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("point_encoder."):
            new_k = k.replace("point_encoder.", "")
            pc_state_dict[new_k] = v

    # -------------------------------------------------
    # 5. Load into PointcloudEncoder
    # -------------------------------------------------
    point_encoder.load_state_dict(
        pc_state_dict,
        strict=False
    )

    print("✅ Uni3D PC encoder loaded")
    # print(f"Missing keys   : {len(missing)}")
    # print(f"Unexpected keys: {len(unexpected)}")

    return point_encoder


# -------------------------------------------------
# Example usage
# -------------------------------------------------
if __name__ == "__main__":
    ckpt_path = "/media/ivsr/IVSR/Trung/Lidar-LLM/MiniGPT-3D/minigpt4/models/uni3d/weights/uni3dtiny/model.pt"

    pc_encoder = load_uni3d_pc_encoder(ckpt_path)

    # Optional: freeze backbone
    for name, p in pc_encoder.named_parameters():
        if not name.startswith("trans2embed"):
            p.requires_grad = False

    print("🚀 PC encoder ready to use")
