import argparse
import torch
from optimization.run_optimization import main
from argparse import Namespace
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import os

def run_styleclip(description, latent_path, optimization_steps):
    result_name = os.path.splitext(os.path.basename(latent_path))[0]
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{result_name}_styleclip.png")

    args = {
        "description": description,
        "ckpt": "stylegan2-ffhq-config-f.pt",
        "stylegan_size": 1024,
        "lr_rampup": 0.05,
        "lr": 0.1,
        "step": optimization_steps,
        "mode": "edit",
        "l2_lambda": 0.008,
        "id_lambda": 0.005,
        "work_in_stylespace": False,
        "latent_path": latent_path,
        "truncation": 0.7,
        "save_intermediate_image_every": 1,
        "results_dir": output_dir,
        "ir_se50_weights": "model_ir_se50.pth"
    }
    torch.manual_seed(1)
    print("\n[StyleCLIP] Running with args:", args)
    result = main(Namespace(**args))
    result_image = ToPILImage()(make_grid(result.detach().cpu(), normalize=True, padding=0))
    result_image.save(out_path)
    print(f"[StyleCLIP] Saved result image to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--latent_path", type=str, required=True)
    parser.add_argument("--optimization_steps", type=int, default=40)
    args = parser.parse_args()
    run_styleclip(args.description, args.latent_path, args.optimization_steps)
