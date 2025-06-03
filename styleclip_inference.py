import argparse
import torch
from optimization.run_optimization import main
from argparse import Namespace
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import os

def run_styleclip(description, latent_path, optimization_steps, l2_lambda, id_lambda, lr, lr_rampup, truncation):
    result_name = os.path.splitext(os.path.basename(latent_path))[0]
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{result_name}_styleclip.png")

    args = {
        "description": description,
        "ckpt": "stylegan2-ffhq-config-f.pt",
        "stylegan_size": 1024,
        "lr_rampup": lr_rampup,
        "lr": lr,
        "step": optimization_steps,
        "mode": "edit",
        "l2_lambda": l2_lambda,
        "id_lambda": id_lambda,
        "work_in_stylespace": False,
        "latent_path": latent_path,
        "truncation": truncation,
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
    # New arguments with defaults (matching your previous values)
    parser.add_argument("--l2_lambda", type=float, default=0.008, help="L2 regularization lambda")
    parser.add_argument("--id_lambda", type=float, default=0.005, help="Identity loss lambda")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--lr_rampup", type=float, default=0.05, help="LR rampup")
    parser.add_argument("--truncation", type=float, default=0.7, help="Truncation parameter")
    args = parser.parse_args()
    run_styleclip(
        args.description, args.latent_path, args.optimization_steps,
        args.l2_lambda, args.id_lambda, args.lr, args.lr_rampup, args.truncation
    )
