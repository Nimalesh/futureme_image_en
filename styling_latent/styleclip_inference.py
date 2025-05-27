import argparse
import torch
from optimization.run_optimization import main
from argparse import Namespace
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import os
import sys

def run_styleclip(description, latent_path, optimization_steps):
    # Auto-set output name based on input latent
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
    # Set seed for reproducibility
    torch.manual_seed(1)
    # Run StyleCLIP
    print("\n[StyleCLIP] Running optimization with the following arguments:")
    for k, v in args.items():
        print(f"  {k}: {v}")
    result = main(Namespace(**args))
    # Save result image
    result_image = ToPILImage()(make_grid(result.detach().cpu(), normalize=True, padding=0))
    result_image.save(out_path)
    print(f"\n[StyleCLIP] Saved result image to: {out_path}")

def parse_args_or_prompt():
    parser = argparse.ArgumentParser(description="StyleCLIP Inference Script")
    parser.add_argument("--description", type=str, help="Text prompt for editing (e.g., 'A person with purple hair and fuller lips')")
    parser.add_argument("--latent_path", type=str, help="Path to input latent .pt file")
    parser.add_argument("--optimization_steps", type=int, default=40, help="Number of optimization steps (default: 40)")
    args = parser.parse_args()
    # Interactive fallback if not all args provided
    if not args.description:
        args.description = input("Enter a text description (e.g. 'A person with purple hair and fuller lips'): ")
    if not args.latent_path:
        args.latent_path = input("Enter the path to your latent .pt file: ")
    if not args.optimization_steps:
        try:
            steps = input("Enter the number of optimization steps [default 40]: ")
            args.optimization_steps = int(steps) if steps else 40
        except Exception:
            args.optimization_steps = 40
    return args

if __name__ == "__main__":
    # Ensure we are in the StyleCLIP directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, "StyleCLIP"))
    args = parse_args_or_prompt()
    # Sanity check: does latent file exist?
    if not os.path.isfile(args.latent_path):
        print(f"[Error] Latent file not found: {args.latent_path}")
        sys.exit(1)
    run_styleclip(args.description, args.latent_path, args.optimization_steps)
