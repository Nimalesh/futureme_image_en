import os
import subprocess
import argparse
import shutil
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    image_name = os.path.basename(args.image_path)

    # Ensure output folders exist
    os.makedirs('output/aligned_images', exist_ok=True)
    os.makedirs('output/inversion', exist_ok=True)
    os.makedirs('output/raw_images', exist_ok=True)  # For raw image backup, optional

    # Copy input image to output/raw_images (optional, keeps everything in output/)
    raw_image_path = os.path.join('output', 'raw_images', image_name)
    shutil.copy(args.image_path, raw_image_path)
    print(f"Copied {args.image_path} to {raw_image_path}")

    # Also copy to stylegan-encoder/raw_images (required for script compatibility)
    sg_raw_path = os.path.join('stylegan-encoder', 'raw_images', image_name)
    shutil.copy(args.image_path, sg_raw_path)

    # Align images, output to output/aligned_images
    subprocess.run([
        sys.executable, "align_images.py", "../output/raw_images", "../output/aligned_images"
    ], cwd='stylegan-encoder')
    print("Alignment complete.")

    # Copy aligned image to encoder4editing/input/
    aligned_img_path = os.path.join('output', 'aligned_images', image_name)
    encoder_input_path = os.path.join('encoder4editing', 'input', image_name)
    shutil.copy(aligned_img_path, encoder_input_path)
    print(f"Copied aligned image to {encoder_input_path}")

    # Run encoder4editing inference, output to output/inversion
    subprocess.run([
        sys.executable, "-m", "scripts.inference",
        "--images_dir", "input",
        "--save_dir", "../output/inversion",
        "--align",
        "pretrained_models/e4e_ffhq_encode.pt"
    ], cwd="encoder4editing")
    print("Inversion complete. See output/inversion for latent files.")

if __name__ == "__main__":
    main()
