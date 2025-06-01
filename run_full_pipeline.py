import os
import subprocess
import argparse
import shutil
import sys
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    image_name = os.path.basename(args.image_path)

    # Copy to stylegan-encoder/raw_images
    sg_raw_path = os.path.join('stylegan-encoder', 'raw_images', image_name)
    shutil.copy(args.image_path, sg_raw_path)
    print(f"Copied {args.image_path} to {sg_raw_path}")

    # Align images
    subprocess.run([
        sys.executable, "align_images.py", "raw_images", "aligned_images"
    ], cwd='stylegan-encoder')
    print("Alignment complete.")

    # Find the aligned image(s)
    image_name_wo_ext = os.path.splitext(image_name)[0]
    aligned_img_pattern = os.path.join('stylegan-encoder', 'aligned_images', f"{image_name_wo_ext}_*.png")
    aligned_files = glob.glob(aligned_img_pattern)

    if not aligned_files:
        print(f"No aligned images found for {image_name}")
        sys.exit(1)

    aligned_img_path = aligned_files[0]  # Use the first aligned face found
    encoder_input_path = os.path.join('encoder4editing', 'input', os.path.basename(aligned_img_path))
    shutil.copy(aligned_img_path, encoder_input_path)
    print(f"Copied aligned image to {encoder_input_path}")

    # Run encoder4editing inference
    subprocess.run([
        sys.executable, "-m", "scripts.inference",
        "--images_dir", "input",
        "--save_dir", "outputs/inversion",
        "--align",
        "pretrained_models/e4e_ffhq_encode.pt"
    ], cwd="encoder4editing")
    print("Inversion complete. See encoder4editing/outputs/inversion")

if __name__ == "__main__":
    main()
