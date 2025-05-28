import os
import subprocess

def main():
    if not os.path.isdir("encoder4editing"):
        subprocess.run(["git", "clone", "https://github.com/Nimalesh/encoder4editing.git"])
    os.makedirs("encoder4editing/input", exist_ok=True)
    os.makedirs("encoder4editing/pretrained_models", exist_ok=True)
    os.makedirs("encoder4editing/outputs/inversion", exist_ok=True)

    # Download e4e pretrained model if not exists
    pt_path = "encoder4editing/pretrained_models/e4e_ffhq_encode.pt"
    if not os.path.exists(pt_path):
        subprocess.run([
            "wget", "-O", pt_path,
            "https://huggingface.co/AIRI-Institute/HairFastGAN/resolve/main/pretrained_models/encoder4editing/e4e_ffhq_encode.pt"
        ])

    # Download shape predictor if not present
    predictor_path = "encoder4editing/pretrained_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        os.chdir("encoder4editing/pretrained_models")
        subprocess.run(["wget", "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"])
        subprocess.run(["bunzip2", "shape_predictor_68_face_landmarks.dat.bz2"])
        os.chdir("../../../")
    print("encoder4editing setup complete.")

if __name__ == "__main__":
    main()
