# StyleCLIP Local Pipeline

This repository enables you to run StyleCLIP (text-based editing of faces) **locally or in VS Code**, with simple scripts.

---

## 1. Clone and Setup

```sh
git clone https://github.com/Nimalesh/StyleCLIP.git
# Place all scripts and requirements.txt from this repo into the same folder.
python styleclip_setup.py
```
installs requirements from requirements.txt

Downloads StyleGAN2 and IRSE-50 weights automatically to the StyleCLIP folder

Run your inversion pipeline (e.g. encoder4editing) to produce a .pt latent code, such as output/inversion/pic_01.pt

### Run StyleCLIP Inference
```sh
python styleclip_inference.py --description "A person with purple hair and fuller lips" --latent_path output/inversion/pic_01.pt --optimization_steps 40
```

### Default
```sh
python styleclip_inference.py
```

### Results
Output images will appear in StyleCLIP/results/ as pic_01_styleclip.png (matching your latent filename).

### Notes
All weights are downloaded automatically by the setup script.

You can edit any text prompt, latent file, or optimization steps as needed.

Make sure to run scripts in a Python 3.7+ environment with pip.

