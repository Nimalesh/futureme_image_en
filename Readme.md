# Encoder Pipeline: StyleGAN-Encoder + Encoder4Editing

A command-line workflow for facial image alignment and inversion using [StyleGAN-Encoder](https://github.com/Nimalesh/stylegan-encoder) and [encoder4editing](https://github.com/Nimalesh/encoder4editing), ready for use in VS Code or any terminal.  
This mirrors a Colab workflow, but is easy to run and reproduce anywhere.


## Setup

### 1. Clone Repositories

```sh
git clone https://github.com/Nimalesh/stylegan-encoder.git
git clone https://github.com/Nimalesh/encoder4editing.git
```
### 2. Create a Virtual env

```sh
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Setup

```sh
python setup_stylegan_encoder.py
python setup_encoder4editing.py
```

These scripts:

Create necessary folders in both repos.

Download pretrained model weights and the Dlib shape predictor.

Install the ninja build tool if missing.

### Run

```sh
python run_full_pipeline.py --image_path pic_01.png
```

<!-- This will:

Copy your image to stylegan-encoder/raw_images/

Align the face with StyleGAN-Encoder

Copy the aligned image to encoder4editing/input/

Run encoder4editing inversion -->

