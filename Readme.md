# 1. Clone your repo
```sh
!git clone https://github.com/krisu-ai/future-me.git
%cd future-me
```

# 2. Install requirements (optional but recommended)
```sh
!pip install -r requirements.txt
```

# 3. Upload your image
```sh
from google.colab import files
uploaded = files.upload()
```
# 4. Run setups and pipeline!
```sh
!python setup_stylegan_encoder.py
!python setup_encoder4editing.py
```
# 5. (Assume the uploaded image is named pic.png)
```sh
!python run_full_pipeline.py --image_path pic.png
```
# 6. List outputs
```sh
!ls -lh encoder4editing/outputs/inversion/
```