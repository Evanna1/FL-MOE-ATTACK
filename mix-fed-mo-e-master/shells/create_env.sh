conda create -n flwr python=3.12 -y
conda activate flwr
pip install uv
uv pip install flwr-datasets[vision] flwr[simulation] torch torchvision datasets transformers matplotlib seaborn setuptools modelscope huggingface_hub ipykernel jupyter ipywidgets nvitop evaluate pytest

# download model
export HF_ENDPOINT=https://hf-mirror.com
hf download google/switch-base-8 --local-dir google/switch-base-8