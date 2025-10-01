python3 -m pip install -U "huggingface_hub[cli]"
mkdir -p ../data
hf download nabil-m/bcss --repo-type dataset --local-dir ../data/bcss
