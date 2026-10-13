conda env create -f requirements_litellm_ubuntu22.yml -p venvs/litellm_u22_py312
conda activate venvs/litellm_u22_py312
conda install conda-pack
conda-pack -p ./venvs/litellm_u22_py312 -o litellm_u22_py312.tar.gz
# target machine
mkdir -p ~/venvs/litellm_py312
tar -xzf litellm_u22_py312.tar.gz -C ~/venvs/litellm_py312
source ~/venvs/litellm_py312/bin/activate

