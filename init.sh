#!/usr/bin/env bash

# #################### get env directories

# CONDA ENV
CONDA_NEW_ENV=wbdc
ENV_ROOT=./

# #################### use tsinghua conda sources
# conda config --show channels
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
# conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
# conda config --set show_channel_urls yes

# #################### create conda env and activate
# conda in shell propagation issue - https://stackoverflow.com/questions/52779016/conda-command-working-in-command-prompt-but-not-in-bash-script/52813960#52813960
# shellcheck source=/opt/conda/etc/profile.d/conda.sh

# ###### create env and activate

# create env by prefix
conda create --prefix ${ENV_ROOT}/envs/${CONDA_NEW_ENV} python=3.6
source activate ${ENV_ROOT}/envs/${CONDA_NEW_ENV}

conda info --envs

pip install -r requirements.txt -i https://mirrors.tencentyun.com/pypi/simple