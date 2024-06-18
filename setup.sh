#!/bin/bash

# setup git config
read -p "Enter your Git email: " git_email
git config --global user.email "$git_email"

# install base packages
apt update
apt install -y htop nvtop vim make clang clang-format gnupg

# install nvidia nsight systems
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli

# create virtual environment
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# install python packages
source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv --display-name "Default"
