#!/bin/bash

read -p "Enter your Git email: " git_email
git config --global user.email "$git_email"

apt update
apt install -y htop nvtop vim make

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=venv --display-name "Default"
