#!/bin/bash

# add github email
git config --global user.email "adlangshur@gmail.com"

# install core packages
apt update
apt install -y htop nvtop vim make clang clang-format gnupg

# install libraries
apt install -y libgtest-dev