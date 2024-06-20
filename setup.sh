#!/bin/bash

# install core packages
apt update
apt install -y htop nvtop vim make clang clang-format gnupg

# install libraries
apt install -y libgtest-dev