# GPU Programming

## Setup

Start by cloning the repository:

```bash
git clone https://github.com/alangshur/gpu-programming.git
cd gpu-programming
```

Then, run the `setup.sh` script to configure your virtual environments and run system checks:

```bash
source setup.sh
```

## NVIDIA Versions

Check `nvcc` version (you should have version 11.8+):

```bash
nvcc --version
```

Check the NVIDIA driver version and compute capability (you should have 12.2+ and 8.9+):

```bash
nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Running

To compile our CUDA program, we can use `nvcc` like so:

```bash
nvcc -o build/saxpy src/saxpy.cu
```

Then, to run the executable, you can do the following:

```bash
./build/saxpy
```

## Profiling

You can download NVIDIA Nsight Systems [here](https://developer.nvidia.com/nsight-systems/get-started#latest-version) for Mac. Then, you can connect it to the instance via SSH and target an executable.