# GPU Programming

## Setup

Start by cloning the repository:

```bash
git clone https://github.com/alangshur/gpu-programming.git
cd gpu-programming
```

Then, run the `setup.sh` script to configure the environment:

```bash
source setup.sh
```

## NVIDIA Versions

Check `nvcc` version for version 11.8+:

```bash
nvcc --version
```

Check the NVIDIA driver version and compute capability for versions 12.2+ and 8.9+, respectively:

```bash
nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Running

To run the tests, first use CMake to generate the build system files in the `build` directory:

```bash
mkdir build
cd build
cmake ..
```

Then, build the project:

```bash
make
```

Finally, run the tests:

```bash
ctest
ctest --verbose # for more detailed output with all outputs
ctest --output-on-failure # to show output only on failure
```