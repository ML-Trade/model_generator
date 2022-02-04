# model_generator
Generates models and uploads the best to the model repo

## Data
Data is saved to google drive and locally. Data is initially obtained using the Polygon.io API. It is saved to Google drive in a CSV format

## Prerequisites

- Have python 3.9 installed
- Have VS Build tools 2014+ and Visual C++ build tools (with windows sdk)
- If using Windows, do not run this in WSL, you may run into GPU issues

### Install in this order:

- CUDA toolkit: https://developer.nvidia.com/cuda-toolkit-archive
- CudNN: https://developer.nvidia.com/cudnn (install instructions: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)
- (optional but improves gpu latency) TensorRT: https://developer.nvidia.com/tensorrt-getting-started (install instructions: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-zip)

More up to date instructions may live here: https://www.tensorflow.org/install/gpu