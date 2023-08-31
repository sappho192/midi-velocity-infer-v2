# midi-velocity-infer-v2

MIDI Velocity inference based on Seq2seq and Luong Attention.
This is the implementation code of my recent paper, which is to be presented in the 2023 Autumn meeting of the Acoustical Society of Japan [[Link](https://acoustics.jp/annualmeeting/past-meetings/)].

## 1. Tested running environment

- Operating System: Linux(Ubuntu 20.04 LTS)
- GPU: NVIDIA RTX A6000 48GB
- NVIDIA Driver: 515.65.01
- CUDA: 11.7

## 2. Environment Guide

### 1) Minimal installation (demo.ipynb)

In bash,

- Install .NET 6 (Refer this [[Link](https://docs.microsoft.com/ko-kr/dotnet/core/install/linux-ubuntu)])
- Create Anaconda environment like below and run the `demo.ipynb`

```bash
conda create -y --name tf2p39 python==3.9.0
conda activate tf2p39
pip install matplotlib
pip install pydot
pip install pandas
pip install tensorflow==2.11.0
```

Additionally, you need to setup Jupyter Notebook development environment to run the code interactively.
Or You can do it in Visual Studio Code by simply installing vscode extension `Python`, `Pylance`, `Jupyter`, `Jupyter Notebook Renderers`.

### 2) Full installation (train.ipynb)

Do the minimal installation first, and then,

```
conda activate tf2p39
pip install wandb
```

## 3. How to run

Just run `train.ipynb` for training a model or `demo.ipynb` to create actual prediction result.
After you run all the code in `demo.ipynb`, the result midi file will be saved in `demo_dataset` folder.
To manually add midi data to be used in `demo.ipynb`, copy your midi file to `demo_dataset` folder and then execute `demo.ipynb`.
