# midi-velocity-infer-v2

MIDI Velocity inference based on Seq2seq and Luong Attention.
This is the implementation code of my recent paper, which is to be presented in the 2023 Autumn meeting of the Acoustical Society of Japan [[Link](https://acoustics.jp/annualmeeting/past-meetings/)].

**Try demo at [http://demo1.sapphosound.com](http://demo1.sapphosound.com/)!**

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

Don't forget to unzip the dataset, `dataset/maestro-midi.zip`. After unzipping it, the dataset should be located like `dataset/maestro-midi/test`, `dataset/maestro-midi/train`, `dataset/maestro-midi/validation`.

## 3. How to run

### 1) Data preparation (data.ipynb)

You should generate a `pkl` file which contains the dataset by running the `data.ipynb`.

To do that, you need to change the value of the `root_path` like below, which is the absolute path of this code directory.

```Python
if dataset_name == 'MAESTRO':
    # AMOUNT = '-small'
    # AMOUNT = '-medium'
    AMOUNT = '' # entire dataset

    root_path = '/home/tikim/code/midi-velocity-infer-v2'
    dataset_train_path = f'{root_path}/dataset/maestro-midi{AMOUNT}/train'
    dataset_val_path = f'{root_path}/dataset/maestro-midi{AMOUNT}/validation'
    dataset_test_path = f'{root_path}/dataset/maestro-midi{AMOUNT}/test'
```

At the 3rd block of Jupyter Notebook code, there is a variable `SAMPLE_LENGTH`. Note that this value will be the size of batch size and size of unit of encoder/decoder LSTM.

```python
SAMPLE_LENGTH = 4
FEATURE_NUM = 5
def pad_data(data, feature_num):
```

After running the code, a JSON file and PKL file will be generated. You will use these files in both `train.ipynb` and `demo.ipynb`.

### 2) Training (train.ipynb)

Before running the `train.ipynb`, you need to check the values of some variables.

* `N_HIDDEN` value should be same with the value of `SAMPLE_LENGTH` of `data.ipynb`.

```python
N_HIDDEN = 4
dataset_filename = ''
if dataset_name == 'MAESTRO':
```

* Check the values in wandb config object. But in general, you don't need to change those values.
  * And you should make a new Wandb account if you don't have one. [[Link](https://wandb.ai/)]

```python
wandb.init(
    # set the wandb project where this run will be logged
    project="midi-velocity-infer-v2-attention",

    # track hyperparameters and run metadata with wandb.config
    config={
        "n_hidden": N_HIDDEN,
        "activation_1": "LeakyRelu",
        "dropout": 0.2,
        "optimizer": "adam",
        "loss": "mse_cosine_loss",
        "loss_alpha": 0.15,
        "batchnorm_momentum": 0.60,
        "metric": "mae",
        "epoch": 5,
        "teacher_forcing": False,
        "rnn_type": "LSTM",
        "dataset_name": dataset_name
    }
)
config = wandb.config
```

* Uncomment the proper parameter line and comment the others in the following code.
  * If `N_HIDDEN` and `SAMPLE_LENGTH` value is 4, your code should be like below.

```python
elif (dataset_name == 'MAESTRO'):
    cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0001, decay_steps=176787*config.epoch*lr_decay_alpha, alpha=0.001 # MAESTRO full len4
        # initial_learning_rate=0.0001, decay_steps=176670*config.epoch*lr_decay_alpha, alpha=0.001 # MAESTRO full len8
        # initial_learning_rate=0.0001, decay_steps=176432*config.epoch*lr_decay_alpha, alpha=0.001 # MAESTRO full len16
        # initial_learning_rate=0.0001, decay_steps=175952*config.epoch*lr_decay_alpha, alpha=0.001 # MAESTRO full len32
    )
```

That's it! Now you are good to run the code.

### 3) Testing (demo.ipynb)

Before running the `demo.ipynb`, you need to check the values of some variables.

* Ensure that `SAMPLE_LENGTH` value is same with the value in `data.ipynb`.

```python
ALPHA = 0.15
mse_cosine_loss = make_mse_cosine_loss(ALPHA)
SAMPLE_LENGTH = 4
```

Run `demo.ipynb` to create actual prediction result.
After you run all the code in `demo.ipynb`, the result midi file will be saved in `demo_dataset` folder.
To manually add midi data to be used in `demo.ipynb`, copy your midi file to `demo_dataset` folder and then execute `demo.ipynb`.

### 4) Testing with ONNX

For bonus, you can test this model with ONNX environment!

Check `tf2onnx.ipynb` and `demo_onnx.ipynb` for detail.


## 4. Citation (WIP)

If you cite this code or paper in academic research, refer following bibtex code:

```bibtex
@InProceedings{kim2023music,
author="Kim, Taein
and Kim, Yunho",
title="Piano Velocity Prediction Using a Seq2Seq Model with Attention Mechanism",
booktitle="2023 Autumn Meeting of the Acoustical Society of Japan",
year="2023",
publisher="Acostical Society of Japan",
pages="???--???",
isbn="???"
}
```


## 5. About the dataset

The dataset "MAESTRO" mentioned and used in this repository is originally from following paper.

```
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
  Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling
  Factorized Piano Music Modeling and Generation with the MAESTRO Dataset."
  In International Conference on Learning Representations, 2019.
```

You can access the original dataset from [https://magenta.tensorflow.org/datasets/maestro](https://magenta.tensorflow.org/datasets/maestro).

The dataset is used under a [Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
