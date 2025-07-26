# CrysVCD
Official code repository for "Enhancing Materials Discovery with Valence Constrained Design in Generative Modeling".



## Installation guide

The backbone of this project is built based on DiffCSP ([DiffCSP](https://github.com/jiaor17/DiffCSP)), and the evaluation of this project is based on MatterSim ([MatterSim](https://github.com/microsoft/mattersim)). For a smooth construction of a compatible python environment, we strongly recommend users to follow the steps below:

1. **(Recommended)** Create a conda virtual environment. We don't recommend you specify python version at this stage. If you insist in specifying one, we recommend `python=3.12`.
```
conda create -n crysvcd
conda activate crysvcd
```
2. Install the torch version (recommended with CUDA GPU) 2.4.1. Please check the cuda version most compatible to your machine on the official torch website ([torch website](https://pytorch.org/get-started/previous-versions/)). **Important: please specify `"numpy<2"` in the command.** For example run:
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 "numpy<2" -c pytorch -c nvidia
```
3. Now we install mattersim first. If you followed the recommended steps above, simply running `pip install` works. For more info please check the official documentation of MatterSim ([MatterSim](https://github.com/microsoft/mattersim)). For example run:
```
pip install mattersim
```
4. Install additional dependencies. We have significantly reduced package dependency of DiffCSP to lighten the code. 
- First we can easily install these following packages by pip:
```
pip install pyxtal p_tqdm chemparse
```
- The slightly trickier part is the installation of `torch_scatter`. It was reported that running `pip install torch_scatter` will pop out error. One known solution is to install directly by the prebuilt wheel of pyg-related packages. For example for `torch.2.4.1+cu118` and `python=3.12` we can check https://data.pyg.org/whl/torch-2.4.1%2Bcu118.html, select the corresponding .whl file for your machine (e.g. linux), and then run
```
pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_scatter-2.1.2%2Bpt24cu118-cp312-cp312-linux_x86_64.whl
```
- **(Optional; required for chemical composition generation)** The last step is to install `transformers` powered by Huggingface:
```
pip install transformers
```


## Getting started
To test if your installation is done properly, simply run `main.py` to train a few epochs and generate some crystals:
```
# Training
python main.py -t gpt2_ionic
python main.py -t gpt2_alloy
python main.py -t crygen
# Generation
python main.py -g crygen_alloy -nf 100 # generate 100 chemical formulas and their crystal structures
python main.py -g crygen_ionic -nf 100
```

Stay tuned for more detailed documentation of this code!
