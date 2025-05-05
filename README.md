# Graph-based ATtractor-TArget Control Algorithm (GATTACA)

Source code. trained models implementation of the GATTACA framework introduced in [A. Mizera, J. Zarzycki. Graph Neural Network-Based Reinforcement Learning for Controlling Biological Networks: The GATTACA Framework (2025)](arxiv.com).

This project extends our previous work [pbn-STAC](https://arxiv.org/abs/2402.08491) implemented in [pbn-STAC](https://github.com/jakub-zarzycki2022/pbn-stac) and [gym-pbn-STAC](https://github.com/jakub-zarzycki2022/gym-PBN-stac), which itself was based on the original methods introduced in [G. Papagiannis, S. Moschoyiannis et al., Deep Reinforcement Learning for Stabilization of Large-Scale Probabilistic Boolean Networks (2022)](https://ieeexplore.ieee.org/document/9999487) implemented in [gym-PBN](https://github.com/UoS-PLCCN/gym-PBN/tree/main) and [pbn-rl](https://github.com/UoS-PLCCN/pbn-rl).

# Environment Requirements
- CUDA 11.3+
- Python 3.9+

# Installation
## Local
- Create a python environment using PIP:
    ```sh
    python3 -m venv .env
    source .env/bin/activate
    ```
    For the last line, use `.\env\Scripts\activate` if on Windows.
- Install [PyTorch](https://pytorch.org/get-started/locally/):
    ```sh
    python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```
- Install the package and its dependencies dependencies:
    ```sh
    python -m pip install -r requirements.txt
    ```

# Models
All trained models are available via google drive:
https://drive.google.com/drive/folders/1qLV0IdBfFg-MFj28WtYGdy6pYK63YfUs?usp=sharing

# Running
- Use `train_gattaca.py` to train a DDQN agent. It's a command line utility so you can check out what you can do with it using `--help`.
    E.g.:
    ```sh
     python train_gattaca.py --size 67 --assa-file  bortezomib_fixed.ispl --exp-name example
    ```

- Use `model_tester.py` to get strategies and statistics about the model.
E.g.:
```sh
python model_tester.py -n 67 --assa-file  bortezomib_fixed.ispl --model-path models/pbn67/bdq_final.pt --attractors 10 --runs 10
```

| Argument       | Description                                                          |
| -------------- | -------------------------------------------------------------------- |
| `-n`           | (Required) Number of nodes in the model.                       |
| `--assa-file`  | (Required) Path to the `.ispl` file.     |
| `--model-path` | (Required) Path to the trained PyTorch model `.pt` file.             |
| `--attractors` | (Optional) Number of source attractors to analyse. It may be smaller than the total number of attractors.             |
| `--runs`       | (Optional) Number of test runs for averaging or robustness checking. |
