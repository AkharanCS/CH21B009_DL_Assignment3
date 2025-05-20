# DA6401 - Assignment 3

## Important Links
- **Link to GitHub Repo** : <https://github.com/AkharanCS/CH21B009_DL_Assignment3>
- **Link to wandb report** : <https://wandb.ai/ch21b009-indian-institute-of-technology-madras/Assignment3_Q5/reports/DA6401-Assignment-3-Report--VmlldzoxMjgyNzU1MA?accessToken=hf0u4cqvukop88avyqy82k8ztmy80fb8r1m41141dxx3q0l6v3emy33yrze5yjdc>

## Directories
- **predictions_attention**: contains the predictions for the test dataset made by the attention based seq2seq model.
- **predictions_vanilla**: contains the predictions for the test dataset made by the vanilla seq2seq model.

## Script Files
- **`prepare.py`**: contains the classes and functions required for tokenizing and padding the dataset.
- **`Q1.py`**: contains the Seq2Seq class which includes the architecture and relevant methods for the vanilla seq2seq model.
- **`Q2.py`**: contains code for performing the wandb sweep as required in Q2.
- **`Q4.py`**: contains code for running the best network configuration on the test dataset and analysing its peroformace.
- **`Q5.py`**: contains the Seq2Seq class which includes the architecture and relevant methods for the attention based seq2seq model.
- **`Q5_a.py`**: contains code for performing a wandb sweep for the attention based seq2seq model.
- **`Q5_b.py`**: contains code for running the best network configuration for the attention based model on the test dataset and analysing its peroformace.
- **`Q5_d.py`**: contains code for running the best network configuration for the attention based model and plotting the attention heatmaps.
- **`Q6.py`**: contains code for plotting the connectivity diagram.
- **`config.yaml`**: contains the hyperparameter space used for running wandb sweeps for both the vanilla and attention based seq2seq model.

## Other Important files
- **`attention_heatmap.png`**: contains the attention heatmap as required in Q5 part-d.
- **`connectivity.png`**: contains the connectivity diagram as required in Q6.
- **`requirements.txt`**: contains all the libraries and dependancies required for running both part A and B.

## Steps to run (follow this order)
1. Clone the repository:
   ```bash
   git clone https://github.com/AkharanCS/CH21B009_DL_Assignment3.git
   ```
2. Download the dakshina_dataset_v1.0 dataset.
3. Setup a python virtual environment with the required dependancies using the following commands:
     ```bash
    python -m venv venv
    venv/Scripts/activate
    pip install -r requirements.txt
   ```
4. Run `prepare.py`. <br>
5. Run `Q1.py`. <br>
6. Run `Q5.py`. <br>
7. Save `config.yaml`. <br>
8. All the other files (`Q2.py`,`Q4.py`,`Q5-a.py`,`Q5-b.py`,`Q5-d.py`,`Q6.py`) can be run in any order. <br>
