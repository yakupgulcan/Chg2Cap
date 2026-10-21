<h1 align="center">Changes to Captions: An Attentive Network for Remote Sensing Change Captioning</h1>

<h3 align="center"> Shizhen Chang and <a href="https://www.ai4rs.com/">Pedram Ghamisi</a></h3>
<br>

![](Figure/Flowchart.png)
    
This is the official PyTorch implementation of **[Changes to Captions: An Attentive Network for Remote Sensing Change Captioning](https://arxiv.org/abs/2304.01091)**, a project conducted at the [Institute of Advanced Research in Artificial Intelligence (IARAI)](https://www.iarai.ac.at/).


### Preparation
- Install the required packages using the runtime setup below.
- Download the remote sensing change captioning datasets. We have adopted [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC) in this repository.
- The data structure of LEVIR-CC is organized as follows:

```
├─datasets/LEVIR_CC/
        ├─LevirCCcaptions.json
        ├─images
             ├─train
             │  ├─A
             │  ├─B
             ├─val
             │  ├─A
             │  ├─B
             ├─test
             │  ├─A
             │  ├─B
```
where folder A contains images of pre-phase, folder B contains images of post-phase.

- Extract text files for the change descriptions of each image pair in LEVIR-CC:

```
$ python preprocess_data.py
```

!NOTE: When preparing the text token files, we suggest setting the word count threshold of LEVIR-CC to 5 and Dubai_CC to 0 for fair comparisons.
### Training
- Ensure you have completed the data preparation steps above, and then proceed to train the model as follows:
```
$ python train.py
```

!NOTE: METEOR evaluation requires Java on PATH and its paraphrase resource. Run `python scripts/prepare_meteor.py` before training or evaluation.

Alternatively, you can obtain our pretrained models from [Google Drive](https://drive.google.com/file/d/1phoO1BvPsRwIOIykkLm6acBpIocz9liz/view?usp=drive_link).

### Caption Generation
- To generate captions, run the following command:
```
$ python test.py --checkpoint best.pth
```

### Quantitative Evaluation and Visual Examples

- Quantitative evaluations of Chg2Cap compared to other state-of-the-art (SOTA) methods in LEVIR-CC are illustrated as follows:

![](Figure/Results.png)

Here are some visualized examples of the generated captions in LEVIR-CC:

![](Figure/Visualization.png)

### Runtime setup

The local dependency file pins Python 3.11-compatible packages, including
PyTorch 2.2.2 and torchvision 0.17.2. The implementation requires a CUDA-capable
NVIDIA GPU. Create a virtual environment before installing dependencies.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python scripts/prepare_meteor.py
java -version
```

Install Java separately if it is unavailable. The METEOR preparation script
downloads the paraphrase resource and verifies its SHA-256 checksum. The encoder
uses torchvision's weights API with the original ImageNet V1 weights.

### Data paths and experiment outputs

The default LEVIR-CC root is `datasets/LEVIR_CC`. Use `--dataset_root` to select
another location and `--output_dir` to select the prepared-data directory:

```bash
python preprocess_data.py --dataset LEVIR_CC --dataset_root ./datasets/LEVIR_CC --word_count_threshold 5
```

Preprocessing writes split lists, caption tokens, `vocab.json` and `manifest.json`
to `data/LEVIR_CC/`. Repeated runs rewrite split lists without adding duplicate
rows. Keep the vocabulary and maximum caption length consistent between training
and evaluation.

Training saves `last.pth` with model, optimizer and scheduler states at epoch
boundaries, and `best.pth` with weights selected by validation BLEU-4. It also
records configuration, runtime information and validation metrics. Resume with
`--checkpoint`; `--num_epochs` specifies the total target epoch count:

```bash
python train.py --data_folder ./datasets/LEVIR_CC/images --savepath ./runs/experiment/checkpoints --train_batchsize 2 --workers 0 --num_epochs 20
python train.py --data_folder ./datasets/LEVIR_CC/images --savepath ./runs/experiment/checkpoints --train_batchsize 2 --workers 0 --num_epochs 40 --checkpoint ./runs/experiment/checkpoints/last.pth
```

Resume starts at the next epoch and does not restore within-epoch progress or
random generator states. Batch shuffling and dropout may therefore differ from
uninterrupted training; identical numerical results are not guaranteed. Learned
weights and optimizer/scheduler states are retained. If interrupted during validation, that epoch's
validation is not replayed. Weight-only checkpoints support evaluation but do
not contain optimizer/scheduler state for resume.

For a fresh experiment with a fixed initial seed and early stopping:

```bash
python train.py --data_folder ./datasets/LEVIR_CC/images --savepath ./runs/seed123/checkpoints --train_batchsize 32 --num_epochs 50 --seed 123 --patience 10
```

`--patience` counts consecutive epochs without a strictly higher validation
BLEU-4 score; 0 disables early stopping (the CLI default). Training saves the
last checkpoint before stopping. The counter and best epoch are restored on
resume; older checkpoints without these fields start the counter at zero.
`training_epoch_XXX.json` records the learning rates used, validation metrics
and checkpoint selection. A fixed seed initializes Python, NumPy and PyTorch;
it does not guarantee deterministic CUDA execution or identical resumed runs.
The Colab notebook defaults to seed 123 and patience 10. Early stopping is an
additional experiment setting rather than a reproduction of the paper schedule.

Evaluate the held-out test split and save metrics and generated captions:

```bash
python test.py --data_folder ./datasets/LEVIR_CC/images --savepath ./runs/experiment/checkpoints --checkpoint best.pth --workers 0 --results_dir ./runs/experiment/test_results
```

Evaluation writes `metrics.json`, `predictions.json`, `config.json` and
`environment.json`. Validation and test caption generation require batch size 1.
Adjust training batch size and worker count for the available hardware.

### Google Colab

[Chg2Cap_Colab_Training.ipynb](Chg2Cap_Colab_Training.ipynb) provides a GPU training
and evaluation workflow. Configure the repository URL, branch/tag/commit,
dataset Drive URL or folder/ZIP path, and experiment parameters before running
the cells in order. The selected Git ref must contain the notebook's supporting
scripts.

The notebook retains Colab's installed PyTorch/torchvision pair and installs
additional packages from `requirements-colab.txt`. It checks GPU and METEOR
availability and provides a small training, resume and evaluation smoke check.
Code, images and working checkpoints use runtime storage; uploaded checkpoints,
logs, prepared split lists,
vocabulary and evaluation results are saved to Google Drive. Environment records
include Python, PyTorch, torchvision, CUDA build, GPU/driver and Git commit.

For recovery in a new runtime, reuse the experiment name, enable `RESUME`, and
set the total epoch count beyond the completed count. Checkpoints are downloaded
from Drive and verified before loading. Colab runtimes vary; run the smoke check
before a full training job.

The notebook authenticates the Drive API and passes `--drive_checkpoint_folder_id`
to training with a local `--savepath`. Each upload is verified by size and MD5
checksum before the previous `last.pth` or `best.pth` in that experiment's
checkpoint folder is permanently deleted. Previous files do not enter the trash.
Upload or deletion errors stop training; the verified new checkpoint remains
available if deletion fails. Allow storage for both copies during replacement.
Existing trash and other experiment folders are left untouched. Use one training
process per experiment folder. Smoke-check checkpoints stay on runtime storage;
training records are copied to Drive when the training cell exits.

### Paper
**[Changes to Captions: An Attentive Network for Remote Sensing Change Captioning](https://arxiv.org/abs/2304.01091)**

Please cite the following paper if you find it useful for your research:

```
@article{chg2cap,
  title={Changes to Captions: An Attentive Network for Remote Sensing Change Captioning},
  author={Chang, Shizhen and Ghamisi, Pedram},
  journal={IEEE Trans. Image Process.}, 
  doi={10.1109/TIP.2023.3328224},
  year={2023}
}
```

### Acknowledgement

The authors would like to thank the contributors to the [LEVIR-CC](https://github.com/Chen-Yang-Liu/RSICC/tree/main) and [Dubai-CC](https://disi.unitn.it/~melgani/datasets.html) datasets.

### License
This repo is distributed under [MIT License](https://github.com/ShizhenChang/Chg2Cap/blob/main/LICENSE.txt). The code can be used for academic purposes only.