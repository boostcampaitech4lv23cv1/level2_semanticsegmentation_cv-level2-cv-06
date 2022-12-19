<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Template for semantic segmentation
## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/boostcampaitech4lv23cv1/level2_semanticsegmentation_cv-level2-cv-06.git
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
# this is default setting
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
You can use multiple run at once

```bash
python src/train.py datamodule.batch_size=16,32,64,128 model.arch_name="UnetPlusPlus", "Unet"
```

## How to use

- train 실행 시 test도 실행하고 csv 파일 자동으로 생성
   - python src/train.py

- 학습된 ckpt 파일을 이용하여 test 실행하려면
    - python src/eval.py ckpt_path = "your ckpt path"

- configs폴더에 있는 config들을 CLI 상에서 수정해서 사용 가능
    - python src/train.py datamodule.batch_size=128 model.arch_name="Unet" model.encoder_name="your encoder name"
    - model 관련 config는 [Docs](https://smp.readthedocs.io/en/latest/encoders_timm.html) 참고
- encoder, archtitechtures, loss, optimizer, scheduler 등등 다양한 config 수정 가능
- 더 자세한 내용은 [README](https://github.com/ashleve/lightning-hydra-template) 참고