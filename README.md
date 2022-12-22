# 환경설정

```sh

create conda -n aistage python=3.8
source activate aistage

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .

pip install mlflow

```