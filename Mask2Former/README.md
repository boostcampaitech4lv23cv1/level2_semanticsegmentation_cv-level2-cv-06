## Mask2Former 개발환경
1) conda create --name [NAME] python=3.8
2) cudatoolkitdev=11.0 설치
```bash
conda config --apend channels conda-forge
conda install cudatoolkit=11.0 -c conda-forge
conda install cudatoolkit-dev=11.0 -c conda-forge
```
nvcc -V 버전 뜨는지 확인
<!-- https://kyubumshin.github.io/2022/04/23/tip/conda-cuda-%EC%84%A4%EC%B9%98/ -->
3) detectron2 설치
```bash
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```


<!-- https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-05/blob/main/mask2former/readme.md -->




## Mask2Former 설치법

1) 프로젝트 clone
```bash
git clone https://github.com/boostcampaitech4lv23cv1/level2_semanticsegmentation_cv-level2-cv-06.git
```

2) Mask2Former directory 삭제
```bash
cd
level2_semanticsegmentation_cv-level2-cv-06
rm -rf Mask2Former
```

3. Mask2Former 공식 repository clone
```bash
git clone https://github.com/facebookresearch/Mask2Former.git
```
4. 기존 수정 내용 복구
```bash
git checkout .
```
5. pixel decoder 설정
```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```
만약 g++ 없다고 뜨면,
```bash
apt update
apt install build-essential
```
6. detectron2 package 코드 수정
```bash
conda/envs/[NAME]/lib/python3.8/site-packages/detectron2/project/point_rend/point_features.py
```
- 48 line을 아래와 같이 수정
```python
output = F.grid_sample(input.float(), 2.0 * point_coords - 1.0, **kwargs)
```

## Mask2Former 사용법 (with MLflow)
1) Mask2Former directory 진입
```bash
cd Mask2Former
```

2) 실행
```bash
python train_net_mlflow.py --config-file configs/custom/maskformer2_R50_bs16_160k_mlflow.yaml
```
3) Mask2Former/configs/custom 내의 파일 수정하여 사용