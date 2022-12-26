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