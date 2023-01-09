# Level2 Wrap-up Report

# Semantic Segmentation Competition

### 프로젝트 개요

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

![Untitled](Level2%20Wrap-up%20Report%20451d241232f24d27b27e512b8e2c4a48/Untitled.png)

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

- **Input :** 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. Segmentation Annotation은 COCO format으로 제공됩니다.
- **Output :** 모델은 Pixel 좌표에 따라 카테고리 값을 리턴합니다. 이를 Submission 양식에 맞게 csv 파일을 만들어 제출합니다.
- 프로젝트 팀 구성 및 역할

| 팀원 / 역할 | Streamlit | Detectron2 | Paper Review |
| --- | --- | --- | --- |
| 오주헌 | Data Viewer 레이블 표시 | DiNAT, SegFormer | https://velog.io/@ozoooooh/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0Dilated-Neighborhood-Attention-Transformer |
| 강민수 | Segmentation 시각화 | MaskDino/DenseCRF/Ensemble/ Copy and Paste | https://velog.io/@tec10182/ClassMix |
| 신성윤 | Albumentation demo 통합
제출 파일(image, mask) 시각화 | Mask2Former / Copy and Paste | https://arxiv.org/abs/2112.01527 |
| 나성근 | Data distribution 시각화 | SeMaskMask2Former/Pseudo Labeling | https://arxiv.org/abs/2112.12782 |
| 박시형 | Streamlit Data Viewer Page | SegViT, SeMaskFAPN | https://arxiv.org/abs/2010.01824 |

### 프로젝트 수행 절차 및 방법

1. Streamlit을 이용해서 EDA 진행
    1. EDA 과정을 팀원들과 쉽게 공유
2. Mask2Former을 이용한 Segmentation Model Architecture를 Baseline으로 설정
    1. 여러 모델 실험 (MaskDINO, SegFormer UperNet, SegViT)
3. Backbone을 DiNAT / SeMask로 설정
4. Dense CRF를 이용한 앙상블 전략 실험
5. Pseudo Labeling 실험

- EDA
    
    ![스크린샷 2023-01-09 오전 11.55.21.png](Level2%20Wrap-up%20Report%20451d241232f24d27b27e512b8e2c4a48/%25E1%2584%2589%25E1%2585%25B3%25E1%2584%258F%25E1%2585%25B3%25E1%2584%2585%25E1%2585%25B5%25E1%2586%25AB%25E1%2584%2589%25E1%2585%25A3%25E1%2586%25BA_2023-01-09_%25E1%2584%258B%25E1%2585%25A9%25E1%2584%258C%25E1%2585%25A5%25E1%2586%25AB_11.55.21.png)
    
    - 데이터 직접 살펴보기
        - 통계 분석 과정에서 확인하기 어려운 데이터의 특성 파악
            - Plastic bag 안에 있는 쓰레기를 구분하는가?
            - Mislabel된 데이터는 얼마나 존재하는가?
    - Statistical Analysis
        - Class distribution
        - Color distribution by class
        - Number of object per image
        - Annotation size proportion per image
- Model Architecture
    
    
    |  | Backbone |
    | --- | --- |
    | Mask2Former | Swin-B, Swin-L, DiNAT-L |
    | SeMask | FAPN-Swin-L, Swin-L |
- Augmentation
    
    
    |  | Geometric | Color |
    | --- | --- | --- |
    | SeMask | Multi-Scaling Training(256~1024) Random Horizontal Flip         | Color Jittering |
    | DiNAT | Multi-Scaling Training (320~1280)                                 Random Horizontal Flip | Color Jittering |
- Copy and Paste
    
    Copy & Paste 방식을 이용하여 Validation 성능이 낮은 Class를 증강하는 목표
    
    → Detectron2 내부에서는 Augmentation 구현이 어려워 다음과 같은 방식으로 진행
    
    1. 미리 Copy & Paste 이미지를 만들어 두고 학습 진행
        
         → 이미 성능이 잘 나오는 Class의 데이터도 같이 증강된 것으로 예상되어 성능 하락 
             (Public LB mIoU: 0.77 → 0.72)
        
    2. Evaluation Hook에 Copy & Paste 이미지를 생성하는 코드를 등록 후 Evaluation할 때 마다 400개의 이미지를 랜덤하게 변환
        
        → 지정된 Class에 대한 Validation은 증가하였지만 다른 Class에 대한 성능이 하락한 것으로 예상되어 성능은 비슷하게 유지(Public LB mIoU: 0.77 → 0.77)
        
- Hyperparameter Tuning
    
    
    |  | LR | Optimizer | Scheduler |
    | --- | --- | --- | --- |
    | Mask2Former | 0.0001 | AdamW | WarmupPolyLr+MultiStep |
- Ensemble(Final Submission)
    1. Pixel 단위 Ensemble → Masking Image에 Noise가 발생
    
    ![ensemble.PNG](Level2%20Wrap-up%20Report%20451d241232f24d27b27e512b8e2c4a48/ensemble.png)
    
     b. Noise를 해결하기 위하여 DenseCRF 사용 → 작은 물체가 무시되는 경향이 발생
    
    ![crf.PNG](Level2%20Wrap-up%20Report%20451d241232f24d27b27e512b8e2c4a48/crf.png)
    
     c. b의 문제를 해결하기 위해 다시 Ensemble을 하여 DenseCRF를 제외한 다른 모델들에서 공통적으로 잡아내는 물체에 대하여 Pixel값을 Overwrite
    
- Pseudo Labeling
    - 3가지 모델(SeMask, SeMask-FAPN, DiNAT) 각각의 Best Result를 512x512로 Inference한 결과들을 Ensemble한 뒤, Pseudo Labeling 진행
        
        
        | Model | LB(Single) | LB(Pseudo Labeling) |
        | --- | --- | --- |
        | SeMask | 0.774 | 0.787 |
        | SeMask-FAPN | 07724 | 0.7842 |
        | DiNAT-L | 0.7763 | 0.7838 |
        | Swin-L | 0.7718 | 0.75 |
- Model FLOPs and Parameters
    
    
    | Model | FLOPs(G) | Parameters(M) |
    | --- | --- | --- |
    | Mask2Former-Swin-L | 630 | 215 |
    | Mask2Former-Dinat-L | 510 | 220 |
    | SeMask-Swin-L | 643 | 223 |

### 프로젝트 수행 결과

- 핵심 실험 내용
    
    
    | Experiment | LB(mIoU) |
    | --- | --- |
    | UperNet-Swin-B | 0.6 |
    | UperNet-Swin-L | 0.7 |
    | MaskDino-Swin-L | 0.74 |
    | Mask2Former-Swin-B | 0.7210 |
    | Mask2Former-Swin-L | 0.7718 |
    | SeMask-FAPN-Swin-L | 0.7724 |
    | Mask2Former-Dinat-L | 0.7763 |
    | SeMask-Swin-L | 0.783 |
    | Dense-CRF+Ensemble(DiNAT+SeMask) | 0.7895 |
    | Ensemble-ALL | 0.7989 |
- 최종 제출: Public: 0.7989(2nd) / Private: 0.7831(1st)

### 자체 평가 의견

- 대회 초반에 디테일한 데이터 분석과  문제 정의를 하지 못하여, 가설을 설정하고 그 가설을 검증하는 방식의 체계적인 실험을 진행하지 못했다.
    - 데이터의 특성이나 모델 시각화 결과를 참고해 적절한 모델이나 기법, 하이퍼 파라미터 등을 찾아야 한다.
- Task가 어려워서 알맞은 기법을 찾기 힘들었다.
    - General Trash Class의 범위가 다른 클래스에 비해 너무 넓다.
    - Paper와 Paper pack처럼 비슷한 특징을 가지는 클래스가 존재한다.
    - Mislabeled된 데이터가 많이 존재한다.
- 추론 과정과 결과를 시각화 하는 것이 중요하다.
    - 추론 결과를 시각화하면서 Noise가 생기는 것을 발견했다. (Noise가 점수에 큰 영향을 미쳤다.)
    - 추론 결과를 시각화하면서 모델 별 특징을 알 수 있었다.
    - Dense CRF를 그냥 사용하면 성능이 하락하였지만, 작은 물체가 없어진다는 문제를 발견해 앙상블기법을 적용해 성능을 향상시킬 수 있었다.
- 이번 대회에는 잘 못했지만, 다음부터는 대회 초반에 전체적인 계획을 잡는 과정이 필요하다.
    - 이 과정에서 프로젝트에서 수행해야 할 작업들의 체계를 잡을 수 있다.
    - 세부 작업들을 팀원들에게 잘 분배할 수 있다.
    - 어떤 일을 해야 할지 고민할 시간을 줄일 수 있다.
- SeMask가 왜 성능이 좋았을까?
    - 모델의 추론 결과를 시각화 해보면 Segmentation보다 Classification 성능이 좋지 않았다.
    - 하지만 SeMask은 기존 모델 구조에 Semantic Layer을 추가하여 이미지 단위의 Semantic 정보를 잘 추출하기 때문에, 큰 범위에서 Classification 성능을 좋게 만들어 최종적인 성능은 오히려 좋아졌다고 판단한다.
- DiNAT은 왜 성능이 Swin보다 좋았을까?
    - Swin보다 Receptive Field가 더 넓어서 Global Context를 더 많이 잡고 따라서 Object에 대한 이해도가 더 높았다고 판단한다.
    - 논문에서 나온대로 Mask2former Backbone을 Swin과 비교했을 때 성능이 더 우수했기 때문에 본 대회에서도 성능이 더 좋았을 것이라 판단한다.

### 개인 회고

- 박시형
    
    Semantic Segmentation competition을 하면서 어떤 구조로 이루어져 있는지 또 어떤 모델이 있는지 전반적으로 어떻게 발전해왔는지 알 수 있었습니다. 그리고 streamlit으로 모델 결과의 시각화, 훈련 데이터의 시각화를 하였고 Detectron2라이브러리를 이용해 모델을 학습시키고 MLflow를 통해 로깅을 시도하였습니다.
    
    이번 프로젝트에서 1등을 했지만 두 번의 실패를 해서 개인적으로 많이 아쉬운 대회 였습니다. 모델을 선정할 때  이전 대회를 바탕으로 SOTA가 전반적으로 잘 동작하는 것을 보고 papers with code에서 참고하여 SegViT를 돌려보았습니다. 그러나 만 단위의 iteration을 학습을 해도 일부 클래스만 성능이 오르는 문제점이 있었는데 해결을 못해 결국 사용하지 못했습니다. 또한 전체 갯수가 상대적으로 적은 배터리 클래스를 보고 Class Balanced loss를 적용하고자 하였으나 팀원들이 만든 모델의 confusion matrix의 배터리 클래스를 잡는 성능이 다른 클래스보다 더 좋게 나왔습니다. 그래서 후속으로 나온 논문으로 클래스의 난이도까지 고려하는 CDB loss를 읽고 저희가 사용하는 라이브러리인 detectron2에 적용하고자 하였습니다. 하지만 생각보다 라이브러리에 적용하는게 복잡했고 끝나는 시간은 다가오고 이미 학습하던 모델도 있어 코드 디버깅도 제대로 하지 못해 결국 적용해보지 못했습니다.
    
    팀원들과 매일 회의를 하며 어떤 기법은 되고 이 기법은 왜 안되는지 토의를 하면서 시각이 넓어질 수 있었습니다. 그리고 이전 대회에서는 하지 못했었던, 결과를 바탕으로 다음에 학습할 모델을 어떻게 변경할 지 정하는 시도를 했습니다. 이 방법을 통해 모델에 대한 이해도를 높이고 어떤 부분이 부족한지 예측할 수 있었습니다. 추가적으로 토의한 문제에 대해 논의가 더 필요한 상황이 생길 수 있어 보다 자세히 기록할 필요성을 느꼈습니다.
    
- 신성윤
    - 이번 프로젝트에서 나의 목표는 무엇이었는가?
        - 대회 기간 동안 사용할 유용한 기능들 구현하기
        - 코딩 생산성 늘리기
        - 작업 수행 절차 체계적으로 정하고 행동하기
        - 높은 성능의 모델 빨리 찾기
    - 나는 내 학습 목표를 달성하기 위해 무엇을 어떻게 했는가?
        - 과제에 적합한 Data Augmentation 기법을 추려내기 위해서 다양한 Augmentation 기법을 테스트 해볼 수 있는 기능을 가진 Tool을 탐색했습니다. Albumentations의 Demo에 원하는 기능들이 다 구현되어 있었고 이것을 팀의 시각화 도구에 통합하였습니다.
        - 작성한 코드가 개인적으로 사용하고 버려지는 일회성 코드가 되지 않도록 가능하면 기존에 작성되어 있는 코드를 재사용 하려고 했고, 팀에서 정한 coding convention을 지키고자 했습니다.
        - 매일 TODO List를 작성하였습니다.
        - 지난 기수가 동일한 과제를 수행했기 때문에 지난 기수의 Github Repository를 조사해서 성능이 가장 좋았던 모델이 무엇인지 찾아봤습니다. 그리고 Paperswithcode의 SOTA 모델 목록을 참고했습니다.
    - 나는 어떤 방식으로 모델을 개선했는가?
        - Mask2Former Model Architecture에 Backbone Model을 변경 및 학습하여 성능을 확인하는 실험을 했습니다.
    - 내가 한 행동의 결과로 어떤 지점을 달성하고, 어떠한 깨달음을 얻었는가?
        - 좋은 성능의 모델의 기준을 잡을 수 있었고, 기준을 토대로 대회가 끝날 때 상위권의 팀의 점수가 어떻게 분포될 지 짐작할 수 있었습니다.
        - 대회 초반에 작성한 시각화 도구들을 통해 모델의 추론 결과에 대한 평가를 단순히 mIoU, Acc로만 하는 것이 아니라 물체의 경계면 품질, 잘 맞추지 못하는 항목에 대한 정성적인 요소 등을 파악할 수 있었습니다.
    - 전과 비교해서, 내가 새롭게 시도한 변화는 무엇이고, 어떤 효과가 있었는가?
        - ML Experiment Tracking 도구로 MLflow를 사용했습니다.  MLflow를 집에 남는 PC를 서버로 구축했고, Cloud 방식에 대비하여 On-premise 방식으로 서버를 구축했을 때 생기는 보안 문제, 구축하는데 드는 많은 소요 시간 문제 등을 확인할 수 있었습니다.
    - 마주한 한계는 무엇이며, 아쉬웠던 점은 무엇인가?
        - 시간은 한정되어 있기 때문에, 데이터 시각화 도구 제작, 인프라에 많은 시간을 사용한 만큼 데이터, 모델에 대한 이해를 넓히는 데에 시간 투자를 못했습니다.
        - MLflow의 Tracking 기능을 사용했는데, Detectron2에서 나오는 모든 Log를 Tracking 하도록 설정하여 MLflow UI의 속도 저하 및 확인 불가 문제가 있었습니다. 필요한 Log만 Tracking 하도록 하는 작업이 중요할 것 같습니다.
    - 한계/교훈을 바탕으로 다음 프로젝트에서 스스로 새롭게 시도해볼 것은 무엇일까?
        - Docker를 활용하고, 잘 구성된 MLOps Template을 찾아서 프로젝트 기반 구축에 소요되는 시간을 줄이려고 합니다.
- 강민수
    - 이번 프로젝트에서 나의 목표는 무엇이었는가?
    Segmentation의 전반적인 과정을 이해하고, Segmentation을 하기 위한 여러가지 툴에 대해 이해를 하며 이를 통해 점수를 높여보는 것이었다.
    - 나는 내 학습 목표를 달성하기 위해 무엇을 어떻게 했는가?
    Detectron, mmsegmentation등 여러가지 툴에 대해 배웠고, 이를 직접 만져보고, 이런 툴을 기반으로 만들어진 SOTA 모델들에 대한 실험을 진행해 보았다.
    - 나는 어떤 방식으로 모델을 개선했는가?
    Dense CRF와 Ensemble을 결합해 앙상블을 하면서도, 이로인해 생기는 이미지의 노이즈를 제거하여 성능을 올렸다.
    - 내가 한 행동의 결과로 어떤 지점을 달성하고, 어떠한 깨달음을 얻었는가?
    대회 막바지에 앙상블 전략을 이용하여 최고점을 달성할 수 있었다. 앙상블을 하면서 안되면 왜 안된건지 생각해보는 과정이 중요함을 다시 한 번 깨달았다.
    - 전과 비교해서, 내가 새롭게 시도한 변화는 무엇이고, 어떤 효과가 있었는가?
    사용해 본적 없는 툴을 사용해보려고 하였고, 이로 인해 다양한 모델을 실험해 볼 수 있었다.
    - 마주한 한계는 무엇이며, 아쉬웠던 점은 무엇인가?
    Task에 Specific하게 접근했던 방식들이 모두 실패하여서 안타까웠다.
    - 한계/교훈을 바탕으로 다음 프로젝트에서 스스로 새롭게 시도해볼 것은 무엇일까?
    조금 더 창의적인 문제 해결 방식을 시도해 봐야겠다.
- 나성근
    - 이번 프로젝트에서 나의 목표는 무엇이었는가?
        - 성능이 우수하고 주어진 데이터셋에 잘 맞는 모델을 찾아 학습시켜보고, 하나 이상의 기법을 적용해 결과를 개선시키는 것이었다.
    - 나는 내 학습목표를 달성하기 위해 무엇을 어떻게 했는가?
        - SeMask 구조에 대해 공부하여 Mask2Former 모델에 적용하였다.
        - Pseudo Labeling 기법을 적용했다.
        - DenseCRF 기법 적용을 시도했다.
    - 나는 어떤 방식으로 모델을 개선했는가?
        - pretrained weight을 바꾸어보거나, num_obj_query 등의 하이퍼 파라미터를 수정해보았다.
        - validation set이나 test set을 모두 학습 데이터에 포함시켜보았다.
        - 학습이 끝난 pth 파일을 로드하는 방식으로 총 이터레이션을 200k 이상으로 설정하여, 디폴트 값인 80k보다 학습을 훨씬 많이 해보았다.
    - 내가 한 행동의 결과로 어떤 지점을 달성하고, 어떠한 깨달음을 얻었는가?
        - 이터레이션을 늘려 단일 모델 최고 점수를 달성했다.
        - 학습 횟수가 증가할 수록, 장기적인 텀에서 모델의 성능이 지속적으로 개선되는 것을 체감했다.
    - 전과 비교해서, 내가 새롭게 시도한 변화는 무엇이고, 어떤 효과가 있었는가?
        - Pseudo labeling이나 DenseCRF 등 새로운 기법을 적용했다.
        - Pseudo Labeling의 결과 실험한 모든 모델에 대해 각각의 성능이 향상됐다.
        - DenseCRF의 결과 미세한 성능 하락이 있었다.
    - 마주한 한계는 무엇이며, 아쉬웠던 점은 무엇인가?
        - 대회 마지막 날 제출 서버의 불안정으로 Pseudo Labeling 결과를 제대로 활용하지 못했다.
        - DenseCRF를 적용해보았지만, 이를 응용할 생각을 하지 못했다.
    - 한계/교훈을 바탕으로 다음 프로젝트에서 스스로 새롭게 시도해볼 것은 무엇일까?
        - 마지막 프로젝트에서는 기획과 문제정의를 철저히 하여, 주어진 짧은 시간 내에 높은 퀄리티를 뽑아낼 수 있도록 할 예정이다.
- 오주헌
    - 이번 프로젝트에서 나의 목표는 무엇이었는가?
        - 최신 모델, 기법 적용해보기
        - 논리적으로 문제 해결하기
        - Pytorch Lightning을 적용해 문제 해결하기
    - 나는 내 학습목표를 달성하기 위해 무엇을 어떻게 했는가?
        - 최근 2년 동안 나온 논문들을 학회별로 정리하고 본 대회에 적용할 수 있는 논문을 별도로 정리
        - 왜 특정 모델의 성능이 좋은지 알기 위해 논문을 통해 분석
        - Lightning, Hydra, SMP를 적용한 Baseline Template 구축
        - 문제 해결 시 마주한 이슈에 대해 분석 및 정리([Inference](https://velog.io/@ozoooooh/Kornia%EB%A5%BC-%ED%86%B5%ED%95%B4-Inference-%EB%8D%94-%EB%B9%A0%EB%A5%B4%EA%B2%8C-%ED%95%98%EB%8A%94-%EB%B0%A9%EB%B2%95))
    - 나는 어떤 방식으로 모델을 개선했는가?
        - Panoptic Segmentation Task에 사용된 Pretrained Weight도 적용 및 비교
        - Mask Interpolation 변경을 통해 성능 향상
        - Pseudo Labeling을 통해 성능 향상
    - 내가 한 행동의 결과로 어떤 지점을 달성하고, 어떠한 깨달음을 얻었는가?
        - 대회 데이터 특성을 고려하지 않고 최신 기법을 찾아보다가 생각보다 적용해볼 수 있는 기법이 별로 없었고, 상황에 맞는 기법을 사용하는 것이 더 중요한 것을 깨달음
    - 전과 비교해서, 내가 새롭게 시도한 변화는 무엇이고, 어떤 효과가 있었는가?
        - Streamlit을 통해 Test Data Inference 결과를 시각화해서 분석했고, 특정 상황에서 Noise가 생기는 것을 발견해서 문제 원인 및 해결 효과를 봄
        - 새로운 툴 Detectron2 사용을 통해 다양한 모델을 실험해 볼 수 있었음
        - Inference 속도 개선을 해보려고 노력했고 기존 대비 약 9배 속도 향상을 이룸. 마지막날 Ensemble을 위해서 Inference를 많이 해야 되는 상황에서 효과를 많이 봄
    - 마주한 한계는 무엇이며, 아쉬웠던 점은 무엇인가?
        - Lightning, SMP를 활용해 Template을 구축했지만 mmsegmentation, Detectron2같은 라이브러리들이 최신 연구에 사용되서 Template을 사용하지 않음. 별 효과를 보지 못해 아쉬움.
        - Detectron2가 내가 만든 코드가 아니다 보니 오류 파악이 힘들었음. Configuration에 대한 설명이 부족해서 실험 통제를 제대로 하지 못함.
    - 한계/교훈을 바탕으로 다음 프로젝트에서 스스로 새롭게 시도해볼 것은 무엇일까?
        - 문제 접근부터 해결까지 더 논리적으로 시도해보기
        - 문제 해결 시 발생한 이슈 정리하기