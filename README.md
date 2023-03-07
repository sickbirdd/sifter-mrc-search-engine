# mrc-search-engine
mrc-search-engine은 검색 엔진과 함께 사용되는 기계독해(Machine Reading Comprehension) 모듈을 개발하는 프로젝트입니다.

## 개발 환경 설정 가이드

가상 환경 설정(Conda)
```
conda create --name (이름) python=3.8
```

가상 환경 활성화/비활성화(Conda)
```
conda activate (이름)
conda deactivate
```

패키지 설치(CPU)
```
pip install -r requirements.txt
```

패키지 설치(GPU)
```
pip install -r requirements_gpu.txt
```

requirements.txt 생성
```
pip list --format=freeze > requirements.txt
```

Service program 관련 사항
[서비스 모듈](modules/mrc_service/README.md)를 참조합니다.