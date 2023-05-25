<p align="center">
  <a href="https://www.google.com/search?q=%EB%A7%8C%EB%93%A0+%EA%B2%80%EC%83%89+%EC%82%AC%EC%9D%B4%ED%8A%B8+%EB%84%A3%EC%9C%BC%EB%A9%B4+%EB%90%A9%EB%8B%88%EB%8B%A4.&ei=-AQHZP2aE8yp2roPlaiMkAc&ved=0ahUKEwj99tfcwcn9AhXMlFYBHRUUA3IQ4dUDCA8&uact=5&oq=%EB%A7%8C%EB%93%A0+%EA%B2%80%EC%83%89+%EC%82%AC%EC%9D%B4%ED%8A%B8+%EB%84%A3%EC%9C%BC%EB%A9%B4+%EB%90%A9%EB%8B%88%EB%8B%A4.&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIFCAAQogQyBQgAEKIEOgoIABBHENYEELADOgoIIRCgARDDBBAKOggIIRCgARDDBEoECEEYAFDeAlj6H2DWLGgLcAF4BYABnwGIAd4UkgEEMS4xOZgBAKABAcgBCsABAQ&sclient=gws-wiz-serp"><img width="420px" src="resources/shifter.png"></a>
</p>
<p align="center">
    <em>✨ MRC training modules : post-train & fine-tune ✨</em>
</p>
</p>


---

**문서**: 문서화 페이지 넣으면 됩니다. (TODO)

---

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

## 어떻게 시작하나요?
```
cd modules
python.exe -m main.py post-training
```

훈련 과정상의 변경점을 주고 싶다면 다양한 인자를 추가할 수 있습니다.
다음 명령어를 통해서 이를 확인해 보세요.
```
python.exe -m main.py --help
```

---

## 서비스 모듈
해당 프로젝트는 훈련한 모델을 사용할 수 있는 api server를 동시에 제공합니다.
Service program 관련 사항은 [서비스 모듈](modules/mrc_service/README.md)를 참조합니다.