# mrc-search-engine
mrc-search-engine은 검색 엔진과 함께 사용되는 기계독해(Machine Reading Comprehension) 모듈을 개발하는 프로젝트입니다.

## 개발 환경 설정 가이드

가상 환경 설정
```
python -m venv [이름]
```

가상 환경 활성화/비활성화 (Windows PowerShell)
```
./[이름]/Scripts/activate
deactivate
```

가상 환경 활성화/비활성화 (MacOS)
```
./[이름]/bin/activate
deactivate
```
패키지 매니저(Package Manager) 업데이트
```
python -m pip3 install --upgrade pip
```

패키지 설치
```
pip install -r requirements.txt
```

requirements.txt 생성
```
pip freeze > requirements.txt
```