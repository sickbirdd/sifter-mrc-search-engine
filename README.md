<p align="center">
  <a href="https://www.google.com/search?q=%EB%A7%8C%EB%93%A0+%EA%B2%80%EC%83%89+%EC%82%AC%EC%9D%B4%ED%8A%B8+%EB%84%A3%EC%9C%BC%EB%A9%B4+%EB%90%A9%EB%8B%88%EB%8B%A4.&ei=-AQHZP2aE8yp2roPlaiMkAc&ved=0ahUKEwj99tfcwcn9AhXMlFYBHRUUA3IQ4dUDCA8&uact=5&oq=%EB%A7%8C%EB%93%A0+%EA%B2%80%EC%83%89+%EC%82%AC%EC%9D%B4%ED%8A%B8+%EB%84%A3%EC%9C%BC%EB%A9%B4+%EB%90%A9%EB%8B%88%EB%8B%A4.&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIFCAAQogQyBQgAEKIEOgoIABBHENYEELADOgoIIRCgARDDBBAKOggIIRCgARDDBEoECEEYAFDeAlj6H2DWLGgLcAF4BYABnwGIAd4UkgEEMS4xOZgBAKABAcgBCsABAQ&sclient=gws-wiz-serp"><img width="420px" src="resources/shifter.png"></a>
</p>
<p align="center">
    <em>✨ MRC training modules : post-train & fine-tune ✨</em>
</p>
</p>

# mrc-search-engine
mrc-search-engine은 검색 엔진과 함께 사용되는 기계독해(Machine Reading Comprehension) 모듈을 개발하는 프로젝트입니다.

---
## 서비스 모듈 실행 가이드
## Docker 설치
```
sudo wget -qO- http://get.docker.com/ | sh
```
## Nvidia-docker 설치
### 저장소 및 GPG키 설정
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
### Install Nvidia-docker
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```
### docker 서비스 재시작
```
sudo systemctl restart docker
```

---
## Docker image 불러오기
- https://drive.google.com/file/d/14_IsFGQFzjRMhFAmUYCibwDx4JuOb4BB/view?usp=sharing
```
docker load -i mrc.tar
```
## Docker image 실행하기 (gpu 사용)
```
docker run -it --gpus all -p [PORT NUMBER]:8000 mrc
```

---

## 직접 구축 (Ubuntu 배포판 기준)
JDK 설치 (1.7 버전 이후) 및 JAVA_HOME 환경 변수 설정
```
apt-get install openjdk-17-jdk
```
Mecab 설치
```
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz && \
    tar xvfz mecab-0.996-ko-0.9.2.tar.gz && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure && \
    make && \
    make check && \
    make install && \
    ldconfig
```
Mecab-dictionary 설치
```
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz && \
    tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./configure && \
    make && \
    make install
```
파이썬 라이브러리 설치
```
pip install -r modules/mrc_service/requirements.txt
```
---

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