FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y wget build-essential
RUN apt-get install -y g++ openjdk-17-jdk python3-dev python3-pip curl

ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64
RUN export JAVA_HOME

# Download and install MeCab
RUN wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz && \
    tar xvfz mecab-0.996-ko-0.9.2.tar.gz && \
    cd mecab-0.996-ko-0.9.2 && \
    ./configure && \
    make && \
    make check && \
    make install && \
    ldconfig

# Download and install MeCab - dictionary
RUN wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz && \
    tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz && \
    cd mecab-ko-dic-2.1.1-20180720 && \
    ./configure && \
    make && \
    make install

COPY modules/mrc_service .

RUN pip install --no-cache-dir -r requirements.txt

# Discard unnessary files
RUN rm mecab-0.996-ko-0.9.2.tar.gz
RUN rm -rf mecab-0.996-ko-0.9.2
RUN rm -rf mecab-ko-dic-2.1.1-20180720
RUN rm mecab-ko-dic-2.1.1-20180720.tar.gz
RUN rm requirements.txt

ENV DEBIAN_FRONTEND=null

EXPOSE 8000

CMD ["uvicorn", "--host=0.0.0.0", "--port", "8000", "server:app"]

