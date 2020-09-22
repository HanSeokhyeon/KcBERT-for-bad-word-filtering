FROM        pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
MAINTAINER  sah0322@naver.com

RUN         apt-get -y update && apt-get -y install git
RUN         pip install --upgrade pip
RUN         pip install transformers pytorch_lightning emoji soynlp pandas pymongo

WORKDIR     /opt/project

RUN         git clone https://github.com/e9t/nsmc

ADD         . /opt/project

CMD         ["/bin/bash"]