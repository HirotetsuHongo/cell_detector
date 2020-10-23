FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

MAINTAINER Hirotetsu Hongo

ENV DEBIAN_FRONTEND=noninteractive

# Install python and pipenv
RUN apt update \
 && apt install -y git-core python3.8 python3-pip python3-venv python3-tk \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.8 /usr/bin/python \
 && ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip

RUN pip install pipenv

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# User
## Get user arguments
ARG user
ARG uid
ARG gid

## Add new user
ENV USERNAME $user
RUN useradd -m $USERNAME \
 && echo "$USERNAME:$USERNAME" | chpasswd \
 && usermod --shell /bin/bash $USERNAME \
 && usermod --uid $uid $USERNAME \
 && groupmod --gid $gid $USERNAME

USER $USERNAME
WORKDIR /workspace

CMD ["/bin/bash"]
