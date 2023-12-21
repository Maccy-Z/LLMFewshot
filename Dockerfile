FROM python:3.11
ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y python3-pip git

RUN pip install install install torch torchvision torchaudio
RUN pip install matplotlib scipy pandas scikit-learn tabulate
RUN pip install catboost
RUN pip install seaborn
RUN pip install toml




