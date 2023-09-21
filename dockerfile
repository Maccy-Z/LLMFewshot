#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
#FROM pytorch/pytorch:latest
FROM python:latest
ARG DEBIAN_FRONTEND=noninteractive

#
RUN apt update
RUN apt install -y python3-pip git
#
RUN pip3 install torch torchvision torchaudio

# RUN conda install -c conda-forge faiss-gpu
RUN pip install matplotlib scipy pandas scikit-learn tabulate
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
RUN pip install torch_geometric==2.3
RUN pip install tab-transformer-pytorch catboost networkx[default]
RUN pip install seaborn
RUN pip install toml

RUN pip install torchdiffeq
RUN pip install xgboost
RUN pip install seaborn
RUN pip install lightgbm


