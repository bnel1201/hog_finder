FROM fastdotai/fastai:latest

WORKDIR /code

COPY requirements_docker.txt /code/requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8501

RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

COPY . /code/

CMD streamlit run streamlit_app.py