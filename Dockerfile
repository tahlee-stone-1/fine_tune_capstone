FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY modeling ./modeling

RUN pip install --no-cache-dir \
    transformers==4.44.2 \
    peft==0.13.0 \
    accelerate==1.1.0 \
    datasets==2.20.0 \
    dagshub==0.6.3 \
    mlflow==2.16.0 \
    sentencepiece

ENV MODEL_NAME=
ENV TRAIN_FILES=
ENV HF_TOKEN=""
ENV DAGSHUB_API_TOKEN=""

CMD ["python", "modeling/finetune.py"]
