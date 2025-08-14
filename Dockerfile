FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

ENV TRANSFORMERS_CACHE=/app/models

COPY . /app
RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /app/models && \
    python -c "from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; \
    model_name='Qwen/Qwen2-VL-2B-Instruct'; \
    Qwen2VLForConditionalGeneration.from_pretrained(model_name, cache_dir='/app/models'); \
    AutoProcessor.from_pretrained(model_name, cache_dir='/app/models')"

CMD ["python", "runpod_handler.py"]