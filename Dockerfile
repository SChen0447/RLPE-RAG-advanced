FROM python:3.12-slim

# 环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /naive_rag

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*



# 装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r /naive_rag/requirements.txt

# RUN python -m spacy download en_core_web_sm

# 拷代码
COPY app ./app

# 数据
RUN mkdir -p /naive_rag/data_base

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1


# 生产启动
CMD ["fastapi", "run", "app/main.py", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]