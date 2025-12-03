FROM python:3.10-slim
# FROM python:3.12-bookworm

WORKDIR /app

COPY . .

# 配置阿里云 APT 源
RUN rm -f /etc/apt/sources.list.d/debian.sources && \
    echo "deb https://mirrors.aliyun.com/debian/ bullseye main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bullseye-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ bullseye-security main contrib non-free" >> /etc/apt/sources.list

# 安装系统依赖（matplotlib/scipy 需要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libfreetype6-dev \
        libpng-dev \
        pkg-config \
        fonts-wqy-zenhei \
        curl \
    && rm -rf /var/lib/apt/lists/*

# === 删除 Git 安装行，仅通过 requirements.txt 安装 ===
COPY requirements.txt .

# 使用阿里云 PyPI 镜像安装所有依赖（包括 fastmcp==2.5.1）
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

# 从源码安装 statsmodels（确保完整性）
RUN pip install --no-binary=statsmodels statsmodels==0.14.5

# 创建非 root 用户
RUN useradd --create-home --shell /bin/bash app

# 👇 关键：授权整个 /app 给 app 用户（必须在 USER app 之前！）
RUN chown -R app:app /app

# 切换到 app 用户
USER app

# 👇 关键：由 app 用户自己创建 generated_files（100% 可写）
RUN mkdir -p generated_files

EXPOSE 7766
CMD ["python", "my_mcp_server.py"]