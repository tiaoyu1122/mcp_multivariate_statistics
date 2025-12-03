#!/bin/bash

# MCP 多元统计分析服务部署脚本

set -e

# 默认变量设置
DEFAULT_ECS_DEPLOYMENT=false
DEFAULT_ECS_PUBLIC_IP=""
DEFAULT_BASE_URL="http://localhost:7766"

# 从环境变量或参数获取配置
MCP_ECS_DEPLOYMENT=${MCP_STATS_ECS_DEPLOYMENT:-$DEFAULT_ECS_DEPLOYMENT}
MCP_ECS_PUBLIC_IP=${MCP_STATS_ECS_PUBLIC_IP:-$DEFAULT_ECS_PUBLIC_IP}
MCP_BASE_URL=${MCP_STATS_BASE_URL:-$DEFAULT_BASE_URL}

# 如果是ECS部署但没有提供公网IP，则尝试自动获取
if [[ "$MCP_ECS_DEPLOYMENT" == "true" && -z "$MCP_ECS_PUBLIC_IP" ]]; then
    echo "警告: 未提供ECS公网IP，尝试自动获取..."
    # 尝试多种方式获取公网IP
    MCP_ECS_PUBLIC_IP=$(curl -s --connect-timeout 5 http://whatismyip.akamai.com/ || curl -s --connect-timeout 5 https://checkip.amazonaws.com/ || echo "")
    
    if [[ -z "$MCP_ECS_PUBLIC_IP" ]]; then
        echo "错误: 无法自动获取公网IP，请手动设置MCP_STATS_ECS_PUBLIC_IP环境变量"
        exit 1
    fi
    
    echo "自动获取到公网IP: $MCP_ECS_PUBLIC_IP"
fi

# # 设置BASE_URL
# if [[ "$MCP_ECS_DEPLOYMENT" == "true" && -n "$MCP_ECS_PUBLIC_IP" ]]; then
#     MCP_BASE_URL="http://$MCP_ECS_PUBLIC_IP"
# fi

# 导出变量供后续使用
export MCP_STATS_ECS_DEPLOYMENT
export MCP_STATS_ECS_PUBLIC_IP
export MCP_BASE_URL

# 输出部署信息
echo "开始部署 MCP 多元统计分析服务..."
echo "部署模式: $([[ "$MCP_STATS_ECS_DEPLOYMENT" == "true" ]] && echo "ECS部署" || echo "本地部署")"
echo "基础URL: $MCP_BASE_URL"

# 检查是否安装了 Docker
if ! command -v docker &> /dev/null
then
    echo "未检测到 Docker，请先安装 Docker"
    exit 1
fi

# 检查是否安装了 docker-compose
if ! command -v docker-compose &> /dev/null
then
    echo "未检测到 docker-compose，请先安装 docker-compose"
    exit 1
fi

# 构建 Docker 镜像
echo "构建 Docker 镜像..."
docker build --no-cache -t mcp-multivariate-statistics .

# 停止并移除现有的容器
echo "停止并移除现有容器..."
docker-compose down 2>/dev/null || true

# 启动服务
echo "启动服务..."
if [[ "$MCP_ECS_DEPLOYMENT" == "true" ]]; then
    # 在 ECS 环境中，将服务暴露在 7766 端口
    # 先尝试停止和删除可能存在的容器
    docker stop mcp-multivariate-statistics 2>/dev/null || true
    docker rm mcp-multivariate-statistics 2>/dev/null || true
    
    # 启动容器
    docker run -d -p 7766:7766 \
      --name mcp-multivariate-statistics \
      -e MCP_STATS_BASE_URL="$MCP_BASE_URL" \
      -e MCP_STATS_SERVER_HOST="0.0.0.0" \
      -e MCP_STATS_SERVER_PORT="7766" \
      -e MCP_STATS_SERVER_TRANSPORT="http" \
      -e MCP_STATS_SERVER_PATH="" \
      -v "$(pwd)/generated_files:/app/generated_files" \
      mcp-multivariate-statistics
else
    # 本地部署
    docker-compose up -d
fi

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
if [[ "$MCP_ECS_DEPLOYMENT" == "true" ]]; then
    # ECS 部署检查
    if curl -f http://localhost:7766/health > /dev/null 2>&1; then
        echo "ECS 服务部署成功！"
        echo "访问地址: $MCP_BASE_URL/mcp"
        echo "健康检查: $MCP_BASE_URL/health"
        echo "文件访问示例: $MCP_BASE_URL/generated_files/xxx.json"
    else
        echo "ECS 服务部署失败，请检查日志"
        docker logs mcp-multivariate-statistics
        exit 1
    fi
else
    # 本地部署检查
    if curl -f http://localhost:7766/health > /dev/null 2>&1; then
        echo "本地服务部署成功！"
        echo "访问地址: http://localhost:7766/mcp"
        echo "健康检查: http://localhost:7766/health"
    else
        echo "本地服务部署失败，请检查日志"
        docker-compose logs
        exit 1
    fi
fi