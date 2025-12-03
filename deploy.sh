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
MCP_STATS_BASE_URL=${MCP_STATS_BASE_URL:-$DEFAULT_BASE_URL}

# 如果是ECS部署但没有提供公网IP，则尝试自动获取
if [[ "$MCP_ECS_DEPLOYMENT" == "true" && -z "$MCP_ECS_PUBLIC_IP" ]]; then
    echo "警告: 未提供ECS公网IP，尝试自动获取..."
    # 尝试多种方式获取公网IP
    MCP_ECS_PUBLIC_IP=$(curl -s --connect-timeout 5 http://whatismyip.akamai.com/ || \
                        curl -s --connect-timeout 5 https://checkip.amazonaws.com/ || \
                        echo "")
    
    if [[ -z "$MCP_ECS_PUBLIC_IP" ]]; then
        echo "错误: 无法自动获取公网IP，请手动设置MCP_STATS_ECS_PUBLIC_IP环境变量"
        exit 1
    fi
    
    echo "自动获取到公网IP: $MCP_ECS_PUBLIC_IP"
fi

# 设置BASE_URL（关键：包含端口）
if [[ "$MCP_ECS_DEPLOYMENT" == "true" && -n "$MCP_ECS_PUBLIC_IP" ]]; then
    MCP_STATS_BASE_URL="http://$MCP_ECS_PUBLIC_IP:7766"
fi

# 导出变量供后续使用
export MCP_STATS_ECS_DEPLOYMENT
export MCP_STATS_ECS_PUBLIC_IP
export MCP_STATS_BASE_URL

# 输出部署信息
echo "开始部署 MCP 多元统计分析服务..."
echo "部署模式: $([[ "$MCP_ECS_DEPLOYMENT" == "true" ]] && echo "ECS部署" || echo "本地部署")"
echo "基础URL: $MCP_STATS_BASE_URL"

# 检查是否安装了 Docker
if ! command -v docker &> /dev/null; then
    echo "未检测到 Docker，请先安装 Docker"
    exit 1
fi

# 检查是否安装了 docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "未检测到 docker-compose，请先安装 docker-compose"
    exit 1
fi

# 构建 Docker 镜像
echo "构建 Docker 镜像..."
docker build --no-cache -t mcp-multivariate-statistics .

# 停止并移除现有的容器（兼容两种部署方式）
echo "停止并移除现有容器..."
docker-compose down 2>/dev/null || true
docker stop mcp-multivariate-statistics 2>/dev/null || true
docker rm mcp-multivariate-statistics 2>/dev/null || true

# 启动服务
echo "启动服务..."
if [[ "$MCP_ECS_DEPLOYMENT" == "true" ]]; then
    docker run -d -p 7766:7766 \
        --name mcp-multivariate-statistics \
        -e MCP_STATS_BASE_URL="$MCP_STATS_BASE_URL" \
        -e MCP_STATS_SERVER_HOST="0.0.0.0" \
        -e MCP_STATS_SERVER_PORT="7766" \
        -e MCP_STATS_SERVER_TRANSPORT="http" \
        -e MCP_STATS_SERVER_PATH="" \
        -v "$(pwd)/generated_files:/app/generated_files" \
        mcp-multivariate-statistics
else
    docker-compose up -d
fi

# 等待服务初步启动
echo "等待服务启动（最多 120 秒）..."
sleep 5

# 健康检查函数
check_health() {
    local url="$1"
    local mode="$2"
    local max_retries=12
    local retry=1

    while [ $retry -le $max_retries ]; do
        echo "第 $retry 次健康检查: curl -f -s --max-time 10 '$url'"
        # 使用 -f 让 curl 在 HTTP 4xx/5xx 时也失败
        # 并捕获完整响应
        response=$(curl -f -s --max-time 10 "$url" 2>/dev/null)
        if [ $? -eq 0 ] && echo "$response" | grep -q '"status":"healthy"'; then
            echo "$mode 服务部署成功！"
            return 0
        fi
        echo "健康检查未通过（响应: ${response:-<empty>}），等待 3 秒后重试..."
        sleep 10
        ((retry++))
    done

    echo "$mode 服务部署失败：健康检查始终未通过"
    return 1
}

# 执行健康检查
if [[ "$MCP_ECS_DEPLOYMENT" == "true" ]]; then
    if check_health "http://localhost:7766/health" "ECS"; then
        echo "访问地址: $MCP_STATS_BASE_URL/mcp"
        echo "健康检查: $MCP_STATS_BASE_URL/health"
        echo "文件访问示例: $MCP_STATS_BASE_URL/generated_files/xxx.json"
    else
        docker logs mcp-multivariate-statistics
        exit 1
    fi
else
    if check_health "http://localhost:7766/health" "本地"; then
        echo "访问地址: http://localhost:7766/mcp"
        echo "健康检查: http://localhost:7766/health"
    else
        docker-compose logs
        exit 1
    fi
fi