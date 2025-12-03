# ECS 部署指南

本指南将帮助您在阿里云 ECS 上部署 MCP 多元统计分析服务。

## ECS 实例配置

在创建 ECS 实例时，请按照以下配置进行设置：

1. **付费类型**：建议先选择按量付费，便于测试和后续调整
2. **实例规格**：选择 `ecs.e-c1m2.large`
3. **镜像**：选择 `Alibaba Cloud Linux 3.2104 LTS 64位`
4. **扩展程序**：选择 `Docker 社区版`
5. **公网 IP**：开启分配公网 IPv4 地址
6. **登录凭证**：选择密钥对或自定义密码

## 部署步骤

### 1. 连接到 ECS 实例

使用 SSH 连接到您的 ECS 实例：

```bash
ssh root@your-ecs-public-ip
```

### 2. 安装必要的工具

确保系统已安装 git 和其他必要工具：

```bash
# 更新包管理器
yum update -y

# 安装 git
yum install git -y

# 验证 Docker 是否已安装
docker --version
docker-compose --version
```

### 3. 克隆项目代码

```bash
# 克隆项目到服务器
git clone <项目仓库地址>
cd mcp_multivariate_statistics
```

### 4. 配置环境变量

复制 `.env.example` 文件并根据需要进行修改：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置以下关键配置：

```properties
# 输出文件存储目录
MCP_STATS_OUTPUT_DIR=generated_files

# 基础URL (用于生成文件访问链接)
MCP_STATS_BASE_URL=http://localhost:7766

# 文件生命周期（小时），超过此时间的文件将被自动清理
MCP_STATS_FILE_LIFETIME_HOURS=2

# 清理任务执行间隔（分钟）
MCP_STATS_CLEANUP_INTERVAL_MINUTES=5

# 服务器配置
MCP_STATS_SERVER_HOST=0.0.0.0
MCP_STATS_SERVER_PORT=7766
MCP_STATS_SERVER_PATH=
MCP_STATS_SERVER_TRANSPORT=http
MCP_STATS_SERVER_LOG_LEVEL=info

# ECS部署配置
MCP_STATS_ECS_DEPLOYMENT=true
MCP_STATS_ECS_PUBLIC_IP=121.43.125.2

MCP_STATS_BASE_URL=http://121.43.125.2:7766

```

### 5. 部署应用

使用提供的部署脚本部署：

```bash
# 构建 Docker 镜像

# 删除所有未被使用的镜像
docker system prune -a --volumes

# 查看磁盘使用情况
docker system df


mkdir -p generated_files
chmod a+w generated_files    # 所有用户可写
# 或更宽松（但有效）：
chmod 777 generated_files

# 设置环境变量
export MCP_STATS_ECS_DEPLOYMENT=true

# 部署
./deploy.sh
```

### 6. 查看日志

部署完成后，可以通过以下方式查看日志：

```bash
# (1) 查看日志
docker logs -f

# (2)查看全部日志（不限行数）
docker logs --no-log-prefix

# (3)查看指定服务的完整日志（带时间戳）
docker logs -t mcp-multivariate-statistics

# (4)查看最近 1000 行（默认可能只保留几百行）
docker logs --tail=1000 -f -t mcp-multivariate-statistics

# (5)进入容器（交互式）
docker exec -it mcp_multivariate_statistics-mcp-multivariate-statistics-1 bash
# 手动运行服务（会阻塞终端，显示所有输出）
python my_mcp_server.py
```

如果一切正常，您应该能看到类似以下的响应：

```json
{
  "status": "healthy",
  "service": "multivariate-statistics-mcp-server",
  "version": "1.0.0"
}
```

## 使用服务

部署成功后，您可以通过以下 URL 访问服务：

- **MCP 服务地址**：`http://your-actual-public-ip:7766/mcp`
- **健康检查端点**：`http://your-actual-public-ip:7766/mcp/health`
- **文件访问地址**：`http://your-actual-public-ip:7766/generated_files/filename.json`

## 常见问题

### 1. 如何查看服务日志？

```bash
# 查看容器日志
docker logs mcp-multivariate-statistics
```

### 2. 如何重启服务？

```bash
# 停止并移除容器
docker stop mcp-multivariate-statistics && docker rm mcp-multivariate-statistics

# 重新运行容器
docker run -d -p 7766:7766 \
  --name mcp-multivariate-statistics \
  -e MCP_STATS_BASE_URL=http://your-actual-public-ip \
  mcp-multivariate-statistics
```

### 3. 如何更新服务？

```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker build -t mcp-multivariate-statistics .

# 停止并移除旧容器
docker stop mcp-multivariate-statistics && docker rm mcp-multivariate-statistics

# 运行新容器
docker run -d -p 7766:7766 \
  --name mcp-multivariate-statistics \
  -e MCP_STATS_BASE_URL=http://your-actual-public-ip \
  mcp-multivariate-statistics
```

## 安全建议

1. **防火墙设置**：在安全组中只开放必要的端口（如 7766、22）
2. **定期更新**：定期更新系统和 Docker 镜像以修复安全漏洞
3. **使用 HTTPS**：在生产环境中考虑使用 HTTPS 来保护数据传输
4. **监控和日志**：启用阿里云监控服务并定期检查应用日志