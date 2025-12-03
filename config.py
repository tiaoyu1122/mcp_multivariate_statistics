import os
from dotenv import load_dotenv

load_dotenv()  # 确保加载 .env

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def safe_get_env(key: str, default: str = "") -> str:
    """安全获取环境变量，去除首尾空格，空值返回 default"""
    val = os.environ.get(key, "").strip()
    return val if val != "" else default

# 使用安全获取
OUTPUT_DIR = safe_get_env("MCP_STATS_OUTPUT_DIR", "generated_files")
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR)
SERVER_HOST = safe_get_env("MCP_STATS_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(safe_get_env("MCP_STATS_SERVER_PORT", "7766"))
SERVER_PATH = safe_get_env("MCP_STATS_SERVER_PATH", "/mcp")
SERVER_TRANSPORT = safe_get_env("MCP_STATS_SERVER_TRANSPORT", "http")
SERVER_LOG_LEVEL = safe_get_env("MCP_STATS_SERVER_LOG_LEVEL", "info")

# ECS 配置
ECS_DEPLOYMENT = safe_get_env("MCP_STATS_ECS_DEPLOYMENT", "false").lower() == "true"
ECS_PUBLIC_IP = safe_get_env("MCP_STATS_ECS_PUBLIC_IP", "")

# 自动构建 BASE_URL
if ECS_DEPLOYMENT and ECS_PUBLIC_IP:
    BASE_URL = f"http://{ECS_PUBLIC_IP}:{SERVER_PORT}"
else:
    BASE_URL = safe_get_env("MCP_STATS_BASE_URL", f"http://localhost:{SERVER_PORT}")

# 清理配置
FILE_LIFETIME_HOURS = int(safe_get_env("MCP_STATS_FILE_LIFETIME_HOURS", "2"))
CLEANUP_INTERVAL_MINUTES = int(safe_get_env("MCP_STATS_CLEANUP_INTERVAL_MINUTES", "5"))

# 公共文件 URL
PUBLIC_FILE_BASE_URL = f"{BASE_URL.rstrip('/')}/generated_files"