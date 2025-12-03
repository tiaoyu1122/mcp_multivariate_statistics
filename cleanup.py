import os
import time
import glob
import threading
from datetime import datetime
from config import OUTPUT_DIR, FILE_LIFETIME_HOURS, CLEANUP_INTERVAL_MINUTES


def cleanup_old_files():
    """
    清理超过指定时间的文件（支持任意类型）
    """
    try:
        # 确保目录存在
        if not os.path.exists(OUTPUT_DIR):
            return
            
        # 计算过期时间戳
        cutoff_time = time.time() - FILE_LIFETIME_HOURS * 3600
        cutoff_time_str = datetime.fromtimestamp(cutoff_time).strftime('%Y-%m-%d %H:%M:%S')
        current_time_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[{current_time_str}] 开始检查过期文件，清理截止时间: {cutoff_time_str}")
        
        # 查找所有文件（不限类型）
        pattern = os.path.join(OUTPUT_DIR, "*")
        all_files = glob.glob(pattern)
        
        # 过滤出文件（排除目录）
        files = [f for f in all_files if os.path.isfile(f)]
        
        # 删除过期的文件
        deleted_count = 0
        for file_path in files:
            # 获取文件的修改时间
            mod_time = os.path.getmtime(file_path)
            
            # 如果文件超过指定时间，则删除
            if mod_time < cutoff_time:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"已删除过期文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")
        
        if deleted_count > 0:
            print(f"已清理 {deleted_count} 个过期文件")
        else:
            print("没有发现过期文件")
            
    except Exception as e:
        print(f"清理过期文件时出错: {e}")


def start_cleanup_scheduler():
    """
    启动定时清理任务
    """
    def run_cleanup():
        while True:
            cleanup_old_files()
            # 按配置的间隔时间执行清理
            time.sleep(CLEANUP_INTERVAL_MINUTES * 60)
    
    # 在后台线程中运行清理任务
    cleanup_thread = threading.Thread(target=run_cleanup, daemon=True)
    cleanup_thread.start()
    print(f"文件自动清理任务已启动，每{CLEANUP_INTERVAL_MINUTES}分钟检查一次过期文件")