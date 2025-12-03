import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def set_chinese_font():
    available_fonts = {f.name for f in fm.fontManager.ttflist}  # 用 set 提升查找效率
    chinese_fonts = [
        'WenQuanYi Zen Hei',   # ← 显式添加（最关键！）
        'Noto Sans CJK SC',
        'Microsoft YaHei',
        'SimHei',
        'STHeiti',
        'PingFang SC',
        'Heiti SC',
        'SimSun',
        'Songti SC',
        'STSong',
        'Arial Unicode MS'
    ]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return plt
            
    # fallback: 至少保证英文和数字能显示
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    return plt