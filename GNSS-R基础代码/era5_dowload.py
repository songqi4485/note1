"""
ERA-5数据下载脚本 - 适配 TCNet 复现 (Hurricane Frank 2022)
基于文献：Shi 等 - 2024 - TCNet Triple Collocation-Based Network...
"""

import cdsapi
import os
from datetime import date, timedelta 
import time
import sys
import threading

# ================= 配置区 =================
# 保存路径 (改为相对路径，自动创建在当前脚本目录下的 data/ERA5 文件夹)
OUTPUT_DIR = r"C:\Users\SONGQI\Desktop\1\era5"

# 下载时间范围 (复现 Hurricane Frank 案例)
START_DATE = date(2025, 8, 1)
END_DATE   = date(2025, 8, 31)

# 超时设置（秒）
TIMEOUT = 60  # 5分钟超时
# ==========================================

class ERA5Downloader:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        
        try:
            self.client = cdsapi.Client(timeout=TIMEOUT)
            print("CDS API客户端初始化成功")
        except Exception as e:
            print(f"CDS API客户端初始化失败: {e}")
            print("\n!!! 关键提示 !!!")
            print("请确保你已在用户主目录 (如 C:\\Users\\你的用户名\\) 下创建了 .cdsapirc 文件")
            print("新版 CDS (2024年9月后) 配置格式如下:")
            print("------------------------------------------------")
            print("url: https://cds.climate.copernicus.eu/api")
            print("key: 你的_PERSONAL_ACCESS_TOKEN")
            print("------------------------------------------------")
            print("请去 https://cds.climate.copernicus.eu/user/profile 获取 Key")
            raise

    def download_single_day(self, year, month, day):
        # 构建文件名
        file_name = f'era5_{year}{month:02d}{day:02d}.nc'
        output_file = os.path.join(self.output_dir, file_name)
        
        if os.path.exists(output_file):
            print(f"[跳过] 文件已存在: {file_name}")
            return output_file
        
        print(f"\n正在下载: {year}-{month:02d}-{day:02d}")
        
        # 启动进度动画
        download_finished = threading.Event()
        def show_progress():
            chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            i = 0
            while not download_finished.is_set():
                sys.stdout.write(f'\r请求中... {chars[i % len(chars)]}')
                sys.stdout.flush()
                i += 1
                time.sleep(0.1)
        
        t = threading.Thread(target=show_progress)
        t.daemon = True
        t.start()
        
        try:
            # 根据论文 ，使用 ERA5 10m 风速 (U/V 分量)
            # 空间分辨率 0.25度，时间分辨率 1小时
            request_params = {
                'product_type': 'reanalysis',
                'data_format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', # 只需要这两个计算风速
                    '10m_v_component_of_wind',
                    # 'mean_sea_level_pressure', # 论文未提及用于训练，注释掉以节省空间
                    # 'significant_height_of_combined_wind_waves_and_swell',
                ],
                'year': str(year),
                'month': f'{month:02d}',
                'day': f'{day:02d}',
                'time': [f'{h:02d}:00' for h in range(24)], # 00:00 到 23:00
                'area': [90, -180, -90, 180],  # 北、西、南、东 (论文指定的西北太平洋区域)
                'grid': [0.25, 0.25],         # 论文指定的 0.25度分辨率
            }
            
            print(f"请求参数: {request_params}")
            self.client.retrieve(
                'reanalysis-era5-single-levels',
                request_params,
                output_file
            )
            download_finished.set()
            print(f"\r[成功] 已保存: {file_name}")
            return output_file
            
        except Exception as e:
            download_finished.set()
            print(f"\r[失败] {e}")
            print("可能的原因及解决方案:")
            print("1. 检查您的 .cdsapirc 配置文件是否正确设置")
            print("2. 确认您是否有权限访问该数据集")
            print("3. 检查日期是否有效且数据确实存在")
            print("4. 尝试减少请求的数据量（例如减少天数或区域大小）")
            print("5. 如果是超时问题，可以尝试增大 TIMEOUT 值")
            return None

def main():
    print(f"--- 开始下载 ERA5 数据 ({START_DATE} 至 {END_DATE}) ---")
    print("变量: 10m U-wind, 10m V-wind (用于合成风速)")
    
    downloader = ERA5Downloader(output_dir=OUTPUT_DIR)
    
    current_date = START_DATE
    while current_date <= END_DATE:
        result = downloader.download_single_day(
            year=current_date.year,
            month=current_date.month,
            day=current_date.day
        )
        # 如果下载失败，可以选择暂停一段时间再继续
        if result is None:
            print("等待10秒后继续下一个日期...")
            time.sleep(10)
        else:
            print("等待2秒后继续下一个日期...")
            time.sleep(2)
        current_date += timedelta(days=1)

    print("\n全部任务结束。")

if __name__ == "__main__":
    main()