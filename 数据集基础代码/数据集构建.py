import xarray as xr
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pickle
from tqdm import tqdm


# 时间格式转换函数（将 UTC 时间戳转换为 pandas datetime 格式）
def convert_datetime(time_values):
    return pd.to_datetime(time_values, unit='s')


# 经度转换函数（将经度范围从 [0, 360] 转换到 [-180, 180]）
def shift_lon(lon):
    lon[lon > 180.0] -= 360.0  # 将经度从 [0, 360] 转换为 [-180, 180]
    return lon


# 经度转换函数（将经度范围从 [-180, 180] 转换到 [0, 360]）
def shift_lon_to_360(lon):
    """将经度从[-180, 180]转换为[0, 360]"""
    lon_360 = lon.copy()
    lon_360[lon_360 < 0] += 360.0
    return lon_360


# 新增：检测ERA5经度范围的函数
def detect_era5_lon_range(era5_ds):
    """检测ERA5数据集中经度的范围"""
    lon_values = era5_ds.longitude.values
    lon_min = float(lon_values.min())
    lon_max = float(lon_values.max())

    print(f"ERA5经度范围检测:")
    print(f"  最小经度: {lon_min:.1f}°")
    print(f"  最大经度: {lon_max:.1f}°")

    # 判断经度范围类型
    if lon_min >= -180 and lon_max <= 180:
        coord_system = "[-180, 180]"
        needs_conversion = False
    elif lon_min >= 0 and lon_max <= 360:
        coord_system = "[0, 360]"
        needs_conversion = True
    else:
        coord_system = "混合范围"
        needs_conversion = True
        print(f"  警告: 检测到异常的经度范围，可能包含混合坐标系")

    print(f"  坐标系类型: {coord_system}")
    print(f"  需要转换: {'是' if needs_conversion else '否'}")

    return needs_conversion, coord_system


# 新增：将ERA5经度转换为[-180, 180]的函数
def convert_era5_longitude_to_180(era5_ds):
    """将ERA5数据集的经度从[0, 360]转换为[-180, 180]"""
    print(f"🔧 开始转换ERA5经度坐标系...")

    # 获取原始经度
    original_lon = era5_ds.longitude.values
    print(f"转换前经度范围: {original_lon.min():.1f}° 到 {original_lon.max():.1f}°")

    # 转换经度
    new_lon = original_lon.copy()
    new_lon[new_lon > 180] -= 360

    # 对经度进行排序（从-180到180）
    sorted_indices = np.argsort(new_lon)
    new_lon_sorted = new_lon[sorted_indices]

    print(f"转换后经度范围: {new_lon_sorted.min():.1f}° 到 {new_lon_sorted.max():.1f}°")

    # 重新排列数据以匹配新的经度顺序
    era5_ds_converted = era5_ds.isel(longitude=sorted_indices)
    era5_ds_converted = era5_ds_converted.assign_coords(longitude=new_lon_sorted)

    print(f"✅ ERA5经度坐标系转换完成")
    return era5_ds_converted


class GRDataMod:

    def __init__(self, filename):
        self.rcg_threshold = 1.0  # 范围修正增益（RCG）阈值
        self.variable_list = [
            "prn_code", "track_id", "sp_lat", "sp_lon", "sp_alt",
            "sp_inc_angle", "rx_to_sp_range", "tx_to_sp_range",
            "sp_rx_gain", "ddm_nbrcs", "ddm_les", "quality_flags",
            "ddm_timestamp_utc"  # 时间戳
        ]

        try:
            # 加载数据集
            print(f"正在加载CYGNSS数据文件: {filename}")
            self.ds = xr.open_dataset(filename)
            print(f"数据文件加载成功")

            # 处理数据
            success = self.process_data()
            if not success:
                raise Exception("数据处理失败")

        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise

    def process_data(self):
        if not self.extract_variables():
            return False

        self.ds = self.ds.stack(time=("sample", "ddm"))
        print(f"堆叠后的数据维度: {self.ds.dims}")

        time_converted = convert_datetime(self.ds['ddm_timestamp_utc'].values)

        all_vars = {}
        for var_name in self.ds.variables:
            if var_name != 'time':  # 跳过时间维度本身
                try:
                    all_vars[var_name] = (['time'], self.ds[var_name].values)
                except Exception as e:
                    print(f"警告: 无法处理变量 {var_name}: {e}")
                    continue

        # 创建新的数据集，使用转换后的时间作为坐标
        self.ds = xr.Dataset(all_vars, coords={'time': time_converted})
        print(f"时间格式已转换。")
        print(f"转换后的变量: {list(self.ds.variables.keys())}")

        # 验证关键变量是否存在
        if 'sp_lon' not in self.ds.variables:
            print(f"错误: sp_lon变量丢失！")
            return False
        if 'sp_lat' not in self.ds.variables:
            print(f"错误: sp_lat变量丢失！")
            return False

        # 4. 转换经度到 [-180, 180] 范围
        self.ds['sp_lon'].values = shift_lon(self.ds['sp_lon'].values)
        print(f"CYGNSS经度格式已转换为[-180, 180]。")

        # 5. 筛选海洋中的数据（仅筛选操作，替换原来的简单筛选）
        self.filter_ocean_data()

        # 6. 去除缺失值
        self.ds = self.ds.dropna("time", how="any")
        print(f"去除缺失值后的数据点数: {self.ds['sp_lat'].size}")

        # 7. 根据质量标志和范围修正增益筛选有效数据
        self.filter_data()

        # 8. 将筛选后的数据保存为表格形式的 .pkl 文件
        self.save_data_as_pkl()

        return True

    def extract_variables(self):
        """提取需要的变量"""
        print(f"数据集中所有可用变量: {list(self.ds.variables.keys())}")

        # 检查关键变量
        critical_vars = ["sp_lat", "sp_lon", "ddm_timestamp_utc", "quality_flags"]
        missing_critical = [var for var in critical_vars if var not in self.ds.variables]

        if missing_critical:
            print(f"错误: 缺失关键变量 {missing_critical}")
            print("无法继续处理，请检查输入文件")
            return False

        selected_vars = {}
        missing_vars = []

        for var in self.variable_list:
            if var in self.ds.variables:
                selected_vars[var] = self.ds[var]
                print(f"✓ 成功提取变量: {var}")
            else:
                missing_vars.append(var)
                print(f"✗ 警告: 数据集中没有找到变量 '{var}'")

        if missing_vars:
            print(f"缺失的非关键变量: {missing_vars}")

        self.ds = xr.Dataset(selected_vars)
        print(f"最终提取的变量: {list(self.ds.keys())}")
        return True

    def filter_ocean_data(self):
        """使用位运算筛选开阔海洋数据"""
        quality_flags = self.ds["quality_flags"]

        print(f"\n海洋样本筛选分析:")
        print(f"原始数据点数: {quality_flags.size}")

        # 转换为numpy数组进行位运算
        qf_values = quality_flags.values.astype(np.float64)

        # 处理NaN值
        nan_mask = np.isnan(qf_values)
        if nan_mask.any():
            print(f"发现 {nan_mask.sum()} 个NaN值，将被排除")
            qf_values[nan_mask] = 999999  # 设置一个大值确保被排除

        # 转换为整数进行位运算
        qf_values = qf_values.astype(np.int64)

        # 定义陆地相关的位标识
        LAND_FLAGS = {
            'sp_over_land': 1024,  # 镜面反射点在陆地上
            'sp_very_near_land': 2048,  # 镜面反射点距离陆地<25km
            'sp_near_land': 4096  # 镜面反射点距离陆地<50km
        }

        # 统计陆地标识分布
        total_samples = len(qf_values)
        for flag_name, flag_value in LAND_FLAGS.items():
            flag_count = ((qf_values & flag_value) != 0).sum()
            flag_percentage = (flag_count / total_samples) * 100
            print(f"{flag_name:20}: {flag_count:8d} 个样本 ({flag_percentage:6.2f}%)")

        # 筛选开阔海洋数据：排除所有陆地相关标识
        ocean_mask = ((qf_values & 1024) == 0) & \
                     ((qf_values & 2048) == 0) & \
                     ((qf_values & 4096) == 0)

        # 确保经纬度有效
        valid_location_mask = (~np.isnan(self.ds['sp_lat'].values)) & \
                              (~np.isnan(self.ds['sp_lon'].values))
        final_mask = ocean_mask & valid_location_mask

        ocean_count = final_mask.sum()
        ocean_percentage = (ocean_count / total_samples) * 100

        print(f"开阔海洋样本数量: {ocean_count:8d} ({ocean_percentage:6.2f}%)")

        # 应用筛选掩码 - 这是唯一的筛选操作
        self.ds = self.ds.isel(time=final_mask)
        print(f"筛选后剩余的数据点数: {self.ds['sp_lat'].size}")

    def filter_data(self):
        # 计算范围修正增益（RCG）
        if all(k in self.ds for k in ["sp_rx_gain", "rx_to_sp_range", "tx_to_sp_range"]):
            sp_rx_gain = 10 ** (self.ds["sp_rx_gain"] / 10)  # 转换增益从 dB 到线性值
            inc_range = self.ds["rx_to_sp_range"]
            sca_range = self.ds["tx_to_sp_range"]
            range_corr_gain = sp_rx_gain * 1.e27 / (inc_range * sca_range) ** 2  # 按公式计算 RCG
            self.ds["range_corr_gain"] = range_corr_gain

            # 分析RCG分布
            rcg_values = range_corr_gain.values
            rcg_valid = rcg_values[~np.isnan(rcg_values)]
            print(f"\nRCG分析:")
            print(f"RCG有效值数量: {len(rcg_valid)}")
            if len(rcg_valid) > 0:
                print(f"RCG范围: {rcg_valid.min():.2e} 到 {rcg_valid.max():.2e}")
                print(f"RCG阈值: {self.rcg_threshold:.1f}")

                # 根据 RCG 阈值筛选
                rcg_mask = rcg_values > self.rcg_threshold
                valid_rcg_count = rcg_mask.sum()
                rcg_percentage = (valid_rcg_count / len(rcg_values)) * 100

                print(f"RCG > {self.rcg_threshold} 的样本: {valid_rcg_count} ({rcg_percentage:.1f}%)")

                self.ds = self.ds.isel(time=rcg_mask)
                print(f"按照 RCG 阈值筛选后剩余的数据点数: {self.ds['sp_lat'].size}")
            else:
                print("警告: 没有有效的RCG值")
        else:
            print("警告: 计算 RCG 时缺少必要的变量。")

    def save_data_as_pkl(self):
        # 将数据转换为 DataFrame，每一列是一个变量
        df = self.ds.to_dataframe()
        df.reset_index(inplace=True)  # 确保时间作为列

        # 数据质量检查
        print(f"\n数据质量检查:")
        print(f"DataFrame shape: {df.shape}")
        print(f"缺失值统计:")
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count} 个缺失值")

        print(f"转换为 DataFrame 后的前几行:\n{df.head()}")

        # 将 DataFrame 保存为 .pkl 文件
        output_filename = 'restructured_gr_data_ocean_filtered.pkl'
        df.to_pickle(output_filename)
        print(f"处理后的海洋数据已保存为 '{output_filename}'")


def verify_data_coverage(cygnss_df, era5_ds):
    """验证CYGNSS和ERA5数据的时空覆盖范围"""
    print(f"\n" + "=" * 60)
    print(f"数据覆盖范围验证")
    print(f"=" * 60)

    # ERA5数据范围
    era5_time_range = era5_ds.valid_time.values
    era5_lat_range = [float(era5_ds.latitude.min()), float(era5_ds.latitude.max())]
    era5_lon_range = [float(era5_ds.longitude.min()), float(era5_ds.longitude.max())]

    print(f"\nERA5数据范围:")
    print(f"  时间: {pd.Timestamp(era5_time_range[0])} 到 {pd.Timestamp(era5_time_range[-1])}")
    print(f"  时间点数: {len(era5_time_range)}")
    print(f"  纬度: {era5_lat_range[0]:.1f}° 到 {era5_lat_range[1]:.1f}°")
    print(f"  经度: {era5_lon_range[0]:.1f}° 到 {era5_lon_range[1]:.1f}° (统一坐标系)")

    # CYGNSS数据范围
    cygnss_time_range = [cygnss_df['ddm_timestamp_utc'].min(), cygnss_df['ddm_timestamp_utc'].max()]
    cygnss_lat_range = [float(cygnss_df['sp_lat'].min()), float(cygnss_df['sp_lat'].max())]
    cygnss_lon_range = [float(cygnss_df['sp_lon'].min()), float(cygnss_df['sp_lon'].max())]

    print(f"\nCYGNSS数据范围:")
    print(f"  时间: {cygnss_time_range[0]} 到 {cygnss_time_range[1]}")
    print(f"  纬度: {cygnss_lat_range[0]:.1f}° 到 {cygnss_lat_range[1]:.1f}°")
    print(f"  经度: {cygnss_lon_range[0]:.1f}° 到 {cygnss_lon_range[1]:.1f}° ([-180,180]坐标系)")

    # 检查覆盖情况
    print(f"\n覆盖情况分析:")

    # 时间覆盖
    era5_time_start = pd.Timestamp(era5_time_range[0])
    era5_time_end = pd.Timestamp(era5_time_range[-1])
    time_covered = (cygnss_time_range[0] >= era5_time_start) and (cygnss_time_range[1] <= era5_time_end)
    print(f"  时间覆盖: {'✓' if time_covered else '✗'}")

    # 纬度覆盖
    lat_covered = (cygnss_lat_range[0] >= era5_lat_range[0]) and (cygnss_lat_range[1] <= era5_lat_range[1])
    print(f"  纬度覆盖: {'✓' if lat_covered else '✗'}")

    # 经度覆盖（现在两者都是[-180, 180]坐标系）
    lon_covered = (cygnss_lon_range[0] >= era5_lon_range[0]) and (cygnss_lon_range[1] <= era5_lon_range[1])
    print(f"  经度覆盖: {'✓' if lon_covered else '✗'}")

    if not (time_covered and lat_covered and lon_covered):
        print(f"\n⚠️  警告: 存在覆盖范围不足的情况，可能导致插值失败")
    else:
        print(f"\n✓ 所有维度覆盖完整，预期插值成功率较高")

    return time_covered, lat_covered, lon_covered


# 改进的插值匹配函数
def interpolate_wind_speed(cygnss_df, era5_ds):
    """进行ERA5与CYGNSS数据的插值匹配 - 优化版本（统一坐标系）"""
    print(f"\n开始ERA5-CYGNSS数据匹配...")
    print(f"CYGNSS数据点数: {len(cygnss_df)}")

    # 首先验证数据覆盖范围
    time_covered, lat_covered, lon_covered = verify_data_coverage(cygnss_df, era5_ds)

    # 提取ERA5数据
    time_era5 = era5_ds.valid_time.values  # 时间
    lat_era5 = era5_ds.latitude.values  # 纬度
    lon_era5 = era5_ds.longitude.values  # 经度 （现在已经是[-180, 180]）
    u10_era5 = era5_ds.u10.values  # U10 风速分量
    v10_era5 = era5_ds.v10.values  # V10 风速分量

    print(f"\nERA5数据信息:")
    print(f"  时间点数: {len(time_era5)}")
    print(f"  纬度网格: {len(lat_era5)} 点, 范围 {lat_era5.min():.1f}° 到 {lat_era5.max():.1f}°")
    print(f"  经度网格: {len(lon_era5)} 点, 范围 {lon_era5.min():.1f}° 到 {lon_era5.max():.1f}°")

    # 时间处理
    time_era5 = np.array([pd.Timestamp(t).timestamp() for t in time_era5])
    cygnss_df = cygnss_df.copy()  # 避免修改原始数据
    cygnss_df['ddm_timestamp_utc'] = pd.to_datetime(cygnss_df['ddm_timestamp_utc'])
    cygnss_df['ddm_timestamp_sec'] = cygnss_df['ddm_timestamp_utc'].astype('int64') / 1e9
    cygnss_df['ddm_timestamp_sec'] = np.round(cygnss_df['ddm_timestamp_sec'] * 2) / 2
    cygnss_df['ddm_timestamp_utc'] = pd.to_datetime(cygnss_df['ddm_timestamp_sec'], unit='s')

    # 🔧 优化关键：现在两者都使用[-180, 180]坐标系，无需转换
    print(f"\n✅ 坐标系统一优化: 两个数据集都使用[-180, 180]坐标系")
    print(f"CYGNSS经度范围: {cygnss_df['sp_lon'].min():.1f}° 到 {cygnss_df['sp_lon'].max():.1f}°")
    print(f"ERA5经度范围: {lon_era5.min():.1f}° 到 {lon_era5.max():.1f}°")

    # 检查数据范围
    print(f"\n数据范围检查:")
    time_in_range = ((cygnss_df['ddm_timestamp_sec'] >= time_era5.min()) &
                     (cygnss_df['ddm_timestamp_sec'] <= time_era5.max())).sum()
    lat_in_range = ((cygnss_df['sp_lat'] >= lat_era5.min()) &
                    (cygnss_df['sp_lat'] <= lat_era5.max())).sum()
    lon_in_range = ((cygnss_df['sp_lon'] >= lon_era5.min()) &
                    (cygnss_df['sp_lon'] <= lon_era5.max())).sum()

    total_points = len(cygnss_df)
    print(f"  时间范围内的点: {time_in_range}/{total_points} ({100 * time_in_range / total_points:.1f}%)")
    print(f"  纬度范围内的点: {lat_in_range}/{total_points} ({100 * lat_in_range / total_points:.1f}%)")
    print(f"  经度范围内的点: {lon_in_range}/{total_points} ({100 * lon_in_range / total_points:.1f}%)")

    # 所有维度都在范围内的点
    all_in_range = ((cygnss_df['ddm_timestamp_sec'] >= time_era5.min()) &
                    (cygnss_df['ddm_timestamp_sec'] <= time_era5.max()) &
                    (cygnss_df['sp_lat'] >= lat_era5.min()) &
                    (cygnss_df['sp_lat'] <= lat_era5.max()) &
                    (cygnss_df['sp_lon'] >= lon_era5.min()) &
                    (cygnss_df['sp_lon'] <= lon_era5.max())).sum()

    print(f"  所有维度范围内的点: {all_in_range}/{total_points} ({100 * all_in_range / total_points:.1f}%)")

    # 创建插值函数
    print(f"\n创建插值函数...")
    interp_U10 = RegularGridInterpolator(
        (time_era5, lat_era5, lon_era5), u10_era5,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    interp_V10 = RegularGridInterpolator(
        (time_era5, lat_era5, lon_era5), v10_era5,
        method='linear', bounds_error=False, fill_value=np.nan
    )

    # 进行插值匹配
    print(f"开始插值计算...")
    matched_wind_speeds = []
    valid_matches = 0
    boundary_failures = {'time': 0, 'lat': 0, 'lon': 0, 'interpolation': 0}

    for idx, row in tqdm(cygnss_df.iterrows(), total=len(cygnss_df), desc="Processing CyGNSS Data"):
        time_point = row['ddm_timestamp_sec']
        lat_point = row['sp_lat']
        lon_point = row['sp_lon']  # 🔧 关键优化：直接使用[-180, 180]经度，无需转换

        # 检查边界情况
        time_ok = time_era5.min() <= time_point <= time_era5.max()
        lat_ok = lat_era5.min() <= lat_point <= lat_era5.max()
        lon_ok = lon_era5.min() <= lon_point <= lon_era5.max()

        if not time_ok:
            boundary_failures['time'] += 1
        if not lat_ok:
            boundary_failures['lat'] += 1
        if not lon_ok:
            boundary_failures['lon'] += 1

        if time_ok and lat_ok and lon_ok:
            # 进行插值
            U10_interp = interp_U10((time_point, lat_point, lon_point))
            V10_interp = interp_V10((time_point, lat_point, lon_point))

            if not (np.isnan(U10_interp) or np.isnan(V10_interp)):
                wind_speed = np.sqrt(U10_interp ** 2 + V10_interp ** 2)
                valid_matches += 1
            else:
                wind_speed = np.nan
                boundary_failures['interpolation'] += 1
        else:
            wind_speed = np.nan

        matched_wind_speeds.append(wind_speed)

    # 添加风速到DataFrame
    cygnss_df['WS'] = matched_wind_speeds

    # 详细的失败分析
    print(f"\n" + "=" * 60)
    print(f"插值结果详细分析")
    print(f"=" * 60)

    valid_wind_count = (~np.isnan(matched_wind_speeds)).sum()
    match_percentage = (valid_wind_count / len(cygnss_df)) * 100

    print(f"成功匹配: {valid_wind_count} / {len(cygnss_df)} ({match_percentage:.1f}%)")

    if valid_wind_count > 0:
        print(f"风速范围: {np.nanmin(matched_wind_speeds):.2f} - {np.nanmax(matched_wind_speeds):.2f} m/s")
        print(f"平均风速: {np.nanmean(matched_wind_speeds):.2f} m/s")

    print(f"\n失败原因分析:")
    print(f"  时间超出范围: {boundary_failures['time']} ({100 * boundary_failures['time'] / len(cygnss_df):.1f}%)")
    print(f"  纬度超出范围: {boundary_failures['lat']} ({100 * boundary_failures['lat'] / len(cygnss_df):.1f}%)")
    print(f"  经度超出范围: {boundary_failures['lon']} ({100 * boundary_failures['lon'] / len(cygnss_df):.1f}%)")
    print(
        f"  插值计算失败: {boundary_failures['interpolation']} ({100 * boundary_failures['interpolation'] / len(cygnss_df):.1f}%)")

    # 如果匹配率仍然很低，给出建议
    if match_percentage < 90:
        print(f"\n⚠️  匹配率较低 ({match_percentage:.1f}%)，建议检查:")
        print(f"   1. ERA5数据的时间范围是否覆盖CYGNSS数据")
        print(f"   2. ERA5数据是否包含所需的地理区域")
        print(f"   3. 数据文件是否完整")

    # 重新排序列并只保留需要的列
    desired_columns = [
        'prn_code', 'track_id', 'sp_alt', 'sp_inc_angle',
        'rx_to_sp_range', 'tx_to_sp_range', 'sp_rx_gain',
        'ddm_nbrcs', 'ddm_les', 'quality_flags', 'range_corr_gain',
        'WS', 'ddm_timestamp_utc', 'sp_lat', 'sp_lon'
    ]

    # 检查哪些列存在
    available_columns = [col for col in desired_columns if col in cygnss_df.columns]
    missing_columns = [col for col in desired_columns if col not in cygnss_df.columns]

    if missing_columns:
        print(f"\n警告: 缺失的列: {missing_columns}")

    cygnss_df = cygnss_df[available_columns]
    print(f"重新排序后的列顺序: {list(cygnss_df.columns)}")

    # 🔧 关键修改：去除所有包含缺失值的样本
    print(f"\n" + "=" * 60)
    print(f"🗑️  去除缺失值处理")
    print(f"=" * 60)

    # 统计去除前的情况
    original_count = len(cygnss_df)
    missing_ws_count = cygnss_df['WS'].isna().sum()
    valid_ws_count = original_count - missing_ws_count

    print(f"去除缺失值前的数据统计:")
    print(f"  总样本数: {original_count}")
    print(f"  有效风速样本: {valid_ws_count} ({100 * valid_ws_count / original_count:.1f}%)")
    print(f"  缺失风速样本: {missing_ws_count} ({100 * missing_ws_count / original_count:.1f}%)")

    # 去除WS列中包含NaN的行
    cygnss_df_clean = cygnss_df.dropna(subset=['WS'])

    print(f"\n去除缺失值后的数据统计:")
    print(f"  最终样本数: {len(cygnss_df_clean)}")
    print(f"  去除的样本数: {original_count - len(cygnss_df_clean)}")
    print(f"  数据完整率: 100.0%")

    # 验证数据完整性
    remaining_nan_count = cygnss_df_clean.isna().sum().sum()
    if remaining_nan_count == 0:
        print(f"  ✅ 验证通过：数据集中无任何缺失值")
    else:
        print(f"  ⚠️  警告：仍有 {remaining_nan_count} 个缺失值")

    # 最终数据质量检查
    print(f"\n最终数据质量检查:")
    print(f"最终数据集大小: {cygnss_df_clean.shape}")
    print(f"风速统计:")
    print(f"  风速范围: {cygnss_df_clean['WS'].min():.2f} - {cygnss_df_clean['WS'].max():.2f} m/s")
    print(f"  平均风速: {cygnss_df_clean['WS'].mean():.2f} m/s")
    print(f"  风速标准差: {cygnss_df_clean['WS'].std():.2f} m/s")

    # 检查是否还有其他列的缺失值
    print(f"各列缺失值检查:")
    has_missing = False
    for col in cygnss_df_clean.columns:
        missing_count = cygnss_df_clean[col].isna().sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} 个缺失值 ❌")
            has_missing = True

    if not has_missing:
        print(f"  ✅ 所有列均无缺失值")

    # 保存结果
    output_file = "cygnss_era5_2024.2.11-3.2.pkl"
    cygnss_df_clean.to_pickle(output_file)
    print(f"\n✅ 无缺失值的完整数据集已保存至 {output_file}")

    return cygnss_df_clean


# 主函数
def main(cygnss_file, era5_file):
    print("=" * 60)
    print("CYGNSS-ERA5 海洋数据集构建 (优化坐标系版)")
    print("=" * 60)

    # Step 1: 处理CYGNSS数据并筛选海洋样本
    print("\n步骤1: 处理CYGNSS数据并筛选海洋样本")
    gr_data = GRDataMod(cygnss_file)

    # Step 2: 加载ERA5数据并检测/转换坐标系
    print(f"\n步骤2: 加载ERA5数据并处理坐标系")
    era5_ds = xr.open_dataset(era5_file)
    print(f"ERA5数据集加载完成: {era5_file}")
    print(f"ERA5变量: {list(era5_ds.variables.keys())}")

    # 🔧 新增：检测ERA5经度范围并进行必要的转换
    needs_conversion, coord_system = detect_era5_lon_range(era5_ds)

    if needs_conversion:
        print(f"\n🔄 需要转换ERA5坐标系...")
        era5_ds = convert_era5_longitude_to_180(era5_ds)
    else:
        print(f"\n✅ ERA5已使用[-180, 180]坐标系，无需转换")

    # Step 3: 读取处理后的CYGNSS数据
    print(f"\n步骤3: 读取处理后的CYGNSS数据")
    cygnss_df = pd.read_pickle('restructured_gr_data_ocean_filtered.pkl')
    print(f"海洋筛选后的CYGNSS数据加载完成，共 {len(cygnss_df)} 个数据点")

    # Step 4: 插值匹配并去除缺失值
    print(f"\n步骤4: ERA5-CYGNSS插值匹配并生成无缺失值数据集")
    final_df = interpolate_wind_speed(cygnss_df, era5_ds)

    print("\n" + "=" * 60)
    print("优化后无缺失值数据集构建完成！")
    print("=" * 60)

    # 最终统计
    final_count = len(final_df)
    original_count = len(cygnss_df)
    retention_rate = 100 * final_count / original_count

    print(f"最终结果:")
    print(f"  原始数据点: {original_count}")
    print(f"  最终数据点: {final_count}")
    print(f"  数据保留率: {retention_rate:.1f}%")
    print(f"  风速范围: {final_df['WS'].min():.2f} - {final_df['WS'].max():.2f} m/s")
    print(f"  数据完整性: 100.0% (无任何缺失值)")
    print(f"  坐标系优化: ✅ 统一使用[-180, 180]坐标系，避免重复转换")

    if retention_rate >= 90:
        print(f"  数据保留率优秀！数据集质量高")
    elif retention_rate >= 80:
        print(f"  数据保留率良好，可用于分析")
    else:
        print(f"  数据保留率较低，建议检查数据质量")


if __name__ == "__main__":
    # 设置输入文件路径
    cygnss_file = r"C:\Users\SONGQI\Desktop\1\cyg01.ddmi.s20190801-000000-e20190801-235959.l1.power-brcs.a32.d33.nc" #........................................................................................
    era5_file = r"C:\Users\SONGQI\Desktop\1\91c08d051f82009891390139471e5410.nc"

    # 运行主函数
    main(cygnss_file, era5_file)