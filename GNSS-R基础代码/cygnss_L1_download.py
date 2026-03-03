"""
CYGNSS L1 (CYGNSS_L1_V3.2) 批量下载脚本（稳健版）
- CMR：TLS1.2 + 重试
- 链接解析：过滤真正 data 链接（https, .nc/.nc.md5）
- 下载：优先 S3（boto3.download_file）→ 无 HeadObject 权限时改用 get_object 流式 → 仍失败则走已鉴权 HTTPS
- S3 凭证：兼容不同键名/嵌套返回
- 可选：若 CMR TLS 有问题，自动/手动切换到 earthaccess.search_data + earthaccess.download
"""
#账号：songqi774599
#密码：Wow774599122_
import os  # 用于环境变量设置和文件路径操作
import time  # 用于下载重试时的延时
import json  # 未直接使用，但CMR响应为JSON，可扩展
import pathlib  # 用于路径操作，确保目录创建
import ssl  # 用于自定义TLS上下文，强制TLS1.2
from urllib.parse import urlencode  # 用于构建CMR查询URL参数
import requests  # 用于HTTP请求，包括CMR搜索和备用HTTPS下载
from urllib3.util.retry import Retry  # 用于配置requests重试策略
from requests.adapters import HTTPAdapter  # 用于自定义TLS适配器
import boto3  # 用于S3客户端操作（优先下载方式）
from botocore.exceptions import ClientError  # 用于捕获S3操作错误
import earthaccess  # 用于Earthdata认证、S3凭证获取和备用搜索/下载
from tqdm import tqdm  # 用于显示进度条

# ========= 配置区（按需修改） =========
COLLECTION_SHORT_NAME = "CYGNSS_L1_V3.2"  # 数据集短名，用于CMR/earthaccess搜索
PROVIDER = "POCLOUD" # CMR provider（PO.DAAC on cloud）  # 指定提供者，确保云端数据
PAGE_SIZE = 2000  # CMR分页大小，平衡效率和API限流
START = "2025-8-01T00:00:00Z"   # 搜索起始时间（ISO格式，UTC）
END = "2025-08-31T23:59:59Z"   # 搜索结束时间（小范围示例，可扩展）
OUT_DIR = r"C:\Users\SONGQI\Desktop\2_STG_DNN\cygnss3.2"  # 输出目录，下载文件保存位置
NETRC_PATH = r"C:\Users\SONGQI\account_password\.netrc"  # Earthdata .netrc文件路径，用于非交互认证
S3_REGION = "us-west-2" # PO.DAAC S3 所在区域  # S3客户端区域配置
DAAC = "PODAAC" # 如果用 daac 参数取凭证，必须是 PODAAC（不是 POCLOUD）  # 备用凭证获取参数
# CMR 失败时是否自动改走 earthaccess.search_data + earthaccess.download  # 启用/禁用备用路径
AUTO_FALLBACK_TO_EARTHACCESS = True
SPACECRAFT_FILTER = "cyg01" # 想下所有星就=None，想下单星就="cyg06"
# ==================================

# -------- 网络会话：强制 TLS1.2 + 重试 --------
def build_tls12_retry_session():
    # 自定义适配器：强制TLS1.2上下文，避免CMR的SSL兼容问题
    class TLS12Adapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            ctx = ssl.create_default_context()
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2  # 最低TLS版本
            ctx.maximum_version = ssl.TLSVersion.TLSv1_2  # 最高TLS版本（仅1.2）
            kwargs["ssl_context"] = ctx
            return super().init_poolmanager(*args, **kwargs)
    # 重试策略：总5次，指数退避（backoff_factor=1.0: 1s,2s,4s...），针对特定HTTP错误
    retries = Retry(
        total=5,
        backoff_factor=1.0, # 1s, 2s, 4s...
        status_forcelist=[429, 500, 502, 503, 504],  # 重试的HTTP状态码（限流/服务器错误）
        allowed_methods=["GET"],  # 只对GET请求重试
        raise_on_status=False,  # 不立即抛出异常，允许重试
    )
    s = requests.Session()  # 创建持久会话
    s.mount("https://", TLS12Adapter(max_retries=retries))  # 挂载TLS+重试适配器到HTTPS
    s.headers.update({"User-Agent": "cygnss-python-downloader"})  # 自定义UA，避免被拒
    return s
CMR_SESSION = build_tls12_retry_session()  # 全局CMR专用会话

# -------- CMR 搜索 --------
def cmr_search(short_name, start, end, provider, page_size=PAGE_SIZE):
    # 分页搜索CMR granules，返回每个entry的生成器（yield）
    base = "https://cmr.earthdata.nasa.gov/search/granules.json"  # CMR API端点
    page = 1  # 从第1页开始
    while True:
        params = {
            "short_name": short_name,  # 数据集短名
            "temporal": f"{start},{end}",  # 时间范围（逗号分隔）
            "provider": provider,  # 指定提供者
            "page_size": page_size,  # 每页结果数
            "page_num": page,  # 当前页码
        }
        url = f"{base}?{urlencode(params)}"  # 构建完整查询URL
        resp = CMR_SESSION.get(url, timeout=60)  # 发送GET，超时60s
        resp.raise_for_status()  # 检查HTTP错误，失败抛异常
        js = resp.json()  # 解析JSON响应
        items = js.get("feed", {}).get("entry", []) or []  # 提取granules列表
        if not items:
            break  # 无结果，结束
        for item in items:
            yield item  # 逐个yield entry
        if len(items) < page_size:
            break  # 不足一页，结束
        page += 1  # 下一页

# -------- 解析下载链接（HTTPS）--------
def pick_download_links(entry):
    """
    从 CMR entry 里挑可直接下载的 https 链接（排除 OPeNDAP/ERDDAP 等）
    返回列表 [(url, is_protected), ...]
    """
    links = entry.get("links", []) or []
    out = []
    seen = set()
    for lk in links:
        rel = (lk.get("rel") or "")
        href = (lk.get("href") or "")
        if not href or not href.startswith("https"):
            continue
        if "data#" not in rel:
            continue

        low = href.lower()

        # ======= 新增：按星号过滤，只保留 cyg06 =======
        # 文件名一般形如 cyg06.ddmi.sYYYYMMDDHHMMSS_vX.Y.nc
        # 所以只要检测 URL 中是否包含 'cyg06'
        if SPACECRAFT_FILTER:
            if SPACECRAFT_FILTER.lower() not in low:
                continue
        # ========================================

        if any(s in low for s in ["opendap", "erddap", "dap"]):
            continue
        if not (href.endswith((".nc", ".nc.md5"))):
            continue
        if any(ex in href for ex in ["/s3credentials", "/virtual-directory", "?"]):
            continue
        if href in seen:
            continue
        seen.add(href)
        is_protected = "protected" in href
        out.append((href, is_protected))
    return out


# -------- earthaccess 凭证 → boto3 参数（健壮映射）--------
def boto3_kwargs_from_earthaccess(creds):
    # 兼容形如 {"Credentials": {...}} 的嵌套  # 处理earthaccess返回的嵌套凭证
    if isinstance(creds, dict) and "Credentials" in creds and isinstance(creds["Credentials"], dict):
        creds = creds["Credentials"]
    # 键映射：earthaccess键到boto3标准键，支持多种变体
    mapping = {
        "aws_access_key_id": ["accessKeyId", "AccessKeyId", "aws_access_key_id"],
        "aws_secret_access_key": ["secretAccessKey", "SecretAccessKey", "aws_secret_access_key"],
        "aws_session_token": ["sessionToken", "SessionToken", "token", "aws_session_token"],
    }
    out = {}  # 输出dict
    for boto_key, candidates in mapping.items():
        for k in candidates:
            if k in creds and creds[k]:
                out[boto_key] = creds[k]
                break  # 找到第一个匹配的，停止
    missing = [k for k in mapping if k not in out]  # 检查缺失必需键
    if missing:
        raise KeyError(f"Unexpected S3 creds shape: {list(creds.keys())}, missing: {missing}")
    return out

# -------- 三级回退下载：S3(download_file) → S3(get_object) → HTTPS --------
def download_file(url, out_dir, s3_client, https_session=None, chunk=16 * 1024 * 1024):
    """
    带进度条和自动降级策略的下载函数
    """
    fn = url.split("/")[-1]
    out = pathlib.Path(out_dir) / fn
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # 断点续传/跳过已存在检查（简单版：只检查文件是否存在且非空）
    if out.exists() and out.stat().st_size > 0:
        print(f"Skipping (exists): {fn}")
        return str(out)
    
    # ============================
    # 策略 A: 尝试 S3 下载 (优先)
    # ============================
    if s3_client:
        bucket = "podaac-ops-cumulus-protected" if "protected" in url else "podaac-ops-cumulus-public"
        token = f"/{bucket}/"
        if token in url:
            s3_key = url.split(token, 1)[1]
            try:
                # 获取文件大小用于进度条
                meta = s3_client.head_object(Bucket=bucket, Key=s3_key)
                total_size = meta.get('ContentLength', 0)
                
                # 使用 S3 流式下载以配合进度条
                resp = s3_client.get_object(Bucket=bucket, Key=s3_key)
                body = resp["Body"]
                
                print(f"[S3] Downloading {fn} ...")
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=fn, leave=False) as bar:
                    with open(out, "wb") as f:
                        while True:
                            chunk_bytes = body.read(chunk)
                            if not chunk_bytes:
                                break
                            f.write(chunk_bytes)
                            bar.update(len(chunk_bytes))
                return str(out)
            except Exception as e:
                # S3 失败（权限或网络），静默失败，转入 HTTPS 兜底
                pass

    # ============================
    # 策略 B: 切换 HTTPS 下载 (兜底)
    # ============================
    if https_session is None:
        raise ValueError("HTTPS session missing and S3 failed.")
    
    print(f"[HTTPS] Downloading {fn} ...")
    try:
        # stream=True 允许流式下载
        with https_session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            # 获取总大小
            total_size = int(r.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=fn, leave=False) as bar:
                with open(out, "wb") as f:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            f.write(part)
                            bar.update(len(part))
        return str(out)
    except Exception as e:
        # 如果下载中途出错，删除未完成的文件，避免下次误判
        if out.exists():
            out.unlink()
        raise e

# -------- 备用：完全交给 earthaccess 来检索+下载（HTTPS/S3 自动）--------
def fallback_download_via_earthaccess(short_name, start, end, provider, out_dir):
    """
    作为 CMR/TLS 异常的兜底：用 earthaccess.search_data + earthaccess.download
    返回 (found, saved) 统计。
    """
    # 搜索granules，与CMR参数一致
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(start, end),  # 元组格式
        provider=provider,
    )
    if not results:
        return 0, 0  # 无结果
    # 自动下载，返回本地路径列表（earthaccess内部处理S3/HTTPS）
    paths = earthaccess.download(results, out_dir)
    saved = len([p for p in paths if p])  # 计数非空/成功路径
    # 尝试从 results 里估计数据文件数量（过滤非 .nc/.nc.md5）  # 粗略found计数
    found = 0
    for gran in results:
        try:
            for href in gran.data_links():  # 获取每个granule的数据链接
                if href.endswith((".nc", ".nc.md5")):
                    found += 1
        except Exception:
            pass  # 忽略链接解析错误
    return found, saved

# -------- 主流程 --------
def main():
    print(f"Search {COLLECTION_SHORT_NAME} from {START} to {END} ...")

    os.environ["NETRC"] = NETRC_PATH
    try:
        earthaccess.login(strategy="netrc")
        print("Earthdata 登录成功 (netrc)")
    except Exception as e:
        print(f"netrc 登录失败: {e} - 尝试 interactive...")
        earthaccess.login(strategy="interactive")
        print("Earthdata 登录成功 (interactive)")

    # --- S3 客户端（保持你原来的这段即可） ---
    s3_client = None
    try:
        creds = earthaccess.get_s3_credentials(provider=PROVIDER)
        print("获取 S3 临时凭证成功，将使用高速 S3 通道")
        s3_client = boto3.client(
            "s3",
            region_name=S3_REGION,
            **boto3_kwargs_from_earthaccess(creds),
        )
    except Exception as e:
        print(f"注意：S3 模式不可用 (错误: {e})")
        print("系统已自动切换为 HTTPS 普通下载模式，请耐心等待...")
        s3_client = None
    # ---------------------------------------

    https_session = earthaccess.get_requests_https_session()

    found = 0      # 通过 pick_download_links 拿到的 URL 数
    saved = 0      # 成功下载的文件数
    gran_total = 0 # CMR 返回的 granule 总数
    gran_kept = 0  # 经过 cyg06 过滤后保留下来的 granule 数

    filter_tag = (SPACECRAFT_FILTER or "").lower()

    try:
        for g in cmr_search(COLLECTION_SHORT_NAME, START, END, PROVIDER):
            gran_total += 1
            # 从 entry 里拿 granule 名称（文件名）
            gid = (g.get("producer_granule_id")
                   or g.get("title")
                   or "").lower()

            # 如果设置了星号过滤，只保留包含 cyg06 的 granule
            if filter_tag:
                if filter_tag not in gid:
                    continue
                gran_kept += 1

            # 可选：打印一下当前保留下来的 granule 名称，方便检查
            print(f"Granule kept: {gid}")

            for url, _ in pick_download_links(g):
                found += 1
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        path = download_file(
                            url,
                            OUT_DIR,
                            s3_client,
                            https_session=https_session,
                        )
                        saved += 1
                        print(f"[{saved}] {path}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"重试 {url} (尝试 {attempt+1}/{max_retries}): {e}")
                            time.sleep(2 ** attempt)
                        else:
                            print(f"!! fail: {url} -> {e}")
    except requests.exceptions.SSLError as e:
        print(f"CMR TLS/SSL 异常：{e}")
        if AUTO_FALLBACK_TO_EARTHACCESS:
            print("切换到 earthaccess.search_data + earthaccess.download 兜底路径...")
            f2, s2 = fallback_download_via_earthaccess(
                COLLECTION_SHORT_NAME, START, END, PROVIDER, OUT_DIR
            )
            found += f2
            saved += s2

    print(
        f"done. links={found}, saved={saved}, "
        f"granules_kept={gran_kept}/{gran_total}, "
        f"filter={SPACECRAFT_FILTER}, out_dir={OUT_DIR}"
    )


if __name__ == "__main__":
    main()  # 入口点