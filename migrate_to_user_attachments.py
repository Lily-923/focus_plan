#!/usr/bin/env python3
"""
一次性把仓库里所有本地图片转成 GitHub user-attachments 外链
并替换所有 .md 里的旧路径
"""
import os, re, pathlib, mimetypes, time, requests
from tqdm import tqdm

# ========== 用户配置 ==========
TOKEN   = "ghp_xxxxxxxxxxxxxxxxxxxx"   # 你的 GitHub token
USER    = "你的用户名"
REPO    = "你的仓库名"
BRANCH  = "main"
# 图片在仓库里的根目录，也可以改成 "." 表示全仓库
IMG_ROOT = "imgs"
# ==============================

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

def upload_one_image(file_path: pathlib.Path):
    """上传单张图，返回 user-attachments 外链"""
    url = f"https://uploads.github.com/repos/{USER}/{REPO}/releases/1/assets"
    filename = file_path.name
    mime, _ = mimetypes.guess_type(filename)
    params  = {"name": filename}
    headers = {"Content-Type": mime or "application/octet-stream"}
    with file_path.open("rb") as f:
        # GitHub 要求 POST 数据是原始二进制
        r = SESSION.post(url, params=params, headers=headers, data=f)
    if r.status_code == 201:
        return r.json()["browser_download_url"]
    else:
        print(f"上传失败 {file_path} : {r.status_code} {r.text}")
        return None

def build_mapping():
    """建立 旧相对路径 → 新外链 的映射"""
    mapping = {}
    img_root = pathlib.Path(IMG_ROOT)
    files = list(img_root.rglob("*")) if img_root.exists() else []
    img_files = [f for f in files if f.is_file() and f.suffix.lower() in {".png",".jpg",".jpeg",".gif",".svg",".webp"}]
    for f in tqdm(img_files, desc="uploading"):
        new_url = upload_one_image(f)
        if new_url:
            # 保留两种可能写法：imgs/xxx 或 ./imgs/xxx
            old_key1 = str(f).replace("\\", "/")               # imgs/2025-07-30/a.png
            old_key2 = "./" + old_key1                         # ./imgs/2025-07-30/a.png
            mapping[old_key1] = new_url
            mapping[old_key2] = new_url
        time.sleep(0.5)  # 简单限流
    return mapping

def replace_in_md(mapping):
    """批量替换所有 .md 文件"""
    for md in pathlib.Path(".").rglob("*.md"):
        txt = md.read_text(encoding="utf8")
        txt2 = txt
        for old, new in mapping.items():
            # 匹配 ![...](old) 或 < img src="old">
            txt2 = re.sub(r'(!\[.*?\]\(|<img[^>]+src=([\'"]))' + re.escape(old) + r'(\)|\2)',
                          rf'\g<1>{new}\g<3>', txt2)
        if txt2 != txt:
            md.write_text(txt2, encoding="utf8")
            print("fixed", md)

def main():
    mapping = build_mapping()
    if not mapping:
        print("没有需要处理的图片")
        return
    replace_in_md(mapping)
    print("全部替换完成，请检查 git diff 然后 commit + push")

if __name__ == "__main__":
    main()
