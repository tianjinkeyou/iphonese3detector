# 手机停放监测器

Ubuntu + Python + 外置 USB 摄像头，在本地判断手机是否在固定停放区。
手机持续离开 5 秒后累计估算使用时间，持续放回 5 秒后停止；每两分钟保存一张照片。
**统计的是离开停放区的时间，并非真实屏幕使用时间。** 遮挡、漏检会影响结果。
默认使用 YOLO11n 的 `cell phone` 类别，不专门识别 iPhone SE3。

## Ubuntu 安装

在本地终端执行（第一次安装需要联网）：

```bash
cd "/home/hua/ドキュメント/APPS/iphonese3detector"
git pull origin main
sudo apt update
sudo apt install -y python3-venv libgl1 libglib2.0-0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 安装 CPU 版本，避免下载不需要的 CUDA 组件
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

使用 Ubuntu 桌面会话启动，需要显示器和摄像头。首次启动会从 Ultralytics 的 GitHub release 下载模型 `yolo11n.pt`；之后可离线运行。不要同时用其他应用占用摄像头。

```bash
source .venv/bin/activate
python -m phonewatch --camera 0
```

如果打开的是电脑内置摄像头，尝试 `--camera 1`、`--camera 2`，或使用 `--camera /dev/video2`。可用 `ls /dev/video*` 查看设备节点。

## 操作与记录

- 按 **R**，拖动框选手机停放区，按 Enter/空格确认；C 取消。选择期间不计时。
- 终端会打印 `--roi X Y WIDTH HEIGHT`，下次启动加上这几个参数可复用区域。
- 按 **Q** 或 Esc 退出。
- `PRESENT`：手机在区域内；`ABSENT`：手机离开，累计时间；`UNKNOWN`：刚启动或摄像头异常，不累计时间。
- 摄像头读取失败时暂停计时，恢复后重新确认状态。程序关闭期间不计时。
- `data/usage.sqlite3` 保存每日秒数和状态变化；`data/photos/` 保存原始全画面照片。
- 数据和模型已加入 `.gitignore`。照片可能包含家庭隐私，不应上传；请与孩子明确说明监测用途。
- 照片没有自动删除功能，需定期清理以避免磁盘占满。

更多参数：

```bash
python -m phonewatch --help
python -m phonewatch --camera 1 --delay 5 --photo-interval 120 --roi 100 100 400 300
```

手机建议平放在固定位置、保持均匀照明。先观察放入/拿走各十次，检查检测框、状态和计时是否符合实际，再正式使用。若背面朝上或遮挡导致漏检，可调整角度、阈值 `--confidence 0.15`；降低阈值也会增加误报。

## 开发验证

```bash
python -m unittest discover -s tests -v
```

云端可测试计时与数据库逻辑，但无法访问你本地的 USB 摄像头。识别稳定性、桌面显示、真实拍照需要在 Ubuntu 实机验证。
