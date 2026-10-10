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

## 倾斜摆放与弱光

默认 `--detection-mode robust`：先检测原图；漏检时，对暗画面做亮度对比度增强，并尝试 +45°、-45°、90° 的旋转画面。旋转使用扩展画布避免裁掉手机边角，检测框会映射回原画面。窗口底部显示成功的检测方式和每帧检测耗时。

```bash
python -m phonewatch --camera /dev/video2 --confidence 0.15 --delay 10
```

旋转处理主要针对手机在画面中斜着摆放；手机靠在支架上造成的透视变形，或几乎只看到侧边，不能靠旋转完全解决。亮度增强也无法恢复摄像头没拍到的细节。保持白色背景、适当补光，并让摄像头尽量看到手机完整正面或背面。框选区域应留出手机边角，不要贴边裁切。

漏检时每帧最多进行 5 次推理，老电脑可能变慢。可选 `--detection-mode fast` 恢复单次检测。降低置信度会增加误报。请分别测试倾斜放回和拿走至少十次，确认空停放区不会被误判，不能把算法增强当作准确率保证。

## 手机斜靠在支架上：四点透视校准

1. 固定摄像头和支架，将手机放到正常停放位置，尽量露出完整正面或背面。
2. 按 **P**。在冻结画面中依次点击手机面的 **左上、右上、右下、左下** 四个角，再按 Enter。Esc 取消。
3. 程序将这块倾斜平面变换为矩形，再进行手机检测；检测框仍显示在原摄像头画面上。
4. 校准自动保存至本地 `data/parking-plane.json`，下次自动加载。摄像头或支架移动后，重新按 P 校准。测试手机放回和拿走，特别注意空支架是否误报。

按 R 重新选择普通区域，会清除四点校准。`--ignore-plane` 可临时忽略保存的校准；显式传入 `--roi` 也优先使用普通区域。换摄像头时使用 `--ignore-plane` 并重新按 P。拍照仍保存原始全画面。

透视校正只是近似：输出采用 iPhone SE3 机身宽高比例（67.3:138.4），四个角应按手机竖向正面理解；请竖放手机进行校准。这不是完整的相机标定。它无法还原严重侧视、遮挡或极暗环境中没有拍到的细节，不能保证识别成功。过于侧视时需要调整摄像头角度。

## 黑屏手机识别不稳定：固定支架参考图模式（推荐）

```bash
python -m phonewatch --camera /dev/video2 --detector reference --delay 5
```

此模式不需要下载/运行 YOLO 模型；未完成两张参考图前显示 UNKNOWN，不计时。

1. 按 R 框选手机占据的区域，保留少量固定背景和支架边缘作对照，不要只框住黑色屏幕；尽量排除旁边瓶子、灯和移动物体。也可保留 P 的透视校准，但要先完成区域校准再保存参考图。
2. 拿走手机，手离开画面、曝光稳定后按 **E**，保存空支架。
3. 放回黑屏手机，手离开画面、曝光稳定后按 **F**，保存手机放好。
4. 测试放回和拿走，观察 PRESENT/ABSENT 是否正确。两张图过于相似、光线大幅变化或画面与两张参考图均不匹配时显示 UNKNOWN，暂停计时。

参考图保存在 `data/references/`，下次自动加载。默认 `--detector auto` 在两张参考图齐全时也自动使用参考图检测。按 R/P 改变检测区域会清除旧参考图，必须重新按 E/F。换摄像头、移动支架或显著改变照明也应重新校准。

这是固定位置的外观比较，并非手机身份识别：相似物体、遮挡和背景变化仍可能误判。首次测试前的累计时间包含旧模型漏检，不代表真实使用时间；原有记录保留，不自动删除。

## Home 按钮局部结构检测（黑屏 iPhone SE3）

```bash
python -m phonewatch --camera /dev/video2 --detector home --delay 5
```

1. 手机正常放好、屏幕关闭，按 **H**，框选 **完整 Home 按钮圆环和周围少量黑色边框**，按 Enter。至少 12×12 像素；不要框整个屏幕。如果圆环看不见，需要补光或调整摄像头。
2. 拿走手机，手离开画面后按 **E**，保存同位置的空支架背景。
3. 放回手机，测试 PRESENT；拿走后测试 ABSENT。显示 UNKNOWN 表示局部结构不清楚、光线变化或无法确认空支架，不计时。

此模式对选中的按钮区域提取梯度边缘，在原位置附近及 0.9/1.0/1.1 倍尺度上做归一化相关匹配；不依赖屏幕内容，不运行 YOLO。它是用户提供的局部结构模板，不是能从任意图像自动认出 Home 按钮的训练模型。相似圆环仍可能误报，手机大幅移位或转向则需重新按 H。再次拿走手机按 E 更新背景。

`data/home/` 保存按钮特征图、位置和空背景。默认 auto 在 H 校准存在时优先使用 Home 模式；明确 `--detector reference` 可恢复参考图模式。Home 模式独立使用原画面位置，不需要 R/P 校准；换摄像头、分辨率或移动支架后重新 H/E。

**全黑屏幕与全黑摄像头画面不同。** 只要按钮和边缘可见，可以提取特征；整幅摄像头画面几乎全黑时无法确认手机，Home 模式显示 UNKNOWN。不要通过降低阈值把不可见的物体当作可靠检测。
