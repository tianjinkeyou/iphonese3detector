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

- 按 **R**，拖动框选手机停放区，按 Enter/空格确认；C 取消。选择期间按 UNKNOWN 计时。
- 终端会打印 `--roi X Y WIDTH HEIGHT`，下次启动加上这几个参数可复用区域。
- 按 **Q** 或 Esc 退出。
- `PRESENT`（绿色）：手机在区域内，不计时；`ABSENT`（红色）：手机离开，累计时间；`UNKNOWN`/`UNCERTAIN`（黄色）：未知或不确定，也累计时间。
- 摄像头读取失败时显示黄色 UNKNOWN 并继续计时，恢复后重新确认状态。程序关闭期间不计时。
- `data/usage.sqlite3` 的 daily_state 表按天分别保存 absent/unknown 秒数，events 表保存状态变化；`data/photos/` 保存原始全画面照片。
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

此模式不需要下载/运行 YOLO 模型；未保存至少一张 E 和一张 F 前显示黄色 UNKNOWN，也计时。

1. 按 R 框选手机占据的区域，保留少量固定背景和支架边缘作对照，不要只框住黑色屏幕；尽量排除旁边瓶子、灯和移动物体。也可保留 P 的透视校准，但要先完成区域校准再保存参考图。
2. 拿走手机，手离开画面、曝光稳定后按 **E**，保存空支架。
3. 放回黑屏手机，手离开画面、曝光稳定后按 **F**，保存手机放好。
4. 测试放回和拿走，观察 PRESENT/ABSENT 是否正确。当前匹配的 E/F 样本过于相似、光线大幅变化或画面与E/F 两类参考图均不匹配时显示 UNKNOWN（黄色），继续计时。

参考图保存在 `data/references/`，下次自动加载。新版默认 `--detector auto` 优先智能检测，无法确认时自动使用E/F 两类参考图。按 R/P 改变检测区域会清除旧参考图，必须重新按 E/F。换摄像头、移动支架或显著改变照明也应重新校准。

这是固定位置的外观比较，并非手机身份识别：相似物体、遮挡和背景变化仍可能误判。首次测试前的累计时间包含旧模型漏检，不代表真实使用时间；原有记录保留，不自动删除。

## 新版默认：智能优先，失败回退 R/E/F

已撤回固定 Home 按钮模式；无需 H，旧 `data/home/` 不再读取，原数据保留。

```bash
python -m phonewatch --camera /dev/video2 --detector smart --delay 5
```

智能流程按每帧依次尝试：

1. YOLO 手机检测（包含已有旋转/弱光增强），成功则判定手机在停放区域。
2. 利用 E/F 两张图的差异提取手机上的 SIFT 局部特征，用比例检验和 RANSAC 验证多点的一致移动；允许小幅平移、旋转和缩放，不绑定一个 Home 按钮位置。黑屏手机上如果没有足够纹理，此步骤会跳过。
3. 仍无法确认时，自动使用 R/E/F 参考图比较；回退比较允许约 6% 的区域平移及 ±5° 小角度旋转。仍不确定则 UNKNOWN（黄色），继续计时。模型下载/加载失败也会继续特征和参考图模式，并在终端报告原因。

首次仍建议 **R 框选支架及手机附近区域（留出移动余量）→拿走手机按 E→放回手机按 F**。R 的区域和参考图均自动保存，重启后恢复；R/P 重新选择区域会清除原参考图，需要重新 E/F。窗口底部显示 `smart YOLO`、`smart: moving phone features` 或 `fallback R/E/F`，让你知道当前依据。

想完全使用上一个版本的参考图方法，执行：

```bash
python -m phonewatch --camera /dev/video2 --detector reference --delay 5
```

此模式不加载模型，也不使用智能特征匹配。默认 auto 与 smart 相同。智能方式仍只判断区域内是否存在手机/学习到的外观，并非手机身份认证；相似物品仍可能误报。手机必须留在框选区域内，较大摄像头移位、明显改变角度或光照需重新 R/E/F，不能保证任何移动都能识别。首轮 CPU 模型加载和漏检时的多角度推理可能较慢，可加 `--detection-mode fast` 降低负载。

新版计时规则：每日分别统计 ABSENT 和 UNKNOWN/UNCERTAIN 的运行时间（包括首次确认、校准等待和摄像头异常）；只有确认 PRESENT 时停止。未知状态立即显示黄色，PRESENT/ABSENT 仍使用 --delay 防抖。程序关闭期间不计时；历史记录保留，不追溯修改。

## 分开统计离开时间和不确定时间

画面分别显示红色 `ABSENT today` 和黄色 `UNKNOWN / UNCERTAIN`。UNKNOWN 与 UNCERTAIN 合并为一项，不加入 ABSENT 时间；PRESENT 不计时。跨日会分别归入当天，重启保留累计值。状态切换前的间隔归属于旧状态，避免切换时误记。

升级前的混合累计保留在原 daily 表，画面单独显示灰色 `Legacy unsplit`，不会假装拆分为新类别，也不会清空历史数据。新数据保存到 daily_state 表，不再写入旧 daily 表。

## 拿起次数与 AI 分析用事件文本

新版在 `auto`、`smart`、`reference`、`yolo`、`smart-watch` 模式中都自动统计拿起次数，无需额外开启。默认 `--pickup-mode parking` 独立比较 E 空支架与 F 停放手机的机身证据，判断手机是否仍在原停放位置；R 中其他位置检测到手机，不直接算作已经放回。有可靠空位证据时，手机离开支架但仍在摄像头画面中，也能记录拿起。E/F 缺失、与当前区域不兼容，或无法从 E/F 区分机身和空支架时，回退原手机检测依据；`--pickup-mode presence` 可显式使用原方式。若画面显示回退手机检测的依据，建议用 R 包含机身边框及附近背景，再分别保存手已离开的 E/F。

画面显示 `PICKUPS today`，按天保存、重启保留；`Pickup evidence` 另显示 `PARKED`、`EMPTY` 或 `UNCERTAIN` 及判断方式，与原有 PRESENT/ABSENT/UNKNOWN 计时状态分开。一次拿起是**确认 PARKED → 确认 EMPTY 的变化**：先确认手机停放，再确认空位，才加一次；持续空位不会重复加次数，确认放回后才能开始下一次。拿起/放回使用独立的 `--pickup-delay`，默认 1 秒；原有离开时间的 `--delay` 仍默认 5 秒，算法和计时规则保留。因此较短的拿走也可能记录一次，而离开计时状态尚未切换。

默认 parking 模式还支持快速拿起：已经确认 PARKED 后，只要至少一帧处理画面严格匹配高置信空支架 E，就立即记录一次拿起并保存照片，事件带 `rapid_empty: true`；单帧立即确认时 `confirmed_after_seconds` 为 0。普通空位、回退检测和放回仍使用 `--pickup-delay`，启动空位或摄像头恢复不会凭一帧建立拿起记录。快速换手机时如果拍到这种空位帧，也可以记一次；整个动作发生在两帧之间、被手遮住或没有可靠空位证据时，仍无法可靠计数。相同或相似手机的身份不能靠这一条事件确定。

短暂 UNKNOWN（最多 2 秒）允许桥接前后的确认状态，但 UNKNOWN 不计入拿起/放回的确认时间，并会在事件中保留 `unknown_seconds`。较长 UNKNOWN 后，只有摄像头始终有效、此前确认过 PARKED、随后确认符合补证据条件的高置信空位时，才能补记一次估算拿起；事件标记 `gap_recovered: true`、`timing_uncertain: true`，保留未知时长。它不能确定拿起的准确时间，也不能还原未知期间发生了几次拿起/放回；这种事件的 `away_seconds` 从恢复后确认空位时开始，不计入未知间隔。UNKNOWN 本身或弱空位证据不会增加次数。

摄像头断连、程序重启或校准操作仍会重新建立基线，不补记期间的拿起次数。启动时手机已不在区域内，不算一次拿起；旧版记录也不会追溯生成次数。原有启动命令可继续使用。

每次确认拿起、以及该次拿起后确认放回时，触发原始全画面快照，保存在 `data/photos/pickup_*.jpg`、`return_*.jpg`。事件中的 `photo_path` 相对于 `--data-dir`，例如 `photos/pickup_example.jpg`。照片保存失败时保留 `photo_path: null` 和 `photo_error`，已确认的拿起次数仍会保存。拿起次数和事件以同一个 `data/usage.sqlite3` 数据库的 `activity_events` 表为准，不依赖文本行数统计。放回事件的 `away_seconds` 表示从确认拿起到确认放回的间隔，可能包含短暂 UNKNOWN；它不等于已确认的使用时间或 ABSENT 计时。

拿起时还保存最近一次确认停放的原始画面 `before_pickup_*.jpg`；事件中的 `before_photo_path` 指向它，`before_photo_time` 保留原画面的采集时间。对应的拿起与放回事件共享 `cycle_id`，便于比较拿走前和放回后的手机外观。未能补证据的长 UNKNOWN 恢复时，记录 `unobserved_change` 和恢复后的照片；若此前有确认停放的画面，也保存 `before_unobserved_change_*.jpg` 供前后核查，事件本身不增加拿起次数。这些照片可以辅助发现外观变化，仍不能证明手机身份。

事件文本自动写入 `data/events/YYYY-MM-DD.txt`，使用 UTF-8：文件开头说明字段，之后每行是一条 JSON 事件，便于直接交给 AI 分析。每次运行保留独立的 `session` 标识；记录手机状态变化、拿起/放回、摄像头异常及恢复、E/F 样本和区域校准操作，以及启用 `smart-watch` 后的疑似使用开始/结束。约每 5 分钟、以及程序正常退出时，写入一次 `summary`，包含当天拿起次数、离开秒数、不确定秒数、疑似使用秒数和本次运行时间。重启追加新记录，不覆盖旧记录。文本导出失败时，SQLite 中的事件仍然保留，终端会打印诊断；后续成功导出时会恢复文本记录。

例如，下面一行表示摄像头确认手机离开，时间包含本地 UTC 偏移：

```json
{"id": 12, "time": "2026-10-10T19:30:02+09:00", "event": "pickup", "session": "example-session", "details": {"from_state": "present", "to_state": "absent", "method": "fallback R/E/F", "photo_path": "photos/pickup_example.jpg", "unknown_seconds": 0}}
```

可以将 TXT 和需要核查的对应照片交给 AI，辅助寻找“连续几小时没有拿起”“频繁短暂拿起”“大量 UNKNOWN”“手机放着但疑似使用持续”等事件组合。程序只在本地输出，不自动上传给 AI。日志和照片可能包含家庭隐私，分享前请检查内容。

相似的旧手机放在支架上，尤其没有拍到明确的拿走/放回过程时，现有算法不能确认手机身份，也不能仅凭拿起次数为零判定作弊。记录用于辅助核查，应结合照片、未知间隔、摄像头在线时间和实际情况分析。

需要更敏感时，可使用 `--pickup-delay 0.5`；短暂遮挡和漏检也更容易变成拿起记录。建议先在本地验证：

1. 手机放好并等待确认，拿走超过 1 秒，再放回等待确认；次数应增加 1，TXT 应出现 `pickup` 和 `return`，并有两张事件照片。
2. 手机离开支架但仍放在摄像头画面其他位置；有正确 E/F 和可靠空位证据时，`Pickup evidence` 应变为 EMPTY，次数增加 1，不要求原有计时状态一定变为 ABSENT。手机拿走后再启动程序，不能凭启动状态增加拿起次数。
3. 手机放好后断开摄像头、拿走手机，再恢复摄像头；应记录摄像头异常和 UNKNOWN，不能推测断连期间的拿起次数。再次正常放回、拿走后，才记录新的一次。

## 智能检测与 E 空支架冲突的处理

新版智能检测不会无条件覆盖 E/F：与 E 高度一致且明显不同于 F 时，即使 YOLO 或特征匹配误报手机，也判为 ABSENT；参考图偏向 E 但证据不够强、与智能结果冲突时判为黄色 UNKNOWN，单独计入不确定时间。没有空支架冲突时继续优先采用智能检测。底部会显示 `E empty confirmed; smart rejected` 或 `CONFLICT smart vs E` 及距离数值。

E 必须在手机拿走且手离开画面时保存，F 必须在手机放好且手离开画面时保存；两个标签不能反过来。ABSENT/PRESENT 切换仍有 --delay 防抖，请等待确认时间。原有校准可继续使用。

## 多张 E / F 样本（不同姿势和照明）

现在每按一次 **E** 或 **F** 都新增一张样本，不覆盖旧样本。旧版 `data/references/empty.png`、`phone.png` 自动作为首批样本加载，原文件保留。

1. **R 只需框选一次**，区域包括手机各种摆放姿势，并留出移动余量。
2. 手机拿走、手离开画面后按 E；不同光线或空支架姿势下可以重复按 E。
3. 手机放好、手离开画面后按 F；更换角度、位置、屏幕开/关、光线，再分别按 F。
4. 画面底部显示 `E samples` / `F samples` 数量，重启自动加载全部样本。

参考图模式与智能回退都在全部兼容样本中寻找最相似的 E 和 F；智能特征匹配也会从多张 F 学习，并排除多张 E 中的背景特征。某个 E 高度匹配时仍能否决智能误报；同时匹配 E 和 F 的冲突样本返回黄色 UNKNOWN。

新文件分别保存在 `data/references/empty/` 和 `data/references/phone/`。按 **C 两次（3 秒内）** 清空全部 E/F 样本并重新采集；不会清空计时、照片或区域设置。重新按 R/P 确认区域会清空旧区域的全部样本，请不要为了每个姿势重新按 R。

建议先保存 3～5 张有明显区别的 E、5～10 张不同姿势的 F，再测试未保存过的姿势。E 必须没有手机，F 必须有手机；误标样本可能降低准确度，错误时清空重采。样本更多会增加比较耗时，反复保存完全相同的画面帮助不大。程序没有重新训练 YOLO 模型，只扩充本地参考和特征库；相似物品或极暗画面仍可能无法判断。

## 可选 smart-watch：支架上疑似使用监测

- `--detector smart`：原有手机检测、R/E/F 回退和计时。
- `--detector smart-watch`：上述功能加上手指接近监测、疑似时间单独统计、触发拍照。可再加 `--watch-screen` 监测屏幕变化。默认不启用，不影响 smart 模式。

可选依赖支持 Python 3.10–3.12。首次启用安装：

```bash
source .venv/bin/activate
sudo apt install -y libportaudio2
python -m pip install -r requirements-watch.txt
python -m phonewatch --camera /dev/video2 --detector smart-watch --delay 5 --watch-delay 3
```

已有 R/E/F 区域与样本可继续使用。必须用 R 框选手机停放区域（保留少量移动余量），或使用已有 P 校准/--roi；没有指定区域时仍显示手部检测结果，但不触发疑似计时。

只有手机当前检测和稳定状态均为 PRESENT，且手指接近、局部手部候选或启用的屏幕变化持续至少 `--watch-delay` 秒，才记录一次疑似使用开始，并立即保存 `data/photos/suspect_*.jpg`。所有线索消失后停止；摄像头失败、手机检测变为 ABSENT/UNKNOWN、进入 R/P/E/F/C/S 操作时立即停止，不把不可确认的间隔算作疑似使用。程序退出也会结束事件，不计算关闭期间的时间。

画面紫色 `SUSPECTED today` 单独显示疑似时间。数据库 `watch_daily` / `watch_events` 分别保存每日疑似秒数和开始/结束事件、照片路径；不并入 ABSENT 或 UNKNOWN，也不改写旧记录。原有每两分钟拍照继续运行。MediaPipe 手部模型随可选 Python 包安装，本地处理，不上传画面。

默认 `--hand-mode robust` 先检测全画面，再在手机附近扩大搜索区域、增强对比度并尝试旋转，避免横放手机或画面只露出部分手时直接放弃。完整手部仍检测不到时，利用已有 F 样本寻找新增的肤色区域，标为橙色 `partial` 候选；F 要在没有手时保存。紫色指尖点和 `Hands` 数量无论手机状态是否 PRESENT 都会显示，便于区分手部漏检和手机状态阻断。

局部候选只是外观线索：肤色的屏幕图片或新物体也可能误触发，手套、严重遮挡和弱光仍可能漏检。`--hand-mode landmarks` 可关闭局部候选及裁剪搜索，保留完整手模型，减轻计算负担。家长的手、擦拭/整理手机也可能触发。如果手遮挡机身导致 UNKNOWN，只有不确定时间继续记录，疑似监测暂停。

### 可选屏幕变化检测（横放/竖放均支持）

```bash
python -m phonewatch --camera /dev/video2 --detector smart-watch --watch-screen --delay 5 --watch-delay 3
```

1. 保留已有 R/E/F。手机换成横放等新姿势后，手离开、曝光稳定时再按 F 增加样本；R 区域需要包含完整手机及少量背景。不要只框屏幕，否则缺少机身证据。
2. 按 **S**，在冻结画面依次点击**实际显示区域**的左上、右上、右下、左下四角，按 Enter 保存，Esc 取消。不要把支架、Home 按钮或整台手机框进去。S 不会清除 E/F；选择期间按 UNKNOWN 计时。
3. 滚动页面或播放视频。`SCREEN: changing` 表示屏幕变化被检测到；旁边的 `change` 显示变化像素比例，默认门槛为 2.5%。手机确认 PRESENT 后，持续活动约 3 秒会显示 `SUSPECTED USE - screen motion`、拍照并单独计时，即使没有看到手。

屏幕四角保存在 `data/screen-plane.json`，重启自动加载。S 是当前姿势的固定区域，移动手机、支架或摄像头后需要重新 S；多张 F 不会自动切换屏幕四角。S 应位于 R/P 停放区内部。横放手机建议使用 R，旧 P 采用竖向机身比例。

比较仅覆盖屏幕内部，利用周围背景过滤全画面曝光变化和小幅摄像头抖动；短暂停顿最多保留 1.5 秒的活动线索，单次变化无法独自满足默认 3 秒确认。屏幕黑暗、过曝、摄像头中断时不保留线索。屏幕变亮造成原智能检测 UNKNOWN 时，还会排除屏幕内容，比较可见机身的 E/F 特征；紧贴手机的 P/R 区域没有固定背景时，只采用可辨认的边框、Home 按钮等空间细节。机身全黑且缺少细节时仍可能 UNKNOWN，建议 R 留出固定背景；明确 ABSENT 不会被屏幕变化覆盖。

如果变化比例持续低于 2.5%，确认 S 准确后可尝试 `--screen-threshold 0.01`（1%）；降低门槛也会增加噪声误报。画面显示总处理耗时，老电脑运行较慢时可加 `--detection-mode fast`，或使用 `--hand-mode landmarks`。屏幕变化属于疑似使用：视频、通知、自动动画和反光都可能触发，静止阅读可能漏检，不能判断是谁操作或证明触屏。请在实机分别验证静止屏幕、持续滚动、短暂通知、远处挥手和摄像头断连。
