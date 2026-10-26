基于深度学习YOLOv12的杂草识别检测系统（YOLOv12+YOLO数据集+UI界面+登录注册界面++模型）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2049a0a828df460ca377bbc6bc9a348c.png)


针对农田杂草高效识别与精准防控的需求，本研究基于深度学习目标检测算法YOLOv12，开发了一套杂草识别检测系统。系统以12类常见杂草（如鳢肠、牵牛、牛筋草等）为检测对象，构建包含2796张训练图像和523张验证图像的YOLO格式数据集，通过数据增强与迁移学习优化模型性能。系统集成Python开发的交互式UI界面，支持用户登录注册、实时检测与结果可视化，测试集平均精度（mAP@0.5）达94.1%。实验表明，YOLOv12在复杂农田场景下对小目标杂草具有鲁棒性，可为智慧农业杂草治理提供自动化解决方案。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06c89e34e0574ffd9e98ae27518b1ec8.png)


✅ 用户登录注册：支持密码检测和安全性验证。
✅ 三种检测模式：基于YOLOv12模型，支持图片、视频和实时摄像头三种检测，精准识别目标。
✅ 双画面对比：同屏显示原始画面与检测结果。
✅ 数据可视化：实时表格展示检测目标的类别、置信度及坐标。
✅智能参数调节：提供置信度滑块，动态优化检测精度，适应不同场景需求。
✅科幻风交互界面：深色主题搭配动态光效，减少视觉疲劳，提升操作体验。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0e9d05853dbc44df960cfb0212dd34bb.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df3853a0e3cb49469fe313b2aa7031ac.png)
1
 **基于 YOLOv12 的杂草识别检测系统** 的完整技术说明、UI 功能解析及 **Python 项目源码结构与核心代码实现**，包含登录注册、三种检测模式、双画面对比、参数调节等全部功能。

---

## ✅ 一、系统概述

| 项目 | 内容 |
|------|------|
| **系统名称** | 基于深度学习 YOLOv12 的杂草识别检测系统 |
| **核心算法** | YOLOv12（假设为 YOLOv8/v10/v11 的演进版本，实际使用 `ultralytics` 库） |
| **目标类别** | **12 类常见农田杂草**（如鳢肠、牵牛、牛筋草、马齿苋等） |
| **数据集规模** | - 训练集：2,796 张- 验证集：523 张- 格式：YOLO 格式（`.txt` + `.jpg`） |
| **mAP@0.5** | **94.1%**（测试集表现） |
| **开发语言** | Python 3.9+ |
| **UI 框架** | PyQt5 / PyQt6（科幻风深色主题） |
| **部署平台** | Windows/Linux/macOS |

---

## ✅ 二、12 类杂草类别表（来自截图）

根据界面显示的检测结果，推断 12 类杂草如下：

| 类别编号 | 中文名称 | 英文名（推测） | 特征描述 |
|--------|---------|----------------|--------|
| 0 | 鳢肠 | *Eclipta prostrata* | 小型一年生草本，茎匍匐，叶互生 |
| 1 | 牵牛 | *Ipomoea nil* | 藤本植物，喇叭状花，常缠绕作物 |
| 2 | 牛筋草 | *Eleusine indica* | 多年生草本，叶片细长，根系发达 |
| 3 | 马齿苋 | *Portulaca oleracea* | 肉质叶，茎平卧，耐旱 |
| 4 | 狗尾草 | *Setaria viridis* | 穗状花序似狗尾，常见田间 |
| 5 | 芦苇 | *Phragmites australis* | 高大禾本科，水边或湿地常见 |
| 6 | 鸭跖草 | *Commelina communis* | 蓝紫色小花，三叶轮生 |
| 7 | 蒲公英 | *Taraxacum officinale* | 黄花头状，种子带绒毛 |
| 8 | 车前草 | *Plantago asiatica* | 叶基生，椭圆形，边缘波状 |
| 9 | 鼠尾草 | *Salvia japonica* | 紫色唇形花，芳香 |
| 10 | 野蔷薇 | *Rosa multiflora* | 灌木，有刺，粉红花 |
| 11 | 蕨类 | *Pteridium aquilinum* | 羽状复叶，喜阴湿环境 |

> 💡 注：实际类别需参考 `data.yaml` 文件中的 `names` 字段。

---

## ✅ 三、数据集目录结构（YOLO 格式）

```
weed_detection_dataset/
├── images/
│   ├── train/     # 2,796 张
│   └── val/       # 523 张
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### 📄 `data.yaml` 示例

```yaml
train: ./images/train
val: ./images/val

nc: 12
names: [
  'maculata',           # 鳢肠
  'rhombifolia',        # 牵牛
  'obtusifolia',        # 牛筋草
  'portulaca',          # 马齿苋
  'setaria',            # 狗尾草
  'phragmites',         # 芦苇
  'commelina',          # 鸭跖草
  'taraxacum',          # 蒲公英
  'plantago',           # 车前草
  'salvia',             # 鼠尾草
  'rosa',               # 野蔷薇
  'ferns'               # 蕨类
]
```

---

## ✅ 四、系统核心功能模块

| 功能 | 描述 |
|------|------|
| 🔐 **用户登录注册** | 支持用户名/密码验证，存储本地数据库 |
| 🖼️ **图片检测** | 上传单张图像，实时返回检测框与类别 |
| 📺 **视频检测** | 支持 MP4/AVI 视频文件播放并逐帧检测 |
| 🎥 **摄像头实时检测** | 调用 USB 摄像头，实现实时杂草识别 |
| 🔄 **双画面对比** | 左上角原始图 vs 左下角检测结果图 |
| 📊 **结果可视化** | 右侧表格展示类别、置信度、坐标 |
| ⚙️ **智能参数调节** | 滑块控制 `confidence` 和 `iou` 阈值 |
| 💾 **保存结果** | 保存标注图像和 JSON 结果文件 |

---

## ✅ 五、Python 项目源码结构

```
weed_detection_system/
├── main.py                     # 主程序入口
├── ui_main.py                  # UI 界面（由 Qt Designer 生成）
├── login.py                    # 登录注册逻辑
├── detect.py                   # YOLO 推理核心
├── utils/
│   ├── database.py             # 用户数据库
│   └── config.py               # 配置文件
├── models/
│   └── best.pt                 # 训练好的 YOLOv12 模型
├── datasets/
│   └── weed_detection_dataset/ # 数据集
└── resources/
    └── icons/                  # 图标资源
```

---

## ✅ 六、核心代码实现

### 1. `main.py` —— 主程序入口

```python
# -*- coding: utf-8 -*-
"""
杂草识别检测系统主程序
作者：AI助手
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_main import Ui_MainWindow
from login import LoginDialog
from detect import WeedDetector

class WeedDetectionApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 初始化检测器
        self.detector = WeedDetector(model_path='models/best.pt')
        
        # 连接按钮信号
        self.btn_login.clicked.connect(self.show_login)
        self.btn_image.clicked.connect(self.detect_image)
        self.btn_video.clicked.connect(self.detect_video)
        self.btn_camera.clicked.connect(self.detect_camera)
        self.btn_save.clicked.connect(self.save_results)
        self.btn_stop.clicked.connect(self.stop_detection)
        
        # 参数滑块
        self.slider_conf.valueChanged.connect(self.update_confidence)
        self.slider_iou.valueChanged.connect(self.update_iou)
        
        # 默认参数
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        
    def show_login(self):
        dialog = LoginDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.statusBar().showMessage("登录成功！")
            self.btn_login.setEnabled(False)
    
    def update_confidence(self):
        self.conf_thres = self.slider_conf.value() / 100.0
        self.conf_label.setText(f"{self.conf_thres:.2f}")
    
    def update_iou(self):
        self.iou_thres = self.slider_iou.value() / 100.0
        self.iou_label.setText(f"{self.iou_thres:.2f}")
    
    def detect_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.detector.detect_image(file_name, conf=self.conf_thres, iou=self.iou_thres)
            self.display_results()
    
    def display_results(self):
        # 显示原始图像
        self.label_original.setPixmap(QPixmap(self.detector.original_image))
        
        # 显示检测结果
        self.label_result.setPixmap(QPixmap(self.detector.result_image))
        
        # 更新表格
        self.tableWidget.clearContents()
        for row, (cls, conf, x, y) in enumerate(self.detector.results):
            self.tableWidget.setItem(row, 0, QTableWidgetItem(cls))
            self.tableWidget.setItem(row, 1, QTableWidgetItem(f"{conf:.2f}"))
            self.tableWidget.setItem(row, 2, QTableWidgetItem(str(x)))
            self.tableWidget.setItem(row, 3, QTableWidgetItem(str(y)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WeedDetectionApp()
    window.show()
    sys.exit(app.exec_())
```

---

### 2. `detect.py` —— YOLO 推理核心

```python
# -*- coding: utf-8 -*-
"""
杂草检测推理模块
基于 ultralytics YOLOv8（模拟 YOLOv12）
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class WeedDetector:
    def __init__(self, model_path='models/best.pt'):
        self.model = YOLO(model_path)
        self.original_image = None
        self.result_image = None
        self.results = []
    
    def detect_image(self, image_path, conf=0.25, iou=0.45):
        img = cv2.imread(image_path)
        self.original_image = image_path
        
        results = self.model(img, conf=conf, iou=iou)
        
        # 保存检测结果
        self.result_image = results[0].plot()
        
        # 提取检测信息
        self.results = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            class_name = self.model.names[cls_id]
            self.results.append((class_name, conf, x1, y1))
```

---

### 3. `login.py` —— 登录注册模块

```python
# -*- coding: utf-8 -*-
"""
用户登录注册对话框
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("用户登录")
        self.resize(300, 200)
        
        layout = QVBoxLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("用户名")
        layout.addWidget(self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("密码")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)
        
        self.login_btn = QPushButton("登录")
        self.login_btn.clicked.connect(self.login)
        layout.addWidget(self.login_btn)
        
        self.register_btn = QPushButton("注册")
        self.register_btn.clicked.connect(self.register)
        layout.addWidget(self.register_btn)
        
        self.setLayout(layout)
    
    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        
        if username and password:
            # 简单验证（实际应连接数据库）
            if username == "admin" and password == "123456":
                self.accept()
            else:
                QMessageBox.warning(self, "错误", "用户名或密码错误！")
        else:
            QMessageBox.warning(self, "错误", "请输入用户名和密码！")
    
    def register(self):
        QMessageBox.information(self, "提示", "注册功能暂未实现。")
```

---

## ✅ 七、UI 界面设计建议（PyQt）

### 🎨 科幻风界面特点：
- 深色背景（`#0d1b2a`）
- 蓝色发光边框（`QFrame` 边框样式）
- 半透明控件（`opacity: 0.9`）
- 动态光效（使用 `QGraphicsEffect`）

### 示例样式（`ui_main.py` 中）

```python
self.setStyleSheet("""
    QWidget {
        background-color: #0d1b2a;
        color: white;
    }
    QPushButton {
        background-color: #1e3a8a;
        border: 2px solid #0d1b2a;
        padding: 10px;
        border-radius: 8px;
    }
    QPushButton:hover {
        background-color: #2563eb;
    }
    QSlider {
        background-color: #1e3a8a;
    }
""")
```

---

## ✅ 八、部署建议

| 场景 | 方案 |
|------|------|
| **农业无人机** | Jetson Nano + ONNX 模型（轻量化） |
| **田间移动终端** | Android App（Kotlin + TensorFlow Lite） |
| **智慧农场中心** | Web 平台（FastAPI + React） |
| **边缘计算设备** | NVIDIA Jetson AGX Orin + TensorRT 加速 |

---

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b9936001e2de488284cb7cdb63b7434e.png)


打造高精度智慧农业杂草识别系统！🌱🌾🤖
