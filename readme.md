# 目标框标注工具

支持系统：Windows, Linux，Mac系统

标注文件为yolo格式

> ```
> <object-class> <x> <y> <width> <height>
> ```
>
> Where `x`, `y`, `width`, and `height` are relative to the image's width and height

## 安装依赖

```shell
pip install numpy opencv-python -t thirdpartys/lib/site-packages
```

## 运行

提供json格式的配置文件，示例如voc.json

```shell
python labeled_tool.py -c voc.json
```

![example](./pics/example.jpg)

## 功能

* 目标框标注
* 目标框删除
* 目标框移动
* 目标框标注id修改
* 撤销当前删除
* 类别过滤标注
* 鼠标左键单击切换id

## 说明

* 按键说明: (字母按键不区分大小写, 输入法需要切换到英文模式)

  * Esc: 退出程序

  * Space: 自动播放与暂停

  * -: 切换到删除模式

  * +: 切换到移动模式

  * Backspace: 切换到撤销模式

  * 1\~9: 切换类别label_id为 0\~8(越界时为最大类别号)

  * 0: 切换类别label_id为9(越界时为最大类别号)

  * Q/↑: 向前切换类别

  * W/↓: 向后切换类别

  * A/←: 上一张图片

  * D/→: 下一张图片

  * L: 删除当前图片和标注文件

* 鼠标事件:
  * 标注模式:

    ​    鼠标左键拖动进行目标框标注, 按下与松开分别对应左上点和右下点的位置

    ​    Windows鼠标右键(Linux, Mac左键双击)进行进行目标框类别label_id的切换, 切换离当前鼠标位置最近的框(更新为鼠标点击需要在框内，防止误操作)

  * 删除模式:

    ​    Windows鼠标右键(Linux, Mac左键双击)删除目标框, 删除离当前鼠标位置最近的框 

  * 移动模式:

    ​    鼠标左键拖动来移动目标框，移动离当前鼠标位置最近的框

  * 撤销模式:

    ​    Windows鼠标右键(Linux, Mac左键双击)撤销删除操作, 撤销对当前图片的一次删除操作