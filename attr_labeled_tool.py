import os
import sys
work_root = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(work_root, "thirdpartys/lib/site-packages"))
import glob
import json
import hashlib
import re
import random
import numpy as np
import cv2
import argparse
from collections import OrderedDict
from pynput.keyboard import Key, Controller
"""
图像目标框属性附加标注脚本
Author: Heroinlj
按键说明: (字母按键不区分大小写, 输入法需要切换到英文模式)
    Esc: 退出程序
    Space: 自动播放与暂停
    -: 切换到删除模式
    +: 切换到移动模式
    Backspace: 切换到撤销模式
    \| : 切换到修正模式
    1~9: 切换类别label_id为 0~8(越界时为最大类别号)
    0: 切换类别label_id为9(越界时为最大类别号)
    W/↑: 向前切换类别
    S/↓: 向后切换类别
    A/←: 上一张图片
    D/→: 下一张图片
    L: 删除当前图片和标注文件
    Q: 切换到上一属性列表
    E: 切换到下一属性列表
鼠标事件:
    标注模式:
        鼠标左键拖动进行目标框标注, 按下与松开分别对应左上点和右下点的位置
        Windows鼠标右键(Linux, Mac左键双击)进行进行目标框类别label_id的切换, 切换离当前鼠标位置最近的框
    删除模式:
        鼠标左键点击, 高亮离当前鼠标位置最近的框 
        Windows鼠标右键(Linux, Mac左键双击)删除目标框, 删除离当前鼠标位置最近的框
        可以切换属性后进行目标框和属性的一起标注添加
    移动模式:
        鼠标左键点击, 高亮离当前鼠标位置最近的框
        鼠标左键拖动来移动目标框，移动离当前鼠标位置最近的框
    撤销模式:
        Windows鼠标右键(Linux, Mac左键双击)撤销操作, 撤销对当前图片的一次操作, 可自定义设置最大撤销记录个数，默认为10
    修正模式：
        鼠标左键点击, 高亮离当前鼠标位置最近的框
        按住Alt的同时鼠标左键点击, 高亮所选框最近的顶点(左上点或右下点)
​        单击鼠标中键切换高亮的点
        在高亮的情况下, Windows鼠标右键(Linux, Mac左键双击), 修正所选点的位置
参数文件说明：
    可在对应参数文件(如voc_attr.json)中设置参数
    windows_name： 标注程序窗口命名
    dataset_path： 标注数据文件夹路径, 路径下格式为
                   - dataset_path
                     - images
                     - labels   (可选)
    decay_time: 自动播放等待时间， 单位ms
    pixel_size： 图像显示最大像素
    select_type： 选择框模式，默认取离左上点最近框, 取值为1可改为取离中心最近框
    checkpoint_name: 已标注记录文件, 用来支持多人同时标注
    total_class_names： 标注所有类别列表
    class_names： 当前标注只展示的类别列表
    colors: 标注颜色列表
    attrs： 标注所有属性列表
"""
ix, iy = -1, -1
is_mouse_lb_down = False
highlight_idx = -1
select_idx = -1


# 目标框标注程序
class CLabeled:
    def __init__(self, image_folder):
        # 存放需要标注图像的文件夹
        self.image_folder = image_folder
        # 需要标注图像的总数量
        self.total_image_number = 0
        # 需要标注图像的地址列表
        self.images_list = list()
        # 当前标注图片的索引号，也是已标注图片的数量
        self.current_label_index = 0
        # 整个窗口图片，包含图片标注区域和按键区域
        self.win_image = None
        # 待标注图片
        self.image = None
        # 属性类型索引
        self.attr_type_idx = 0
        # 目标框的分类索引号
        self.label_index = 0
        # 需要移动的分类索引号
        self.move_idx = -1
        # 标注框信息
        self.boxes = list()
        # 标注属性信息
        self.attrs = list()
        # 缓存上次操作的框，以进行恢复
        self.undo_boxes = []
        # 记录撤销操作的最大个数
        self.undo_boxes_max_len = 10
        # 过滤不显示的框
        self.fliter_boxes = []
        # 是否保存过滤的框
        self.fliter_flag = False
        # 当前图片
        self.current_image = None
        # 标注框的保存文件地址
        self.label_path = None
        # 记录历史标注位置的文本文件地址
        self.checkpoint_path = os.path.join(image_folder, "checkpoint")
        self.annotation = None
        # 类别数
        self.class_num = 0
        # 原始所有类别名称
        self.src_total_class_names = None
        # 原始展示类别名称
        self.src_class_names = None
        # 所有类别名称
        self.total_class_names = None
        # 类别列表
        self.class_table = None
        # 标注展示类别名称
        self.class_names = None
        # 属性字典
        self.attrs_dict = None
        # 类别对应的颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(max(1, self.class_num))]
        # 高亮颜色
        self.highlight_colors = [[180, 130, 70], [160, 158, 95]]
        # 右侧类别名显示的宽度
        self.class_width = 400
        # 图像宽
        self.width = 720
        # 图像高
        self.height = 576
        # 显示窗口的名称
        self.windows_name = "image"
        self.font_type = cv2.FONT_HERSHEY_SIMPLEX
        # 是否有进行操作
        self.attr_flag = True
        self.operate_flag = False
        self.auto_play_flag = False
        # 默认自动播放等待时间, ms为单位
        self.default_decay_time = 1000
        self.decay_time = self.default_decay_time if self.auto_play_flag else 0
        # 默认显示最大分辨率
        self.pixel_size = 1920
        # 删除选框方式, 默认左上点, 1为中心点
        self.select_type = 0
        self._may_make_dir()
        # 模拟按键事件
        self.keyboard = Controller()

    # 重置
    def _reset(self):
        global is_mouse_lb_down, highlight_idx, select_idx
        is_mouse_lb_down = False
        highlight_idx = -1
        select_idx = -1
        self.image = None
        self.current_image = None
        self.label_path = None
        self.boxes = list()
        self.attrs = list()
        self.operate_flag = False
        self.move_idx = -1

    # 参数检查，确保代码可运行
    def _check(self):
        if self.class_num < 1:
            self.class_num = 1
        if self.total_class_names is None:
            self.total_class_names = range(self.class_num)
        if self.attr_type_idx:
            attrs_list = []
            for attr_dict in self.attrs_dict.values():
                for attr_list in attr_dict.values():
                    attrs_list.append(attr_list)
            self.total_class_names = attrs_list[self.attr_type_idx - 1].copy()
            self.class_names = self.total_class_names.copy()
            # self.total_class_names.extend(["delete"])
            # self.class_names.extend(["delete"])
        else:
            self.total_class_names = self.src_total_class_names.copy()
            self.class_names = self.src_class_names.copy()
        if isinstance(self.total_class_names, list):
            self.total_class_names.extend(["delete", "move", "undo", "fix"])
        if self.class_names is None:
            self.class_names = self.total_class_names
        else:
            self.class_names.extend(["delete", "move", "undo", "fix"])
        self.class_num = len(self.class_names)
        self.class_table = [
            self.total_class_names.index(name) for name in self.class_names
        ]
        self.label_index = 0
        # 判断当前颜色列表是否够用, 不够的话进行随机添加
        if isinstance(self.colors, list) and len(self.colors) < self.class_num:
            self.colors.extend(
                [[random.randint(0, 255) for _ in range(3)]
                 for _ in range(max(1, self.class_num - len(self.colors)))])

    # 统计所有图片个数
    def _compute_total_image_number(self):
        self.total_image_number = len(self.images_list)

    # 判断是否需要新建文件夹
    def _may_make_dir(self):
        if not os.path.exists(self.image_folder):
            print(self.image_folder, " does not exists! please check it !")
            exit(-1)
        path = os.path.join(self.image_folder, "labels")
        if not os.path.exists(path):
            os.makedirs(path)

    # 当前标注位置倒退一个
    def _backward(self):
        self.current_label_index -= 1
        self.current_label_index = max(0, self.current_label_index)

    # 限定鼠标坐标区域的大小
    def _roi_limit(self, x, y):
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.width:
            x = self.width
        if y > self.height:
            y = self.height
        return x, y

    # 限定坐标框的移动不超界
    def _move_delta_limit(self, delta_x, delta_y, box):
        if (delta_x + box[0] * self.width) < 0:
            delta_x = -box[0] * self.width
        if (delta_x + box[2] * self.width) > self.width:
            delta_x = self.width * (1 - box[2])
        if (delta_y + box[1] * self.height) < 0:
            delta_y = -box[1] * self.height
        if (delta_y + box[3] * self.height) > self.height:
            delta_y = self.height * (1 - box[3])
        x1 = int(box[0] * self.width + delta_x)
        y1 = int(box[1] * self.height + delta_y)
        x2 = int(box[2] * self.width + delta_x)
        y2 = int(box[3] * self.height + delta_y)
        return x1, y1, x2, y2

    # box标准化
    def box_fix(self, box):
        left_top_x = min(box[0], box[2])
        left_top_y = min(box[1], box[3])
        right_bottom_x = max(box[0], box[2])
        right_bottom_y = max(box[1], box[3])
        box[0:4] = [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
        return box

    # 根据坐标点与所有框中心距离的远近, 获取框索引的排序
    def _get_sort_indices(self, x, y):
        current_point = np.array([x / self.width, y / self.height])
        current_left_top_point = np.array([box[0:2]
                                           for box in self.boxes])  # 左上点
        current_center_point = (np.array([box[0:2] for box in self.boxes]) +
                                np.array([box[2:4]
                                          for box in self.boxes])) / 2  # 中心点
        current_select_point = current_center_point if self.select_type else current_left_top_point
        square1 = np.sum(np.square(current_select_point), axis=1)
        square2 = np.sum(np.square(current_point), axis=0)
        squared_dist = - 2 * \
            np.matmul(current_select_point,
                        current_point.T) + square1 + square2
        sort_indices = np.argsort(squared_dist)
        return sort_indices

    # 判断点(x, y) 是否在框内
    def _point_in_box(self, x, y, box):
        if (x / self.width) < box[0]:
            return False
        if (y / self.height) < box[1]:
            return False
        if (x / self.width) > box[2]:
            return False
        if (y / self.height) > box[3]:
            return False
        return True

    # 鼠标左键点击切换id事件
    def _event_lbuttondown(self, x, y, dst):
        global highlight_idx
        highlight_idx = -1
        if self.width < x <= (self.width +
                              self.class_width) and 0 <= y <= self.height:
            per_class_h = int(min(self.height, 2000) / (self.class_num + 3))
            select_mode = max(int((y - 5) / per_class_h), 0)
            if select_mode > self.class_num:
                if (x - self.width) < self.class_width / 2:
                    self.keyboard.press('a')
                    self.keyboard.release('a')
                elif (x - self.width) >= self.class_width / 2:
                    self.keyboard.press('d')
                    self.keyboard.release('d')
            else:
                self.label_index = min(select_mode,
                                    self.class_num - 1)
        x, y = self._roi_limit(x, y)
        self._update_win_image(dst)
        cv2.imshow(self.windows_name, self.win_image)
        if self.class_names[self.label_index] == "move":
            global move_box
            move_box = None
            cv2.setMouseCallback(self.windows_name, self._move_roi)
        elif self.class_names[self.label_index] == "delete":
            cv2.setMouseCallback(self.windows_name, self._delete_roi)
        elif self.class_names[self.label_index] == "undo":
            cv2.setMouseCallback(self.windows_name, self._undo_roi)
        elif self.class_names[self.label_index] == "fix":
            cv2.setMouseCallback(self.windows_name, self._fix_roi)
        elif not self.attr_type_idx:
            cv2.setMouseCallback(self.windows_name, self._draw_roi)
        elif self.attr_type_idx:
            cv2.setMouseCallback(self.windows_name, self._attr_map)

    # 标注感兴趣区域
    def _draw_roi(self, event, x, y, flags, param, mode=True):
        global ix, iy, move_ix, move_iy, is_mouse_lb_down
        box_border = round(self.width / 400)
        dst = self.image.copy()
        self._draw_box_on_image(dst, self.boxes)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                is_mouse_lb_down = True
        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE and not (flags
                                                   and cv2.EVENT_FLAG_LBUTTON):
            x, y = self._roi_limit(x, y)
            if mode:
                cv2.line(dst, (x, 0), (x, self.height),
                         self.colors[self.label_index], 1, 8)
                cv2.line(dst, (0, y), (self.width, y),
                         self.colors[self.label_index], 1, 8)
            else:
                cv2.circle(dst, (x, y), 5, (0, 0, 255), -1)
            self._update_win_image(dst)
            cv2.imshow(self.windows_name, self.win_image)
        # 按住鼠标左键进行移动
        elif event == cv2.EVENT_MOUSEMOVE and (flags
                                               and cv2.EVENT_FLAG_LBUTTON):
            x, y = self._roi_limit(x, y)
            if mode:
                cv2.rectangle(dst, (ix, iy), (x, y),
                              self.colors[self.label_index], 1)
            else:
                cv2.circle(dst, (x, y), 5, self.colors[self.label_index], -1)
            self._update_win_image(dst)
            cv2.imshow(self.windows_name, self.win_image)
        elif event == cv2.EVENT_LBUTTONUP and is_mouse_lb_down:  # 鼠标左键松开
            if len(self.undo_boxes) > self.undo_boxes_max_len:
                del self.undo_boxes[0]
            self.undo_boxes.append(self.boxes.copy())
            x, y = self._roi_limit(x, y)
            if mode:
                if abs(x - ix) > 10 and abs(y - iy) > 10:
                    cv2.rectangle(self.current_image, (ix, iy), (x, y),
                                  self.colors[self.label_index], box_border)
                    label_id = self.class_table[self.label_index]
                    box = [
                        ix / self.width, iy / self.height, x / self.width,
                        y / self.height, label_id
                    ]
                    box = self.box_fix(box)
                    if self.attr_flag:
                        box.extend([0] * len(self.attrs_dict))
                    self.boxes.append(box)
            else:
                cv2.circle(self.current_image, (x, y), 5, (0, 0, 255), -1)
            # print(self.boxes)
            self._draw_box_on_image(self.current_image, self.boxes)
            self.operate_flag = True
            is_mouse_lb_down = False
        # elif event == cv2.EVENT_RBUTTONDOWN:  # 撤销上一次标注
        #     self.current_image = self.image.copy()
        #     if len(self.boxes):
        #         del self.boxes[-1]
        #         self._draw_box_on_image(self.current_image, self.boxes)
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (
                sys.platform in ["linux", "darwin"] and event
                == cv2.EVENT_LBUTTONDBLCLK):  # 修改(中心点或左上点)距离当前鼠标最近的框的label_id
            if len(self.undo_boxes) > self.undo_boxes_max_len:
                del self.undo_boxes[0]
            self.undo_boxes.append(self.boxes.copy())
            x, y = self._roi_limit(x, y)
            self.current_image = self.image.copy()
            change_idx = 0
            if len(self.boxes):
                if len(self.boxes) > 1:
                    # 优先修改(中心点或左上点)距离当前鼠标最近的框的label_id
                    sort_indices = self._get_sort_indices(x, y)
                    change_idx = sort_indices[0]
                change_box = self.boxes[change_idx].copy()
                if self._point_in_box(x, y, change_box):
                    label_id = self.class_table[self.label_index]
                    change_box[4] = label_id
                    self.boxes[change_idx] = change_box
                    self._draw_box_on_image(self.current_image, self.boxes)
                self.operate_flag = True

    # 属性标注
    def _attr_map(self, event, x, y, flags, param, mode=True):
        global ix, iy, move_ix, move_iy, is_mouse_lb_down
        box_border = round(self.width / 400)
        dst = self.image.copy()
        self._draw_box_on_image(dst, self.boxes)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                is_mouse_lb_down = True
        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE and not (flags
                                                   and cv2.EVENT_FLAG_LBUTTON):
            x, y = self._roi_limit(x, y)
            if mode:
                cv2.line(dst, (x, 0), (x, self.height),
                         self.colors[self.label_index], 1, 8)
                cv2.line(dst, (0, y), (self.width, y),
                         self.colors[self.label_index], 1, 8)
            else:
                cv2.circle(dst, (x, y), 5, (0, 0, 255), -1)
            self._update_win_image(dst)
            cv2.imshow(self.windows_name, self.win_image)
        # 按住鼠标左键进行移动
        elif event == cv2.EVENT_MOUSEMOVE and (flags
                                               and cv2.EVENT_FLAG_LBUTTON):
            x, y = self._roi_limit(x, y)
            if mode:
                cv2.rectangle(dst, (ix, iy), (x, y),
                              self.colors[self.label_index], 1)
            else:
                cv2.circle(dst, (x, y), 5, self.colors[self.label_index], -1)
            self._update_win_image(dst)
            cv2.imshow(self.windows_name, self.win_image)
        elif event == cv2.EVENT_LBUTTONUP and is_mouse_lb_down:  # 鼠标左键松开
            if len(self.undo_boxes) > self.undo_boxes_max_len:
                del self.undo_boxes[0]
            self.undo_boxes.append(self.boxes.copy())
            x, y = self._roi_limit(x, y)
            if mode:
                if abs(x - ix) > 10 and abs(y - iy) > 10:
                    cv2.rectangle(self.current_image, (ix, iy), (x, y),
                                  self.colors[self.label_index], box_border)
                    attr_type_name = list(
                        self.attrs_dict.keys())[self.attr_type_idx - 1]
                    label_name = list(
                        self.attrs_dict[attr_type_name].keys())[0]
                    label_id = self.src_total_class_names.index(label_name)
                    box = [
                        ix / self.width, iy / self.height, x / self.width,
                        y / self.height, label_id
                    ]
                    box = self.box_fix(box)
                    box.extend([0] * len(self.attrs_dict))
                    box[4 + self.attr_type_idx] = self.label_index + 1
                    self.boxes.append(box)
            else:
                cv2.circle(self.current_image, (x, y), 5, (0, 0, 255), -1)
            # print(self.boxes)
            self._draw_box_on_image(self.current_image, self.boxes)
            self.operate_flag = True
            is_mouse_lb_down = False
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (
                sys.platform in ["linux", "darwin"] and event
                == cv2.EVENT_LBUTTONDBLCLK):  # 修改(中心点或左上点)距离当前鼠标最近的框的label_id
            if len(self.undo_boxes) > self.undo_boxes_max_len:
                del self.undo_boxes[0]
            self.undo_boxes.append(self.boxes.copy())
            x, y = self._roi_limit(x, y)
            self.current_image = self.image.copy()
            change_idx = 0
            if len(self.boxes):
                if len(self.boxes) > 1:
                    # 优先修改(中心点或左上点)距离当前鼠标最近的框的label_id
                    sort_indices = self._get_sort_indices(x, y)
                    change_idx = sort_indices[0]
                change_box = self.boxes[change_idx].copy()
                if self._point_in_box(x, y, change_box):
                    label_id = self.class_table[self.label_index]
                    change_box[4 + self.attr_type_idx] = label_id + 1
                    self.boxes[change_idx] = change_box
                    self._draw_box_on_image(self.current_image, self.boxes)
                self.operate_flag = True

    # 对选择的框进行移动
    def _move_roi(self, event, x, y, flags, param):
        global ix, iy, move_box, is_mouse_lb_down, highlight_idx
        dst = self.image.copy()
        if highlight_idx >= 0:
            self._draw_box_highlight_on_image(dst, self.boxes[highlight_idx])
        self._draw_box_on_image(dst, self.boxes)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            move_box = None
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                is_mouse_lb_down = True
        elif event == cv2.EVENT_MOUSEMOVE and (flags
                                               and cv2.EVENT_FLAG_LBUTTON):
            x, y = self._roi_limit(x, y)
            if len(self.boxes) and is_mouse_lb_down:
                if len(self.boxes) > 1 and self.move_idx == -1:
                    sort_indices = self._get_sort_indices(x, y)
                    self.move_idx = sort_indices[0]
                boxes = self.boxes.copy()
                del boxes[self.move_idx]
                move_box = self.boxes[self.move_idx]
                self._draw_box_on_image(dst, boxes)
                x1, y1, x2, y2 = self._move_delta_limit(
                    x - ix, y - iy, move_box)
                cv2.rectangle(dst, (x1, y1), (x2, y2),
                              self.colors[self.label_index], 1)
                self._update_win_image(dst)
                cv2.imshow(self.windows_name, self.win_image)
        elif event == cv2.EVENT_LBUTTONUP and is_mouse_lb_down:  # 鼠标左键松开
            x, y = self._roi_limit(x, y)
            is_mouse_lb_down = False
            highlight_idx = -1
            if len(self.boxes):
                highlight_idx = 0
                if len(self.boxes) > 1:
                    # 优先修改(中心点或左上点)距离当前鼠标最近的框的label_id
                    sort_indices = self._get_sort_indices(x, y)
                    highlight_idx = sort_indices[0]
                if not self._point_in_box(x, y, self.boxes[highlight_idx]):
                    highlight_idx = -1
            if move_box is not None:
                if len(self.undo_boxes) > self.undo_boxes_max_len:
                    del self.undo_boxes[0]
                self.undo_boxes.append(self.boxes.copy())
                label_idx = self.boxes[self.move_idx][4]
                if self.attr_flag:
                    attr_idxs = self.boxes[self.move_idx][5:5 +
                                                          len(self.attrs_dict)]
                del self.boxes[self.move_idx]
                x1, y1, x2, y2 = self._move_delta_limit(
                    x - ix, y - iy, move_box)
                box = [
                    x1 / self.width, y1 / self.height, x2 / self.width,
                    y2 / self.height, label_idx
                ]
                if self.attr_flag:
                    box.extend(attr_idxs)
                self.boxes.insert(self.move_idx, box)
                self._draw_box_on_image(self.image.copy(), self.boxes)
                self.move_idx = -1
                move_box = None
                self.operate_flag = True

    # 对选择的框进行删除
    def _delete_roi(self, event, x, y, flags, param):
        global is_mouse_lb_down, highlight_idx
        dst = self.image.copy()
        if highlight_idx >= 0:
            self._draw_box_highlight_on_image(dst, self.boxes[highlight_idx])
        self._draw_box_on_image(dst, self.boxes)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                is_mouse_lb_down = True
        # 鼠标左键松开, 高亮选中的框
        elif event == cv2.EVENT_LBUTTONUP and is_mouse_lb_down:
            x, y = self._roi_limit(x, y)
            is_mouse_lb_down = False
            highlight_idx = -1
            if len(self.boxes):
                highlight_idx = 0
                if len(self.boxes) > 1:
                    # 优先修改(中心点或左上点)距离当前鼠标最近的框的label_id
                    sort_indices = self._get_sort_indices(x, y)
                    highlight_idx = sort_indices[0]
                if not self._point_in_box(x, y, self.boxes[highlight_idx]):
                    highlight_idx = -1
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (
                sys.platform in ["linux", "darwin"]
                and event == cv2.EVENT_LBUTTONDBLCLK):  # 删除(中心点或左上点)距离当前鼠标最近的框
            x, y = self._roi_limit(x, y)
            self.current_image = self.image.copy()
            del_index = 0
            if len(self.boxes):
                if len(self.boxes) > 1:
                    # 优先删除(中心点或左上点)距离当前鼠标最近的当前类别框
                    sort_indices = self._get_sort_indices(x, y)
                    del_index = sort_indices[0]
                if len(self.undo_boxes) > self.undo_boxes_max_len:
                    del self.undo_boxes[0]
                self.undo_boxes.append(self.boxes.copy())
                del self.boxes[del_index]
                self._draw_box_on_image(self.current_image, self.boxes)
                self.operate_flag = True
                highlight_idx = -1

    # 对选择的框进行修正
    def _fix_roi(self, event, x, y, flags, param):
        global is_mouse_lb_down, highlight_idx, select_idx
        box_border = round(self.width / 400)
        dst = self.image.copy()
        if highlight_idx >= 0:
            self._draw_box_highlight_on_image(dst, self.boxes[highlight_idx])
        self._draw_box_on_image(dst, self.boxes)
        if highlight_idx >= 0:
            self._draw_point_highlight_on_image(dst, self.boxes[highlight_idx], select_idx=max(select_idx, 0))
        # ALT + 按下鼠标左键
        if flags == (cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_ALTKEY):
            x, y = self._roi_limit(x, y)
            if highlight_idx >= 0:
                select_idx = self._draw_point_highlight_on_image(
                    dst, self.boxes[highlight_idx], x, y)
            cv2.line(dst, (x, 0), (x, self.height),
                     self.colors[self.label_index], 1, 8)
            cv2.line(dst, (0, y), (self.width, y),
                     self.colors[self.label_index], 1, 8)
            self._update_win_image(dst)
            cv2.imshow(self.windows_name, self.win_image)
        # 按下鼠标中键切换高亮点
        elif flags == (cv2.EVENT_FLAG_MBUTTON):
            x, y = self._roi_limit(x, y)
            if highlight_idx >= 0:
                select_idx = int(not max(select_idx, 0))
                self._draw_point_highlight_on_image(
                    dst, self.boxes[highlight_idx], select_idx=select_idx)
            cv2.line(dst, (x, 0), (x, self.height),
                     self.colors[self.label_index], 1, 8)
            cv2.line(dst, (0, y), (self.width, y),
                     self.colors[self.label_index], 1, 8)
            self._update_win_image(dst)
            cv2.imshow(self.windows_name, self.win_image)
        # 高亮开启且按下右键
        elif highlight_idx >= 0 and ("win32" in sys.platform
                                     and event == cv2.EVENT_RBUTTONDOWN) or (
                                         sys.platform in ["linux", "darwin"]
                                         and event == cv2.EVENT_LBUTTONDBLCLK):
            if len(self.undo_boxes) > self.undo_boxes_max_len:
                del self.undo_boxes[0]
            self.undo_boxes.append(self.boxes.copy())
            x, y = self._roi_limit(x, y)
            highlight_box = self.boxes[highlight_idx]
            pt1 = (int(dst.shape[1] * highlight_box[0]),
                   int(dst.shape[0] * highlight_box[1]))
            pt2 = (int(dst.shape[1] * highlight_box[2]),
                   int(dst.shape[0] * highlight_box[3]))
            if select_idx == 0:
                pt1 = (x, y)
            elif select_idx == 1:
                pt2 = (x, y)
            select_idx = max(select_idx, 0)
            fix_box = self.boxes[highlight_idx].copy()
            fix_box[select_idx * 2] = x / dst.shape[1]
            fix_box[select_idx * 2 + 1] = y / dst.shape[0]
            fix_box = self.box_fix(fix_box)
            self.boxes[highlight_idx] = fix_box
            self.current_image = self.image.copy()
            self._draw_box_on_image(self.current_image, self.boxes)
            highlight_idx = -1
            self.operate_flag = True
        elif event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                is_mouse_lb_down = True
        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE:
            x, y = self._roi_limit(x, y)
            cv2.line(dst, (x, 0), (x, self.height),
                     self.colors[self.label_index], 1, 8)
            cv2.line(dst, (0, y), (self.width, y),
                     self.colors[self.label_index], 1, 8)
            self._update_win_image(dst)
            cv2.imshow(self.windows_name, self.win_image)
        # 鼠标左键松开, 高亮选中的框
        elif event == cv2.EVENT_LBUTTONUP and is_mouse_lb_down:
            x, y = self._roi_limit(x, y)
            is_mouse_lb_down = False
            highlight_idx = -1
            if len(self.boxes):
                highlight_idx = 0
                if len(self.boxes) > 1:
                    # 优先修改(中心点或左上点)距离当前鼠标最近的框的label_id
                    sort_indices = self._get_sort_indices(x, y)
                    highlight_idx = sort_indices[0]
                if not self._point_in_box(x, y, self.boxes[highlight_idx]):
                    highlight_idx = -1

    # 恢复上一次删除的框
    def _undo_roi(self, event, x, y, flags, param):
        dst = self.image.copy()
        self._draw_box_on_image(dst, self.boxes)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            self._event_lbuttondown(x, y, dst)
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (
                sys.platform in ["linux", "darwin"] and event
                == cv2.EVENT_LBUTTONDBLCLK):  # 撤销删除(中心点或左上点)距离当前鼠标最近的框
            x, y = self._roi_limit(x, y)
            self.current_image = self.image.copy()
            if len(self.undo_boxes):
                self.boxes = self.undo_boxes[-1]
                del self.undo_boxes[-1]
                self.operate_flag = True
            self._draw_box_on_image(self.current_image, self.boxes)

    # 将标注框显示到图像上
    def _draw_box_on_image(self, image, boxes):
        box_border = round(self.width / 400)
        font_size = max(1, int(min(self.width, self.height) / 600))
        for box in boxes:
            if not len(box):
                continue
            pt1 = (int(image.shape[1] * box[0]), int(image.shape[0] * box[1]))
            pt2 = (int(image.shape[1] * box[2]), int(image.shape[0] * box[3]))
            if len(box) > 4:
                label_id = box[4 + self.attr_type_idx]
            else:
                label_id = 0
            if self.attr_type_idx and label_id:
                label_id -= 1
            label_index = self.class_table.index(label_id)
            cv2.rectangle(image, pt1, pt2, self.colors[label_index],
                          box_border)
            if pt1[1] < 10:
                cv2.putText(image, self.class_names[label_index],
                            (pt1[0], pt1[1] + 10), self.font_type,
                            min(font_size * 0.4,
                                1.0), self.colors[label_index], font_size)
            else:
                cv2.putText(image,
                            self.class_names[label_index], pt1, self.font_type,
                            min(font_size * 0.4,
                                1.0), self.colors[label_index], font_size)
        self._update_win_image(image)
        cv2.imshow(self.windows_name, self.win_image)

    # 将标注框高亮显示到图像上
    def _draw_box_highlight_on_image(self, image, box):
        box_border = round(self.width / 400)
        font_size = max(1, int(min(self.width, self.height) / 600))
        pt1 = (int(image.shape[1] * box[0]), int(image.shape[0] * box[1]))
        pt2 = (int(image.shape[1] * box[2]), int(image.shape[0] * box[3]))
        if len(box) > 4:
            label_id = box[4 + self.attr_type_idx]
        else:
            label_id = 0
        if self.attr_type_idx and label_id:
            label_id -= 1
        label_index = self.class_table.index(label_id)
        cv2.rectangle(image, pt1, pt2, self.highlight_colors[0],
                      box_border * 4)
        cv2.rectangle(image, pt1, pt2, self.colors[label_index], box_border)
        if pt1[1] < 10:
            cv2.putText(image, self.class_names[label_index],
                        (pt1[0], pt1[1] + 10), self.font_type,
                        min(font_size * 0.4,
                            1.0), self.colors[label_index], font_size)
        else:
            cv2.putText(image,
                        self.class_names[label_index], pt1, self.font_type,
                        min(font_size * 0.4,
                            1.0), self.colors[label_index], font_size)
        self._update_win_image(image)
        cv2.imshow(self.windows_name, self.win_image)

    # 将标注框上选择的点高亮显示到图像上
    def _draw_point_highlight_on_image(self, image, box, x=None, y=None, select_idx=0):
        box_border = round(self.width / 400)
        font_size = max(1, int(min(self.width, self.height) / 600))
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        if x is not None and y is not None:
            current_select_point = np.array([pt1, pt2])
            current_point = np.array([x / self.width, y / self.height])
            square1 = np.sum(np.square(current_select_point), axis=1)
            square2 = np.sum(np.square(current_point), axis=0)
            squared_dist = - 2 * \
                np.matmul(current_select_point,
                            current_point.T) + square1 + square2
            sort_indices = np.argsort(squared_dist)
            select_idx = sort_indices[0]
        select_point = pt1 if select_idx == 0 else pt2
        select_point = (int(image.shape[1] * select_point[0]),
                        int(image.shape[0] * select_point[1]))
        cv2.circle(image, select_point, 5, self.highlight_colors[1], -1,
                   cv2.LINE_AA)
        self._update_win_image(image)
        cv2.imshow(self.windows_name, self.win_image)
        return select_idx

    # 更新整个窗口的显示
    def _update_win_image(self, image):
        per_class_h = int(min(self.height, 2000) / (self.class_num + 3))
        font_size = max(1, int(min(self.width, self.height) / 600))
        self.win_image = np.zeros(
            [self.height, self.class_width + self.width, 3], dtype=np.uint8)
        self.win_image[:, self.width:self.width + self.class_width, :] = 255
        self.win_image[(self.label_index + 1) * per_class_h -
                       min(per_class_h, 10):5 +
                       (self.label_index + 1) * per_class_h,
                       self.width:self.width + self.class_width] = [
                           255, 245, 152
                       ]
        for idx in range(self.class_num):
            show_msg = str(idx + 1) + ": " + self.class_names[idx]
            if self.class_names[idx] == "delete":
                show_msg = " - : delete"
            elif self.class_names[idx] == "move":
                show_msg = " + : move"
            elif self.class_names[idx] == "undo":
                show_msg = " Backspace : undo"
            elif self.class_names[idx] == "fix":
                show_msg = " \| : fix"
            cv2.putText(self.win_image, show_msg,
                        (self.width + 5, (idx + 1) * per_class_h),
                        self.font_type, min(font_size * 0.4,
                                            1.0), self.colors[idx], font_size)
        cv2.putText(self.win_image,
                    f"{self.current_label_index}/{self.total_image_number}",
                    (self.width + 5, (idx + 2) * per_class_h), self.font_type,
                    min(font_size * 0.4, 1.0), (0, 0, 0), font_size)
        cv2.putText(self.win_image,
                    f"PgUp",
                    (self.width + 5, (idx + 3) * per_class_h), self.font_type,
                    min(font_size * 0.4, 1.0), (0, 0, 255), font_size)
        cv2.putText(self.win_image,
                    f"PgDown",
                    (self.width + self.class_width//2, (idx + 3) * per_class_h), self.font_type,
                    min(font_size * 0.4, 1.0), (0, 255, 0), font_size)

        self.win_image[:, :self.width, :] = image

    # 从文本读取标注框信息
    def read_label_file(self, label_file_path):
        boxes = []
        fliter_boxes = []
        begin_index = 0
        with open(label_file_path) as label_file:
            for line in label_file:
                if len(line.strip().split()) > 4:
                    begin_index = 1
                x, y, w, h = [float(e) for e in line.strip().split()
                              ][begin_index:begin_index + 4]
                label_id = int(
                    line.strip().split()[0]) if begin_index == 1 else 0
                if len(line.strip().split()) > 5:
                    attrs = [
                        int(e) for e in line.strip().split()[begin_index + 4:]
                    ]
                else:
                    attrs = [0] * len(self.attrs_dict)
                rest_num = len(self.attrs_dict) - len(attrs)
                if rest_num < 0:
                    attrs = attrs[0:len(self.attrs_dict)]
                elif rest_num > 0:
                    attrs.extend([0] * rest_num)
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, 1)
                y2 = min(y2, 1)
                box = [x1, y1, x2, y2, label_id]
                if self.attr_flag:
                    box.extend(attrs)
                if not self.attr_flag or not self.attr_type_idx:
                    if label_id in self.class_table:
                        boxes.append(box)
                    else:
                        fliter_boxes.append(box)
                else:
                    self.operate_flag = True
                    attr_type_name = list(
                        self.attrs_dict.keys())[self.attr_type_idx - 1]
                    # 过滤出要进行属性标注的主分类框
                    if self.src_total_class_names[label_id] == list(
                            self.attrs_dict[attr_type_name].keys())[0]:
                        # 属性部分的分类号由1开始记， 0留给非此类目标使用
                        box[4 + self.attr_type_idx] = max(
                            1, box[4 + self.attr_type_idx])
                        boxes.append(box)
                    else:
                        fliter_boxes.append(box)
        self.boxes = boxes
        self.fliter_boxes = fliter_boxes

    # 将标注框信息保存到文本
    def write_label_file(self, label_file_path, boxes):
        label_file = open(label_file_path, "w")
        attrs = []
        for box in boxes:
            box = self.box_fix(box)
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]
            if len(box) > 4:
                label_id = box[4]
            else:
                label_id = 0
            label_file.writelines("{} {} {} {} {}".format(
                label_id, center_x, center_y, width, height))
            if self.attr_flag:
                if len(box) == 5:
                    box.extend([0] * len(self.attrs_dict))
                attr_type_name = list(
                    self.attrs_dict.keys())[self.attr_type_idx - 1]
                # 过滤出要进行属性标注的主分类框
                if self.src_total_class_names[label_id] == list(
                        self.attrs_dict[attr_type_name].keys())[0]:
                    box[4 + self.attr_type_idx] = max(
                        1, box[4 + self.attr_type_idx])
                attrs = box[5:]
                for attr in attrs:
                    label_file.writelines(" {}".format(attr))
            label_file.writelines("\n")
            # label_file.writelines("{} {} {} {}\n".format(center_x, center_y, width, height))
        label_file.close()

    # 记录当前已标注位置，写到文本
    def write_checkpoint(self, checkpoint_path):
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        checkpoint_file = open(checkpoint_path, "w")
        checkpoint_file.writelines(str(self.current_label_index))

    # 从文本读取当前已标注位置
    def read_checkpoint(self, checkpoint_path):
        checkpoint_file = open(checkpoint_path, "r")
        for line in checkpoint_file.readlines():
            self.current_label_index = int(line.strip())
        checkpoint_file.close()

    # 标注程序运行部分
    def labeled(self):
        self._check()
        self.images_list = sorted(
            glob.glob("{}/*/*.[jp][pn]g".format(self.image_folder)))
        self.images_list = [image_path.replace("_fix", "") for image_path in self.images_list]
        self._compute_total_image_number()
        print("需要标注的图片总数为： ", self.total_image_number)
        if os.path.exists(self.checkpoint_path):
            self.read_checkpoint(self.checkpoint_path)
        while self.current_label_index < self.total_image_number:
            print("已标注的图片数量为： ", self.current_label_index)
            self.current_label_index = min(self.current_label_index,
                                           self.total_image_number - 1)
            self.write_checkpoint(self.checkpoint_path)
            self._reset()
            # print(self.images_list[self.current_label_index])
            self.image = cv2.imdecode(
                np.fromfile(self.images_list[self.current_label_index],
                            dtype=np.uint8), 1)
            h_w_rate = self.image.shape[0] / self.image.shape[1]
            resize_w = min(self.pixel_size, self.image.shape[1])
            self.image = cv2.resize(self.image,
                                    (resize_w, int(resize_w * h_w_rate)))
            self.current_image = self.image.copy()
            self.label_path = self.images_list[
                self.current_label_index].replace("images", "labels").replace(
                    ".jpg", ".txt").replace(".png", ".txt")
            if os.path.exists(self.label_path):
                self.read_label_file(self.label_path)

            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
            self.class_width = self.width // 5
            # cv2.imshow(self.windows_name, self.image)
            cv2.namedWindow(self.windows_name, 0)
            self._draw_box_on_image(self.image.copy(), self.boxes)
            if self.attr_type_idx:  # 开启属性标注模式
                if self.class_names[self.label_index] == "move":
                    cv2.setMouseCallback(self.windows_name, self._move_roi)
                elif self.class_names[self.label_index] == "delete":
                    cv2.setMouseCallback(self.windows_name, self._delete_roi)
                elif self.class_names[self.label_index] == "undo":
                    cv2.setMouseCallback(self.windows_name, self._undo_roi)
                elif self.class_names[self.label_index] == "fix":
                    cv2.setMouseCallback(self.windows_name, self._fix_roi)
                else:
                    cv2.setMouseCallback(self.windows_name, self._attr_map)
            else:
                if self.class_names[self.label_index] == "move":
                    cv2.setMouseCallback(self.windows_name, self._move_roi)
                elif self.class_names[self.label_index] == "delete":
                    cv2.setMouseCallback(self.windows_name, self._delete_roi)
                elif self.class_names[self.label_index] == "undo":
                    cv2.setMouseCallback(self.windows_name, self._undo_roi)
                elif self.class_names[self.label_index] == "fix":
                    cv2.setMouseCallback(self.windows_name, self._fix_roi)
                else:
                    cv2.setMouseCallback(self.windows_name, self._draw_roi)
            # key = cv2.waitKey(self.decay_time)
            key = cv2.waitKeyEx(self.decay_time)  # 开启方向键功能
            # print(key)
            # print(self.boxes, self.fliter_boxes)
            if not self.fliter_flag:
                self.boxes.extend(self.fliter_boxes)
            if self.operate_flag:
                self.write_label_file(self.label_path, self.boxes)
            if 48 < key <= 57:  # 按键 0-9
                self.label_index = min(key - 49, self.class_num - 1)
                continue
            if key == 48:  # 按键 0
                self.label_index = min(9, self.class_num - 1)
                continue
            if key == 8:  # Backspace
                self.label_index = self.class_names.index("undo")
                continue
            if key == 92 or key == 47:  # 按键 \|
                self.label_index = self.class_names.index("fix")
                continue
            if key == 45:  # 按键 _-
                self.label_index = self.class_names.index("delete")
                continue
            if key == 61 or key == 43:  # 按键 =+
                self.label_index = self.class_names.index("move")
                continue
            if key == 32:
                self.auto_play_flag = not self.auto_play_flag
                self.decay_time = self.default_decay_time if self.auto_play_flag else 0
            if key == ord('a') or key == ord('A') or key == 2424832:  # 后退一张
                self._backward()
                self.undo_boxes = []
                continue
            if key == ord('w') or key == ord('W') or key == 2490368:
                self.label_index = max(0, self.label_index - 1)
                continue
            if key == ord('s') or key == ord('S') or key == 2621440:
                self.label_index = min(self.class_num - 1,
                                       self.label_index + 1)
                continue
            if self.attr_flag and key == ord('q') or key == ord(
                    'Q'):  # 向后切换属性表
                self.attr_type_idx = (self.attr_type_idx -
                                      1) % (len(self.attrs_dict) + 1)
                self._check()
                continue
            if self.attr_flag and key == ord('e') or key == ord(
                    'E'):  # 向前切换属性表
                self.attr_type_idx = (self.attr_type_idx +
                                      1) % (len(self.attrs_dict) + 1)
                self._check()
                continue
            if key == ord('l') or key == ord('L'):  # 删除当前图
                os.remove(self.images_list[self.current_label_index])
                os.remove(self.label_path)
                del self.images_list[self.current_label_index]
                self._compute_total_image_number()
                # self._backward()
                continue
            if key == 27:  # 退出
                break
            if key == ord('d') or key == ord('D') or key == 2555904:  # 前进一张
                self.current_label_index += 1
                # 不在 self._reset()中是为了避免其他按键导致撤销失效
                self.undo_boxes = [] 
                continue
            # if key in [0, 16, 17, 20, 65505, 65513]:
            #     continue
            if self.auto_play_flag:
                self.undo_boxes = []
                self.current_label_index += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--cfg',
                        type=str,
                        help='json file path',
                        default=os.path.join(work_root,
                                             "voc_attr.json"))
    args = parser.parse_args()
    json_path = args.cfg
    json_file = open(json_path, 'r')
    cfgs = json.load(json_file)
    json_file.close()
    dataset_path = cfgs["dataset_path"]
    task = CLabeled(dataset_path)
    task.windows_name = cfgs["windows_name"]
    task.src_total_class_names = cfgs["total_class_names"]
    task.total_class_names = cfgs["total_class_names"]
    task.src_class_names = cfgs["class_names"]
    task.class_names = cfgs["class_names"]
    task.colors = cfgs["colors"]
    task.attrs_dict = cfgs["attrs"]
    task.default_decay_time = int(cfgs.get("decay_time", 1000))
    task.pixel_size = int(cfgs.get("pixel_size", 1920))
    task.select_type = int(cfgs.get("select_type", 0))
    task.checkpoint_path = os.path.join(dataset_path, cfgs.get("checkpoint_name", "checkpoint"))
    return task


def main():
    task = parse_args()
    task.labeled()


if __name__ == '__main__':
    main()
