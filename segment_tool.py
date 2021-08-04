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

"""
图像语义分割标注脚本, 生成对应mask标注文件
Author: Heroinlj
按键说明: (字母按键不区分大小写, 输入法需要切换到英文模式)
    Esc: 退出程序
    P: 自动播放与暂停
    -: 切换到撤销模式
    +: 切换到恢复模式
    Space: 切换到擦除模式
    Backspace: 切换到重置掩码模式
    1~9: 切换类别label_id为 0~8(越界时为最大类别号)
    0: 切换类别label_id为9(越界时为最大类别号)
    W/↑: 向前切换类别
    S/↓: 向后切换类别
    A/←: 上一张图片
    D/→: 下一张图片
    L: 删除当前图片和标注文件
    Q: 缩小描绘点像素大小
    E: 放大描绘点像素大小
鼠标事件:
    标注模式:
        鼠标左键拖动进行掩码标注, 按下或按下移动进行涂抹掩码标注
    撤销模式:
        Windows鼠标右键(Linux, Mac左键双击)撤销当前操作 
    恢复模式:
        Windows鼠标右键(Linux, Mac左键双击)恢复上次撤销操作 
    擦除模式:
        鼠标左键拖动进行掩码标注, 按下或按下移动进行涂抹掩码擦除标注
    重置模式
        鼠标右键单击(Linux, Mac左键双击)进行掩码的重置, 可通过鼠标左键拖动标注需要重置的区域, 默认重置全图
    展示模式:
        Windows鼠标右键(Linux, Mac左键双击)进行掩码映射或取消映射, 按住鼠标左键只显示掩码

"""
ix, iy = -1, -1
is_mouse_lb_down = False
reset_box = None


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
        # 目标框的分类索引号
        self.label_index = 0
        # 标注框信息
        self.masks = None
        # 缓存操作，以进行撤销
        self.undo_masks = []
        # 缓存操作，以进行恢复
        self.redo_masks = []
        # 记录撤销操作的最大个数
        self.undo_masks_max_len = 10
        # 过滤不显示的掩码
        self.fliter_masks = []
        # 是否保存过滤的掩码
        self.fliter_flag = False
        # 当前图片
        self.current_image = None
        # 标注框的保存文件地址
        self.mask_path = None
        # 记录历史标注位置的文本文件地址
        self.checkpoint_path = os.path.join(image_folder, "checkpoint")
        self.annotation = None
        # 类别数
        self.class_num = 0
        # 所有类别名称
        self.total_class_names = None
        # 类别列表
        self.class_table = None
        # 标注展示类别名称
        self.class_names = None
        # 类别对应的颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(max(1, self.class_num))]
        # 右侧类别名显示的宽度
        self.class_width = 400
        # 图像宽
        self.width = 720
        # 图像高
        self.height = 576
        # 涂抹像素范围大小
        self.pixel_size = 5
        # 显示窗口的名称
        self.windows_name = "image"
        self.font_type = cv2.FONT_HERSHEY_SIMPLEX
        # 是否有进行操作
        self.operate_flag = False
        # 开启语义分割模式
        self.instance_flag = True
        # 是否将掩码映射到图上
        self.show_mask = True
        self.auto_play_flag = False
        self.decay_time = 1 if self.auto_play_flag else 0
        self._may_make_dir()

    # 重置
    def _reset(self):
        self.image = None
        self.current_image = None
        self.label_path = None
        self.masks = None
        self.operate_flag = False

    # 参数检查，确保代码可运行
    def _check(self):
        if self.class_num < 1:
            self.class_num = 1
        if self.total_class_names is None:
            self.total_class_names = range(self.class_num)
        if isinstance(self.total_class_names, list):
            self.total_class_names.extend(["undo", "redo", "eraser", "reset", "show"])
        if self.class_names is None:
            self.class_names = self.total_class_names
        else:
            self.class_names.extend(["undo", "redo", "eraser", "reset", "show"])
        self.class_num = len(self.class_names)
        self.class_table = [self.total_class_names.index(name) for name in self.class_names]
        # 判断当前颜色列表是否够用, 不够的话进行随机添加
        if isinstance(self.colors, list) and len(self.colors) < self.class_num:
            self.colors.extend([[random.randint(0, 255) for _ in range(3)] 
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

    # 鼠标左键点击切换id事件
    def _event_lbuttondown(self, x, y, dst):
        if self.width < x <= (self.width + self.class_width) and 0 <= y <= self.height:
            per_class_h = int(min(self.height, 600) / (self.class_num+2))
            self.label_index = min(max(int((y - 5) / per_class_h), 0), self.class_num-1)
        x, y = self._roi_limit(x, y)
        self._apply_mask_on_image(dst, self.masks)
        cv2.imshow(self.windows_name, self.win_image)
        if self.class_names[self.label_index] == "redo":
            cv2.setMouseCallback(self.windows_name, self._redo_roi)
        elif self.class_names[self.label_index] == "eraser":
            cv2.setMouseCallback(self.windows_name, self._eraser_roi)
        elif self.class_names[self.label_index] == "undo":
            cv2.setMouseCallback(self.windows_name, self._undo_roi)
        elif self.class_names[self.label_index] == "reset":
            cv2.setMouseCallback(self.windows_name, self._reset_roi)
        elif self.class_names[self.label_index] == "show":
            cv2.setMouseCallback(self.windows_name, self._show_roi)
        else:
            cv2.setMouseCallback(self.windows_name, self._draw_roi)
                
    # 标注感兴趣区域
    def _draw_roi(self, event, x, y, flags, param):
        global ix, iy, move_ix, move_iy, is_mouse_lb_down
        box_border = round(self.width / 400)
        dst = self.image.copy()
        if self.masks is None:
            self.masks = np.zeros_like(self.image, np.uint8)
        self._apply_mask_on_image(dst, self.masks)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                is_mouse_lb_down = True
                if len(self.undo_masks) > self.undo_masks_max_len:
                    del self.undo_masks[0]
                self.undo_masks.append(self.masks.copy())
                cv2.circle(self.masks, (x, y), self.pixel_size, self.colors[self.label_index], -1, cv2.LINE_AA)
        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE:
            x, y = self._roi_limit(x, y)
            cv2.line(dst, (x, 0), (x, self.height), self.colors[self.label_index], 1, 4)
            cv2.line(dst, (0, y), (self.width, y), self.colors[self.label_index], 1, 4)
            cv2.circle(dst, (x, y), self.pixel_size, self.colors[self.label_index], -1, cv2.LINE_AA)
            if is_mouse_lb_down:
                cv2.circle(self.current_image, (x, y), self.pixel_size, self.colors[self.label_index], -1, cv2.LINE_AA)
                cv2.circle(self.masks, (x, y), self.pixel_size, self.colors[self.label_index], -1, cv2.LINE_AA)
                # cv2.line(self.masks, (ix, iy), (x, y), color=self.colors[self.label_index], thickness=self.pixel_size, lineType=cv2.LINE_AA)
                ix, iy = self._roi_limit(x, y)
                self._apply_mask_on_image(self.current_image, self.masks)
            else:
                self._apply_mask_on_image(dst, self.masks)
            cv2.imshow(self.windows_name, self.win_image)
        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            x, y = self._roi_limit(x, y)
            is_mouse_lb_down = False
            self.operate_flag = True
            self.current_image = self.image.copy()

    # 擦除掩码操作
    def _eraser_roi(self, event, x, y, flags, param):
        global ix, iy, is_mouse_lb_down
        dst = self.image.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                is_mouse_lb_down = True
                if len(self.undo_masks) > self.undo_masks_max_len:
                    del self.undo_masks[0]
                self.undo_masks.append(self.masks.copy())
                cv2.circle(self.masks, (x, y), self.pixel_size, [0, 0, 0], -1, cv2.LINE_AA)
        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE:
            x, y = self._roi_limit(x, y)
            cv2.line(dst, (x, 0), (x, self.height), self.colors[self.label_index], 1, 8)
            cv2.line(dst, (0, y), (self.width, y), self.colors[self.label_index], 1, 8)
            cv2.circle(dst, (x, y), self.pixel_size, self.colors[self.label_index], -1, cv2.LINE_AA)
            if is_mouse_lb_down:
                cv2.circle(self.current_image, (x, y), self.pixel_size, self.colors[self.label_index], -1, cv2.LINE_AA)
                cv2.circle(self.masks, (x, y), self.pixel_size, [0, 0, 0], -1, cv2.LINE_AA)
                # cv2.line(self.masks, (ix, iy), (x, y), color=[0, 0, 0], thickness=self.pixel_size)
                ix, iy = self._roi_limit(x, y)
                self._apply_mask_on_image(self.current_image, self.masks)
            else:
                self._apply_mask_on_image(dst, self.masks)
            cv2.imshow(self.windows_name, self.win_image)
        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            x, y = self._roi_limit(x, y)
            is_mouse_lb_down = False
            self.operate_flag = True
            self.current_image = self.image.copy()

    # 撤销上一次操作的mask
    def _undo_roi(self, event, x, y, flags, param):
        dst = self.image.copy()
        self._apply_mask_on_image(dst, self.masks)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            self._event_lbuttondown(x, y, dst)
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (sys.platform in ["linux", "darwin"] and event == cv2.EVENT_LBUTTONDBLCLK):  # 撤销删除(中心点或左上点)距离当前鼠标最近的框
            x, y = self._roi_limit(x, y)
            self.current_image = self.image.copy()
            if len(self.undo_masks):
                self.redo_masks.append(self.masks)
                self.masks = self.undo_masks[-1]
                del self.undo_masks[-1]
                self.operate_flag = True
            self._apply_mask_on_image(self.current_image, self.masks)
    
    # 重置mask， 支持画框只重置框内区域
    def _reset_roi(self, event, x, y, flags, param):
        global ix, iy, reset_box
        box_border = round(self.width / 400)
        dst = self.image.copy()
        self._apply_mask_on_image(dst, self.masks)
        if reset_box is not None:
            self._draw_box_on_image(dst, reset_box)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE and not (flags and cv2.EVENT_FLAG_LBUTTON):
            x, y = self._roi_limit(x, y)
            cv2.line(dst, (x, 0), (x, self.height), self.colors[self.label_index], 1, 8)
            cv2.line(dst, (0, y), (self.width, y), self.colors[self.label_index], 1, 8)
            self._apply_mask_on_image(dst, self.masks)
            cv2.imshow(self.windows_name, self.win_image)
        # 按住鼠标左键进行移动
        elif event == cv2.EVENT_MOUSEMOVE and (flags and cv2.EVENT_FLAG_LBUTTON):
            x, y = self._roi_limit(x, y)
            cv2.rectangle(dst, (ix, iy), (x, y), self.colors[self.label_index], 1)
            self._apply_mask_on_image(dst, self.masks)
            cv2.imshow(self.windows_name, self.win_image)
        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            x, y = self._roi_limit(x, y)
            if abs(x - ix) > 10 and abs(y - iy) > 10:
                cv2.rectangle(dst, (ix, iy),
                                (x, y), self.colors[self.label_index], box_border)
                label_id = self.class_table[self.label_index]
                reset_box = [ix/self.width, iy/self.height, x/self.width, y/self.height, label_id]
            # print(self.boxes)
            self._draw_box_on_image(dst, reset_box)
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (sys.platform in ["linux", "darwin"] and event == cv2.EVENT_LBUTTONDBLCLK):  # 撤销删除(中心点或左上点)距离当前鼠标最近的框
            x, y = self._roi_limit(x, y)
            self.current_image = self.image.copy()
            self.undo_masks.append(self.masks.copy())
            self.redo_masks.append(self.masks.copy())
            if self.masks is not None and reset_box is not None:
                h, w = self.masks.shape[0:2]
                self.masks[int(reset_box[1]*h): int(reset_box[3]*h), int(reset_box[0]*w): int(reset_box[2]*w)] *= 0
            else:
                self.masks = np.zeros_like(self.image, np.uint8)
            self._apply_mask_on_image(self.current_image, self.masks)
            reset_box = None
            self.operate_flag = True

    # 恢复上一次操作的mask
    def _redo_roi(self, event, x, y, flags, param):
        dst = self.image.copy()
        self._apply_mask_on_image(dst, self.masks)
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            self._event_lbuttondown(x, y, dst)
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (sys.platform in ["linux", "darwin"] and event == cv2.EVENT_LBUTTONDBLCLK):  # 撤销删除(中心点或左上点)距离当前鼠标最近的框
            x, y = self._roi_limit(x, y)
            self.current_image = self.image.copy()
            if len(self.redo_masks):
                self.undo_masks.append(self.masks)
                self.masks = self.redo_masks[-1]
                del self.redo_masks[-1]
                self.operate_flag = True
            self._apply_mask_on_image(self.current_image, self.masks)

    # 显示或关闭mask
    def _show_roi(self, event, x, y, flags, param):
        dst = self.image.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键
            ix, iy = self._roi_limit(x, y)
            self._event_lbuttondown(x, y, dst)
            if x == ix and y == iy:
                self._update_win_image(self.masks)
                cv2.imshow(self.windows_name, self.win_image)
        elif ("win32" in sys.platform and event == cv2.EVENT_RBUTTONDOWN) or (sys.platform in ["linux", "darwin"] and event == cv2.EVENT_LBUTTONDBLCLK):  # 撤销删除(中心点或左上点)距离当前鼠标最近的框
            x, y = self._roi_limit(x, y)
            self.show_mask = not self.show_mask
            self.current_image = self.image.copy()
            if self.show_mask:
                self._apply_mask_on_image(self.current_image, self.masks)
            else:
                self._update_win_image(self.current_image)
                cv2.imshow(self.windows_name, self.win_image)
        elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            self.current_image = self.image.copy()
            if self.show_mask:
                self._apply_mask_on_image(self.current_image, self.masks)
            else:
                self._update_win_image(self.current_image)
                cv2.imshow(self.windows_name, self.win_image)

    # 将标注框显示到图像上
    def _apply_mask_on_image(self, image, mask):
        if mask is not None:
            if mask.ndim != 3:
                mask = np.repeat(np.asarray(mask)[:, :, None], 3, axis=2)
            # print(image.shape)
            # print(mask.shape)
            image = cv2.addWeighted(image, 1.0, mask, 0.5, 1)
        self._update_win_image(image)
        cv2.imshow(self.windows_name, self.win_image)

    def _draw_box_on_image(self, image, box):
        box_border = round(self.width / 400)
        font_size = max(1, int(min(self.width, self.height) / 600))
        if box is not None:
            pt1 = (int(image.shape[1] * box[0]), int(image.shape[0] * box[1]))
            pt2 = (int(image.shape[1] * box[2]), int(image.shape[0] * box[3]))
            label_id = box[4] 
            label_index = self.class_table.index(label_id)
            cv2.rectangle(image, pt1, pt2, self.colors[label_index], box_border)
            cv2.putText(image, self.class_names[label_index],
                        pt1, self.font_type, min(font_size*0.4, 1.0), self.colors[label_index], font_size)
        self._update_win_image(image)
        cv2.imshow(self.windows_name, self.win_image)

    # 更新整个窗口的显示
    def _update_win_image(self, image):
        per_class_h = int(min(self.height, 600) / (self.class_num + 2))
        font_size = max(1, int(min(self.width, self.height) / 600))
        self.win_image = np.zeros(
            [self.height, self.class_width + self.width, 3], dtype=np.uint8)
        self.win_image[:, self.width:self.width+self.class_width, :] = 255
        self.win_image[(self.label_index+1)*per_class_h - min(per_class_h, 10):5+(self.label_index+1)*per_class_h, self.width:self.width+self.class_width] = [255, 245, 152]
        for idx in range(self.class_num):
            show_msg = str(idx + 1) + ": " + self.class_names[idx]
            if self.class_names[idx] == "eraser":
                show_msg = " Space: eraser"
            elif self.class_names[idx] == "redo":
                show_msg = " + : redo"
            elif self.class_names[idx] == "undo":
                show_msg = " - : undo"
            elif self.class_names[idx] == "reset":
                show_msg = " Backspace : reset"
            elif self.class_names[idx] == "show":
                show_msg = " Enter : show"
            cv2.putText(self.win_image, show_msg,
                        (self.width+5, (idx+1)*per_class_h), self.font_type, min(font_size*0.4, 1.0), self.colors[idx], font_size)
        cv2.putText(self.win_image, f"{self.current_label_index}/{self.total_image_number}",
                        (self.width+5, (idx+2)*per_class_h), self.font_type, min(font_size*0.4, 1.0), (0, 0, 0), font_size)
        
        self.win_image[:, :self.width, :] = image

    # 从文本读取标注框信息
    def read_mask_file(self, mask_file_path):
        masks = cv2.imread(mask_file_path)
        if not self.instance_flag:
            lower_white1 = np.array([1,0,0])
            lower_white2 = np.array([0,1,0])
            lower_white3 = np.array([0,0,1])
            upper_white = np.array([255,255,255])
            mask1 = cv2.inRange(masks, lower_white1, upper_white)
            mask2 = cv2.inRange(masks, lower_white2, upper_white)
            mask3 = cv2.inRange(masks, lower_white3, upper_white)
            masks = cv2.bitwise_or(mask1, mask2, mask3)
        if masks.ndim != 3:
                masks = np.repeat(np.asarray(masks)[:, :, None], 3, axis=2)
        self.masks = masks
        
    # 将标注框信息保存到文本
    def save_mask_file(self, mask_file_path, masks):
        masks = masks.astype(np.uint8)
        if not self.instance_flag:
            lower_white1 = np.array([1,0,0])
            lower_white2 = np.array([0,1,0])
            lower_white3 = np.array([0,0,1])
            upper_white = np.array([255,255,255])
            mask1 = cv2.inRange(masks, lower_white1, upper_white)
            mask2 = cv2.inRange(masks, lower_white2, upper_white)
            mask3 = cv2.inRange(masks, lower_white3, upper_white)
            masks = cv2.bitwise_or(mask1, mask2, mask3)
        cv2.imwrite(mask_file_path, masks)

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
            glob.glob("{}/*/*.jpg".format(self.image_folder)))
        self._compute_total_image_number()
        print("需要标注的图片总数为： ", self.total_image_number)
        if os.path.exists(self.checkpoint_path):
            self.read_checkpoint(self.checkpoint_path)
        while self.current_label_index < self.total_image_number:
            print("已标注的图片数量为： ", self.current_label_index)
            self.current_label_index = min(
                self.current_label_index, self.total_image_number - 1)
            self.write_checkpoint(self.checkpoint_path)
            self._reset()
            # print(self.images_list[self.current_label_index])
            self.image = cv2.imdecode(np.fromfile(
                self.images_list[self.current_label_index], dtype=np.uint8), 1)
            h_w_rate = self.image.shape[0]/self.image.shape[1]
            resize_w = min(1920, self.image.shape[1])
            self.image = cv2.resize(self.image, (resize_w, int(resize_w*h_w_rate)))
            self.current_image = self.image.copy()
            self.mask_path = self.images_list[self.current_label_index].replace(
                "images", "masks") + ".png"
            if os.path.exists(self.mask_path):
                self.read_mask_file(self.mask_path)

            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
            self.class_width = self.width // 5
            # cv2.imshow(self.windows_name, self.image)
            cv2.namedWindow(self.windows_name, 0)
            self._apply_mask_on_image(self.image.copy(), self.masks)
            if self.class_names[self.label_index] == "redo":
                cv2.setMouseCallback(self.windows_name, self._redo_roi)
            elif self.class_names[self.label_index] == "eraser":
                cv2.setMouseCallback(self.windows_name, self._eraser_roi)
            elif self.class_names[self.label_index] == "undo":
                cv2.setMouseCallback(self.windows_name, self._undo_roi)
            elif self.class_names[self.label_index] == "reset":
                cv2.setMouseCallback(self.windows_name, self._reset_roi)
            elif self.class_names[self.label_index] == "show":
                cv2.setMouseCallback(self.windows_name, self._show_roi)
            else:
                cv2.setMouseCallback(self.windows_name, self._draw_roi)
            # key = cv2.waitKey(self.decay_time)
            key = cv2.waitKeyEx(self.decay_time) # 开启方向键功能
            if self.operate_flag:
                self.save_mask_file(self.mask_path, self.masks)
            if 48 < key <= 57:  # 按键 0-9
                self.label_index = min(key - 49, self.class_num - 1)
                continue
            if key == 48:  # 按键 0
                self.label_index = min(9, self.class_num - 1)
                continue
            if key == 8:  # Backspace
                self.label_index = self.class_names.index("reset")
                continue
            if key == 45:  # 按键 _-
                self.label_index = self.class_names.index("undo")
                continue
            if key == 61 or key == 43:  # 按键 =+
                self.label_index = self.class_names.index("redo")
                continue
            if key == ord('p') or key == ord('P'):
                self.auto_play_flag = not self.auto_play_flag
                self.decay_time = 300 if self.auto_play_flag else 0
            if key == 32:
                self.label_index = self.class_names.index("eraser")
                continue
            if key == 13 :  # 按键 Enter
                self.label_index = self.class_names.index("show")
                continue
            if key == ord('a') or key == ord('A') or key == 2424832:  # 后退一张
                self._backward()
                continue
            if key == ord('w') or key == ord('W') or key == 2490368:
                self.label_index = max(0, self.label_index-1)
                continue
            if key == ord('s') or key == ord('S') or key == 2621440:
                self.label_index = min(self.class_num-1, self.label_index+1)
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
            if key == ord('e') or key == ord('E'):
                self.pixel_size += 2
                self.pixel_size = min(self.pixel_size, 50)
            if key == ord('q') or key == ord('Q'):
                self.pixel_size -= 2
                self.pixel_size = max(self.pixel_size, 1)
            if key == ord('d') or key == ord('D') or key == 2555904:  # 前进一张
                self.current_label_index += 1
                self.undo_masks = []
                self.redo_masks = []
                continue
            # if key in [0, 16, 17, 20, 65505, 65513]:
            #     continue
            if self.auto_play_flag:
                self.current_label_index += 1
                self.undo_masks = []
                self.redo_masks = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--cfg',
                        type=str,
                        help='json file path',
                        default=os.path.join(work_root,
                                             "voc.json"))
    args = parser.parse_args()
    json_path = args.cfg
    json_file = open(json_path, 'r')
    cfgs = json.load(json_file)
    json_file.close()
    dataset_path = cfgs["dataset_path"]
    task = CLabeled(dataset_path)
    task.windows_name = cfgs["windows_name"]
    task.total_class_names = cfgs["total_class_names"]
    task.class_names = cfgs["class_names"]
    task.colors = cfgs["colors"]
    return task


def main():
    task = parse_args()
    task.labeled()


if __name__ == '__main__':
    main()
