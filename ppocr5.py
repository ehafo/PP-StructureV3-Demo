import os
import cv2
import numpy as np
import onnxruntime
import sys
import gc
import math
import traceback
import copy
import locale

class simple_ppocr5():
    def __init__(
        self,ppocr5_onnx_det=r'models/ppocr5_m_det.onnx',ppocr5_onnx_cls=r'models/ppocr5_m_cls.onnx',ppocr5_onnx_rec=r'models/ppocr5_m_rec.onnx',ppcor5_dict=r'models/ppocr5_dict.txt',use_gpu=False):
        self.modal_ready=False
        self.dev_mode=False
        #parameters
        self.limit_side_len=1920
        # 归一化参数
        # ImageNet数据集的均值和标准差来归一化图像，即使用mean = [0.485, 0.456, 0.406]
        # 和std = [0.229, 0.224, 0.225]
        self.scale = np.float32(1. / 255.)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        #检测框体处理参数
        self.det_db_thresh = 0.3
        self.use_dilation = False
        self.det_db_box_thresh = 0.5
        self.det_db_unclip_ratio = 1.8
        self.max_batch_size = 10
        self.use_angle_cls = True
        self.cls_batch_num = 6
        self.cls_thresh = 0.9
        self.label_list = ["0", "180"]
        self.rec_image_shape = [3, 48, 320]
        self.max_text_length = 25
        self.rec_batch_num = 6
        self.dt_boxes=[]
        self.lang=self.checklanguage()
        
        try:
            # 使用gpu
            if use_gpu:
                onnx_providers = ['CUDAExecutionProvider']
            else:
                onnx_providers = ['CPUExecutionProvider']
            options = onnxruntime.SessionOptions()
            options.log_severity_level = 3 # 3 = ERROR, 2 = WARNING, 1 = INFO, 0 = VERBOSE Do not print other level info
            self.det_net = onnxruntime.InferenceSession(ppocr5_onnx_det,sess_options=options,providers=onnx_providers)
            self.rec_net = onnxruntime.InferenceSession(ppocr5_onnx_rec, sess_options=options,providers=onnx_providers)
            self.cls_net = onnxruntime.InferenceSession(ppocr5_onnx_cls,sess_options=options,providers=onnx_providers)
            self.logger(("Model loaded", "模型已经装载")[self.lang])
            self.det_net_input_name=self.det_net.get_inputs()[0].name

            self.character_str = []
            with open(ppcor5_dict, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
                self.character_str.append(" ")  # use_space_char
            self.dict_character = list(self.character_str)
            self.dict_character = ["blank"] + self.dict_character
            self.character = self.dict_character
            self.logger(("Initialization complete", "初始化完成")[self.lang])
        except Exception as e:
            print(("=== Detailed Exception ===", "=== 详细异常信息 ===")[self.lang])
            print(f"{('Exception Type', '异常类型')[self.lang]}: {type(e).__name__}")
            print(f"{('Exception Message', '异常信息')[self.lang]}: {str(e)}")
            print(f"\n{('=== Stack Trace ===', '=== 堆栈跟踪 ===')[self.lang]}")
            traceback.print_exc()
           # 或者获取堆栈信息作为字符串
            print(f"\n{('=== Stack Info String ===', '=== 堆栈信息字符串 ===')[self.lang]}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback_str = traceback.format_exc()
            print(traceback_str)

    def checklanguage(self):
        try:
            lang, _ = locale.getlocale()
            
            if lang is None:
                locale.setlocale(locale.LC_ALL, '')
                lang, _ = locale.getlocale()
                
            if lang:
                lang = lang.lower()
                #  Windows: "chinese (simplified)_china" 
                #  Linux: "zh_cn" or "zh_sg"
                if 'chinese' in lang or 'zh' in lang:
                    return 1
        except:
            return 0
        return 0
    def unload_model(self):
        """卸载模型并释放资源"""
        if self.det_net is not None:
            del self.det_net
            self.det_net = None
        if self.rec_net is not None:
            del self.rec_net
            self.rec_net = None
        if self.cls_net is not None:
            del self.cls_net
            self.cls_net = None
        self.logger(("Model unloaded", "模型已经卸载")[self.lang])
        # 强制垃圾回收
        if gc is not None and hasattr(gc, 'collect'):
            gc.collect()

    #box处理函数
    @staticmethod
    def get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1],
            points[index_2],
            points[index_3],
            points[index_4],
        ]
        return box, min(bounding_box[1])
    @staticmethod
    def box_score_fast(bitmap, _box):
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin: ymax + 1, xmin: xmax + 1], mask)[0]

    @staticmethod
    def unclip(box, unclip_ratio):
        # 1. 确保输入为 numpy 数组并转换为 float32 格式（OpenCV 要求）
        box = np.array(box).astype(np.float32)
        
        # 2. 计算面积和周长 (OpenCV 原生函数)
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, True)
        
        # 3. 计算偏移距离
        distance = area * unclip_ratio / (length + 1e-6)
        
        # 4. 几何外扩逻辑 (顶点法向量平移)
        # 计算边向量
        edges = np.roll(box, -1, axis=0) - box
        edge_lengths = np.sqrt(np.sum(edges**2, axis=1))
        
        # 计算法向量 (dy, -dx)
        norm_edges = edges / (edge_lengths[:, None] + 1e-6)
        v_side = np.column_stack((norm_edges[:, 1], -norm_edges[:, 0]))
        
        # 计算顶点平移向量并补偿夹角 (Miter Limit)
        v_vertex = v_side + np.roll(v_side, 1, axis=0)
        cos_theta = np.sum(v_side * np.roll(v_side, 1, axis=0), axis=1)
        scale = np.sqrt(2 / (1 + cos_theta + 1e-6))
        
        # 限制最大缩放倍数，防止极端尖角无限延伸
        scale = np.clip(scale, 0, 2) 
        
        expanded_box = box + v_vertex * (distance * scale[:, None])
        
        return expanded_box.astype(np.int32) # 通常返回整数像素坐标

    def boxes_from_bitmap(self,pred, _bitmap, dest_width, dest_height):
        """
        _bitmap: single map with shape (1, H, W),
                 whose values are binarized as {0, 1}
        """
        max_candidates = 1000
        min_size = 3
        score_mode = "fast"
        box_thresh = self.det_db_box_thresh
        unclip_ratio = self.det_db_unclip_ratio

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if len(outs) == 3:
            _, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))

            if self.det_db_box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height
            )
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores
    @staticmethod
    def order_points_clockwise(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect
    @staticmethod
    def clip_det_res(points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self,dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def sorted_boxes(self,dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2], sort_indices(list): mapping from sorted index to original index
        """
        num_boxes = dt_boxes.shape[0]
        indexed_boxes = [(dt_boxes[i], i) for i in range(num_boxes)]
        sorted_indexed_boxes = sorted(
            indexed_boxes, key=lambda x: (x[0][0][1], x[0][0][0])
        )

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(
                        sorted_indexed_boxes[j + 1][0][0][1]
                        - sorted_indexed_boxes[j][0][0][1]
                ) < 10 and (
                        sorted_indexed_boxes[j + 1][0][0][0]
                        < sorted_indexed_boxes[j][0][0][0]
                ):
                    tmp = sorted_indexed_boxes[j]
                    sorted_indexed_boxes[j] = sorted_indexed_boxes[j + 1]
                    sorted_indexed_boxes[j + 1] = tmp
                else:
                    break

        sorted_boxes_list = [item[0] for item in sorted_indexed_boxes]
        sort_indices = [item[1] for item in sorted_indexed_boxes]

        return sorted_boxes_list, sort_indices

    def get_rotate_crop_image(self,img, points):
        """
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        """
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2]),
            )
        )
        pts_std = np.float32(
            [
                [0, 0],
                [img_crop_width, 0],
                [img_crop_width, img_crop_height],
                [0, img_crop_height],
            ]
        )
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def resize_norm_img_cls(self,img):
        cls_image_shape = [3, 48, 192]
        imgC, imgH, imgW = cls_image_shape
        h = img.shape[0]
        w = img.shape[1]

        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        if cls_image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_rec(self,img, max_wh_ratio):
        rec_image_shape = [3, 48, 320]
        rec_algorithm = "SVTR_LCNet"
        imgC, imgH, imgW = rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        w = self.rec_net.get_inputs()[0].shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def get_bounding_box(self,points):
        """
        输入：四边形的四个顶点（numpy array, shape=(4, 2)）
        输出：水平外接矩形的左上角和右下角坐标 (x_min, y_min), (x_max, y_max)
        """
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        return np.array([[x_min, y_min], [x_max, y_min],[x_max, y_max],[x_min, y_max]])

    def logger(self,info):
        if self.dev_mode==True:print(info)
    def __del__(self):
        """析构函数，自动清理资源"""
        self.unload_model()
    def run(self, img, det=True, rec=True, cls=True):
        try:
            self.dt_boxes=[]
            if isinstance(img, str) and os.path.isfile(img):
                self.img=cv2.imread(img)
            elif isinstance(img, np.ndarray):
                self.img=img
            elif isinstance(img, (bytes, bytearray)):
                nparr = np.frombuffer(img, np.uint8)
                # 使用cv2读取图像
                self.img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                raise(ValueError("无法解码图像数据"))
            #开始调节原图像大小
            h, w, c = self.img.shape
            if max(h, w) > self.limit_side_len:
                if h > w:
                    ratio = float(self.limit_side_len) / h
                else:
                    ratio = float(self.limit_side_len) / w
            else:
                ratio = 1.0
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)
            resize_h = max(int(round(resize_h / 32) * 32), 32)
            resize_w = max(int(round(resize_w / 32) * 32), 32)
            ratio_h = resize_h / float(h)
            ratio_w = resize_w / float(w)
            img_resize = cv2.resize(self.img, (int(resize_w), int(resize_h)))
            shape_list = np.expand_dims(np.array([h, w, ratio_h, ratio_w]), axis=0)
            #完成调节图像大小
            # 归一化图像（通过减去图像的均值并除以标准差，可以消除图像中的公共部分，凸显个体之间的差异和特征）
            shape = (1, 1, 3)  # hwc排布
            para_mean = np.array(self.mean).reshape(shape).astype("float32")
            para_std = np.array(self.std).reshape(shape).astype("float32")
            img_det = (
                              img_resize.astype("float32") * self.scale - para_mean
                      ) / para_std

            # 图片转化CHW模式
            img_chw = img_det.transpose((2, 0, 1))
            img_ret = np.expand_dims(img_chw, axis=0)
            outputs=self.det_net.run(None, {self.det_net_input_name: img_ret})
            #preds = {}
            #preds["maps"] = outputs[0]
            pred = outputs[0][:, 0, :, :]
            segmentation = pred > self.det_db_thresh
            boxes_batch = []
            for batch_index in range(pred.shape[0]):
                src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
                mask = segmentation[batch_index]
                boxes, scores = self.boxes_from_bitmap(
                    pred[batch_index], mask, src_w, src_h
                )
                boxes_batch.append({"points": boxes})
            dt_boxes = boxes_batch[0]["points"]
            dt_boxes = self.filter_tag_det_res(dt_boxes, self.img.shape)
            img_crop_list = []
            dt_boxes, sort_indices = self.sorted_boxes(dt_boxes)
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = self.get_rotate_crop_image(self.img, tmp_box)
                img_crop_list.append(img_crop)
            cls_res = None
            mg_crop_list, angle_list = None, None
            img_list = copy.deepcopy(img_crop_list)
            if self.use_angle_cls:
                #img_list = copy.deepcopy(img_crop_list)
                img_num = len(img_list)
                     # Calculate the aspect ratio of all text bars
                width_list = []
                for img_cls in img_list:
                    width_list.append(img_cls.shape[1] / float(img_cls.shape[0]))
                # Sorting can speed up the cls process
                indices = np.argsort(np.array(width_list))
                cls_res = [["", 0.0]] * img_num
                batch_num = self.cls_batch_num
                elapse = 0
                for beg_img_no in range(0, img_num, batch_num):
                    end_img_no = min(img_num, beg_img_no + batch_num)
                    norm_img_batch = []
                    max_wh_ratio = 0
                    for ino in range(beg_img_no, end_img_no):
                        h, w = img_list[indices[ino]].shape[0:2]
                        wh_ratio = w * 1.0 / h
                        max_wh_ratio = max(max_wh_ratio, wh_ratio)
                    for ino in range(beg_img_no, end_img_no):
                        norm_img = self.resize_norm_img_cls(img_list[indices[ino]])
                        norm_img = norm_img[np.newaxis, :]
                        norm_img_batch.append(norm_img)
                    norm_img_batch = np.concatenate(norm_img_batch)
                    norm_img_batch = norm_img_batch.copy()
                    input_dict = {}
                    outputs = self.cls_net.run(None, {self.cls_net.get_inputs()[0].name: norm_img_batch})
                    prob_out = outputs[0]
                    label_list = self.label_list
                    if label_list is None:
                        label_list = {idx: idx for idx in range(preds.shape[-1])}
                    preds = prob_out
                    pred_idxs = preds.argmax(axis=1)
                    decode_out = [
                        (label_list[idx], preds[i, idx]) for i, idx in enumerate(pred_idxs)
                    ]
                    cls_result = decode_out
                    for rno in range(len(cls_result)):
                        label, score = cls_result[rno]
                        cls_res[indices[beg_img_no + rno]] = [label, score]
                        if "180" in label and score > self.cls_thresh:
                            img_list[indices[beg_img_no + rno]] = cv2.rotate(
                                img_list[indices[beg_img_no + rno]], 1
                            )
            img_num = len(img_list)
            width_list = []
            for img_rec in img_list:
                width_list.append(img_rec.shape[1] / float(img_rec.shape[0]))
                # Sorting can speed up the recognition process
            indices = np.argsort(np.array(width_list))
            rec_res = [["", 0.0]] * img_num
            batch_num = self.rec_batch_num
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                norm_img_batch = []
                imgC, imgH, imgW = self.rec_image_shape[:3]
                max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
                for ino in range(beg_img_no, end_img_no):
                    h, w = img_list[indices[ino]].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                    # print("max_wh_ratio", h,w,wh_ratio, max_wh_ratio)
                for ino in range(beg_img_no, end_img_no):
                    # print('img_list[indices[ino]], max_wh_ratio',img_list[indices[ino]].shape, max_wh_ratio)
                    norm_img = self.resize_norm_img_rec(
                        img_list[indices[ino]], max_wh_ratio
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                norm_img_batch = np.concatenate(norm_img_batch)
                norm_img_batch = norm_img_batch.copy()

                outputs = self.rec_net.run(
                    None, {self.rec_net.get_inputs()[0].name: norm_img_batch}
                )
                preds_rec = outputs[0]
                if isinstance(preds_rec, tuple) or isinstance(preds_rec, list):
                    preds_rec = preds_rec[-1]
                # np.save("preds_rec.npy", preds_rec) #此处验证相同
                preds_idx = preds_rec.argmax(axis=2)
                preds_prob = preds_rec.max(axis=2)

                ignored_tokens = [0]
                text_index = preds_idx
                text_prob = preds_prob

                result_list = []
                ignored_tokens = [0]
                batch_size = len(text_index)
                # print("batch_size",len(text_index))
                for batch_idx in range(batch_size):
                    selection = np.ones(len(text_index[batch_idx]), dtype=bool)
                    for ignored_token in ignored_tokens:
                        selection &= text_index[batch_idx] != ignored_token

                    char_list = [
                        self.character[text_id]
                        for text_id in text_index[batch_idx][selection]
                    ]
                    if text_prob is not None:
                        conf_list = text_prob[batch_idx][selection]
                    else:
                        conf_list = [1] * len(selection)
                    if len(conf_list) == 0:
                        conf_list = [0]

                    text = "".join(char_list)

                    # for arabic rec should Do text = pred_reverse(text)

                    result_list.append((text, np.mean(conf_list).tolist()))
                rec_result = result_list
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            self.results = []
            for i, (text, score) in enumerate(rec_res):
                if score > 0.6:
                    self.results.append({
                        'text': text,
                        'rec_pos': self.get_bounding_box(dt_boxes[i])
                    })
            boxes = [item['rec_pos'] for item in self.results]
            self.dt_boxes=boxes
        except Exception as e:
            traceback.print_exc()
            return None

    def displaybox(self,winname):
        try:
            box_img= self.img.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for i,box in enumerate(self.dt_boxes):
                color=colors[i % len(colors)]
                points_array = np.array(box, dtype=np.int32)
                cv2.polylines(box_img, [points_array], isClosed=True, color=color, thickness=2)
            cv2.imshow(winname,box_img)
        except Exception as e:
            traceback.print_exc()
            return None


if __name__ == "__main__":
    print("This is simple_ppocr5 src file")
