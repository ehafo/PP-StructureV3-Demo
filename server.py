"""PP-StructureV3 版面结构化演示
- 版面检测：PP-DocLayoutV3 (HuggingFace transformers + PyTorch)
- 文本提取：PyMuPDF（文本型 PDF）/ PP-OCRv5 ONNX（扫描件，全页 OCR + 按区域归类）
- 表格提取：PyMuPDF（文本型 PDF）/ PP-TableMagic V2 完整产线（扫描件）
  · 表格分类（有线/无线自动区分）
  · 单元格检测 (RT-DETR-L)
  · 结构识别 (SLANeXt)
  · 文字检测+识别 (PP-OCRv4)
  · 降级方案：RapidTable (SLANet_plus ONNX)"""

import re
import time
import uuid
import shutil
import logging
import threading
from pathlib import Path

import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image
from transformers import RTDetrImageProcessor, AutoModelForObjectDetection
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pp-struct")

app = FastAPI(title="PP-StructureV3 Demo")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── 版面检测模型 ──────────────────────────────────────────────────
_MODEL_NAME = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_MODEL_DIR = Path("model")
_THRESHOLD = 0.35
_processor = None
_model = None
_model_status = "loading"  # loading | ready | error
_model_error = ""


def _load_model():
    global _processor, _model, _model_status, _model_error
    try:
        t0 = time.time()
        local = _MODEL_DIR / "config.json"

        if local.exists():
            src = str(_MODEL_DIR)
            log.info(f"从本地加载版面模型: {src}")
        else:
            src = _MODEL_NAME
            log.info(f"首次启动，从 HuggingFace 下载 {src} ...")

        _processor = RTDetrImageProcessor.from_pretrained(src)
        _model = AutoModelForObjectDetection.from_pretrained(src)

        if not local.exists():
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            _processor.save_pretrained(str(_MODEL_DIR))
            _model.save_pretrained(str(_MODEL_DIR))
            log.info(f"版面模型已保存到 {_MODEL_DIR}/")

        if torch.cuda.is_available():
            _model.to("cuda")
        _model.eval()
        _model_status = "ready"
        log.info(f"版面模型加载完成: {time.time() - t0:.2f}s")
    except Exception as e:
        _model_status = "error"
        _model_error = str(e)
        log.error(f"版面模型加载失败: {e}")


def _ensure_model():
    if _model is None:
        raise RuntimeError("模型尚未加载完成，请稍候")


# ── OCR 模型（PP-OCRv5 ONNX）─────────────────────────────────────
_OCR_DIR = Path("model/ocr")
_ocr_engine = None
_ocr_model_type = ""  # "server" | "mobile" | ""

# HuggingFace 下载映射：(repo_id, remote_filename) → local_filename
_OCR_MODELS = {
    "ppocr5_server_det.onnx": ("marsena/paddleocr-onnx-models", "PP-OCRv5_server_det_infer.onnx"),
    "ppocr5_server_rec.onnx": ("marsena/paddleocr-onnx-models", "PP-OCRv5_server_rec_infer.onnx"),
    "ppocr5_cls.onnx":        ("SWHL/RapidOCR", "PP-OCRv1/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    "ppocr5_dict.txt":        ("AXERA-TECH/PPOCR_v5", "ppocrv5_dict.txt"),
}


def _download_ocr_models():
    """Download OCR models from HuggingFace if not present."""
    from huggingface_hub import hf_hub_download

    _OCR_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, (repo_id, remote_name) in _OCR_MODELS.items():
        local_path = _OCR_DIR / local_name
        if local_path.exists():
            continue
        log.info(f"下载 OCR 模型: {local_name} ← {repo_id}/{remote_name}")
        downloaded = hf_hub_download(repo_id=repo_id, filename=remote_name)
        # Copy to local dir with our naming convention
        import shutil as _shutil
        _shutil.copy2(downloaded, str(local_path))
        log.info(f"OCR 模型已保存: {local_path}")


def _get_ocr():
    global _ocr_engine, _ocr_model_type
    if _ocr_engine is not None:
        return _ocr_engine

    # Auto-download if missing
    if not (_OCR_DIR / "ppocr5_server_det.onnx").exists():
        try:
            _download_ocr_models()
        except Exception as e:
            log.error(f"OCR 模型下载失败: {e}")

    cls = str(_OCR_DIR / "ppocr5_cls.onnx")
    dic = str(_OCR_DIR / "ppocr5_dict.txt")
    use_gpu = torch.cuda.is_available()

    # 优先 server 版（精度高），没有则 mobile 版
    if (_OCR_DIR / "ppocr5_server_det.onnx").exists():
        det = str(_OCR_DIR / "ppocr5_server_det.onnx")
        rec = str(_OCR_DIR / "ppocr5_server_rec.onnx")
        _ocr_model_type = "server"
        log.info(f"使用 PP-OCRv5 server 版模型 (GPU={use_gpu})")
    elif (_OCR_DIR / "ppocr5_mobile_det.onnx").exists():
        det = str(_OCR_DIR / "ppocr5_mobile_det.onnx")
        rec = str(_OCR_DIR / "ppocr5_mobile_rec.onnx")
        _ocr_model_type = "mobile"
        log.info("使用 PP-OCRv5 mobile 版模型")
    else:
        log.warning("OCR 模型未找到，扫描件将无法识别文字")
        return None

    from ppocr5 import simple_ppocr5
    _ocr_engine = simple_ppocr5(
        ppocr5_onnx_det=det, ppocr5_onnx_cls=cls,
        ppocr5_onnx_rec=rec, ppcor5_dict=dic, use_gpu=use_gpu,
    )
    # 保持默认参数（unclip=1.8, box_thresh=0.5）以避免检测框过紧切掉笔画复杂的字
    log.info(f"OCR 模型加载完成 (GPU={use_gpu})")
    return _ocr_engine


# ── 按区域 OCR：对每个版面区域单独裁剪识别 ──────────────────────
def ocr_blocks(image_path: str, blocks: list[dict]) -> dict[int, list[dict]]:
    """对每个版面区域单独裁剪 OCR，避免多栏混排。
    返回 {block_index: [{"text":..., "bbox":...}]}
    bbox 是图片全局坐标（裁剪坐标 + 区域偏移）。
    """
    import cv2
    engine = _get_ocr()
    if engine is None:
        return {}
    img = cv2.imread(image_path)
    if img is None:
        return {}

    result = {}
    for idx, block in enumerate(blocks):
        label = block["label"]
        # 只对需要文字的区域做 OCR
        if label in ("number", "figure", "table"):
            continue
        x1, y1, x2, y2 = block["bbox"]
        # 适度扩大裁剪范围，避免边缘字符被裁切
        pad = 5
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(img.shape[1], x2 + pad)
        cy2 = min(img.shape[0], y2 + pad)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        engine.run(crop)
        if not engine.results:
            continue

        lines = []
        for r in engine.results:
            pos = r["rec_pos"]
            # 转换为全局坐标
            bbox = [
                int(pos[0][0]) + cx1, int(pos[0][1]) + cy1,
                int(pos[2][0]) + cx1, int(pos[2][1]) + cy1,
            ]
            lines.append({"text": r["text"], "bbox": bbox})

        # NMS 去重：对重叠检测框只保留面积最大的（包含最完整文字）
        lines = _nms_ocr_lines(lines)
        # 文字级去重：合并同一行相邻 bbox 的前后缀重叠
        lines = _merge_suffix_prefix_lines(lines)

        lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        result[idx] = lines
        log.info(f"  区域 #{idx}({label}) OCR: {len(lines)} 行")

    return result


def _nms_ocr_lines(lines: list[dict], iou_threshold: float = 0.5,
                   contain_threshold: float = 0.7) -> list[dict]:
    """对 OCR 检测行做 NMS 去重。
    规则：两个框 IoU>阈值 或 一个框被另一个包含 >阈值 时，保留文字较长的那一个。
    """
    if len(lines) <= 1:
        return lines

    # 按文字长度降序（长文字优先，因为包含更完整信息）
    sorted_lines = sorted(lines, key=lambda l: -len(l["text"]))
    kept = []
    for line in sorted_lines:
        drop = False
        for k in kept:
            if _iou(line["bbox"], k["bbox"]) > iou_threshold:
                drop = True
                break
            # 检查一个框是否大部分被另一个包含
            if _contain_ratio(line["bbox"], k["bbox"]) > contain_threshold:
                drop = True
                break
        if not drop:
            kept.append(line)
    return kept


def _contain_ratio(a: list, b: list) -> float:
    """a 被 b 包含的比例（a∩b / a 的面积）"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if area_a == 0:
        return 0
    return inter / area_a


def _merge_suffix_prefix_lines(lines: list[dict], max_overlap: int = 6) -> list[dict]:
    """合并同一行内相邻 bbox 的字符重叠。
    规则：两个 line 同一行（y 中心差 < 15px）且水平相邻（左 bbox 右边缘与右 bbox 左边缘在 bbox 宽度内重叠），
    若左 line 的后缀与右 line 的前缀相同，合并并去重重叠部分。
    """
    if len(lines) <= 1:
        return lines

    # 先按 (y, x) 排序
    sorted_lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))

    merged = []
    for line in sorted_lines:
        if not merged:
            merged.append(dict(line))
            continue

        prev = merged[-1]
        prev_y_center = (prev["bbox"][1] + prev["bbox"][3]) / 2
        cur_y_center = (line["bbox"][1] + line["bbox"][3]) / 2
        same_row = abs(prev_y_center - cur_y_center) < 15

        if same_row:
            # 检查水平重叠（两个 bbox 的 x 有交集）
            x_overlap = min(prev["bbox"][2], line["bbox"][2]) - max(prev["bbox"][0], line["bbox"][0])
            if x_overlap > 0:
                # 寻找前后缀重叠
                prev_text = prev["text"]
                cur_text = line["text"]
                overlap_len = 0
                max_k = min(len(prev_text), len(cur_text), max_overlap)
                for k in range(max_k, 0, -1):
                    if prev_text.endswith(cur_text[:k]):
                        overlap_len = k
                        break
                if overlap_len > 0:
                    # 合并：保留 prev + cur 去掉前缀
                    new_text = prev_text + cur_text[overlap_len:]
                    new_bbox = [
                        min(prev["bbox"][0], line["bbox"][0]),
                        min(prev["bbox"][1], line["bbox"][1]),
                        max(prev["bbox"][2], line["bbox"][2]),
                        max(prev["bbox"][3], line["bbox"][3]),
                    ]
                    prev["text"] = new_text
                    prev["bbox"] = new_bbox
                    continue
        merged.append(dict(line))
    return merged


def ocr_single_block(image_path: str, block: dict) -> list[dict]:
    """对单个版面区域做 OCR，返回 [{"text":..., "bbox":...}]。用于文字层乱码回退。"""
    result = ocr_blocks(image_path, [block])
    return result.get(0, [])



# ── 表格结构识别（扫描件）── PP-TableMagic V2 ────────────────────
# 优先用 PP-TableMagic V2 完整产线（表格分类+单元格检测+结构识别+OCR），
# 没有 PaddlePaddle 则降级为 RapidTable (SLANet_plus ONNX)
_table_engine = None
_table_engine_type = ""  # "paddle" | "rapid" | ""


def _get_table_engine():
    global _table_engine, _table_engine_type
    if _table_engine is not None:
        return _table_engine

    # 优先尝试 PaddlePaddle PP-TableMagic 完整产线（V2）
    # 注意：当前 paddlepaddle-gpu 3.x 尚未发布，使用 CPU 版 Paddle 3.0 推理。
    # 版面检测（PyTorch）和 OCR（ONNX）仍走 GPU，仅表格结构识别走 Paddle CPU。
    try:
        from paddleocr import TableRecognitionPipelineV2
        _table_engine = TableRecognitionPipelineV2(
            device="cpu",
            use_layout_detection=False,          # 已有 PP-DocLayoutV3，不需要 V2 二次检测
            use_doc_orientation_classify=False,   # 裁切后的表格区域不需要方向矫正
            use_doc_unwarping=False,              # 裁切后的表格区域不需要弯曲矫正
        )
        _table_engine_type = "paddle"
        log.info("表格识别引擎: PP-TableMagic V2 (cpu) — "
                 "表格分类+单元格检测+结构识别+OCR")
        return _table_engine
    except ImportError:
        log.info("PaddlePaddle 未安装，尝试 RapidTable 降级方案")
    except Exception as e:
        log.warning(f"PP-TableMagic V2 初始化失败: {e}，尝试 RapidTable")

    # 降级为 RapidTable（ONNX）
    try:
        from rapid_table import ModelType, RapidTable, RapidTableInput
        input_args = RapidTableInput(model_type=ModelType.SLANETPLUS)
        _table_engine = RapidTable(input_args)
        _table_engine_type = "rapid"
        log.info("表格识别引擎: RapidTable (SLANet_plus ONNX)")
    except Exception as e:
        log.error(f"表格识别引擎加载失败: {e}")

    return _table_engine


def recognize_table_structure(image_path: str, table_bbox: list) -> dict | None:
    """对扫描件中的表格区域做结构识别。返回 {"cells": [...]} 或 None。"""
    engine = _get_table_engine()
    if engine is None:
        return None

    if _table_engine_type == "paddle":
        return _recognize_table_paddle(engine, image_path, table_bbox)
    else:
        return _recognize_table_rapid(engine, image_path, table_bbox)


def _recognize_table_paddle(engine, image_path: str, table_bbox: list) -> dict | None:
    """用 PP-TableMagic V2 完整产线识别表格。

    V2 产线内置：表格分类（有线/无线）→ 单元格检测 (RT-DETR-L)
    → 结构识别 (SLANeXt) → 文字检测+识别 (PP-OCRv4)。
    用 pred_html 获取行列结构，用 cell_box_list 获取坐标，
    用 table_ocr_pred 的 rec_texts/rec_boxes 恢复单元格内换行。
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None

    x1, y1, x2, y2 = table_bbox
    pad = 5
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img.shape[1], x2 + pad)
    cy2 = min(img.shape[0], y2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    try:
        results = list(engine.predict(crop))
        if not results:
            return None

        res = results[0]

        # 从 json 结构中提取各项数据
        data = res.json if hasattr(res, 'json') else {}
        inner = data.get("res", data) if isinstance(data, dict) else {}
        table_res_list = inner.get("table_res_list", [])

        pred_html = ""
        cell_boxes = []
        ocr_data = {}
        if table_res_list:
            pred_html = table_res_list[0].get("pred_html", "")
            cell_boxes = table_res_list[0].get("cell_box_list", [])
            ocr_data = table_res_list[0].get("table_ocr_pred", {})

        # 回退：从 res.html 取 HTML
        if not pred_html and hasattr(res, 'html') and isinstance(res.html, dict):
            for key in sorted(res.html):
                pred_html = res.html[key]
                break

        if not pred_html:
            log.info("PP-TableMagic V2: 未检测到表格")
            return None

        # ── 第1步：用 pred_html 解析行列结构，用 cell_box_list 提供 bbox ──
        bboxes_batch = None
        if cell_boxes:
            quads = []
            for cb in cell_boxes:
                x1b, y1b, x2b, y2b = cb[0], cb[1], cb[2], cb[3]
                quads.append([x1b, y1b, x2b, y1b, x2b, y2b, x1b, y2b])
            bboxes_batch = [quads]

        cells = _parse_html_table(pred_html, bboxes_batch, cx1, cy1)

        # ── 第2步：用 OCR 逐行���字 + 坐标恢复单元格内换行 ──
        ocr_texts = ocr_data.get("rec_texts", []) if isinstance(ocr_data, dict) else []
        ocr_boxes = ocr_data.get("rec_boxes", []) if isinstance(ocr_data, dict) else []

        if cell_boxes and ocr_texts and ocr_boxes:
            # 把每条 OCR 文字按中心点分配到最近的单元格
            # 格式: {cell_idx: [(y_center, text), ...]}
            text_by_cell = {}
            for t_idx, txt in enumerate(ocr_texts):
                if t_idx >= len(ocr_boxes):
                    break
                tb = ocr_boxes[t_idx]  # [x1, y1, x2, y2]
                tcx = (tb[0] + tb[2]) / 2
                tcy = (tb[1] + tb[3]) / 2
                # 找包含���文字中心点的单元格
                best_cell = -1
                for ci, cb in enumerate(cell_boxes):
                    if cb[0] <= tcx <= cb[2] and cb[1] <= tcy <= cb[3]:
                        best_cell = ci
                        break
                if best_cell == -1:
                    # 中心点不在任何单元格内，找最近的
                    min_dist = float('inf')
                    for ci, cb in enumerate(cell_boxes):
                        ccx = (cb[0] + cb[2]) / 2
                        ccy = (cb[1] + cb[3]) / 2
                        dist = (tcx - ccx) ** 2 + (tcy - ccy) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_cell = ci
                if best_cell >= 0:
                    text_by_cell.setdefault(best_cell, []).append((tcy, txt))

            # 用按 Y 排序的 OCR 文字替换 HTML 解析出的文字（恢复换行）
            for ci, lines in text_by_cell.items():
                if ci < len(cells):
                    lines.sort(key=lambda x: x[0])  # 按 Y 坐标从上到下
                    cells[ci]["text"] = "\n".join(t for _, t in lines)

        log.info(f"PP-TableMagic V2 表格识别完成: {len(cells)} 个单元格")
        return {"cells": cells} if cells else None

    except Exception as e:
        log.error(f"PP-TableMagic V2 表格识别失败: {e}")
        return None


def _recognize_table_rapid(engine, image_path: str, table_bbox: list) -> dict | None:
    """用 RapidTable (ONNX) 识别表格"""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return None

    x1, y1, x2, y2 = table_bbox
    pad = 5
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(img.shape[1], x2 + pad)
    cy2 = min(img.shape[0], y2 + pad)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    # 用 PP-OCRv5 做文字识别
    ocr = _get_ocr()
    ocr_results = None
    if ocr:
        ocr.run(crop)
        if ocr.results:
            boxes = np.array([r["rec_pos"] for r in ocr.results])
            txts = tuple(r["text"] for r in ocr.results)
            scores = tuple(0.9 for _ in ocr.results)
            ocr_results = [(boxes, txts, scores)]

    try:
        result = engine(crop, ocr_results=ocr_results)
        html = result.pred_htmls[0] if result.pred_htmls else ""
        if not html:
            return None

        cells = _parse_html_table(html, result.cell_bboxes, cx1, cy1)
        log.info(f"RapidTable 表格识别完成: {len(cells)} 个单元格")
        return {"cells": cells}
    except Exception as e:
        log.error(f"RapidTable 表格识别失败: {e}")
        return None


def _parse_html_table(html: str, cell_bboxes_batch, offset_x: int, offset_y: int) -> list[dict]:
    """将 RapidTable 的 HTML 表格解析为 [{row, col, text, bbox}] 格式"""
    from html.parser import HTMLParser

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.cells = []
            self.row = -1
            self.col = 0
            self.in_td = False
            self.current_text = ""
            self.col_spans = {}  # row -> set of occupied cols

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row += 1
                self.col = 0
                # 跳过被 rowspan 占用的列
                while (self.row, self.col) in self.col_spans:
                    self.col += 1
            elif tag == "td":
                self.in_td = True
                self.current_text = ""
                # 跳过被 rowspan 占用的列
                while (self.row, self.col) in self.col_spans:
                    self.col += 1
                # 处理 colspan/rowspan
                colspan = 1
                rowspan = 1
                for k, v in attrs:
                    if k == "colspan":
                        colspan = int(v)
                    elif k == "rowspan":
                        rowspan = int(v)
                self._colspan = colspan
                self._rowspan = rowspan

        def handle_data(self, data):
            if self.in_td:
                self.current_text += data

        def handle_endtag(self, tag):
            if tag == "td" and self.in_td:
                self.in_td = False
                self.cells.append({
                    "row": self.row,
                    "col": self.col,
                    "text": self.current_text.strip(),
                })
                # 标记 rowspan 占用的位置
                for dr in range(self._rowspan):
                    for dc in range(self._colspan):
                        if dr > 0 or dc > 0:
                            self.col_spans[(self.row + dr, self.col + dc)] = True
                self.col += self._colspan

    parser = TableParser()
    parser.feed(html)

    # 分配 bbox：先转为全局坐标，再按空间位置匹配
    bboxes = []
    if cell_bboxes_batch and len(cell_bboxes_batch) > 0:
        bboxes = []
        for cb in cell_bboxes_batch[0]:
            # cb 是 8 值四边形 [x1,y1,x2,y2,x3,y3,x4,y4]，取外接矩形
            xs = [float(cb[i]) for i in range(0, len(cb), 2)]
            ys = [float(cb[i]) for i in range(1, len(cb), 2)]
            bboxes.append([
                int(min(xs)) + offset_x, int(min(ys)) + offset_y,
                int(max(xs)) + offset_x, int(max(ys)) + offset_y,
            ])

    for i, cell in enumerate(parser.cells):
        if i < len(bboxes):
            cell["bbox"] = bboxes[i]
        else:
            cell["bbox"] = [0, 0, 0, 0]

    return parser.cells


# ── 目录行解析 ────────────────────────────────────────────────────
def _dedupe_repeated_chars(text: str) -> str:
    """去除目录文本中连续重复的同一字符（OCR 常见错误）。
    例: "第二二节牙本质" → "第二节牙本质"
    例: "第一节节口腔黏膜的的基本结构" → "第一节口腔黏膜的基本结构"
    白名单：保留某些常见叠词（避免误删合法重叠）。
    """
    # 常见叠词白名单（在 TOC 中极少出现，但保留以防万一）
    WHITELIST = {"看看", "好好", "慢慢", "常常", "刚刚", "渐渐", "个个",
                 "早早", "远远", "静静", "暗暗", "默默", "偷偷", "种种",
                 "处处", "时时", "天天", "年年", "月月", "日日"}
    if len(text) < 2:
        return text
    out = [text[0]]
    i = 1
    while i < len(text):
        prev = out[-1]
        cur = text[i]
        # 仅对中文字符做去重（不处理英文、数字、标点、空格、引导线）
        if cur == prev and '\u4e00' <= cur <= '\u9fff':
            # 检查白名单
            pair = prev + cur
            if pair not in WHITELIST:
                # 跳过这个重复字符
                i += 1
                continue
        out.append(cur)
        i += 1
    return "".join(out)


def _parse_toc_line(line: str) -> dict | None:
    """解析目录行，拆分标题和页码。
    例: '第一章 牙体组织……1' → {"title": "第一章 牙体组织", "page": 1}
    """
    # 先做字符去重（修复 OCR 重复字符错误）
    line = _dedupe_repeated_chars(line)
    line = line.strip()
    if not line:
        return None

    # 纯页码行（如 "100"）跳过
    if re.fullmatch(r"\d{1,4}", line):
        return None

    # 匹配：标题 + 引导线 + 页码
    m = re.match(r"^(.+?)\s*[\.\.…·\-—─]{2,}\s*(\d+)\s*$", line)
    if m:
        title = _clean_text(m.group(1))
        page = int(m.group(2))
        if title:
            return {"title": title, "page": page}

    # 匹配：标题 + 空格/点 + 页码（短引导线）
    m = re.match(r"^(.+?)[\s·\.…]+(\d{1,4})\s*$", line)
    if m:
        title = _clean_text(m.group(1))
        page = int(m.group(2))
        if title and len(title) > 1:
            return {"title": title, "page": page}

    # 无页码的标题行（如"第一篇基础知识"）
    if re.search(r"第.{1,3}[章节篇]", line):
        title = _clean_text(re.sub(r"[\.\.…·]+$", "", line))
        if title:
            return {"title": title, "page": None}

    return None


def _merge_toc_lines(ocr_items: list[dict]) -> list[dict]:
    """将纯页码行合并到前一个标题行。
    如 ["第一章牙体组织", "1"] → ["第一章牙体组织 1"]
    同时将被引导线截断的行合并。
    """
    merged = []
    for item in ocr_items:
        text = item["text"].strip()
        if not text:
            continue
        # 纯页码行或纯引导线+页码，合并到前一行
        if re.fullmatch(r"[\d\s\.…·]+", text) and merged:
            merged[-1]["text"] = merged[-1]["text"].rstrip("·.… ") + " " + text.lstrip("·.… ")
            continue
        # 以引导线开头的残行（如"……60"），合并到前一行
        if re.match(r"^[\.…·\-—]+\s*\d", text) and merged:
            merged[-1]["text"] = merged[-1]["text"].rstrip("·.… ") + text
            continue
        merged.append(dict(item))
    return merged


def _is_toc_block(texts: list[str]) -> bool:
    """判断一组文本行是否像目录"""
    if len(texts) < 3:
        return False
    # 方式1：标题+页码行占比
    parsed = sum(1 for t in texts if _parse_toc_line(t) is not None)
    if parsed / len(texts) > 0.3:
        return True
    # 方式2：大量"第X章/节"模式
    chapter_pat = sum(1 for t in texts if re.search(r"第.{1,3}[章节篇]", t))
    if chapter_pat >= 3:
        return True
    return False


# ── 判断页面是否有文字层 ─────────────────────────────────────────
def is_garbage_text(text: str, threshold: float = 0.3) -> bool:
    """检测提取的文字是否为乱码（CID 字体解码失败、编码错误等）。
    threshold: 有意义字符占比低于此值判定为乱码。
    """
    if not text or len(text) < 3:
        return False

    n = len(text)

    # U+FFFD 替换字符过多
    if text.count('\ufffd') / n > 0.1:
        return True

    # Private Use Area 字符过多（CID 字体解码失败的典型特征）
    pua = sum(1 for c in text if '\ue000' <= c <= '\uf8ff')
    if pua / n > 0.15:
        return True

    # 有意义字符比例过低
    meaningful = sum(1 for c in text if (
        c.isalnum()
        or '\u4e00' <= c <= '\u9fff'   # CJK 基本区
        or '\u3400' <= c <= '\u4dbf'   # CJK 扩展 A
        or '\u3000' <= c <= '\u303f'   # CJK 符号
        or '\uff00' <= c <= '\uffef'   # 全角字符
        or c in ' \n\r\t.,;:!?()[]{}"\'-/\\·、。，；：！？（）【】《》""''—…'
    ))
    if n > 10 and meaningful / n < threshold:
        return True

    return False


def page_has_text(pdf_path: str, page_num: int) -> bool:
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    text = page.get_text("text").strip()
    doc.close()

    if not text or len(text) < 5:
        return False

    printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    ratio = printable / len(text) if text else 0
    if ratio <= 0.5:
        return False

    if is_garbage_text(text):
        log.info(f"第 {page_num} 页文字层检测为乱码，将回退 OCR")
        return False

    return True


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_load_model, daemon=True).start()
    threading.Thread(target=_download_ocr_models, daemon=True).start()
    threading.Thread(target=_get_table_engine, daemon=True).start()


@app.get("/api/model-status")
async def model_status():
    ocr_available = (_OCR_DIR / "ppocr5_server_det.onnx").exists() or (_OCR_DIR / "ppocr5_mobile_det.onnx").exists()
    if _ocr_model_type:
        ocr_model = _ocr_model_type
    elif (_OCR_DIR / "ppocr5_server_det.onnx").exists():
        ocr_model = "server"
    elif (_OCR_DIR / "ppocr5_mobile_det.onnx").exists():
        ocr_model = "mobile"
    else:
        ocr_model = ""
    gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu else ""
    return {
        "status": _model_status, "error": _model_error,
        "ocr_available": ocr_available, "ocr_model": ocr_model,
        "gpu": gpu, "gpu_name": gpu_name,
        "table_engine": _table_engine_type or "none",
    }


# ── PDF 保存 + 按需渲染 ──────────────────────────────────────────────
def pdf_save_and_count(pdf_bytes: bytes, doc_dir: Path) -> int:
    """Save PDF and return page count without rendering any images."""
    (doc_dir / "source.pdf").write_bytes(pdf_bytes)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


def ensure_page_image(doc_dir: Path, page_num: int) -> Path:
    """Render a single page PNG on demand, return its path."""
    img_path = doc_dir / f"page_{page_num:03d}.png"
    if not img_path.exists():
        pdf_path = doc_dir / "source.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"source.pdf not found in {doc_dir}")
        doc = fitz.open(str(pdf_path))
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pix.save(str(img_path))
        doc.close()
    return img_path


def _sort_reading_order(blocks: list[dict], y_tolerance: int = 30) -> list[dict]:
    """按阅读顺序排序：将 y 坐标相近的 block 分到同一行，行内按 x 排序。
    解决双栏布局中右列 y 略小于左列导致排序错误的问题。
    """
    if not blocks:
        return blocks

    sorted_by_y = sorted(blocks, key=lambda b: b["bbox"][1])

    rows = []
    current_row = [sorted_by_y[0]]
    row_y = sorted_by_y[0]["bbox"][1]

    for b in sorted_by_y[1:]:
        if b["bbox"][1] - row_y <= y_tolerance:
            current_row.append(b)
        else:
            rows.append(current_row)
            current_row = [b]
            row_y = b["bbox"][1]
    rows.append(current_row)

    result = []
    for row in rows:
        row.sort(key=lambda b: b["bbox"][0])
        result.extend(row)
    return result


# ── 版面检测 ───────────────────────────────────────────────────────
def detect_layout(image_path: str) -> list[dict]:
    _ensure_model()
    img = Image.open(image_path).convert("RGB")
    device = next(_model.parameters()).device

    inputs = _processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    target_sizes = torch.tensor([img.size[::-1]], device=device)
    results = _processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=_THRESHOLD
    )[0]

    id2label = _model.config.id2label
    blocks = []
    for i, (score, label_id, box) in enumerate(
        zip(results["scores"], results["labels"], results["boxes"])
    ):
        blocks.append({
            "bbox": [round(c) for c in box.tolist()],
            "label": id2label[label_id.item()],
            "score": round(score.item(), 3),
            "order": i,
        })

    blocks = [b for b in blocks if b["label"] != "number"]

    blocks.sort(key=lambda b: -b["score"])
    kept = []
    for b in blocks:
        if any(_iou(b["bbox"], k["bbox"]) > 0.5 for k in kept):
            continue
        kept.append(b)
    blocks = kept

    blocks = _sort_reading_order(blocks)
    for i, b in enumerate(blocks):
        b["order"] = i

    log.info(f"检测到 {len(blocks)} 个区域")
    for b in blocks:
        log.info(f"  [{b['order']}] {b['label']:12s} score={b['score']:.3f} bbox={b['bbox']}")
    return blocks


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


# ── 表格单元格提取（PyMuPDF）──────────────────────────────────────
def extract_table_cells(pdf_path: str, page_num: int, table_bbox: list,
                        img_width: int, img_height: int) -> list[dict]:
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    scale = 2.0
    pad = 10
    page_rect = page.rect
    pdf_rect = fitz.Rect(
        max(0, table_bbox[0] / scale - pad),
        max(0, table_bbox[1] / scale - pad),
        min(page_rect.width, table_bbox[2] / scale + pad),
        min(page_rect.height, table_bbox[3] / scale + pad),
    )

    tables = page.find_tables(clip=pdf_rect)
    cells = []
    for table in tables:
        rows_text = table.extract()
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell_rect in enumerate(row.cells):
                if cell_rect is None:
                    continue
                text = ""
                if row_idx < len(rows_text) and col_idx < len(rows_text[row_idx]):
                    text = rows_text[row_idx][col_idx] or ""
                cells.append({
                    "bbox": [
                        round(cell_rect[0] * scale),
                        round(cell_rect[1] * scale),
                        round(cell_rect[2] * scale),
                        round(cell_rect[3] * scale),
                    ],
                    "row": row_idx,
                    "col": col_idx,
                    "text": text,
                })
    doc.close()

    log.info(f"提取到 {len(cells)} 个单元格")
    return cells


# ── 文本提取（PDF 文字层）────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str, page_num: int, bbox: list) -> str:
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    scale = 2.0
    rect = fitz.Rect(bbox[0] / scale, bbox[1] / scale,
                     bbox[2] / scale, bbox[3] / scale)
    text = page.get_text("text", clip=rect).strip()
    doc.close()
    return text


# ── 结构化输出 ────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    return text.strip()


def _split_value(text: str):
    text = _clean_text(text)
    raw = [l.strip() for l in text.split("\n") if l.strip()]
    lines = []
    for l in raw:
        if re.fullmatch(r'[\W_]+', l) and lines:
            lines[-1] += l
        else:
            lines.append(l)
    return lines if len(lines) > 1 else (lines[0] if lines else "")


def build_structured(blocks: list, pdf_path: str | None, page_num: int,
                     image_path: str, has_text_layer: bool,
                     ocr_by_block: dict[int, list[str]] | None = None) -> dict:
    """构建结构化数据。
    ocr_by_block: 全页 OCR 结果按区域归类后的字典 {block_idx: [text_lines]}
    """
    content = []

    def get_text(block_idx: int, bbox: list) -> str:
        """获取区域文字：优先用 OCR 回退结果，否则有文字层用 PyMuPDF"""
        if ocr_by_block and block_idx in ocr_by_block:
            return "\n".join(it["text"] for it in ocr_by_block[block_idx])
        if has_text_layer and pdf_path:
            return extract_text_from_pdf(pdf_path, page_num, bbox)
        return ""

    for idx, block in enumerate(blocks):
        label = block["label"]

        # 标题类
        if label in ("paragraph_title", "document_title", "title"):
            text = get_text(idx, block["bbox"])
            if text:
                lines = [_clean_text(l) for l in text.split("\n") if l.strip()]
                for line in lines:
                    if not any(c["type"] == "title" and c["text"] == line for c in content):
                        content.append({"type": "title", "text": line})

        # 表格（有单元格）
        elif label == "table" and block.get("cells"):
            cells = block["cells"]
            max_row = max(c["row"] for c in cells)
            max_col = max(c["col"] for c in cells)

            grid = [[None] * (max_col + 1) for _ in range(max_row + 1)]
            for c in cells:
                grid[c["row"]][c["col"]] = c["text"] or ""

            headers = [_clean_text(grid[0][c] or f"列{c+1}") for c in range(max_col + 1)]

            rows = []
            prev = [""] * (max_col + 1)
            for r in range(1, max_row + 1):
                row_data = {}
                for c in range(max_col + 1):
                    val = grid[r][c]
                    if val is None or val == "":
                        val = prev[c]
                    else:
                        prev[c] = val
                    row_data[headers[c]] = _split_value(val)
                rows.append(row_data)

            content.append({"type": "table", "headers": headers, "rows": rows})

        # 表格（扫描件，无结构识别）
        elif label == "table" and block.get("cells") is None:
            text = get_text(idx, block["bbox"])
            if text:
                content.append({"type": "table_text", "text": _clean_text(text),
                                "note": "扫描件表格，仅提取文字，无法解析单元格结构"})

        # 文本/内容类
        elif label in ("text", "content", "abstract", "reference",
                        "header", "footer", "algorithm",
                        "figure_title", "table_title"):
            ocr_items = []  # [{"text":..., "bbox":...}]
            if ocr_by_block and idx in ocr_by_block:
                ocr_items = ocr_by_block[idx]
            elif has_text_layer and pdf_path:
                raw = extract_text_from_pdf(pdf_path, page_num, block["bbox"])
                if raw:
                    ocr_items = [{"text": l.strip(), "bbox": None}
                                 for l in raw.split("\n") if l.strip()]

            if not ocr_items:
                continue

            # 合并被拆分的目录行（页码行合并回标题行）
            ocr_items = _merge_toc_lines(ocr_items)
            text_list = [it["text"] for it in ocr_items]

            # 判断是否为目录
            if _is_toc_block(text_list):
                toc_items = []
                for it in ocr_items:
                    parsed = _parse_toc_line(it["text"])
                    if parsed:
                        parsed["bbox"] = it["bbox"]
                        toc_items.append(parsed)
                    else:
                        cleaned = _clean_text(it["text"])
                        if cleaned and len(cleaned) > 1:
                            toc_items.append({"title": cleaned, "page": None, "bbox": it["bbox"]})
                if toc_items:
                    block["toc_items"] = toc_items
                    content.append({"type": "toc", "items": toc_items})
            else:
                text = _clean_text("\n".join(text_list))
                if text:
                    content.append({"type": "text", "text": text})

    return {"page": page_num, "content": content}



def _table_to_grid(cells: list[dict]) -> tuple[list[list[str]], int, int]:
    """将 cells 列表转为二维网格，返回 (grid, max_row, max_col)"""
    if not cells:
        return [], -1, -1
    max_row = max(c["row"] for c in cells)
    max_col = max(c["col"] for c in cells)
    grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
    for c in cells:
        grid[c["row"]][c["col"]] = c.get("text") or ""
    return grid, max_row, max_col


def _row_similarity(row_a: list[str], row_b: list[str]) -> float:
    """计算两行文字的相似度（0~1），用于判断续表表头是否重复"""
    if len(row_a) != len(row_b):
        return 0.0
    if not row_a:
        return 0.0
    matches = sum(1 for a, b in zip(row_a, row_b)
                  if _clean_text(a) == _clean_text(b))
    return matches / len(row_a)


def _col_widths_ratio(cells: list[dict]) -> list[float]:
    """计算每列宽度占比，用于判断两个表格列结构是否一致"""
    if not cells:
        return []
    max_col = max(c["col"] for c in cells)
    col_bounds = {}
    for c in cells:
        col = c["col"]
        x1, _, x2, _ = c["bbox"]
        if col not in col_bounds:
            col_bounds[col] = [x1, x2]
        else:
            col_bounds[col][0] = min(col_bounds[col][0], x1)
            col_bounds[col][1] = max(col_bounds[col][1], x2)

    widths = []
    for col in range(max_col + 1):
        if col in col_bounds:
            widths.append(col_bounds[col][1] - col_bounds[col][0])
        else:
            widths.append(0)
    total = sum(widths) or 1
    return [w / total for w in widths]


def _col_structure_similar(cells_a: list[dict], cells_b: list[dict],
                           tolerance: float = 0.15) -> bool:
    """判断两个表格的列结构是否相似（列数相同且列宽比例接近）"""
    max_col_a = max(c["col"] for c in cells_a) if cells_a else -1
    max_col_b = max(c["col"] for c in cells_b) if cells_b else -1
    if max_col_a != max_col_b or max_col_a < 0:
        return False

    ratios_a = _col_widths_ratio(cells_a)
    ratios_b = _col_widths_ratio(cells_b)
    for ra, rb in zip(ratios_a, ratios_b):
        if abs(ra - rb) > tolerance:
            return False
    return True


def merge_cross_page_tables(pages_result: list[dict]) -> list[dict]:
    """跨页表格合并：检测连续页的续表并合并到第一个表格所在页。
    使用滑动指针追踪待续表格，支持三页及以上的连续合并。
    """
    merge_info = []
    # 滑动指针：当前"待续"表格
    open_page_idx = None   # 待续表格所在页的 index
    open_block = None      # 待续表格的 block 引用

    for i in range(len(pages_result)):
        page = pages_result[i]
        blocks = page.get("blocks", [])
        page_h = page.get("height", 0)

        # 找本页第一个有 cells 的 table（续表候选）
        first_table = None
        for b in blocks:
            if b["label"] == "table" and b.get("cells"):
                first_table = b
                break

        # 尝试与 open_block 合并
        if open_block is not None and first_table is not None:
            cells_a = open_block["cells"]
            cells_b = first_table["cells"]

            # 位置：续表应在页面上半部
            table_b_top = min(c["bbox"][1] for c in cells_b)
            position_ok = page_h <= 0 or table_b_top < page_h * 0.5

            if position_ok and _col_structure_similar(cells_a, cells_b):
                grid_a, max_row_a, _ = _table_to_grid(cells_a)
                grid_b, max_row_b, _ = _table_to_grid(cells_b)

                # 检测并跳过重复表头
                start_row_b = 0
                if max_row_b >= 0 and _row_similarity(grid_a[0], grid_b[0]) > 0.8:
                    start_row_b = 1
                    log.info(f"跨页表格合并: 第{page['page']}页续表表头与前页相同，跳过")

                if start_row_b <= max_row_b:
                    # 执行合并
                    new_row_offset = max_row_a + 1
                    for c in cells_b:
                        if c["row"] < start_row_b:
                            continue
                        open_block["cells"].append({
                            "bbox": c["bbox"],
                            "row": c["row"] - start_row_b + new_row_offset,
                            "col": c["col"],
                            "text": c.get("text", ""),
                        })

                    open_page = pages_result[open_page_idx]
                    open_block["merged_from_pages"] = open_block.get(
                        "merged_from_pages", [open_page["page"]])
                    open_block["merged_from_pages"].append(page["page"])
                    first_table["_merged_into"] = (open_page_idx,)

                    rows_added = max_row_b + 1 - start_row_b
                    merge_info.append({
                        "from_page": page["page"],
                        "into_page": open_page["page"],
                        "rows_added": rows_added,
                        "header_skipped": start_row_b > 0,
                    })
                    log.info(
                        f"跨页表格合并: 第{page['page']}页续表 → 第{open_page['page']}页"
                        f" (+{rows_added}行"
                        f"{'，跳过重复表头' if start_row_b > 0 else ''})")

                    # 合并后 open_block 继续保持（支持三页+连续合并）
                    # 但需检查本页是否还有其他表格在底部
                    last_table_this_page = None
                    for b in reversed(blocks):
                        if b["label"] == "table" and b.get("cells") and not b.get("_merged_into"):
                            last_table_this_page = b
                            break
                    # 如果本页底部还有另一个表格（不是被合并的那个），更新 open
                    if last_table_this_page and last_table_this_page is not first_table:
                        tb = max(c["bbox"][3] for c in last_table_this_page["cells"])
                        if page_h <= 0 or tb >= page_h * 0.5:
                            open_page_idx = i
                            open_block = last_table_this_page
                    # 否则 open_block 保持不变（继续用合并后的表格）
                    continue  # 跳过下面的 open_block 更新

        # 更新 open_block：找本页最后一个在底部的表格
        open_page_idx = None
        open_block = None
        for b in reversed(blocks):
            if b["label"] == "table" and b.get("cells") and not b.get("_merged_into"):
                tb = max(c["bbox"][3] for c in b["cells"])
                if page_h <= 0 or tb >= page_h * 0.5:
                    open_page_idx = i
                    open_block = b
                break

    # 重建受影响页的 structured content
    if merge_info:
        for page_data in pages_result:
            blocks = page_data.get("blocks", [])

            # 重建合并源表格的 structured
            for block in blocks:
                if block["label"] == "table" and block.get("merged_from_pages"):
                    cells = block["cells"]
                    max_row = max(c["row"] for c in cells)
                    max_col = max(c["col"] for c in cells)
                    grid = [[""] * (max_col + 1) for _ in range(max_row + 1)]
                    for c in cells:
                        grid[c["row"]][c["col"]] = c.get("text") or ""

                    headers = [_clean_text(grid[0][c] or f"列{c+1}") for c in range(max_col + 1)]
                    rows = []
                    prev = [""] * (max_col + 1)
                    for r in range(1, max_row + 1):
                        row_data = {}
                        for ci in range(max_col + 1):
                            val = grid[r][ci]
                            if val is None or val == "":
                                val = prev[ci]
                            else:
                                prev[ci] = val
                            row_data[headers[ci]] = _split_value(val)
                        rows.append(row_data)

                    for idx, item in enumerate(page_data["structured"]["content"]):
                        if item["type"] == "table":
                            page_data["structured"]["content"][idx] = {
                                "type": "table", "headers": headers, "rows": rows,
                                "merged_from_pages": block["merged_from_pages"],
                            }
                            break

            # 移除被合并走的表格
            has_merged_away = any(b.get("_merged_into") for b in blocks if b["label"] == "table")
            if has_merged_away:
                page_data["structured"]["content"] = [
                    item for item in page_data["structured"]["content"]
                    if not (item["type"] == "table" and
                            not item.get("merged_from_pages") and
                            item.get("headers"))
                ]

    return merge_info


# ── API 路由 ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("index.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = uuid.uuid4().hex[:12]
    doc_dir = UPLOAD_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    is_pdf = file.filename.lower().endswith(".pdf")

    if is_pdf:
        page_count = pdf_save_and_count(content, doc_dir)
        filenames = [f"page_{i:03d}.png" for i in range(1, page_count + 1)]
    else:
        name = f"page_001{Path(file.filename).suffix}"
        (doc_dir / name).write_bytes(content)
        filenames = [name]

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": filenames,
        "is_pdf": is_pdf,
    }


@app.get("/api/images/{doc_id}/{filename}")
async def get_image(doc_id: str, filename: str):
    doc_dir = UPLOAD_DIR / doc_id
    path = doc_dir / filename
    if not path.exists():
        if filename.startswith("page_") and filename.endswith(".png"):
            page_num = int(filename.split("_")[1].split(".")[0])
            try:
                ensure_page_image(doc_dir, page_num)
            except (FileNotFoundError, IndexError):
                return {"error": "not found"}
        else:
            return {"error": "not found"}
    return FileResponse(path)


@app.get("/api/text-source/{doc_id}/{page_num}")
async def text_source(doc_id: str, page_num: int):
    """快速判断某页文字来源（pdf/ocr），不做检测"""
    doc_dir = UPLOAD_DIR / doc_id
    pdf_path = doc_dir / "source.pdf"
    if pdf_path.exists():
        has_text = page_has_text(str(pdf_path), page_num)
        return {"text_source": "pdf" if has_text else "ocr"}
    return {"text_source": "ocr"}


@app.post("/api/detect/{doc_id}/{page_num}")
async def detect(doc_id: str, page_num: int):
    doc_dir = UPLOAD_DIR / doc_id
    try:
        img_path = ensure_page_image(doc_dir, page_num)
    except FileNotFoundError:
        return {"error": "page not found"}

    blocks = detect_layout(str(img_path))

    with Image.open(img_path) as img:
        w, h = img.size

    pdf_path = doc_dir / "source.pdf"
    is_pdf = pdf_path.exists()

    # 逐页判断是否有文字层
    has_text_layer = False
    if is_pdf:
        has_text_layer = page_has_text(str(pdf_path), page_num)

    text_source = "pdf" if has_text_layer else "ocr"
    log.info(f"第 {page_num} 页文字来源: {text_source}")

    # 扫描件：全页 OCR 一次，按区域归类
    ocr_by_block = None
    if not has_text_layer:
        ocr_by_block = ocr_blocks(str(img_path), blocks)

    for block in blocks:
        if block["label"] == "table":
            if has_text_layer:
                cells = extract_table_cells(
                    str(pdf_path), page_num, block["bbox"], w, h
                )
                garbage_cells = sum(1 for c in cells if is_garbage_text(c.get("text", "")))
                if cells and garbage_cells > len(cells) * 0.3:
                    log.info(f"表格单元格乱码 ({garbage_cells}/{len(cells)})，回退 PP-TableMagic V2")
                    table_result = recognize_table_structure(str(img_path), block["bbox"])
                    if table_result and table_result["cells"]:
                        block["cells"] = table_result["cells"]
                    else:
                        block["cells"] = cells
                else:
                    block["cells"] = cells
            else:
                table_result = recognize_table_structure(str(img_path), block["bbox"])
                if table_result and table_result["cells"]:
                    block["cells"] = table_result["cells"]
                else:
                    block["cells"] = None

    # 给每个 block 挂上提取到的文字（供前端可视化显示）
    # 文字层乱码时逐 block 回退 OCR
    if ocr_by_block is None:
        ocr_by_block = {}
    for idx, block in enumerate(blocks):
        if block.get("cells") or block.get("table_html"):
            continue
        if has_text_layer and is_pdf:
            text = extract_text_from_pdf(str(pdf_path), page_num, block["bbox"])
            if is_garbage_text(text):
                log.info(f"区域 #{idx}({block['label']}) 文字乱码，回退 OCR")
                ocr_result = ocr_single_block(str(img_path), block)
                if ocr_result:
                    ocr_by_block[idx] = ocr_result
                    block["ocr_text"] = "\n".join(it["text"] for it in ocr_result)
                    block["text_fallback"] = "ocr"
                else:
                    block["ocr_text"] = text
            else:
                block["ocr_text"] = text
        elif idx in ocr_by_block:
            block["ocr_text"] = "\n".join(it["text"] for it in ocr_by_block[idx])

    structured = build_structured(
        blocks, str(pdf_path) if is_pdf else None, page_num,
        str(img_path), has_text_layer, ocr_by_block
    )

    return {
        "page": page_num, "width": w, "height": h,
        "blocks": blocks, "is_pdf": is_pdf,
        "text_source": text_source,
        "structured": structured,
    }


@app.post("/api/detect-all/{doc_id}")
async def detect_all(doc_id: str):
    doc_dir = UPLOAD_DIR / doc_id
    if not doc_dir.exists():
        return {"error": "document not found"}

    pdf_path = doc_dir / "source.pdf"
    is_pdf = pdf_path.exists()

    if is_pdf:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        doc.close()
    else:
        total_pages = len(list(doc_dir.glob("page_*.*")))

    pages_result = []
    for page_num in range(1, total_pages + 1):
        try:
            img_path = ensure_page_image(doc_dir, page_num)
        except FileNotFoundError:
            continue
        blocks = detect_layout(str(img_path))
        with Image.open(img_path) as img:
            w, h = img.size

        has_text_layer = False
        if is_pdf:
            has_text_layer = page_has_text(str(pdf_path), page_num)

        ocr_by_block = None
        if not has_text_layer:
            ocr_by_block = ocr_blocks(str(img_path), blocks)

        for block in blocks:
            if block["label"] == "table":
                if has_text_layer:
                    cells = extract_table_cells(
                        str(pdf_path), page_num, block["bbox"], w, h
                    )
                    garbage_cells = sum(1 for c in cells if is_garbage_text(c.get("text", "")))
                    if cells and garbage_cells > len(cells) * 0.3:
                        log.info(f"[detect_all] 表格单元格乱码，回退 PP-TableMagic V2")
                        table_result = recognize_table_structure(str(img_path), block["bbox"])
                        if table_result and table_result["cells"]:
                            block["cells"] = table_result["cells"]
                        else:
                            block["cells"] = cells
                    else:
                        block["cells"] = cells
                else:
                    table_result = recognize_table_structure(str(img_path), block["bbox"])
                    if table_result and table_result["cells"]:
                        block["cells"] = table_result["cells"]
                    else:
                        block["cells"] = None

        # 逐 block 乱码回退
        if ocr_by_block is None:
            ocr_by_block = {}
        for idx, block in enumerate(blocks):
            if block.get("cells") or block.get("table_html"):
                continue
            if has_text_layer and is_pdf:
                text = extract_text_from_pdf(str(pdf_path), page_num, block["bbox"])
                if is_garbage_text(text):
                    log.info(f"[detect_all] 区域 #{idx}({block['label']}) 文字乱码，回退 OCR")
                    ocr_result = ocr_single_block(str(img_path), block)
                    if ocr_result:
                        ocr_by_block[idx] = ocr_result
                        block["text_fallback"] = "ocr"

        structured = build_structured(
            blocks, str(pdf_path) if is_pdf else None, page_num,
            str(img_path), has_text_layer, ocr_by_block
        )

        pages_result.append({
            "page": page_num, "width": w, "height": h,
            "blocks": blocks, "text_source": "pdf" if has_text_layer else "ocr",
            "structured": structured,
        })

    # 跨页表格合并
    merge_info = merge_cross_page_tables(pages_result)
    if merge_info:
        log.info(f"跨页表格合并完成: {len(merge_info)} 处")

    return {"doc_id": doc_id, "pages": pages_result, "is_pdf": is_pdf,
            "table_merges": merge_info if merge_info else None}


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str):
    doc_dir = UPLOAD_DIR / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3003)
