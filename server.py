"""PP-StructureV3 版面结构化演示
- 版面检测：PP-DocLayoutV3 (HuggingFace transformers + PyTorch)
- 文本提取：PyMuPDF（文本型 PDF）/ PP-OCRv5 ONNX（扫描件，全页 OCR + 按区域归类）
- 表格提取：PyMuPDF"""

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


def _get_ocr():
    global _ocr_engine, _ocr_model_type
    if _ocr_engine is not None:
        return _ocr_engine

    cls = str(_OCR_DIR / "ppocr5_cls.onnx")
    dic = str(_OCR_DIR / "ppocr5_dict.txt")
    use_gpu = torch.cuda.is_available()

    # GPU 用 server 版（精度高），CPU 用 mobile 版（更稳定）
    if use_gpu and (_OCR_DIR / "ppocr5_server_det.onnx").exists():
        det = str(_OCR_DIR / "ppocr5_server_det.onnx")
        rec = str(_OCR_DIR / "ppocr5_server_rec.onnx")
        _ocr_model_type = "server"
        log.info("使用 PP-OCRv5 server 版模型 (GPU)")
    elif (_OCR_DIR / "ppocr5_mobile_det.onnx").exists():
        det = str(_OCR_DIR / "ppocr5_mobile_det.onnx")
        rec = str(_OCR_DIR / "ppocr5_mobile_rec.onnx")
        _ocr_model_type = "mobile"
        log.info("使用 PP-OCRv5 mobile 版模型 (CPU)")
    elif (_OCR_DIR / "ppocr5_server_det.onnx").exists():
        det = str(_OCR_DIR / "ppocr5_server_det.onnx")
        rec = str(_OCR_DIR / "ppocr5_server_rec.onnx")
        _ocr_model_type = "server"
        log.info("使用 PP-OCRv5 server 版模型 (fallback)")
    else:
        log.warning("OCR 模型未找到，扫描件将无法识别文字")
        return None

    from ppocr5 import simple_ppocr5
    _ocr_engine = simple_ppocr5(
        ppocr5_onnx_det=det, ppocr5_onnx_cls=cls,
        ppocr5_onnx_rec=rec, ppcor5_dict=dic, use_gpu=use_gpu,
    )
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
        # 稍微扩大裁剪范围
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
        lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
        result[idx] = lines
        log.info(f"  区域 #{idx}({label}) OCR: {len(lines)} 行")

    return result


# ── 目录行解析 ────────────────────────────────────────────────────
def _parse_toc_line(line: str) -> dict | None:
    """解析目录行，拆分标题和页码。
    例: '第一章 牙体组织……1' → {"title": "第一章 牙体组织", "page": 1}
    """
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
def page_has_text(pdf_path: str, page_num: int) -> bool:
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    text = page.get_text("text").strip()
    doc.close()

    if not text or len(text) < 5:
        return False

    printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    ratio = printable / len(text) if text else 0
    return ratio > 0.5


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_load_model, daemon=True).start()


@app.get("/api/model-status")
async def model_status():
    ocr_available = (_OCR_DIR / "ppocr5_server_det.onnx").exists() or (_OCR_DIR / "ppocr5_mobile_det.onnx").exists()
    if _ocr_model_type:
        ocr_model = _ocr_model_type
    elif torch.cuda.is_available() and (_OCR_DIR / "ppocr5_server_det.onnx").exists():
        ocr_model = "server"
    elif (_OCR_DIR / "ppocr5_mobile_det.onnx").exists():
        ocr_model = "mobile"
    elif (_OCR_DIR / "ppocr5_server_det.onnx").exists():
        ocr_model = "server"
    else:
        ocr_model = ""
    return {"status": _model_status, "error": _model_error, "ocr_available": ocr_available, "ocr_model": ocr_model}


# ── PDF → 图片 + 保留原始 PDF ──────────────────────────────────────
def pdf_to_images(pdf_bytes: bytes, doc_dir: Path) -> list[str]:
    (doc_dir / "source.pdf").write_bytes(pdf_bytes)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    filenames = []
    matrix = fitz.Matrix(2.0, 2.0)
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=matrix)
        name = f"page_{i:03d}.png"
        pix.save(str(doc_dir / name))
        filenames.append(name)
    doc.close()
    return filenames


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

    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    for i, b in enumerate(blocks):
        b["order"] = i

    log.info(f"检测到 {len(blocks)} 个区域")
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
        """获取区域文字：有文字层用 PyMuPDF，否则从全页 OCR 结果取"""
        if has_text_layer and pdf_path:
            return extract_text_from_pdf(pdf_path, page_num, bbox)
        elif ocr_by_block and block_idx in ocr_by_block:
            return "\n".join(it["text"] for it in ocr_by_block[block_idx])
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

        # 表格（扫描件，无单元格）
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
        filenames = pdf_to_images(content, doc_dir)
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
    path = UPLOAD_DIR / doc_id / filename
    if not path.exists():
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
    name = f"page_{page_num:03d}.png"
    img_path = doc_dir / name
    if not img_path.exists():
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
                block["cells"] = extract_table_cells(
                    str(pdf_path), page_num, block["bbox"], w, h
                )
            else:
                block["cells"] = None

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

    pages_result = []
    for img_path in sorted(doc_dir.glob("page_*.png")):
        page_num = int(img_path.stem.split("_")[1])
        blocks = detect_layout(str(img_path))
        with Image.open(img_path) as img:
            w, h = img.size

        has_text_layer = False
        if is_pdf:
            has_text_layer = page_has_text(str(pdf_path), page_num)

        ocr_by_block = None
        if not has_text_layer:
            ocr_lines = ocr_full_page(str(img_path))
            ocr_by_block = assign_ocr_to_blocks(ocr_lines, blocks)

        for block in blocks:
            if block["label"] == "table":
                if has_text_layer:
                    block["cells"] = extract_table_cells(
                        str(pdf_path), page_num, block["bbox"], w, h
                    )
                else:
                    block["cells"] = None

        structured = build_structured(
            blocks, str(pdf_path) if is_pdf else None, page_num,
            str(img_path), has_text_layer, ocr_by_block
        )

        pages_result.append({
            "page": page_num, "width": w, "height": h,
            "blocks": blocks, "text_source": "pdf" if has_text_layer else "ocr",
            "structured": structured,
        })

    return {"doc_id": doc_id, "pages": pages_result, "is_pdf": is_pdf}


@app.delete("/api/documents/{doc_id}")
async def delete_doc(doc_id: str):
    doc_dir = UPLOAD_DIR / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
