"""PP-StructureV3 版面结构化演示
- 版面检测：PP-DocLayoutV3 (HuggingFace transformers + PyTorch)
- 文本提取：PyMuPDF（文本型 PDF）/ PP-OCRv5 ONNX（扫描件）
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


def _get_ocr():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    det = str(_OCR_DIR / "ppocr5_server_det.onnx")
    cls = str(_OCR_DIR / "ppocr5_cls.onnx")
    rec = str(_OCR_DIR / "ppocr5_server_rec.onnx")
    dic = str(_OCR_DIR / "ppocr5_dict.txt")

    if not Path(det).exists():
        log.warning("OCR 模型未找到，扫描件将无法识别文字")
        return None

    from ppocr5 import simple_ppocr5
    use_gpu = torch.cuda.is_available()
    _ocr_engine = simple_ppocr5(
        ppocr5_onnx_det=det, ppocr5_onnx_cls=cls,
        ppocr5_onnx_rec=rec, ppcor5_dict=dic, use_gpu=use_gpu,
    )
    log.info(f"OCR 模型加载完成 (GPU={use_gpu})")
    return _ocr_engine


def ocr_image(image_path: str) -> str:
    """对图片进行 OCR，返回全文"""
    engine = _get_ocr()
    if engine is None:
        return ""
    engine.run(image_path)
    if not engine.results:
        return ""
    # 按 y 坐标排序，拼接文本
    items = sorted(engine.results, key=lambda r: (r["rec_pos"][0][1], r["rec_pos"][0][0]))
    return "\n".join(r["text"] for r in items)


def ocr_region(image_path: str, bbox: list) -> str:
    """对图片中指定区域进行 OCR"""
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return ""
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    engine = _get_ocr()
    if engine is None:
        return ""
    engine.run(crop)
    if not engine.results:
        return ""
    items = sorted(engine.results, key=lambda r: (r["rec_pos"][0][1], r["rec_pos"][0][0]))
    return "\n".join(r["text"] for r in items)


# ── 判断页面是否有文字层 ─────────────────────────────────────────
def page_has_text(pdf_path: str, page_num: int) -> bool:
    """判断 PDF 某页是否有可用的文字层（非乱码）"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    text = page.get_text("text").strip()
    doc.close()

    if not text or len(text) < 5:
        return False

    # 检查乱码：如果大量不可识别字符，视为无效
    printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
    ratio = printable / len(text) if text else 0
    return ratio > 0.5


@app.on_event("startup")
async def startup_event():
    threading.Thread(target=_load_model, daemon=True).start()


@app.get("/api/model-status")
async def model_status():
    ocr_available = (_OCR_DIR / "ppocr5_server_det.onnx").exists()
    return {"status": _model_status, "error": _model_error, "ocr_available": ocr_available}


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


# ── 文本提取（自动选择 PyMuPDF 或 OCR）───────────────────────────
def extract_text_from_region(pdf_path: str | None, page_num: int,
                             bbox: list, image_path: str,
                             has_text_layer: bool) -> str:
    """从指定区域提取文字。有文字层用 PyMuPDF，否则用 OCR。"""
    if has_text_layer and pdf_path:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        scale = 2.0
        rect = fitz.Rect(bbox[0] / scale, bbox[1] / scale,
                         bbox[2] / scale, bbox[3] / scale)
        text = page.get_text("text", clip=rect).strip()
        doc.close()
        return text
    else:
        return ocr_region(image_path, bbox)


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
                     image_path: str, has_text_layer: bool) -> dict:
    content = []

    for block in blocks:
        label = block["label"]

        # 标题类
        if label in ("paragraph_title", "document_title", "title"):
            text = extract_text_from_region(
                pdf_path, page_num, block["bbox"], image_path, has_text_layer
            )
            if text:
                lines = [_clean_text(l) for l in text.split("\n") if l.strip()]
                for line in lines:
                    if not any(c["type"] == "title" and c["text"] == line for c in content):
                        content.append({"type": "title", "text": line})

        # 表格
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

        # 表格（扫描件，无单元格但可 OCR 整块）
        elif label == "table" and block.get("cells") is None:
            text = extract_text_from_region(
                pdf_path, page_num, block["bbox"], image_path, has_text_layer
            )
            if text:
                content.append({"type": "table_text", "text": _clean_text(text),
                                "note": "扫描件表格，仅提取文字，无法解析单元格结构"})

        # 文本类（text, content, abstract, header 等）
        elif label in ("text", "content", "abstract", "reference",
                        "header", "footer", "algorithm",
                        "figure_title", "table_title"):
            text = extract_text_from_region(
                pdf_path, page_num, block["bbox"], image_path, has_text_layer
            )
            if text:
                content.append({"type": "text", "text": _clean_text(text)})

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
        str(img_path), has_text_layer
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
            str(img_path), has_text_layer
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
