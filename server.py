"""PP-StructureV3 版面结构化演示 — 仅做版面检测，不做 OCR
使用 HuggingFace transformers + PyTorch 加载 PP-DocLayoutV3 模型
对 PDF 中的表格区域，使用 PyMuPDF 提取单元格结构"""

import time
import uuid
import shutil
import logging
from pathlib import Path

import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import RTDetrImageProcessor, AutoModelForObjectDetection
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pp-struct")

app = FastAPI(title="PP-StructureV3 Demo")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── 版面检测模型（懒加载）──────────────────────────────────────────
_MODEL_NAME = "PaddlePaddle/PP-DocLayoutV3_safetensors"
_THRESHOLD = 0.35
_processor = None
_model = None


def _ensure_model():
    global _processor, _model
    if _model is not None:
        return
    t0 = time.time()
    log.info(f"正在加载 {_MODEL_NAME} ...")
    _processor = RTDetrImageProcessor.from_pretrained(_MODEL_NAME)
    _model = AutoModelForObjectDetection.from_pretrained(_MODEL_NAME)
    if torch.cuda.is_available():
        _model.to("cuda")
    _model.eval()
    log.info(f"模型加载完成: {time.time() - t0:.2f}s")


# ── PDF → 图片 + 保留原始 PDF ──────────────────────────────────────
def pdf_to_images(pdf_bytes: bytes, doc_dir: Path) -> list[str]:
    # 保存原始 PDF 供后续表格提取
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

    # 过滤掉页码
    blocks = [b for b in blocks if b["label"] != "number"]

    # 按置信度降序排列，NMS 去除重叠框（IoU > 0.5 保留高分的）
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
    """计算两个 bbox 的 IoU"""
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
    """从 PDF 中提取指定区域内的表格单元格。
    table_bbox 是图片坐标系下的 [x1,y1,x2,y2]，需要转换回 PDF 坐标。
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]

    # 图片是 2x 渲染的，PDF 坐标 = 图片坐标 / 2
    scale = 2.0
    pdf_rect = fitz.Rect(
        table_bbox[0] / scale, table_bbox[1] / scale,
        table_bbox[2] / scale, table_bbox[3] / scale,
    )

    tables = page.find_tables(clip=pdf_rect)
    cells = []
    for table in tables:
        rows_text = table.extract()
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell_rect in enumerate(row.cells):
                if cell_rect is None:
                    continue  # 合并单元格的非主格，跳过
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


# ── 提取标题文本（PyMuPDF）────────────────────────────────────────
def extract_title_text(pdf_path: str, page_num: int, bbox: list) -> str:
    """从 PDF 对应区域提取文字"""
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
    """清理文本中多余的空格（如"单 元"→"单元"）"""
    import re
    # 中文字符间的空格去掉
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    return text.strip()


def _split_value(text: str):
    """单值保持字符串，多行拆成数组"""
    text = _clean_text(text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines if len(lines) > 1 else (lines[0] if lines else "")


def build_structured(blocks: list, pdf_path: str | None, page_num: int) -> dict:
    """将检测结果转为干净的结构化数据"""
    content = []

    for block in blocks:
        label = block["label"]

        # 标题类
        if label in ("paragraph_title", "document_title", "title"):
            text = ""
            if pdf_path:
                text = extract_title_text(pdf_path, page_num, block["bbox"])
            if text:
                # 按行拆分，每行一个独立标题
                lines = [_clean_text(l) for l in text.split("\n") if l.strip()]
                for line in lines:
                    # 去重
                    if not any(c["type"] == "title" and c["text"] == line for c in content):
                        content.append({"type": "title", "text": line})

        # 表格
        elif label == "table" and block.get("cells"):
            cells = block["cells"]
            max_row = max(c["row"] for c in cells)
            max_col = max(c["col"] for c in cells)

            # 构建二维网格
            grid = [[None] * (max_col + 1) for _ in range(max_row + 1)]
            for c in cells:
                grid[c["row"]][c["col"]] = c["text"] or ""

            # 第一行做表头
            headers = [_clean_text(grid[0][c] or f"列{c+1}") for c in range(max_col + 1)]

            # 后续行：向上继承合并单元格的值
            rows = []
            prev = [""] * (max_col + 1)
            for r in range(1, max_row + 1):
                row_data = {}
                for c in range(max_col + 1):
                    val = grid[r][c]
                    if val is None or val == "":
                        val = prev[c]  # 继承上方合并单元格
                    else:
                        prev[c] = val
                    row_data[headers[c]] = _split_value(val)
                rows.append(row_data)

            content.append({"type": "table", "headers": headers, "rows": rows})

        # 其他文本类
        elif label == "text":
            text = ""
            if pdf_path:
                text = extract_title_text(pdf_path, page_num, block["bbox"])
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

    # 如果有原始 PDF，对 table 区域提取单元格
    pdf_path = doc_dir / "source.pdf"
    is_pdf = pdf_path.exists()

    for block in blocks:
        if block["label"] == "table":
            if is_pdf:
                block["cells"] = extract_table_cells(
                    str(pdf_path), page_num, block["bbox"], w, h
                )
            else:
                block["cells"] = None  # 图片模式，无法提取单元格

    # 构建结构化数据
    structured = build_structured(
        blocks, str(pdf_path) if is_pdf else None, page_num
    )

    return {"page": page_num, "width": w, "height": h, "blocks": blocks,
            "is_pdf": is_pdf, "structured": structured}


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

        for block in blocks:
            if block["label"] == "table":
                if is_pdf:
                    block["cells"] = extract_table_cells(
                        str(pdf_path), page_num, block["bbox"], w, h
                    )
                else:
                    block["cells"] = None

        pages_result.append(
            {"page": page_num, "width": w, "height": h, "blocks": blocks}
        )

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
