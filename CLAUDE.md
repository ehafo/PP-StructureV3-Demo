# CLAUDE.md — PP-StructureV3 Demo

## 项目概述

PDF 文档版面结构化分析工具。上传 PDF → 版面检测 → 提取标题/表格/正文 → 输出结构化 JSON。
单文件架构：`server.py`（FastAPI 后端）+ `index.html`（原生 JS 前端）。

## 架构与模型管线

```
PDF 上传
  ├─ PyMuPDF 渲染页面为 PNG
  ├─ PP-DocLayoutV3 (PyTorch, GPU) 版面检测 → blocks [{bbox, label}]
  │
  ├─ 文本型 PDF：PyMuPDF 直接提取文字/表格（毫秒级）
  └─ 扫描件 PDF：
       ├─ PP-OCRv5 ONNX (onnxruntime) 全页 OCR → 按区域归类
       └─ 表格区域 → PP-TableMagic V2 (PaddlePaddle) 表格结构识别
                      ├─ 表格分类（有线/无线自动区分，PP-LCNet）
                      ├─ 单元格检测 (RT-DETR-L)
                      ├─ 结构识别 (SLANeXt wired/wireless)
                      └─ 文字检测+识别 (PP-OCRv4 server)
                      降级方案：RapidTable (SLANet_plus ONNX)
```

## 关键文件

- `server.py` — 全部后端逻辑（版面检测、OCR、表格识别、API）
- `index.html` — 全部前端（上传、可视化、双向联动、表格/树形视图）
- `model/` — PP-DocLayoutV3 模型缓存（首次启动自动下载）
- `model/ocr/` — PP-OCRv5 ONNX 模型缓存
- `~/.paddlex/official_models/` — PP-TableMagic V2 子模型缓存（8 个模型）

## 运行方式

```bash
source .venv/bin/activate
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True uvicorn server:app --host 0.0.0.0 --port 3003 --reload
```

环境变量 `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` 跳过 PaddleX 启动时的网络连通性检查（慢）。

## 依赖版本关键约束

### PaddlePaddle 版本问题（重要教训）

- **paddlepaddle-gpu 3.x 尚未公开发布**（截至 2026-04），只有 CPU 版 `paddlepaddle 3.x`
- **paddlepaddle-gpu 2.6.2** 与 **paddlex 3.4.x** 不兼容：
  - Python 层：`AnalysisConfig.set_optimization_level` 不存在（3.x 新增 API）
  - C++ 层：即使 monkey-patch Python 层，推理时 segfault
- **paddlepaddle (CPU) 3.0.0** 可以正常运行 paddlex 3.4.3 / paddleocr 3.4.0
- **paddlepaddle (CPU) 3.3.1** 有 oneDNN + PIR 的 bug（`ConvertPirAttribute2RuntimeAttribute` NotImplementedError），3.0.0 没有此问题

**当前方案**：`paddlepaddle==3.0.0` (CPU) + `paddleocr==3.4.0` + `paddlex[ocr]==3.4.3`
- 版面检测（PyTorch）和 OCR（ONNX）仍走 GPU (RTX 4090)
- 仅表格结构识别走 Paddle CPU 推理

### PP-TableMagic V2 初始化参数

```python
TableRecognitionPipelineV2(
    device="cpu",
    use_layout_detection=False,       # 已有 PP-DocLayoutV3，关闭 V2 内置版面检测
    use_doc_orientation_classify=False, # 裁切后的表格区域不需要
    use_doc_unwarping=False,           # 裁切后的表格区域不需要
)
```

**为什么关闭 `use_layout_detection`**：我们先用 PP-DocLayoutV3 检测表格位置并裁切，再传给 V2。如果 V2 再次做版面检测，会对裁切后的图片二次检测，导致表格结构错乱（列混淆、表头丢失）。

**为什么关闭 `use_doc_orientation_classify` 和 `use_doc_unwarping`**：同理，裁切后的表格区域方向已确定，不需要文档级预处理。如果未来改为传全页图给 V2，可以考虑重新开启。

## V2 输出结构

```python
res.json = {
    "res": {
        "table_res_list": [{
            "cell_box_list": [[x1,y1,x2,y2], ...],    # RT-DETR-L 检测的单元格框
            "pred_html": "<html><body><table>...</table>",  # 含 rowspan/colspan 的 HTML
            "table_ocr_pred": {
                "rec_texts": ["单元", "细目", ...],    # 每行文字
                "rec_boxes": [[x1,y1,x2,y2], ...],    # 每行文字坐标
                "rec_scores": [0.99, ...],
                "rec_polys": [[[x,y], ...], ...],
            }
        }],
        "layout_det_res": {...},
        "overall_ocr_res": {...},
    }
}
res.html = {"table_1": "<html>...</html>"}  # 便捷访问
```

### 如何恢复单元格内换行

pred_html 不保留换行。用 `table_ocr_pred.rec_texts` + `rec_boxes` 按 Y 坐标排序，同一单元格内多行文字用 `\n` 连接。已实现。

### 合并单元格

`pred_html` 包含 `rowspan`/`colspan` 属性，`_parse_html_table` 已在内部使用（计算 row/col 位置），但**当前未输出到 cell 数据中**。如需前端渲染合并效果，需要：
1. `_parse_html_table` 输出 `rowspan`/`colspan` 字段
2. 前端 `renderGrid` 跳过被占用的格子，`<td>` 加 `rowspan`/`colspan` 属性

代码已写过一次但用户要求撤回，可随时恢复。

## 乱码检测与 OCR 回退机制

三层防御：
1. **页级**：`page_has_text()` 调用 `is_garbage_text()` 检测全页文字质量，整页乱码直接走 OCR 路径
2. **Block 级**：文字型 PDF 逐区域检查提取文字，乱码区域单独 OCR（`ocr_single_block()`），结果写入 `ocr_by_block`
3. **表格级**：PyMuPDF 提取的表格单元格超过 30% 乱码，回退 PP-TableMagic V2

`build_structured()` 的 `get_text()` 优先读 `ocr_by_block`（即使 `has_text_layer=True`），确保回退结果被使用。

`is_garbage_text()` 检测规则：U+FFFD 替换字符 >10%、PUA 字符 >15%、有意义字符 <30%。

## 开发注意事项

- 服务启动时后台线程初始化三个引擎：版面模型、OCR 模型、表格引擎
- 表格引擎优先用 PP-TableMagic V2，import 失败或初始化失败时降级为 RapidTable
- 前端状态栏轮询 `/api/model-status`，表格引擎就绪后显示 "表格: PP-TableMagic V2"
- `uvicorn --reload` 开发模式下文件改动自动重载
- 前端 `.cells-table td` 需要 `white-space: pre-line` 才能渲染 `\n` 换行
