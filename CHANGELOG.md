# Changelog

## 2026-04-16

### fix: 双栏布局阅读顺序修正
- 新增 `_sort_reading_order()` 函数，将 y 坐标相近（≤30px）的 block 分为同一行，行内按 x 排序
- 修复双栏目录中右列排在左列前面的问题（右列 y1 略小于左列 y1 导致排序错误）

### feat: 全部检测 + 跨页表格合并前端
- 新增「全部检测」按钮，一键检测所有页面并触发跨页表格合并
- 逐页检测进度条实时显示进度
- 检测完成后显示合并结果摘要（合并数量、涉及页面）
- 跨页合并的表格在可视化视图中标注"跨页合并表格（来自第 X、Y 页）"
- 被合并走的续表标注"此表格已合并至前页的表格中"
- 新增「全文档」JSON 视图 tab，展示所有页结构化数据的完整 JSON
- 状态缓存到 localStorage，刷新不丢失

### feat: 跨页表格合并
- 新增 `merge_cross_page_tables()` 后处理函数，在 `detect_all()` 所有页面处理完后自动检测并合并续表
- 滑动指针算法：支持三页及以上的连续表格合并
- 智能表头去重：续表第一行与前页表头相似度 >80% 时自动跳过
- 列结构匹配：通过列数 + 列宽比例判断是否为同一表格（容差 15%）
- 位置约束：仅合并"页面下半部表格 → 下页上半部表格"的情况，避免误合并
- 合并后自动重建 structured JSON，续表所在页的表格内容被移除
- API 返回 `table_merges` 字段记录合并详情
- 辅助函数：`_table_to_grid`、`_row_similarity`、`_col_widths_ratio`、`_col_structure_similar`

### feat: 乱码检测 + OCR 智能回退
- 新增 `is_garbage_text()` 函数，检测 CID 字体解码失败、PUA 字符、替换字符等乱码特征
- `page_has_text()` 增强：文字层提取出乱码时自动判定为扫描件，整页回退 OCR
- 逐 block 乱码回退：文字型 PDF 中个别区域乱码时，仅对该区域启动 OCR，不影响其他正常区域
- 表格乱码回退：PyMuPDF 提取的表格单元格超过 30% 乱码时，自动回退 PP-TableMagic V2
- `build_structured()` 的 `get_text()` 优先使用 OCR 回退结果，确保结构化输出正确
- 修复 `detect_all()` 调用未定义函数 `ocr_full_page` / `assign_ocr_to_blocks` 的 bug
- `detect_all()` 同步补齐扫描件表格识别（之前只设 `cells=None`，现在调用 PP-TableMagic V2）

## 2026-04-11

### feat: 可视化面板显示所有区域文字内容 + 扫描件表格渲染
- 所有文字区域（text/content/header/title 等）在可视化面板直接显示提取到的文字
- 扫描件表格展开后渲染 HTML 表格（RapidTable 识别结果）

### fix: 修复 RapidTable API 调用格式
- ocr_results 参数改为列表格式，输出属性改为 batch 格式

### feat: 集成 RapidTable 表格结构识别
- 新增 RapidTable (SLANet_plus) 表格结构识别，纯 ONNX 推理
- 扫描件中的表格：PP-OCRv5 识别文字 + SLANet_plus 识别行列结构 → 输出 HTML
- 结构化 JSON 新增 `table_html` 类型

### feat: OCR 全面优化
- OCR 改为按版面区域裁剪后逐块识别，解决双栏混排问题
- 目录自动检测：识别"第X章/节"模式，拆分标题和页码
- 纯页码行/引导线残行自动合并回前一个标题行
- GPU 自动用 server 版模型，CPU 自动用 mobile 版
- OCR 参数优化：limit_side_len=1920, det_db_box_thresh=0.5, unclip_ratio=1.8
- 可视化：目录条目独立覆盖层 + 双向点击高亮
- 可视化：文字来源标签（PDF 文字层 / OCR 识别）检测前即显示
- 顶部显示 OCR 模型类型（mobile/server）
- 新增 `/api/text-source` 轻量接口

### feat: 集成 PP-OCRv5 ONNX
- 新增 PP-OCRv5 ONNX 推理引擎（来自 simple_ppocr5）
- 逐页判断 PDF 是否有文字层：有文字走 PyMuPDF，无文字走 OCR
- 支持区域级 OCR，扫描件表格输出 `table_text` 类型
- 返回 `text_source` 字段标识文字来源

## 2026-04-10

### feat: 模型本地化
- 首次启动从 HuggingFace 下载模型并保存到 `./model/`，之后离线加载（0.25s）
- 拷贝整个项目目录即可分发

### feat: 启动时预加载模型 + 镜像源说明
- 服务启动时后台线程自动下载/加载模型
- 页面顶部显示模型状态（加载中/就绪/失败）
- README 添加 `HF_ENDPOINT` 镜像源配置说明

### fix: 表格裁剪加 padding 防丢列 + 纯标点行合并
- `extract_table_cells` 的 clip 区域四周扩大 10pt，防止 bbox 偏小丢失列
- `_split_value` 将纯标点行（如"。"）合并回上一行
- 中间栏和右侧栏默认 flex:1 各占一半

### feat: 结构化 JSON 输出
- 生成干净的结构化数据：标题按行拆分去重，表格用表头做 key
- 多行值（如要点列）拆为数组，单值保持字符串
- 合并单元格自动向上继承
- 中文字符间多余空格自动清理
- JSON 视图切换（可视化 / JSON）

### feat: 表格单元格提取 + 双向联动高亮
- PyMuPDF `find_tables()` 提取单元格级结构（行、列、bbox、文字）
- 表格支持「网格」和「树形」两种视图切换
- 树形视图：按列层级展开，多行值逐行配对标签
- 双向点击高亮：右侧结果 ↔ 图片区域联动定位
- 结果缓存到 localStorage，刷新不丢失
- 右侧面板宽度可拖动

### init: PP-StructureV3 版面结构化分析演示
- PP-DocLayoutV3 版面检测（HuggingFace transformers + PyTorch）
- PDF 上传 + 渲染 + 版面区域标注
- 三栏布局：页面列表 | 图片预览 | 检测结果
