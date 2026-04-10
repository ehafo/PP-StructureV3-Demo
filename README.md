# PP-StructureV3 Demo

基于 PP-DocLayoutV3 的 PDF 文档版面结构化分析工具。上传 PDF 后自动检测版面区域（标题、表格、文本等），并对表格提取单元格级结构，输出干净的结构化 JSON 数据。

**不做 OCR**，仅做版面检测 + PDF 原生文本提取。

## 功能

- **版面检测**：使用 HuggingFace 上的 PP-DocLayoutV3 模型，识别标题、表格、文本、公式等区域
- **表格结构提取**：对 PDF 中的表格，使用 PyMuPDF 提取单元格级别的行列结构和文本内容
- **双向联动**：点击右侧结果列表高亮图片对应区域，点击图片区域定位到右侧结果
- **多种展示**：表格支持「网格」和「树形」两种视图切换
- **结构化 JSON 输出**：生成干净的结构化数据，标题按行拆分，表格要点/要求按 `\n` 拆为数组，合并单元格自动继承
- **结果缓存**：检测结果缓存到 localStorage，刷新不丢失

## 技术栈

| 组件 | 技术 |
|------|------|
| 版面检测模型 | [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) (HuggingFace transformers + PyTorch) |
| PDF 处理 | PyMuPDF (页面渲染 + 表格提取) |
| 后端 | FastAPI |
| 前端 | 单文件 HTML (原生 JS，无框架依赖) |

## 快速开始

```bash
# 创建虚拟环境（需要 Python 3.11）
python3.11 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动
python server.py
```

访问 http://localhost:3001

启动时会自动从 HuggingFace 下载 PP-DocLayoutV3 模型（约 200MB），页面顶部会显示加载状态。下载完成后显示「模型就绪」即可使用。

### 国内用户加速下载

如果 HuggingFace 访问缓慢，可以使用镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python server.py
```

## 结构化输出示例

```json
{
  "page": 1,
  "content": [
    { "type": "title", "text": "口腔内科学考试大纲" },
    { "type": "title", "text": "基础知识" },
    {
      "type": "table",
      "headers": ["单元", "细目", "要点", "要求"],
      "rows": [
        {
          "单元": "一、牙体组织",
          "细目": "1．釉质",
          "要点": ["⑴ 理化特性", "⑵ 组织学特点", "⑶ 临床意义"],
          "要求": "掌握"
        }
      ]
    }
  ]
}
```

## 项目结构

```
├── server.py          # FastAPI 后端（版面检测 + 表格提取 + 结构化输出）
├── index.html         # 单文件前端（上传 + 可视化 + 双向联动）
├── requirements.txt   # Python 依赖
└── .gitignore
```
