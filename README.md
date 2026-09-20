# EquipMind A1 设备检修知识检索与作业辅助系统

EquipMind A1 是面向制造业设备检修场景开发的 B/S 系统。系统将设备手册、检修案例、标准作业流程、故障图片和设备关系数据统一管理，为检修人员提供资料查询、故障初筛、作业指引、工单生成和知识维护功能。

未配置大模型服务时，系统仍可使用本地知识检索、图片案例匹配、SOP、工单和知识维护功能；配置 DeepSeek、通义千问、Ollama 或其他 OpenAI-compatible 服务后，可基于本地检索资料生成结构化建议。

## 主要功能

- 智能检修对话：结合多轮上下文检索设备手册、案例和 SOP，并展示引用来源。
- 智能检修问答：按设备、型号、故障现象和报警代码检索相关知识。
- 图文联合诊断：融合故障图片、设备类型和现场描述匹配案例，并联动文本知识库。
- 标准化作业指引：提供工具、防护用品、操作步骤、风险提示和合规检查。
- 智能工单生成：形成包含风险、依据、步骤、验收要求和签字栏的检修工单。
- 案例上传与审核：一线案例经审核后进入正式知识库并参与后续检索。
- 人工反馈与纠错：保留问题、回答、引用资料和修正内容，审核通过后再沉淀。
- 知识图谱与统计：展示设备、部件、故障、原因、参数和处置动作之间的关系。
- 大模型配置与知识库管理：支持模型连通性检查、资料导入和索引重建。

## Windows 运行

1. 安装 64 位 Python 3.10 或 3.11，并勾选 `Add Python.exe to PATH`。
2. 将项目解压到路径较短的目录，例如 `D:\EquipMind\equipmind_a1`。
3. 双击 `run_windows.bat`。
4. 浏览器访问 `http://127.0.0.1:8501`。

也可以在命令提示符中手动运行：

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python scripts\prepare_demo.py
python -m streamlit run app.py
```

## Linux / 银河麒麟运行

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_demo.py
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

## 大模型配置

```bash
python scripts/init_deepseek.py
python scripts/test_llm.py
```

模型接口不可用时，系统自动使用本地证据摘要模式，其他业务功能不受影响。真实 API Key 仅保存在本机 `.env` 中，不应提交到代码仓库或公开压缩包。

## 项目文档

`docs/` 目录仅保留以下五份正式 Word 文档：

- `软件功能需求分析文档.docx`
- `软件功能设计文档.docx`
- `软件产品说明书.docx`
- `软件功能测试报告.docx`
- `软件安装包及部署文档.docx`

## 检查与打包

```bash
python scripts/run_all_checks.py
python scripts/make_submission_package.py
```

打包脚本默认不包含 `.env`、虚拟环境、缓存文件和运行日志。

## 使用边界

系统输出属于检修辅助信息，不能替代企业制度、作业票、持证人员判断和现场负责人指令。证据不足时，应补充设备型号、报警代码、工况或测量值，并按企业安全制度执行停机、断电、泄压和挂牌等措施。
