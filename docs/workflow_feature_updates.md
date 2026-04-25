# Workflow 开发跟踪

## 2026-04-01

### 已完成

- 新建 `WorkflowDocument` 架构参考文件
- 为前端增加 workflow 文档 schema 基础结构
- 为后端增加 workflow 文档 schema 基础结构
- 为 JSON 导入/导出增加新旧格式兼容层
- 将项目 JSON 导出升级为 `flowhamster_workflow.json`
- 新文档导入兼容旧版 `nodes/edges` 项目文件
- 将训练配置纳入 Zustand 全局状态
- 新增训练配置面板，支持任务类型、loss、optimizer、scheduler、device、epochs、batch size、lr 等参数
- 本地代码生成器开始读取训练配置，生成基础训练脚手架
- 侧边栏隐藏旧式 Loss / Optimizer / Scheduler 节点入口，开始迁移到配置范式
- 后端 `/api/generate` 与 `/api/export-notebook` 已接入 `training_config`
- 输入节点增加契约编辑能力，可直接维护 `name/shape/dtype`
- 训练脚手架生成已前后端对齐，并兼容多输出主分支选择
- 新增独立数据图基础设施：`dataGraph` store、节点注册表、独立 sidebar/canvas/preview
- 主界面已支持 `Model/Data` 工作区切换
- 工作流导入导出已开始携带 `dataGraph` 内容
- 新增 binding 面板，可把数据字段映射到模型输入和训练目标
- `bindings` 已纳入全局状态与工作流导入导出
- 工作流文档现在同时保存 `modelGraph / dataGraph / trainingConfig / bindings`

### 阶段备份

- `2026-04-01 00:57:36` → `/Users/Zhuanz1/Desktop/file/DeepForge/workspace/FlowHamster_backup_20260401_005736.tar.gz`
- `2026-04-01 01:01:31` → `/Users/Zhuanz1/Desktop/file/DeepForge/workspace/FlowHamster_backup_20260401_010131.tar.gz`
- `2026-04-01 01:07:10` → `/Users/Zhuanz1/Desktop/file/DeepForge/workspace/FlowHamster_backup_20260401_010710.tar.gz`

### 下一步

- 构建 data compiler 最小闭环
- 让 bindings 真正参与导出代码与运行时喂数
- 为数据图增加更严格的字段校验和契约检查

### 验证结果

- `node ./node_modules/typescript/bin/tsc --noEmit` ✅
- `python3.11 -m compileall backend` ✅
- `python3 test_robustness.py` ✅
- `python3 test_training_bugs.py` ✅
