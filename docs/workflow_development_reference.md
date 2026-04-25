# FlowHamster Workflow 开发参考

## 目标边界

本阶段采纳以下设计原则：

1. **模型网络结构** 使用节点图构建，负责表达 `nn.Module` 与 `forward` 逻辑。
2. **数据加载与预处理** 使用独立节点图构建，负责表达数据读取、拆包、解析、清洗、变换、批处理。
3. **损失函数 / 优化器 / 调度器 / Trainer** 不使用节点图，统一收敛到配置面板。
4. **模型图与数据图完全解耦**，仅通过输入/输出契约与绑定关系关联，不使用跨图硬连线。
5. **前端画布不再直接拼接 Python**，统一先编译到 `WorkflowDocument / WorkflowIR`，后端再生成代码。

---

## 当前项目判断

当前 FlowHamster 已具备：

- 模型节点图编辑
- 分组、多选、复制粘贴
- 基础代码生成
- 后端 FastAPI 执行与预览
- 可读性更好的变量命名

但当前仍主要停留在“模型子图 -> Python 脚本”的层级，距离完整 workflow 还缺少：

- 数据源管理
- 数据预处理流水线
- 数据图与模型图绑定
- 训练配置面板
- 运行目录、日志、checkpoint、恢复机制
- 真正可导出的完整训练项目结构

---

## 目标架构

### 1. WorkflowDocument

统一文档结构：

- `modelGraph`
- `dataGraph`
- `trainingConfig`
- `bindings`
- `metadata`

### 2. ModelGraph

只负责模型结构：

- Input / Output
- Conv / Linear / Attention / Merge / Reshape 等

生成目标：

- `model.py`

### 3. DataGraph

只负责数据读取与预处理：

- Source
- Read
- Parse
- Unpack
- Map
- Filter
- SelectFields
- Rename
- Split
- Shuffle
- Batch
- Collate

生成目标：

- `dataset.py`
- `transforms.py`
- `dataloader.py`

### 4. TrainingConfig

统一配置：

- task type
- loss
- optimizer
- scheduler
- metrics
- device
- epoch
- batch size
- amp
- gradient clip
- checkpoint / early stop

生成目标：

- `config.yaml`
- `trainer.py`

### 5. Bindings

用于连接数据图和模型图：

- `data.output.image -> model.input.image`
- `data.output.label -> training.target.label`

生成目标：

- `runner.py` 中的数据解包与喂入逻辑

---

## 分阶段计划

### 阶段 1：文档模型与迁移

目标：

- 建立 `WorkflowDocument`
- 保持旧版 `nodes/edges` JSON 可导入
- 建立前后端共享的 schema 概念

交付：

- 新文档类型
- 旧版迁移器
- 导入/导出兼容层

测试：

- 旧 JSON 导入成功
- 新 JSON 导出可重复导入
- 缺字段 / 非法字段容错

### 阶段 2：训练配置面板

目标：

- 用配置替代训练节点

交付：

- `TrainingConfig` 数据结构
- 训练配置 UI
- loss / optimizer / scheduler 默认模板

测试：

- 缺省配置可正常生成
- 非法超参可提示
- 不同 task type 切换稳定

### 阶段 3：模型图契约化

目标：

- 为模型图 input / output 增加稳定契约

交付：

- 输入输出名字、dtype、shape hint、semantic role
- 模型导出只依赖模型图

测试：

- 多输入 / 多输出命名稳定
- 契约缺失时可检测

### 阶段 4：数据图最小闭环

目标：

- 先打通最小可运行数据流

首批节点：

- `folder_source`
- `csv_source`
- `read_image`
- `read_lines`
- `unpack`
- `select_fields`
- `rename_fields`
- `map_transform`
- `train_val_split`
- `shuffle`
- `batch`
- `collate`

测试：

- 路径不存在
- 字段不存在
- batch 维度不匹配
- 多源字段冲突

### 阶段 5：绑定层

目标：

- 让数据图输出和模型输入解耦绑定

交付：

- 绑定配置界面
- 类型/字段校验
- 绑定导出到运行层

测试：

- 未绑定输入报错
- label 绑定错误报错
- 重名字段冲突提示

### 阶段 6：编译器拆分

目标：

- 从 `graph -> code` 升级为 `graph -> IR -> code`

交付：

- model compiler
- data compiler
- runtime compiler

测试：

- IR round-trip
- 编译顺序稳定
- 部分图缺失时错误可读

### 阶段 7：运行时与产物

目标：

- 从临时脚本执行升级为 run-based runtime

交付：

- `runs/<run_id>/`
- 配置快照
- stdout/stderr 日志
- metrics
- checkpoints

测试：

- 中断恢复
- 超时
- OOM
- checkpoint 恢复

### 阶段 8：导出完整工程

目标：

- 导出后可直接训练

交付目录：

- `model.py`
- `dataset.py`
- `transforms.py`
- `trainer.py`
- `runner.py`
- `config.yaml`
- `requirements.txt`

测试：

- 冷启动运行
- 无 GUI 环境运行
- 导出后二次迁移

---

## 当前推荐执行顺序

1. `WorkflowDocument + 迁移器`
2. `TrainingConfig` 固化
3. 模型图契约化
4. 数据图最小闭环
5. 绑定层
6. 编译器拆分
7. 运行时目录化
8. 导出完整工程

---

## 设计约束

- 禁止继续把所有新能力塞进旧 `params`
- 禁止继续直接从画布对象拼接最终运行脚本
- 模型图、数据图、训练配置必须分层
- 所有新功能必须带最小鲁棒性校验
- 所有导出文件命名必须可读、稳定、可追踪

---

## 当前已开始落地的内容

- 新增 `WorkflowDocument` 基础 schema
- 新增旧版项目 JSON -> 新文档格式迁移入口
- 升级项目 JSON 导入导出逻辑，面向后续完整 workflow 兼容

