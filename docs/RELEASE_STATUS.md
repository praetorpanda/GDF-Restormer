# 整理与验证状态

训练流程只保留论文 GDF、基础 PSFDegField 和标准 G5 Wavelet-LL MoE 三组。其他早期结构与第三方模型仅保留 `models/` 实现。没有顶层 `train.py`、`run_*.py` 或 `engines/training.py` 别名。

## 主要调整

- 论文模型入口为 `train_strict_global_degfield.py`，独立引擎为 `engines/engine_strict_global_degfield.py`，直接使用原 `models/Res_Strict.py`。
- PSFDegField 保留基础版本，Wave 保留 G5 LL、hidden channels 16 的配置；不保留 V2Light、低秩 basis、G5Light、LLHF 或其他 MoE 训练入口。
- 已移除其余 14 组入口与 engine，以及无关的 RGB/meta 数据管线、退化生成工具和历史指标汇总。论文日志继续保留。
- 模型库的完整历史依赖表移入 `models/requirements-reference.txt`，当前依赖表补充 PSFDegField engine 使用的 `ptflops`。
- 原始源码文件不因整理而修改网络实现；原始来源、目标文件和摘要见 `source_manifest.json`。其中 `open_source/` 来源表示上一版整理时保留的日志或说明快照。
- 对照原 GDF runner 后修正开源入口的配对预检查：只统计目录直属、文件名和扩展名一致的配对，与 FullCanvas Dataset 的读取规则一致。详见 `GDF_AUDIT.md`。

## 验证范围

`tools/check_release.py` 检查所有 Python 文件语法、三组入口/engine 命名与导入、单模型选择、模型符号是否存在，以及所有入口的 `--help`。源码来源清单已同步删除不再保留的文件。

当前可用 Python 为 3.13，未安装 PyTorch 等运行依赖，因此没有进行模型前向、GPU 训练、旧模型运行适配或数值复现。静态检查通过不等同于所有历史配置都能在新环境中直接运行。

## 复现与发布材料

FullCanvas 原始数据模块已包含。仍需准备原始数据、固定划分清单、论文 checkpoint，并在论文环境核对参数量、权重严格加载与完整图像指标。日志参数量为 25,971,477，本次未运行参数计数。Strict 原始 runner 不在源工程中，论文入口由可用的 GlobalDegField runner 与 Strict 注册配置整合，应结合原实验环境核对。

正式名称、论文作者与引用、适用许可证尚需作者补充；原 README 的 UWFormer 信息作为历史来源保存，不代表当前论文信息。
