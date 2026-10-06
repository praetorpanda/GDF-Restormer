# GDF 原脚本对照检查

参照文件：原工程根目录 `run_Ablation_GlobalDegField_4Level_6500_FullCanvas_FastVal_WarmStart.py`。
检查目标：`train_strict_global_degfield.py` 及其模型、引擎、数据和配置依赖。

## 结论

开源版保留原 runner 的训练参数传递和引擎计算流程，按作者此前指定的论文指标选择 Strict GDF。它并非原 GlobalDegField 模型的纯改名版本。未发现整理引入的训练计算差异；本次修正了继承自原 runner 的配对预检查问题，以及文档对多次启动日志的误读。

## 模型与结果对应

| 项目 | 指定原 runner | 当前开源 GDF |
| --- | --- | --- |
| 注册键 | `GlobalDegField_4L` | `StrictGlobalDegField_4L` |
| 模型类 | `Restormer_AblationGlobalDegField` | `Restormer_StrictGlobalDegField` |
| 模型实现 | `Res_Ablation` 包装 `Res_Global` / `Res_Psfbasis` | `Res_Strict` |
| 编码结构 | 四级 encoder，最深一级另有 latent | 三级 encoder，第四级仅作为 latent |
| 日志参数量 | 40,637,989 | 25,971,477 |
| 完整图像 PSNR / SSIM | 31.5882 / 0.8094 | **31.5631 / 0.8090** |

两类模型传入的 15 项配置值一致，包括 dim=48、blocks=[4,6,6,8]、heads=[1,2,4,8]、BiasFree、deg_ch=16、deg_mid_ch=32、退化场下采样倍率 4。但配置相同不代表拓扑相同：Strict 模型使用 `Resbase` 基础组件，原 PSF 模型使用 `Restormer` 基础组件，且编码阶段安排不同。应使用各自的 checkpoint，不能假定权重可直接互换。

模型与指标分别由 `result_Ablation/result_Ablation_GlobalDegField_4L_fixed6500_1729_unified.txt` 和 `result_Ablation/result_StrictGlobalDegField_4L_fixed6500_1729_unified.txt` 核对。上述参数量来自历史日志，本次未运行模型计数。

## 已核对的一致项

- 原与新 runner 的 32 项命令行参数定义在排除帮助文案后，表达式 AST 一致。模型名与默认输入/输出路径常量的变更单独列于下表。
- `train_variant_6500(...)` 的 50 项关键字实参及传参表达式 AST 完全一致。
- `COMMON_REG` 完全一致：deg smoothness=1e-4，其余启用权重与原 runner 一致。
- engine 去除模块说明并仅归一化明确的文案替换后，整棵 AST 一致。没有修改优化器、反向传播、调度器、验证计算、分块推理或 checkpoint 选择。
- `models/Res_Strict.py`、`models/Resbase.py`、`models/Res_Psfbasis.py`、`models/Restormer.py`、FullCanvas 数据文件、`config.yml`、配置类、loss 和 utils 文件与原工程逐字节一致。

关键运行设置：batch size=2（runner 显式传入，优先于配置文件中的 1）；100 epochs；AdamW，lr=2e-4，weight decay=0；CosineAnnealingLR，最低 lr=1e-6；随机种子 3407。模型输入为 meta，目标为 clean；虽然配置中的旧字段名称容易混淆，引擎实际处理方向未改变。损失为 L1 + 0.2×SSIMLoss 加原有先验正则。

## 路径与接口调整

| 项目 | 开源版处理 |
| --- | --- |
| 数据默认路径 | 原服务器绝对路径改为 `open_source/datasets`，可用 `--data_root` 指定原数据位置 |
| 固定划分 | 仍为 `split_6500_1729/train/{gt,meta}` 与 `val/{gt,meta}` |
| 结果日志 | 默认 `open_source/results/strict_global_degfield.txt` |
| checkpoint | 默认 `open_source/checkpoints/StrictGlobalDegField_4L/`；engine 仍会在 checkpoint_root 下追加 model_name |
| 导入 | 改为 `engines.engine_strict_global_degfield` 和 Strict 注册表，延迟到 main 内，以允许独立查看 `--help` |
| 配对预检查 | 本次修正为目录直属文件、完整文件名相同，与 FullCanvas Dataset 一致 |

原配对预检查递归扫描并忽略扩展名，因此 `gt/a.png` 与 `meta/a.jpg`、或嵌套目录下的同名图像都会被计入，但 Dataset 不会读取这些配对。已用临时目录验证修正后的计数：同名文件、不同扩展名、子目录和未配对文件均符合 Dataset 规则。有效的原始平铺数据集不受此修正影响。

原 fixed split manifest 仍只报告是否存在，不校验文件内容。6500/1729 数量检查不能替代原始划分一致性核验。

## 验证与 warm start 口径

- 训练从整个补边画布随机裁剪 256×256；默认快速验证为 center。
- 每 20 epochs 的周期 grid9 loader 使用 FullCanvas Dataset，分布在完整补边画布上。引擎原日志把它称为 RectROI-grid9，但实际采样不受 ROI_W/H 限制。
- 最终可选 grid9 评估使用引擎的 RectROI 路径（768×640），与周期 FullCanvas grid9 不是同一个口径。默认 center checkpoint 选择不依赖周期 grid9 分数。
- 默认完整图像评估开启，tile=256、overlap=32，采用最佳 center checkpoint；最终 grid9 默认关闭。
- 论文日志有两个启动头：第一次 `EVAL_GRID9=False`，第二次 `EVAL_GRID9=True`。第二次记录之后是最终结果，故日志没有先前文档所述的矛盾。要同时得到日志中的 full 与 grid9 汇总，使用 `--eval_grid9`。
- `--pretrained_ckpt=None` 为默认，论文日志也记录无 warm start。指定此参数时仅加载模型参数，优化器和调度器重建；不是完整断点续训。
- 权重加载沿用原引擎的 `strict=False`，缺失/多余键只输出提示；这不代表不同拓扑 checkpoint 可兼容，形状不匹配仍可能报错。

## 验证限制

已完成源码差异、AST、文件一致性、参数与日志对照、配对规则样例，以及三组入口的发布静态检查。当前环境没有 PyTorch 等运行依赖，未执行网络前向、真实数据 dry run、权重加载、GPU 训练或数值复现。未修改原工程文件或训练计算。
