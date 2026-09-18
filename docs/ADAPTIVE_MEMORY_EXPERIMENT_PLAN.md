# Adaptive memory 替代 SlowFast：实验与 GPU 前准备计划

更新日期：2026-09-18。状态：**模型与 annotations 已准备、CPU 预检已执行；缺 MP3D 场景，尚未运行导航实验。**

本轮执行证据见 [资产准备状态报告](NAV_ASSET_STATUS_20260918.md)，使用方法见
[CPU-only 资产准备](NAV_ASSET_PREPARATION.md)。这不代表模型推理或 Habitat 已跑通。

本文是 `juexZZ/LightNav-0` 研究 fork 的执行依据。上游 README 的结果是作者报告，
不是本 fork 已复现的结果。更新任务状态时必须记录证据，不能把“脚本已写好”标成
“训练／评测已验证”。

## 1. 已确认的研究问题与边界

研究问题：**用 adaptive memory 替代 LightNav 的 SlowFast 固定历史压缩规则，
是否能在相近资源预算下提高导航效果，或在相近效果下降低成本？**

- 第一优先级是复现原始 LightNav 在 **R2R / RxR 的 VLN-CE 连续环境版本**上的表现。
- 随后替换历史压缩机制；不是在原有完整 SlowFast 历史之上额外拼接 memory。
- 第一训练阶段只训练新 memory 模块及其内部必要的读出／投影层。
  冻结视觉编码器、LightNav LLM、原有 embedding/LM head 与 action tokenizer。
- 下一阶段才考虑开放 LLM；届时加入同训练预算的 `SlowFast + LLM 微调` 对照。
- **当前不使用 VSI memory checkpoint 初始化，也不研究 VSI 权重迁移。**
  复用模块设计／代码与复用训练权重是两回事。
- 本阶段不扩展 ObjectNav、EVT-Bench、真实机器人，也不重训 RVQ action tokenizer。

### 资源硬约束

当前 8 GPU 留给更高优先级的 VSI 实验。GPU 明确获准使用前：

- 不加载模型到 GPU，不运行导航训练、真实推理、GPU smoke 或特征提取。
- 不启动 Habitat simulator/EGL 渲染；“只启动环境服务器”也不是 CPU-only 工作。
- 不修改 VSI 代码、环境、进程、训练控制器、checkpoint 或备份流程。
- 不借用或升级 `/root/spmem/.venv`，不更改全局 CUDA、驱动或系统库。
- 不自动监听空闲 GPU 并抢占运行；必须经用户确认后显式指定获准的 GPU。
- CPU 准备也要限制线程、下载并发和磁盘 I/O；禁止无约束并行解压／特征生成。
  小型 CPU 验证默认 1 个工作进程、1–2 个线程；大文件下载先检查剩余空间和 VSI I/O。

## 2. 实验矩阵与评测协议

| ID | 方法 | 训练范围 | 用途 |
|---|---|---|---|
| B0 | 发布的 LightNav + 原始 SlowFast | 无 | 首先复现；保留原始处理与动作路径 |
| M1 | LightNav + adaptive memory | 仅 memory | 当前核心实验 |
| B2 | LightNav + SlowFast | LLM；视觉与 tokenizer 冻结 | 后续联合训练的公平对照 |
| M2 | LightNav + adaptive memory | memory + LLM；视觉与 tokenizer 冻结 | 下一阶段，非当前启动项 |

原始 SlowFast 就是固定规则基线，无需重复创建一组“固定池化”方法。
B0 先使用发布配置；随后在单独实验配置中调整 SlowFast 历史预算，与 M1 的写入率／
读取预算形成少量可比预算点。不得覆盖发布 checkpoint 内的 `eval_config.json`。

### 数据划分与不变量

- R2R：`R2R_VLNCE_v1-3_preprocessed`；RxR：`RxR_VLNCE_v0` 的 guide annotations。
- 首轮 RxR 明确使用 `en-US`、`en-IN`，不与全语言或 follower 结果混报。
- `train` 用于训练与教师轨迹采集；使用预先划定的训练 holdout 或 `val_seen` 调参。
  `val_unseen` 用作锁定的报告集，不用于梯度、教师训练数据或预算／超参数选择。
- 在生成训练集前锁定 R2R/RxR 混合比例、轨迹采样规则与开发集；这些参数目前未选定。
- 锁定 checkpoint revision、代码版本、场景／episode 清单、语言过滤、相机设置、
  观测频率、解码策略、控制器、STOP 规则及 500-step 上限。
- 上游文档的参考计数是 R2R `val_unseen` 1,839 episodes、RxR English 3,669 episodes。
  下载后必须重新审计实际计数、唯一键、GT 路径和场景覆盖，不能仅信文件名。
- 环境用于度量／专家标签的深度、GT pose、reference path 不得进入学生或教师策略输入。
  策略输入保持原始 RGB + 指令协议；标签可使用未来轨迹，输入必须严格因果。

### 效果与效率

- 效果：SR、SPL、nDTW、NE；同时记录 STOP／强制 STOP、无效动作、超时和失败类型。
- 按轨迹长度分桶，并保留逐 episode 结果；使用配对 episode 分析与不确定性估计。
  资源允许时采用多个训练随机种子；单次结果不得表述为稳定提升。
- 成本：历史／当前／总输入 token 的逐步、累计、平均、p95、峰值统计；持久 memory
  与 writer KV 大小；GPU allocated/reserved 峰值；冷启动和稳态 p50/p95 决策延迟。
- 时间分解至少含预处理、ViT、memory 更新、LLM prefill/decode 和动作后处理。
  simulator 渲染、网络及磁盘 I/O 单列；不得把这些开销悄悄只排除一组。
- 不要求 adaptive memory 每步固定 token 数；比较同一批任务的预算分布与效率—效果曲线。
  token 数减少不是端到端加速的充分证据，尤其要计入新增 writer 和视觉编码成本。
- 闭环轨迹可能分叉；用同一 episode 集做闭环评测，另以相同因果观测序列做固定回放 profiling。
- B0 可先用上游推荐的 vLLM 复现；M1 若先使用 HF，则必须补齐 **HF B0 vs HF M1**
  的同后端比较。不能把 vLLM 与未优化 HF 的耗时差归因于压缩机制。

## 3. 已核实的版本与本地状态

以下是 2026-09-18 首次计划阶段的只读审计快照；准备后的状态见第 6 节及资产状态报告：

| 项目 | 核实结果 | 尚未完成 |
|---|---|---|
| 上游与用户 fork | 两者对应的本地／fork `main` 为 `3015508b70fb6e30bbd66701e94c9b916dcb8df8` | 新研究修改未提交或推送 |
| 官方模型 | `LightOriginsHQ/LightNav-0`，revision `826dc5fbfa37afa8293d2e336d329b6ffc0bfb64` | 权重下载、哈希与真实加载 |
| 模型权限 | HF API 当时返回 `private=false`、`gated=false`；小型配置可匿名读取 | 不能据此保证后续所有文件下载成功 |
| 模型结构 | 配置中 text hidden size / vision out hidden size 都为 2560；temporal patch size 2 | 实际 processor token 布局与后端一致性 |
| 评测代码 | 已有 Habitat server、client、R2R/RxR YAML 和分片汇总脚本 | 环境安装、真实 episode 与完整复现 |
| 训练代码 | 已检查的发布脚本／入口中没有 memory trainer 或训练 rollout 生成器 | 自行实现、测试并建立标签来源 |
| 本地资产 | repo 的 `.venv/`、`data/`、`checkpoints/` 及常用 HF LightNav cache 路径未发现 | 尚未进行全盘数据清点；不等于其他位置一定没有资产 |

### 发布 checkpoint 的真实 VLN-CE 配置

已从上述固定 revision 的 `eval_config.json` 读取，而不是采用代码中的示例 tiers：

- `video_size=[256,448]`、`max_seq_len=8192`、`pool_stage=post_vit`、`pool_mode=avg`。
- `video_fps=4`、`timestamp_relative=true`、`prompt_style=unified_traj`。
- `num_history_frames=64`，但启用 SlowFast 时不能把它解释为仅保留最近 64 帧。
- tiers：current age 0–1 / dense / pool 1；fast 2–33 / dense / pool 2；
  mid 34–89 / burst stride 6 / pool 2；long 90–1,000,000 / span 14 pairs / pool 4；
  anchor 固定起点 2 frames / pool 4。
- RVQ：horizon 10、三级各 256 项、`se2_diff`、`weighted_diff`；发布 STOP codes
  为 `[6,122,174]`。保留模型的完整输出 token 协议及原有动作解码行为。

仅有元数据不等于模型已安装。本轮已随后下载固定版本并保存完整配置及校验清单，
但真实模型加载仍待 GPU 验证。

## 4. GPU 空闲前可以完成的准备

以下按执行优先级排列。P0、本轮公开模型／annotation 资产准备及 CPU 资产预检已完成；
MP3D 场景、完整模型／Habitat 环境、评测 wrapper、memory 与 trainer 仍待完成。

| 优先级 | 工作包 | 交付与 CPU 验收 | 边界／依赖 |
|---|---|---|---|
| P0 | fork 与协议维护 | `origin` 指用户 fork，`upstream` 保留作者仓库；本文、版本记录和任务状态 | 不直接向 upstream 推送；提交／推送按用户授权执行 |
| P1 | 数据／模型资产准备 | 固定 revision、下载清单、校验和、split/scene 覆盖报告、RVQ 文件完整性检查 | MP3D 访问需人类授权；大下载限并发 |
| P1 | 独立环境准备 | 推理／训练与 Habitat 分环境，记录依赖锁；CPU import、配置解析通过 | 不动 VSI 环境，不启动 EGL，不以 import 成功宣称 GPU 可用 |
| P2 | 原始评测预检 | R2R/RxR wrapper、dry-run、资产检查、episode 分片和汇总测试 | GPU 前仅构造命令／fake-env 测试，不启动现有 eval shell |
| P2 | 训练数据协议 | rollout schema、因果前缀 collator、标签／STOP 校验、合成样本测试 | 真实 RGB 渲染、教师标注与特征提取等 GPU 释放 |
| P3 | memory 接口与 CPU 实现 | 状态对象、增量写入／读出、预算诊断、reset 和 session 隔离测试 | 无 VSI 权重；模块源码许可与公开发布范围先审计 |
| P3 | trainer 与 profiler 骨架 | freeze 审计、合成小模型梯度检查、checkpoint/resume、计时字段 | 不用真实大模型 CPU 训练代替 GPU 验证 |
| P4 | GPU 启动 runbook | 串行 smoke → baseline → 小规模训练 → 完整实验的明确 gate | 不自动启动、不假定需要／可以占满 8 卡 |

### 4.1 资产准备的具体范围

建议独立本地根目录 `LIGHTNAV_DATA_ROOT=/root/lightnav_data_v1`，通过运行时路径配置使用，
不把机器绝对路径写死进通用源码。模型、场景、轨迹、缓存和运行输出都不进 Git。

1. 模型：下载固定 revision 的完整 checkpoint，包含 safetensors/index、tokenizer、
   chat template、processor/config/eval_config 和整个 `action_tokenizer/`；保存文件哈希。
   HF API 报告的模型仓库占用约 9.7 GB，下载前另留缓存／解压余量；不加载模型验证。
2. 数据：准备 R2R/RxR 的 `train`、`val_seen`、`val_unseen` annotations 及相应 GT 路径。
   MP3D 场景必须来自用户已有授权资产或正式获准下载；不能代签协议或使用非官方镜像。
3. 检查每个 episode 的场景、起点、指令、语言、reference path／GT、重复键和 split 隔离；
   检查实际 `.glb` 路径和需要的 navmesh。GT 文件缺失不能静默把 nDTW 当作有效结果。
4. RxR 必须保留 `{split}_guide.json.gz` / `{split}_guide_gt.json.gz` 路径。
   现有 `--data-path` 重写规则只生成 `{split}.json.gz`，不能直接用于 RxR guide；
   准备独立运行配置或保持标准目录，不为了本机路径修改上游 YAML。
5. 不为 LightNav 下载传统 VLN baseline 的 BERT／depth-encoder 特征；它们不是当前模型输入。
6. 下载失败按网络／授权／文件损坏区分；公开元数据可访问不等于 MP3D 已获授权。

### 4.2 环境与基线脚本

- LightNav 模型环境遵循本仓库依赖，当前钉住 `transformers==5.8.0`、
  `vllm==0.19.1` 和 `nvidia-cutlass-dsl==4.5.2`；记录适用 GPU 的 torch/CUDA wheel。
- Habitat 使用独立 Python 3.9 环境和 `habitat_server/environment.yml`，不能混装进模型环境。
- 先保留现有 `scripts/eval_habitat.sh`、`src/lightnav/cli/eval_habitat.py` 和 merge 路径，
  在其外准备可审计 wrapper，不重写一套不同指标的 evaluator。
- wrapper 要求显式模型路径、数据路径、输出目录、episode 集／语言和 GPU 列表，提供纯 CPU
  `--dry-run`／preflight；禁止沿用“不指定 GPU 就使用全部可见 GPU”的隐式行为。
- 检查分片是否不重不漏、部分失败能否检测、续跑是否去重、汇总是否按 episode 数加权。
  CPU mock 测试不能取代真实 Habitat 的指标和渲染验证。

### 4.3 训练数据是独立工作，不是下载 annotations 就完成

R2R/RxR 的指令与 reference paths 不是直接可喂给 LightNav 的完整 RGB-prefix / pointing /
RVQ 训练样本。已检查的 `src/lightnav/traj_vocab.py` 提供 RVQ 加载与解码，不能据此
宣称已具备发布模型一致的轨迹编码与 pointing 标注流水线。

先完成 CPU 可验证的样本协议：episode/scene/split/language、指令、时间戳与 frame ids、
观测引用、目标输出 tokens、标签来源与版本、loss mask、末端 STOP、无效样本原因。
完整时序不跨 episode，padding 不参与 loss，输入中不得出现未来帧／专家轨迹。

真实标签来源在训练前必须确定并记录；以下是候选路线，不是已完成的数据：

- 可先用冻结的原始 LightNav，在 **train-only** 因果观测上产生完整输出序列作为压缩适配教师；
  必须标明 teacher/pseudo labels，不能称为专家 GT。生成与特征缓存需要 GPU。
- 若采用专家轨迹监督，须验证 reference-path 到可执行 rollout、10-step SE(2) 目标、
  时间采样、RVQ 编码、pointing 标签和 STOP 的转换；不能用任意最近邻编码冒充官方编码器。
- 可以先完成合成数据 collator 和 trainer 单元测试，但不能在标签方案未验收时启动正式训练。
- 训练目标以有效导航输出 token 监督为主，压缩 rate／distortion 项作为可调辅助项；
  不直接照搬 VSI 的损失权重。最终系数和预算点在开发集上选定后锁定。

### 4.4 接入 adaptive memory 的实现边界

优先提供可独立 CPU 测试的 memory 组件；先通过 HF 路径训练和验证，再考虑 vLLM 的自定义接入。

- 默认保留双方一致的 current 原生视觉输入；用 learned memory 取代其余历史的
  fast/mid/long/anchor 规则，不能偷偷保留完整 SlowFast 历史后再额外加入 memory。
- 同一观测流按实际因果顺序写入。tubelet 重叠／重采样时按明确标识去重，不能每次决策
  把旧历史重复写入；教师与学生可见的观测范围一致，编码计算量单独记录。
- 显式接口包含 state 初始化、单步 update、readout、reset、序列化及 diagnostics；
  state 归属于 episode/session，不能在共享 engine 上串流污染。
- 现有 spmem 路线可作架构参考，但不复制整个旧 Qwen/Transformers fork，不使用
  `/root/spmem` 的运行时导入捷径，不动其工作树；先核实源码许可与可公开范围。
- 导航 token 数必须从实际 processor/tubelet/grid 推导，不能沿用 VSI 的每帧 256 tokens 假设。
  核对 DeepStack、mRoPE、timestamp、attention mask 和混合 memory/native 输入的对应关系。
- memory 初始行为、可训练参数集合与冻结参数列表必须可审计。冻结 LLM 参数不意味着
  可以把整个 LLM forward 包在 `no_grad` 中；memory 必须能收到输出监督的反向梯度。
- append-only 写入不保证有界；分别限制／记录写入状态和 LLM readout 的增长。
  预算超限应显式处理或报错，不能静默截掉当前帧、指令或输出目标。
- memory 缺席时必须回到原始 SlowFast 路径；第一帧、空 memory、全拒绝、全部保留、
  超长 episode、不同 batch 长度、reset 和 checkpoint/resume 都需测试。

### 4.5 训练脚本与 checkpoint

- 提供 memory-only 训练配置；将未来 memory+LLM 配置分离，不允许默认误解冻。
- CPU 小模型验证：memory 有梯度且有更新；冻结的 backbone 没有参数更新；
  loss mask、跨 episode reset、流式／离线前缀一致性和 causal suffix 检查通过。
- 为长轨迹显式定义截断反传、state detach、prefix 采样和缓存策略，记录被截断的梯度范围。
- 只有视觉编码器固定、预处理和权重版本锁定时才复用视觉特征缓存；新特征提取本身等 GPU。
- checkpoint 保存 memory、optimizer、scheduler、RNG、采样进度及配置／资产版本；
  通过小模型中断恢复测试后，仍需真实模型 GPU resume smoke。
- 保存不可变运行清单、逐 episode metrics 和失败日志；敏感路径、密钥、场景和大模型
  不进入 Git。独立备份位置需另行确认，不能借用 VSI 的 `latest` 或共享备份命名空间。

## 5. GPU 获准后的执行顺序与门槛

1. 用户确认可用 GPU ID、时间窗口和显存预算；**不是检测到低利用率就自动启动**。
2. 用单卡小规模检查：原始模型加载、RVQ 解码、Habitat EGL、一个 R2R 和一个 RxR episode；
   检查语言、相机、动作、STOP 和指标。失败先修协议，不直接扩到多卡。
3. 运行锁定配置的 B0 R2R/RxR 完整报告集评测，保存逐 episode 结果和耗时；记录与作者
   报告的差异及配置依据，不能只挑成功子集声明复现。
4. 在获准预算内生成／核验 train-only rollout、教师标签和可选冻结视觉特征缓存。
5. 做 M1 小样本过拟合、真实梯度／冻结检查、checkpoint/resume、短闭环和长时状态测试。
6. 正式训练 M1，在开发集上选择少量预算点，锁定模型后运行报告集并与 B0 比较。
7. 只有 M1 接口与指标可信后，再决定 B2/M2 的 LLM 解冻方式、训练预算与运行资源。

在第 2 步之前，最多只能声称“CPU 预检通过、待 GPU 验证”；不得声称模型或评测已跑通。

## 6. 仓库维护与当前进度

- `origin`：`https://github.com/juexZZ/LightNav-0.git`；
  `upstream`：`https://github.com/lightorigins/LightNav-0.git`。
- 保留上游方法、许可证和基线代码；研究改动可开关、可追溯，不把研究结果混进作者表格。
- 当前计划文档可以提交 Git；数据、权重、环境、特征、runs 和任何 token 不提交。
- 后续按资产预检、memory 接口、trainer、GPU 验证分批维护，变更状态应随证据更新。
  未经明确要求，不创建分支、commit 或 push；配置 remote 不等于 GitHub 已收到新文件。

| 状态 | 事项 |
|---|---|
| 已完成 | 研究协议与 fork 配置；模型 18 文件发布方校验；R2R/RxR 12 个 annotation/GT 文件；独立资产环境；下载／CPU 预检工具；47 项选定 CPU tests |
| 外部依赖 | 用户提供的 HM3D 900 个场景 ID 与所需 MP3D 72 个 ID 零交集；仍需定位授权 MP3D 场景 |
| 待执行 | 完整推理／训练及 Habitat 环境；基线评测 dry-run wrapper；独立持久备份方案 |
| 待实现 | 训练数据流水线、memory 接入、memory-only trainer、效率 profiler 及对应 CPU tests |
| 待 GPU | 真实渲染、原始模型复现、真实训练样本／特征生成、M1 训练与闭环评测 |
| 后续阶段 | LLM 解冻的 B2/M2；VSI 初始化迁移研究另立方案 |

近期执行顺序：**补齐 MP3D／环境及 baseline dry-run → 标签协议与 memory 接口 → trainer／测试 → 等待 GPU gate。**
关键外部依赖是 MP3D 授权资产；关键实现依赖是与发布模型一致的训练输出监督。

## 7. 核对依据

- 本地：[评测协议](EVAL_HABITAT.md)、[Habitat 环境与路径](HABITAT_SERVER.md)、
  [开发与 CPU tests](DEVELOPMENT.md)、[依赖](../pyproject.toml)。
- [用户 fork](https://github.com/juexZZ/LightNav-0)、
  [作者仓库](https://github.com/lightorigins/LightNav-0)。
- [固定版本模型文件](https://huggingface.co/LightOriginsHQ/LightNav-0/tree/826dc5fbfa37afa8293d2e336d329b6ffc0bfb64)：
  `config.json`、`eval_config.json`、`action_tokenizer/manifest.json` 已只读核对。
- [VLN-CE 官方数据说明](https://github.com/jacobkrantz/VLN-CE#data)：
  R2R/RxR annotations、MP3D 来源、目录结构和数据使用条件；annotation 本地审计见状态报告。
