# tfcpu：用 Transformer 搭的一台小 CPU

一个**权重手工构造**的 Transformer，一次前向传播执行一条指令；外层循环只有 `x = model(x)`。
网络里只有 `nn.Linear`、ReLU、hardmax attention 和残差连接，每一步与纯 Python 参考 CPU **比特级一致**。

- 机器：16 位有符号字（回绕），4 个寄存器，64 条指令程序区 + 64 字数据区
- 指令：`HALT LOADI LOAD STORE ADD SUB JZ JNEG JMP`
- 两种微架构（同一台机器、同一种状态编码）：

| | v1 | v2（默认） |
|---|---|---|
| 块数 | 6 | 2 |
| 关键技巧 | 逐级流水 | 投机执行（16 种寄存器组合一次全算）、读端口 + 两个写地址比较器 |
| 精确的精度 | float64 | float64 / float32（可上 Apple GPU） |
| 快速引擎 | — | `FastCPU`：同一套权重，只算会变的行 |

设计文档：`docs/superpowers/specs/2026-11-07-transformer-cpu-design.md`

## 用法

```python
from tfcpu import assemble, build_cpu, run, run_ref
prog  = assemble(open("examples/bubble_sort.asm").read())
model = build_cpu()                       # v2；build_cpu("v1") / build_cpu("v2", torch.float32, "mps")
trace = run(model, prog, max_steps=5000)  # 每一步的机器状态
trace.final.mem[:8]                       # [-20, -3, 0, 4, 5, 7, 7, 12]
assert trace.states == run_ref(prog, max_steps=5000)

from tfcpu.fast import run_fast           # 批量、快速：同一套权重
finals = run_fast(model, [prog] * 1000)
```

## 测速

`.venv/bin/python bench.py`，Apple M 系列 Mac，冒泡排序；每行计时后都抽查结果与参考 CPU 一致：

| 架构 | 精度 | 设备 | 执行方式 | 同时运行的机器数 | 指令/秒 |
|---|---|---|---|---:|---:|
| v1 | float64 | cpu | 完整模型 | 1 | 1,779 |
| v2 | float32 | cpu | 完整模型 | 1 | 2,257 |
| v2 | float32 | cpu | FastCPU | 1 | 8,692 |
| v1 | float64 | cpu | 完整模型 | 512 | 3,966 |
| v2 | float32 | mps | 完整模型 | 4,096 | 76,749 |
| v2 | float32 | cpu | FastCPU | 4,096 | 309,953 |
| v2 | float32 | mps | FastCPU | 16,384 | **1,271,812** |

单个程序时瓶颈是 Python 调度开销（每步约 0.1 ms），MPS 每步还有一次同步，反而比 CPU 慢；批量运行时 FastCPU + MPS 比 v1 快约 700 倍。

演示：`notebooks/demo.ipynb`（权重热力图、attention 落点、寄存器曲线、自己写程序）。

## 环境与测试

venv 复用系统 Python 里已装的 torch / jupyter（避免重新下载）：

```bash
uv venv --system-site-packages .venv && uv pip install -p .venv nbconvert
.venv/bin/python -m pytest -q
```

测试包括：积木穷举、每条指令的单步比特级不变式、4 个示例程序全程对比、
2000 个随机程序 × 300 步差分测试、"只用 Transformer 组件"的纯度检查——
在 v1-float64、v2-float64、v2-float32 三种配置上各跑一遍；外加 FastCPU 与完整模型逐步比特级一致（CPU 与 MPS）。

## 许可证

MIT，见 `LICENSE`。
