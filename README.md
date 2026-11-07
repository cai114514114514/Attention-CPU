# tfcpu：用 Transformer 搭的一台小 CPU

一个**权重手工构造**的 Transformer，一次前向传播执行一条指令；外层循环只有 `x = model(x)`。
网络里只有 `nn.Linear`、ReLU、hardmax attention 和残差连接，每一步与纯 Python 参考 CPU **比特级一致**。

- 机器：16 位有符号字（回绕），4 个寄存器，64 条指令程序区 + 64 字数据区
- 指令：`HALT LOADI LOAD STORE ADD SUB JZ JNEG JMP`
- 模型：6 个块，d_model = 74，129 个 token，约 2.9 万参数，float64

设计文档：`docs/superpowers/specs/2026-11-07-transformer-cpu-design.md`

## 用法

```python
from tfcpu import assemble, build_cpu, run, run_ref
prog  = assemble(open("examples/bubble_sort.asm").read())
model = build_cpu()                       # 普通的 nn.Module
trace = run(model, prog, max_steps=5000)  # 每一步的机器状态
trace.final.mem[:8]                       # [-20, -3, 0, 4, 5, 7, 7, 12]
assert trace.states == run_ref(prog, max_steps=5000)
```

演示：`notebooks/demo.ipynb`（权重热力图、attention 落点、寄存器曲线、自己写程序）。

## 环境与测试

venv 复用系统 Python 里已装的 torch / jupyter（避免重新下载）：

```bash
uv venv --system-site-packages .venv && uv pip install -p .venv nbconvert
.venv/bin/python -m pytest -q          # 48 个测试，约 3 分钟（随机差分测试占大头）
```

测试包括：积木穷举、每条指令的单步比特级不变式、4 个示例程序全程对比、
2000 个随机程序 × 300 步差分测试、以及"只用 Transformer 组件"的纯度检查。

## 许可证

MIT，见 `LICENSE`。
