# Transformer CPU（tfcpu）设计文档

日期：2026-11-07
状态：待审阅

## 1. 目标

用一个**权重手工构造**的 Transformer 实现一台多指令小 CPU，证明 Transformer 真的能做通用计算：

- 一次前向传播 = 执行一条指令；外层循环只做 `x = model(x)`，不含任何计算逻辑。
- 模型内部只由 `nn.Linear`、ReLU、hardmax attention、残差连接组成。
- 结果**比特级精确**：对任意合法机器状态 `s`，`model(encode(s)) == encode(step_ref(s))`（float64 张量逐元素相等）。
- 以 Python 库 + Jupyter notebook 的形式使用。

理论背景：Pérez et al.《Attention is Turing Complete》；Giannou et al. 2023《Looped Transformers as Programmable Computers》。

### 不在范围内（YAGNI）

- 训练 / 梯度学习。
- 可视化前端（notebook 中用 matplotlib 画图即可）。
- softmax 近似版本（可作为后续工作：大温度系数 softmax + 误差界）。
- 中断、I/O 设备、乘除法指令。

## 2. 机器规格

| 项目 | 规格 |
|---|---|
| 字长 | 16 位有符号整数，范围 −32768…32767，算术溢出回绕（mod 2^16） |
| 寄存器 | R0–R3 |
| PC | 整数，0–63 |
| 程序区 | 64 条指令（哈佛结构，程序不可被 STORE 修改） |
| 数据区 | 64 个字 M[0..63] |
| 停机 | `halted` 标志；置位后每一步状态完全不变 |

### 2.1 指令集（9 条）

| 编号 | 指令 | 语义（未写明的 PC 更新均为 PC+1） |
|---|---|---|
| 0 | `HALT` | halted = 1；PC 不变 |
| 1 | `LOADI rd, imm` | R[rd] = wrap(imm) |
| 2 | `LOAD rd, ra, imm` | R[rd] = M[clampA(R[ra] + imm)] |
| 3 | `STORE rs, ra, imm` | M[clampA(R[ra] + imm)] = R[rs] |
| 4 | `ADD rd, ra, rb` | R[rd] = wrap(R[ra] + R[rb]) |
| 5 | `SUB rd, ra, rb` | R[rd] = wrap(R[ra] − R[rb]) |
| 6 | `JZ ra, imm` | 若 R[ra] == 0，PC = clampA(imm) |
| 7 | `JNEG ra, imm` | 若 R[ra] < 0，PC = clampA(imm) |
| 8 | `JMP imm` | PC = clampA(imm) |

定义：

- `wrap(x) = ((x + 32768) mod 65536) − 32768`
- `clampA(x) = min(max(x, 0), 63)`
- `PC + 1`：若 PC = 63 则结果 clampA(64) = 63（即停在最后一条，不回绕）。
- 立即数 `imm` 为 16 位有符号整数（汇编器检查范围）。
- 指令字段：每条指令统一存 `op, rd, ra, rb, imm`；不用的字段为 0。`STORE` 的 rs 存在 `rb` 字段；`JZ/JNEG` 的被测寄存器存在 `ra` 字段。

**全定义语义**：任何状态、任何指令都有唯一确定的下一状态，不存在未定义行为。这使随机程序差分测试成为可能。

### 2.2 汇编器

- 每行一条指令，`;` 开始注释，`name:` 定义标签，标签可用作 `imm`。
- 寄存器写作 `r0`–`r3`。
- `.data` 段：`.data addr v1 v2 ...` 初始化 M[addr..]。
- 程序不足 64 条时用 `HALT` 填充。
- 错误（未知助记符、未定义标签、立即数越界、程序超过 64 条）抛出带行号的 `AsmError`。
- API：`assemble(text) -> Program`，`Program` 含 `instrs: list[Instr]` 与 `mem: list[int]`（长度 64）。

## 3. 状态编码

### 3.1 序列

共 129 个 token，顺序固定：

- token 0：CPU
- token 1–64：PROG[0..63]
- token 65–128：MEM[0..63]

### 3.2 残差流字段（`layout.py`）

每个字段是残差流中一个有名字的维度（或一组维度）。`layout.py` 提供名字 → 下标的映射，所有权重构造都通过名字访问，不出现裸下标。

| 类别 | 字段 | 说明 |
|---|---|---|
| 公共 | `one` | 恒为 1 |
| 公共 | `is_cpu, is_prog, is_mem` | token 类型 |
| 公共 | `a, a2` | 地址 a 与 a²（CPU token 为 0） |
| PROG | `op[0..8]` | opcode one-hot |
| PROG | `rd[0..3], ra[0..3], rb[0..3]` | 寄存器选择 one-hot |
| PROG | `imm` | 立即数 |
| MEM | `val` | 内存值 |
| CPU | `R[0..3], pc, halted` | 寄存器、PC、停机标志 |
| 草稿 | 见 §4 各块 | 每步结束时清零 |

持久字段只在所属类型的 token 上非零；草稿字段在每步开始和结束时全为 0。`encode` 产生满足该不变式的张量，`decode` 读取持久字段并断言不变式成立。

### 3.3 精确寻址

hardmax attention 用二次型做精确地址匹配：query `(2t, −1)`，key `(a, a2)`，点积为 `t² − (t − a)²`，在 `a = t` 时唯一取到最大值。再加上类型偏置（例如 key 的 `is_prog` 乘以大常数 M）限定只匹配某类 token。由于目标 t 已先夹取到 0–63，最大值始终唯一。

### 3.4 Hardmax

`HardmaxAttention`：分数 = QKᵀ，取每行最大值；若有并列，在并列位置上取平均（averaging hard attention）。CPU 和 MEM 的有用查询在构造上无并列；其他 token 的查询结果只写入草稿字段，随后被清零。

## 4. 单步流水线（6 个块）

每个块 = `x = x + Attn(x)`，再 `x = x + MLP(x)`，其中 `MLP(x) = W2·relu(W1·x + b1) + b2`。不需要 attention 的块，attention 权重为零。

| 块 | Attention | MLP |
|---|---|---|
| B1 | CPU 以 `pc` 匹配 PROG 的 `a`，复制 `op, rd, ra, rb, imm` 到草稿 `f_*` | `A = R[ra]`，`B = R[rb]`（门控选择） |
| B2 | — | `ea = clampA(A + imm)`；`sum = wrap(A+B)`；`dif = wrap(A−B)`；`z = [A==0]`；`n = [A<0]`；`npc = clampA(pc+1)`；`jt = clampA(imm)` |
| B3 | CPU 以 `ea` 匹配 MEM 的 `a`，读 `val` 到 `mv` | `res = Σ_k op_k·cand_k`（LOADI→imm，LOAD→mv，ADD→sum，SUB→dif）；`take = op_JMP + op_JZ·z + op_JNEG·n` |
| B4 | — | 写回 `R[i] += [we ∧ rd_i ∧ is_cpu ∧ ¬halted]·(res − R[i])`；`pc += [is_cpu ∧ ¬halted ∧ ¬op_HALT]·(新pc − pc)`；`halted += [is_cpu ∧ op_HALT ∧ ¬halted]`；写内存信号 `sa = ea`（若写使能）否则 −1000，`sv = B` |
| B5 | 每个 MEM token 匹配 CPU token，复制 `sa, sv` 到 `rx_sa, rx_sv` | `hit = [rx_sa == a] ∧ is_mem` |
| B6 | — | `val += hit·(rx_sv − val)`；所有草稿字段清零 |

其中 `we = op_LOADI + op_LOAD + op_ADD + op_SUB`，写内存使能 = `op_STORE ∧ is_cpu ∧ ¬halted`。需要两级运算才能得到的量（例如 B4 中依赖 B3 结果的门控）放在后续块里。实现时如发现某个块需要的层数比表中多，可以拆分或插入额外的零 attention 块，但需在 `build.py` 中注释说明，并保持 §6 的不变式测试通过。

### 4.1 ReLU 积木（`gadgets.py`）

全部只用 `relu` 和线性组合，在整数输入上精确：

- `step_ge(x, t) = relu(x − t + 1) − relu(x − t)` → [x ≥ t]
- `is_zero(x) = relu(x + 1) − 2·relu(x) + relu(x − 1)` → [x == 0]
- `is_neg(x) = relu(−x) − relu(−x − 1)` → [x < 0]
- `clamp(x, lo, hi) = lo + relu(x − lo) − relu(x − hi)`
- `wrap16(x) = x − 65536·[x ≥ 32768] + 65536·[x < −32768]`（输入范围 −65536…65534）
- `gate(x, s₁..s_k) = relu(x − M·(k − Σsᵢ)) − relu(−x − M·(k − Σsᵢ))` → x·∏sᵢ（sᵢ ∈ {0,1}，|x| < M）

常数 `M = 2^20`。所有中间值的绝对值 < 2^24，float64 下精确。

每个积木在代码中表示为"往 W1/b1/W2/b2 里写若干隐藏单元"的函数，按字段名指定输入与输出。

## 5. 代码结构与 API

项目根目录：Mac 上的 `/Users/wangzhe/TF TC`，uv 管理，Python 3.12，依赖 torch、numpy、pytest、jupyter、matplotlib。

```
tfcpu/
  __init__.py   导出 assemble, build_cpu, run, ReferenceCPU, State
  isa.py        Op 枚举、Instr、常量（WORD=16, NREG=4, NPROG=64, NMEM=64）、wrap/clampA
  asm.py        assemble()、AsmError
  reference.py  State（dataclass：regs, pc, halted, mem, program）、step_ref(state) -> State
  layout.py     Layout：字段名 → 下标；d_model
  nn.py         HardmaxAttention、MLP、Block、LoopedTransformer（nn.Module）
  gadgets.py    ReLU 积木：向 MLP 权重中写入隐藏单元
  build.py      build_cpu() -> LoopedTransformer：逐块构造 B1–B6
  state.py      encode(State) -> Tensor[129, d_model]、decode(Tensor) -> State
  run.py        run(model, program, max_steps) -> Trace
examples/       mul.asm  fib.asm  bubble_sort.asm  gcd.asm
notebooks/      demo.ipynb
tests/          test_isa.py  test_asm.py  test_reference.py  test_gadgets.py
                test_step.py  test_programs.py  test_fuzz.py  test_purity.py
```

`run` 的实现：

```python
x = encode(initial_state(program))
for _ in range(max_steps):
    x = model(x)
    trace.append(decode(x))
    if trace[-1].halted: break
```

`Trace`：状态列表，`.final` 为最后状态。`decode` 只用于观察，不参与计算；即使去掉 `decode`，对 `x` 的循环本身就完成全部计算。

### 5.1 Notebook（`demo.ipynb`）

1. 汇编并运行冒泡排序，展示结果。
2. 打印模型结构与参数量，热力图显示若干块的权重矩阵。
3. 选一步，展示 B1 attention 选中的 PROG token、B3 选中的 MEM token。
4. 寄存器 / PC 随步数变化的曲线。
5. 与参考 CPU 逐步对比，输出 "N 步全部一致"。

## 6. 验证

开发采用 TDD。

1. **积木测试**（`test_gadgets.py`）：每个积木在其整个整数输入范围内穷举（如 `wrap16` 测 −65536…65534 全部取值），输出与 Python 精确相等。
2. **参考 CPU 测试**（`test_reference.py`）：每条指令的语义，包括边界（回绕、地址夹取、PC=63、HALT 后不变）。
3. **汇编器测试**（`test_asm.py`）：标签、注释、`.data`、各类错误。
4. **单步不变式**（`test_step.py`）：每条指令 × 随机寄存器/内存/PC/halted，断言 `torch.equal(model(encode(s)), encode(step_ref(s)))`。
5. **示例程序**（`test_programs.py`）：mul、fib、bubble_sort、gcd 全程逐步与参考 CPU 一致，最终结果正确。
6. **随机差分测试**（`test_fuzz.py`）：固定种子生成 ≥ 2000 个随机程序（随机指令、操作数、随机初始内存），每个最多 300 步，每一步都比对。
7. **纯度检查**（`test_purity.py`）：遍历模块树，只允许 `LoopedTransformer / Block / HardmaxAttention / MLP / nn.Linear / nn.ReLU`；`HardmaxAttention.forward` 中只做矩阵乘、取最大、平均。

完成标准：以上测试全部通过，notebook 能从头到尾运行。

## 7. v2 微架构与快速执行引擎（2026-11-07 追加）

v1 保留不变（`build_cpu("v1")`，§4）。v2 是同一台机器、同一套持久字段和编码，只换内部结构，默认 `build_cpu()` 即 v2。

### 7.1 两个块

| 块 | Attention | MLP（一层） |
|---|---|---|
| B1 | CPU 以 `pc` 取指（1 头） | 投机执行：`ea`、`sa` 用带门控的 clamp 同时完成"选寄存器 + 夹取"；`sv = R[rb]`；ADD/SUB 对 16 种 (ra, rb) 组合各算原始和与溢出位（`*_raw, *_hi, *_lo`）；`take` 把判零/判负与 ra 选择、opcode 合并成带门控的 hat；`sen, jt, npc` |
| B2 | 3 头：读端口（CPU 以 `ea` 读 MEM）+ 两个写地址比较器 | 写回寄存器（候选值 `imm / mv / raw ∓ 65536·hi ± 65536·lo`）、PC、halted、`val`；清空草稿 |

两层 attention 是下限：读内存依赖取到的指令。

### 7.2 写端口：两个线性比较器

MEM token i 作为 query，对 CPU 的 key 打分 `ATT_BIG + (sa − aᵢ + ½) − STORE_OFF·(1 − sen)`（头 ge）或 `ATT_BIG + (aᵢ − sa + ½) − …`（头 le）；每个 MEM key 都恰好得 `ATT_BIG`，PROG key 得 0。于是 CPU 赢头 ge ⇔ sa ≥ aᵢ，赢头 le ⇔ sa ≤ aᵢ，与 MEM 分数至少差 ½，不会并列。CPU 输时 query 在 64 个并列的 MEM key 上平均，而它们被复制的值 `(sv, is_cpu)` 全是 0，结果仍精确。B2 的写入门控为 `got_ge ∧ got_le ∧ is_mem`。

只在 sa = aᵢ 处取峰值的单头打分需要在 key 里放 sa²，它不是残差流的线性函数——这正是最初单头写端口的错误（它在 sa ≥ aᵢ 时都可能赢），已由测试发现并改为双比较器。

### 7.3 float32 精确

v1 的一层回绕中，阶跃单元激活（~2¹⁵）乘以 65536 达到 ~2³¹，float32 不精确。v2 在 B1 只输出 0/1 溢出位，在 B2 才乘 65536；门控常数 M = 2²⁰、条件数 ≤ 5，所有中间量 < 2²⁴，float32 精确，可在 MPS 上运行。

### 7.4 FastCPU（`tfcpu/fast.py`）

同一套权重、同样的运算，只算会变的行：每步只有 CPU 行和至多一个被写的 MEM 行会变。缓存 PROG/MEM 的 key/value（取指端口、读端口的 key 永不变化；读端口的 value 只在写入时更新一行）；写端口只算 64 个 MEM query 与 CPU key 的分数，并与缓存的"其余 key 最高分"比较。依赖的不变式由 `tests/test_fast.py` 在 500 个随机程序 × 300 步上逐步与完整模型做比特级比较来保证（CPU 与 MPS）。

### 7.5 验证

所有模型级测试（单步、示例程序、随机差分 2000 × 300、纯度）在 v1-float64、v2-float64、v2-float32 三种配置上运行；另有 FastCPU ≡ 完整模型、构建参数检查。`bench.py` 输出各配置的指令/秒，并在计时后抽查结果正确性。

### 7.6 NativeCPU：C 推理引擎（单机速度目标：Intel 4004）

目标：单个程序的速度达到 Intel 4004 的每秒 46,250–92,500 条指令。PyTorch 版 FastCPU 每步约 118 μs，几乎全是每步约 20 个小算子的调度开销；真正的算术只有约 2 千次非零乘加。

`tfcpu/csrc/engine.c` 用 C 执行与 FastCPU 完全相同的计算：各 Linear 层以 CSR 形式只存非零权重，做稀疏矩阵-向量乘；CPU query 的 hardmax 在缓存的 key 上取唯一最大值；写端口与 FastCPU 同一判定；一次调用跑完 `max_steps` 或到 HALT。float32 下所有量是 < 2²⁴ 的整数或半整数，求和顺序不影响结果；不使用 `-ffast-math`。CPU query 出现并列或写端口出现并列即视为 v2 不变式被破坏，返回错误码而不是猜测。

边界：只跳过精确为零的权重和不会变化的行；不把指令语义编译成查表或分支——每一步都执行网络的注意力与 MLP 运算。

验证：`tests/test_native.py` 在 300 个随机程序 × 200 步上逐步与完整模型比特级比较；示例程序的最终状态和步数与参考 CPU 相同。实测（M4 Max，单核）：一个 100 万步的循环程序 1.44 s（≈ 69 万条/秒）且结果与参考一致；bench 中冒泡排序 ≈ 66 万条/秒，约为 4004 上限的 7 倍。

## 8. 扩展指令集（2026-11-07 追加）

v2 实现完整的 17 条指令；v1 只实现原来的 9 条（`build_cpu("v1").ops`），`run()` 遇到 v1 不支持的指令会报错。原 0–8 号指令编号和语义不变。

| 编号 | 指令 | 语义（未写明的 PC 更新均为 clampA(PC+1)） |
|---|---|---|
| 9 | `ADDI rd, ra, imm` | R[rd] = wrap(R[ra] + imm)；伪指令 `MOV rd, ra` = `ADDI rd, ra, 0` |
| 10 | `JR ra` | PC = clampA(R[ra]) |
| 11 | `CALL imm` | M[clampA(SP−1)] = clampA(PC+1)；SP = clampS(SP−1)；PC = clampA(imm) |
| 12 | `RET` | PC = clampA(M[clampA(SP)])；SP = clampS(SP+1) |
| 13 | `PUSH rs` | M[clampA(SP−1)] = R[rs]；SP = clampS(SP−1)（rs 存在 rb 字段） |
| 14 | `POP rd` | R[rd] = M[clampA(SP)]；SP = clampS(SP+1) |
| 15 | `MUL rd, ra, rb` | R[rd] = wrap(R[ra]·R[rb])，多周期，共 17 步（见下） |
| 16 | `OUT ra` | out = R[ra]，out_flag = 1；其他任何执行的步 out_flag = 0 |

新增持久字段（CPU token）：`sp`（0–64，初值 64 = 空栈，向下增长），`out, out_flag`，`mstep`（0–16），`ma, mb, macc`。`clampS(x) = min(max(x, 0), 64)`。停机后所有字段不变。

### 8.1 多周期 MUL

单周期精确 16 位乘法在"一个实数存一个字"的表示下需要拆字节、部分积与取模，至少 4 层 MLP、约 2000 个神经元。改为微码式多周期：

- `mstep = 0`：ma = R[ra]，mb = R[rb]，macc = 0，mstep = 1，PC 不变；
- `mstep = 1..16`：macc = wrap(2·macc + (mb < 0 ? ma : 0))，mb = wrap(2·mb)（从最高位开始的移位加）；mstep < 16 时 mstep += 1、PC 不变；mstep = 16 时 R[rd] = macc，mstep = 0，PC 前进。

16 次迭代后 macc = a·b mod 2¹⁶（二进制补码乘积的低 16 位与符号无关）。每个微步只需判负、回绕位和二选一：B1 算出 `[mstep==0]`、`[mstep==16]`（带门控的 hat，条件含 op_MUL）、`[mb<0]` 以及 2·macc、2·macc+ma、2·mb 的回绕位（2·macc+ma ∈ [−98304, 98301]，两个阈值即可），B2 用门控写回——仍是 2 个块，约 40 个神经元，float32 精确。

### 8.2 其余指令的电路

都只是在已有的多路选择上增加候选：`ea` 增加 clampA(SP)（POP/RET），`sa` 增加 clampA(SP−1)（PUSH/CALL），`sv` 增加 clampA(PC+1)（CALL），寄存器写回增加 ADDI 与 POP，PC 增加 `jr`（JR）和 clampA(mv)（RET）。"x_new = target·cond − x·cond" 拆成两个独立的门控项，所以 RET 的 clampA(M[SP]) 不需要多一层。

### 8.3 验证

全部模型级测试按架构支持的指令集运行（v1：9 条；v2：17 条，含随机的 sp、mstep、乘法临时值与 out_flag）；参考 CPU 测试覆盖 SP 饱和、返回地址夹取、OUT 只持续一步、MUL 17 步与 300+ 组随机乘积；新增示例 `factorial.asm`（递归、栈、MUL、OUT）与 `primes.asm`。FastCPU 与 NativeCPU 在全指令集随机程序上与完整模型逐步比特级一致；NativeCPU 收集 OUT 输出流。

## 9. 容量与代价解耦：静态路由与向量指令（2026-11-07 追加）

问题：模型做小装不下电路，做大则每一步都要算更多神经元——在 C 引擎里"面积变成时间"。实测每步只有约 6% 的神经元被激活，说明代价本应只取决于活跃电路。

### 9.1 静态路由（`tfcpu/routing.py`，NativeCPU 默认开启）

对每种译码后的指令 (opcode, rd, ra, rb)（18 × 4³ = 1152 种）固定取指得到的 one-hot 字段，其余字段取已知范围（寄存器为 16 位字、pc ∈ 0..63、机器在运行等），用区间算术求每个隐藏单元预激活的上界；上界 ≤ 0 的单元对该指令恒为 0，引擎跳过它——结果逐位相同。B1 活跃单元给出 B2 读取的草稿字段范围，读头给出 mv/amax 的范围；被写的 MEM 行另有一张（与指令无关的）清单。为了让路由更有效，B1 中投机计算的 ADD/SUB/ADDI 单元增加了 opcode 门控条件（相当于时钟门控）。

C 引擎在取指后读取 one-hot 字段得到清单编号，只计算清单中的单元（fc1 按行、fc2 按单元转置存储）。`routed=False` 时清单为全部单元，用于对照。

验证：路由与稠密两种模式均与完整模型逐步比特级一致（300 个随机程序 × 200 步）；另一测试在 2000 个随机状态上检查"引擎实际计算中每个激活的单元都在清单里"。

### 9.2 向量指令 ARGMAX（opcode 17）

`ARGMAX rd`：R[rd] = 内存中最大值的地址，并列时取最大地址。B2 增加一个读头：CPU query 对 MEM key 打分 `128·val + a + 2²³·is_mem`，最大值唯一（float32 下 < 2²⁴，精确），复制地址 `a`。读头（CPU query：读端口、ARGMAX）排在写头（MEM query：两个比较器）之前，attention 模块记录 `n_read_heads = 2`；FastCPU 与 NativeCPU 的读头缓存在写内存时刷新 key 与 value。汇编器新增 `.fill addr count value`。

VSUM 未做：64 个字之和可达 2²¹，超过门控常数 2²⁰ 的保护范围，提高常数又会突破 float32 的 2²⁴ 精确范围。

### 9.3 实测（M4 Max，单核）

| 指标 | 结果 |
|---|---|
| 每条指令平均活跃单元 | B1 30.9 / 312（10%），B2 67.9 / 235（29%） |
| 单个程序速度，稠密 → 路由 | 循环程序 48 万 → 86 万；冒泡排序 46 万 → 78 万；阶乘 31 万 → 85 万（×1.7–2.7） |
| 容量扫描：B1 增加 K 个可证明不活跃的单元 | K = 0 / 1 千 / 1 万 / 10 万：稠密 47 万 / 10.6 万 / 1.3 万 / 1358；路由 87 万 / 85 万 / 66 万 / 85 万 |
| 排序 8 个数 | 冒泡排序 535 步；ARGMAX 选择排序 58 步 |

结论：稠密执行的代价随容量线性增长（容量 × 321 时慢 346 倍），路由后代价与容量无关——"面积变成时间"可以被解开。向量指令则让一步做 64 个内存格的工作。B2 活跃比例较高，主要是对可能非零的草稿字段做清零。
