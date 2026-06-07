# 机制二：先行推理，后采样

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04),
> "Mechanism 2: forward now, sample later" 章节

---

## 1. 依赖关系难题

下一个前向并不依赖于 CPU 对上一步 token 所做的任何处理 ——
前向只需要 token ID，而它已在 GPU 上了。因此原则上，前向可以提前执行。

但是，下一个步骤的**某些**方面确实依赖于上一步已提交的结果：

1. **批量成员**：如果某个请求在步骤 t 完成，它不应出现在步骤 t+1 中。
   这由机制三（僵尸序列）处理。

2. **约束解码掩码**：Moondream 的空间技能（`point`、`detect`、`segment`）
   限制模型在每个步骤可以产生哪些 token。步骤 t+1 的掩码取决于步骤 t
   已采样的 token —— 而当我们想发射步骤 t+1 时，该 token 尚未被 CPU 提交。

## 2. 解决方案：将前向与采样解耦

关键洞见：依赖关系在**采样**中，而非在前向中。
前向本身不需要掩码 —— 它只是计算整个词汇表的 logits。

解决方案是将每个步骤拆分为两个阶段，它们在不同的流水线阶段执行：

```
┌──────────────────────────────────────────────────────────┐
│  Tick N                                                   │
│                                                           │
│  LAUNCH:   Forward(第N步) → [分数]                      │
│            （不需要掩码 — 分数是无偏的）                   │
│                                                           │
│  COMMIT:   等待 token(第N-1步) 到达 CPU。               │
│            推进状态 → 构建 mask(第N步)。                │
│                                                           │
│  FINALIZE: 应用 mask(第N步) → Sample(第N步)。           │
│            （此时掩码已是最新的，因为 COMMIT 刚刚完成）     │
└──────────────────────────────────────────────────────────┘
```

这建立了**"先提交后定稿"的排序**：前向立即执行，但采样必须等到
上一步的提交完成，使每个序列的状态变为最新之后才进行。

## 3. 统一处理

博客文章描述了如何用一个循环优雅地同时处理纯文本和约束解码：

### 3.1 纯文本（无约束解码）

- 无需掩码。
- 前向**和**采样都可以提前一步执行。
- 两者都在 LAUNCH 阶段发生。

### 3.2 约束解码（结构化输出）

- 前向提前一步执行（在 LAUNCH 中）。
- 采样等待上一次提交（在 FINALIZE 中）。
- COMMIT 阶段根据更新后的状态构建掩码，
  然后 FINALIZE 应用掩码并进行采样。

### 3.3 优雅之处

无需任何特例判断。同一套调度器代码同时处理两者：
- LAUNCH 始终执行前向。
- COMMIT 始终处理上一步的结果。
- FINALIZE 始终执行掩码相关的工作。

对于纯文本，FINALIZE 是空操作（掩码为空）。对于约束序列，
FINALIZE 等待刚刚发生的提交。

## 4. 禁止掩码（Disallow Mask）

Moondream 的约束解码使用*禁止掩码*：一个布尔型张量 `[batch, vocab]`，
其中 `True` 表示"在采样前将该 token 的 logit 强制设为负无穷"。

每个序列的掩码由该序列当前的解码状态构建 ——
即它在结构化输出中已经输出了哪些 token。例如：
- `point` 输出总是输出一个坐标 `(x, y)`。
- `detect` 输出遍历一个 `x, y, width, height` 循环。

步骤 t+1 的掩码在步骤 t 的 COMMIT 结束时计算，此时序列状态是最新的。
然后通过异步 H2D 拷贝（在拷贝流上）上传到 GPU 槽位，
使其在 FINALIZE 执行时立即可用。

## 5. 性能影响

引自博客文章"It only pays once the bubble is actually hideable"一节：

> 我们发现了一个 bug，流水线数据却表现为*阻塞*速度，
> 追踪到在构建约束解码掩码时发生了一次意外的同步拷贝。
> 将其移至拷贝流后，在 3090 上获得了 +11% 的提升，
> 在 B200 上获得了 +34% 的提升。

教训：**热路径上的任何同步都会毁掉流水化的收益**。
即使是在关键路径内的单次同步拷贝，也足以破坏重叠效果。

## 6. AMD ROCm 实现说明

在我们的 `photon_amd` 实现中：

- `PipelineCallbacks.do_sample` 在 `_launch()` 期间为纯文本
  （掩码无关的采样）调用，或在 `_finalize()` 期间为约束解码调用。
- 禁止掩码存储在 `DecodeMetaBuffers.disallow_mask` 中，
  并通过拷贝流异步上传到 GPU。
- `slot.mask_ready_event` 标记掩码上传完成，采样内核可等待该事件。

## 7. 参考文献

1. Photon 源码: `kestrel/models/moondream/tokenizer.py`（结构化解码逻辑）
2. 博客文章: [机制二示意图](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-2-forward-now-sample-later)
3. NVIDIA: [Guided Decoding with Logits Processor](https://huggingface.co/docs/transformers/main_classes/logits)
