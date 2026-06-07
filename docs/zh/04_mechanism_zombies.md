# 机制三：僵尸序列 —— 提前定稿，延迟释放

> **来源**: Moondream 博客, "Popping the GPU Bubble" (2026-06-04),
> "Mechanism 3: zombies: finalize early, release late" 章节

---

## 1. 问题

在机制二中，我们指出步骤 t+1 的**批量成员**依赖于步骤 t 的提交结果。
如果某个序列在步骤 t 完成（EOS 或长度封顶），它不应出现在步骤 t+1 的批次中。

但步骤 t+1 是在步骤 t **提交之前**被发射的 —— 这正是流水化的全部意义。
所以已完成的序列**已被写入步骤 t+1 的前向中**。你无法取消已发射的 GPU 工作。

## 2. 朴素方案：飞行中取消

一个可以设想的方案是在飞行中取消该行：
- 将该序列的 token 设为"无关"值。
- 或在注意力内核中跳过它。

但这需要每个内核（注意力、MLP、采样、提交）都有特例逻辑，
形成"一团取消逻辑的乱麻"。

## 3. 优雅方案：让它多跑一次

Photon 的方法是更简单的：**让已完成的序列作为"僵尸"再多跑一次前向**。

```
步骤 t:     序列 X 存活 → 前向计算真实结果
            提交检测到 EOS → X 被标记为 "finalized"
            结果已输出给调用方
            但是：X 的 KV 页**并未释放**

步骤 t+1:   X 已在批次中 → 前向将 X 作为 "僵尸" 一并计算
            提交发现 X 已 finalized → 跳过，不追加 token
            释放 X 的飞行引用计数

Refcount=0: 回收 X 的 KV 页和 LoRA 槽位
```

僵尸占着一个批量槽位，写入一些永远不会被读取的 KV 缓存条目 ——
但这只是一个很小的代价。

## 4. 每序列状态机

每个序列额外携带两个字段：

| 字段 | 类型 | 含义 |
|------|------|------|
| `finalized` | `bool` | 当检测到 EOS 或达到长度上限后设置为 `True`。 |
| `inflight_refs` | `int` | 0、1 或 2 —— 有多少个飞行中的前向引用了该序列。 |

状态转移：

```
IDLE (0 refs, not finalized)
  │
  ├── launch ──► ACTIVE (1 ref)
  │                │
  │                ├── launch ──► ACTIVE (2 refs, 已流水化)
  │                │                │
  │                │                ├── commit (EOS) ──► ZOMBIE (2 refs, finalized)
  │                │                │                      │
  │                │                │                      ├── commit (skip)
  │                │                │                      └── release ──► ZOMBIE (1 ref)
  │                │                │
  │                │                ├── commit (normal) ──► ACTIVE (1 ref)
  │                │                └── release ──► ACTIVE (1 ref)
  │                │
  │                ├── commit (EOS) ──► ZOMBIE (1 ref, finalized)
  │                │                      │
  │                │                      └── release ──► IDLE (已回收)
  │                │
  │                └── commit (normal) ──► ACTIVE (1 ref)
  │
  └── release ──► IDLE (如果 finalized 则回收，否则无操作)
```

## 5. 僵尸税分析

来自博客文章的公式，*僵尸税* $z$ 是浪费前向的比例：

$$z \approx \frac{1}{L \cdot B}$$

其中：
- $L$ = 每个请求平均生成的 token 数
- $B$ = 批量大小

### 5.1 在 batch = 1 时

每 $L$ 个 token 浪费一次前向：$z \approx 1/L$。
在 $L \approx 110$ 时，约为 1% 的开销 —— 可忽略不计。

### 5.2 在 batch > 1 时

僵尸只是**多一行**数据，而该步骤已经在将全部模型权重
流过计算单元。在 decode 的内存带宽受限场景下，
权重传输才是主导因素，多一行几乎不增加任何成本。
当批量大小 ≫ 1 时，僵尸税实际上趋近于零。

### 5.3 实验验证

来自博客文章：

| 硬件 | 并发数 | L | 预测值 | 实测值 |
|------|--------|---|--------|--------|
| 3090 | 1  | 104 | +5.7%  | +6.5%  |
| 3090 | 32 | 113 | +11.1% | +11.6% |
| B200 | 1  | 115 | +17.2% | +17.6% |
| B200 | 32 | 104 | +39.1% | +35.4% |

预测加速比（已考虑僵尸税）与实测值高度吻合，证实了该模型的准确性。

## 6. 资源回收

资源**不会**在序列首次完成时释放。只有当 `inflight_refs` 降到 0 时才释放：

- **KV 缓存页**：归还到页面分配器的空闲列表。
- **LoRA 槽位**：归还到 LoRA 槽位池。
- **批量槽位**：在下一次调度周期中释放给新序列。

这种延迟释放策略是不需要飞行中取消逻辑的代价。

## 7. AMD ROCm 实现说明

在 `photon_amd/zombie.py` 中：

- `ZombieState`：每序列字段（`finalized`、`inflight_refs`、`kv_page_start`、`kv_page_count`、`lora_slot`）。
- `ZombieTracker`：中央管理器，在三个钩子点被调用：
  - `launch_ref()` —— 当序列进入批次时增加引用计数。
  - `commit()` —— 检测 EOS，返回 "finalized" 或 "skip"。
  - `release()` —— 减少引用计数，完成后回收资源。
- `should_skip()` —— 查询某行是否是僵尸。

调度器分别在 `_assemble_batch()`（发射引用）、`_commit()`（提交）
和 `_recycle_done()`（释放）中调用这些钩子。

## 8. 参考文献

1. 博客文章: [机制三示意图](https://moondream.ai/blog/popping-the-gpu-bubble#mechanism-3-zombies-finalize-early-release-late)
2. Photon 源码: `kestrel/models/moondream/zombie.py`（近似路径）
3. 相关: [引用计数在垃圾回收中的应用](https://en.wikipedia.org/wiki/Reference_counting)
