# Learning LLM Inference with Nano-vLLM and Qwen3-0.6B: A Complete Tutorial

> **Goal:** Understand how a modern LLM inference engine works, end to end, by reading and running ~1,200 lines of real, production-quality Python. Every concept is grounded in the actual nano-vllm source code.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Part I -- The Qwen3-0.6B Architecture](#2-part-i----the-qwen3-06b-architecture)
   - [2.1 Model Configuration](#21-model-configuration)
   - [2.2 The Transformer Stack](#22-the-transformer-stack)
   - [2.3 RMSNorm (Pre-Norm)](#23-rmsnorm-pre-norm)
   - [2.4 Multi-Head Attention with Grouped-Query Attention (GQA)](#24-multi-head-attention-with-grouped-query-attention-gqa)
   - [2.5 Rotary Position Embeddings (RoPE)](#25-rotary-position-embeddings-rope)
   - [2.6 SwiGLU Feed-Forward Network](#26-swiGLU-feed-forward-network)
   - [2.7 Vocabulary Embedding and LM Head (Tied Weights)](#27-vocabulary-embedding-and-lm-head-tied-weights)
   - [2.8 Autoregressive Generation](#28-autoregressive-generation)
3. [Part II -- How vLLM-Style Inference Works](#3-part-ii----how-vllm-style-inference-works)
   - [3.1 The Big Picture: Engine, Scheduler, ModelRunner](#31-the-big-picture-engine-scheduler-modelrunner)
   - [3.2 Sequences and Their Lifecycle](#32-sequences-and-their-lifecycle)
   - [3.3 The Two Phases: Prefill vs Decode](#33-the-two-phases-prefill-vs-decode)
   - [3.4 The Scheduler: Continuous Batching](#34-the-scheduler-continuous-batching)
   - [3.5 The ModelRunner: Preparing Inputs and Running the Model](#35-the-modelrunner-preparing-inputs-and-running-the-model)
   - [3.6 Sampling: From Logits to Tokens](#36-sampling-from-logits-to-tokens)
4. [Part III -- Optimizations That Reduce Inference Time](#4-part-iii----optimizations-that-reduce-inference-time)
   - [4.1 Paged Attention and the Block Manager](#41-paged-attention-and-the-block-manager)
   - [4.2 Prefix Caching](#42-prefix-caching)
   - [4.3 Flash Attention](#43-flash-attention)
   - [4.4 CUDA Graphs](#44-cuda-graphs)
   - [4.5 torch.compile](#45-torchcompile)
   - [4.6 Tensor Parallelism](#46-tensor-parallelism)
   - [4.7 Triton Kernels](#47-triton-kernels)
   - [4.8 Fused Residual + LayerNorm](#48-fused-residual--layernorm)
   - [4.9 Merged QKV and Gate-Up Projections](#49-merged-qkv-and-gate-up-projections)
   - [4.10 Preemption and Memory Management](#410-preemption-and-memory-management)
5. [Part IV -- Hands-On Exercises](#5-part-iv----hands-on-exercises)
6. [Appendix: File Map](#6-appendix-file-map)

---

## 1. Setup

```bash
# Clone the repo
git clone https://github.com/GeeeekExplorer/nano-vllm.git
cd nano-vllm

# Create environment (Python 3.10-3.12 required)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install torch first (flash-attn needs it at build time)
uv pip install torch

# Install nano-vllm in editable mode
uv pip install -e . --no-build-isolation

# Download model weights
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

Verify everything works:

```python
from nanovllm import LLM, SamplingParams
llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.6, max_tokens=32))
print(outputs[0]["text"])
```

---

## 2. Part I -- The Qwen3-0.6B Architecture

### 2.1 Model Configuration

Qwen3-0.6B is a decoder-only causal language model from Alibaba's Qwen team. Here are its key parameters:

| Parameter | Value | What It Means |
|---|---|---|
| `num_hidden_layers` | **28** | 28 transformer decoder blocks stacked in sequence |
| `hidden_size` | **1024** | Dimensionality of the residual stream (the "width" of the model) |
| `num_attention_heads` | **16** | Number of query heads in multi-head attention |
| `num_key_value_heads` | **8** | Number of KV heads (GQA: every 2 Q heads share 1 KV head) |
| `head_dim` | **128** | Dimension of each attention head (note: 16 * 128 = 2048 != 1024; Qwen3 uses explicit `head_dim`) |
| `intermediate_size` | **3072** | Hidden size of the feed-forward network (3x the model width) |
| `vocab_size` | **151,936** | Number of tokens in the vocabulary |
| `max_position_embeddings` | **40,960** | Maximum context length supported |
| `rope_theta` | **1,000,000** | Base frequency for RoPE (high value = better long-context extrapolation) |
| `rms_norm_eps` | **1e-6** | Epsilon for RMS normalization |
| `hidden_act` | **silu** | Activation function (SiLU / Swish, used in SwiGLU) |
| `tie_word_embeddings` | **true** | Input embedding and output LM head share the same weight matrix |
| `torch_dtype` | **bfloat16** | Training/inference precision |
| **Total parameters** | **0.6B** | 0.44B non-embedding parameters |

These values are loaded automatically from the HuggingFace `config.json` at model initialization.

### 2.2 The Transformer Stack

The full model is defined in `nanovllm/models/qwen3.py`. Here is the top-level structure:

```
Qwen3ForCausalLM                          (qwen3.py:185)
  ├── Qwen3Model                          (qwen3.py:161)
  │     ├── VocabParallelEmbedding        token_ids -> hidden_states [151936, 1024]
  │     ├── 28x Qwen3DecoderLayer         the core transformer blocks
  │     │     ├── RMSNorm (input)         pre-attention normalization
  │     │     ├── Qwen3Attention          multi-head attention with GQA + RoPE
  │     │     ├── RMSNorm (post-attn)     pre-FFN normalization
  │     │     └── Qwen3MLP               SwiGLU feed-forward network
  │     └── RMSNorm (final)              final normalization before LM head
  └── ParallelLMHead                      hidden_states -> logits [151936]
```

The forward pass (`qwen3.py:172-182`):

```python
def forward(self, input_ids, positions):
    hidden_states = self.embed_tokens(input_ids)   # [seq_len] -> [seq_len, 1024]
    residual = None
    for layer in self.layers:                       # 28 layers
        hidden_states, residual = layer(positions, hidden_states, residual)
    hidden_states, _ = self.norm(hidden_states, residual)  # final norm
    return hidden_states                            # [seq_len, 1024]
```

Each token passes through embedding, 28 identical decoder layers, and a final normalization. Then `compute_logits` projects back to vocabulary size for next-token prediction.

### 2.3 RMSNorm (Pre-Norm)

**File:** `nanovllm/layers/layernorm.py`

Unlike the original Transformer which uses LayerNorm *after* each sub-layer, Qwen3 uses **RMSNorm** (Root Mean Square Normalization) *before* each sub-layer (pre-norm architecture).

RMSNorm is simpler than LayerNorm -- it skips the mean-centering step:

```python
# layernorm.py:17-26
def rms_forward(self, x):
    orig_dtype = x.dtype
    x = x.float()                                   # upcast to float32 for precision
    var = x.pow(2).mean(dim=-1, keepdim=True)       # RMS = sqrt(mean(x^2))
    x.mul_(torch.rsqrt(var + self.eps))              # normalize: x / RMS(x)
    x = x.to(orig_dtype).mul_(self.weight)           # scale by learnable weight
    return x
```

**Why RMSNorm over LayerNorm?** It has one fewer reduction operation (no mean subtraction) and empirically works just as well for LLMs. The `eps=1e-6` prevents division by zero.

The pre-norm pattern with **residual connections** flows through the decoder layer like this (`qwen3.py:145-158`):

```
Input: hidden_states, residual (from previous layer)
  1. hidden_states = RMSNorm(hidden_states + residual)   # fused add+norm
     residual = hidden_states + residual (before norm)    # save for next residual
  2. hidden_states = Attention(hidden_states)
  3. hidden_states = RMSNorm(hidden_states + residual)   # fused add+norm
     residual = hidden_states + residual
  4. hidden_states = MLP(hidden_states)
Output: hidden_states, residual (passed to next layer)
```

### 2.4 Multi-Head Attention with Grouped-Query Attention (GQA)

**File:** `nanovllm/models/qwen3.py:14-87` and `nanovllm/layers/attention.py`

Standard multi-head attention (MHA) has equal numbers of Q, K, and V heads. **Grouped-Query Attention (GQA)** reduces the number of K and V heads. In Qwen3-0.6B:

- **16 query heads** but only **8 KV heads**
- Each KV head is shared by 2 query heads (group size = 2)
- This cuts KV cache memory by 50% with minimal quality loss

The attention forward pass (`qwen3.py:71-87`):

```python
def forward(self, positions, hidden_states):
    # Step 1: Project hidden_states to Q, K, V
    qkv = self.qkv_proj(hidden_states)              # [seq, 1024] -> [seq, (16+8+8)*128]
    q, k, v = qkv.split([self.q_size,               # q: [seq, 16*128=2048]
                          self.kv_size,              # k: [seq, 8*128=1024]
                          self.kv_size], dim=-1)     # v: [seq, 8*128=1024]

    # Step 2: Reshape to per-head format
    q = q.view(-1, self.num_heads, self.head_dim)    # [seq, 16, 128]
    k = k.view(-1, self.num_kv_heads, self.head_dim) # [seq, 8, 128]
    v = v.view(-1, self.num_kv_heads, self.head_dim) # [seq, 8, 128]

    # Step 3: Q/K normalization (Qwen3 specific, when no QKV bias)
    q = self.q_norm(q)                               # RMSNorm per head
    k = self.k_norm(k)

    # Step 4: Apply Rotary Position Embeddings
    q, k = self.rotary_emb(positions, q, k)

    # Step 5: Compute attention (Flash Attention + KV cache)
    o = self.attn(q, k, v)                           # [seq, 16, 128]

    # Step 6: Concatenate heads and project back
    output = self.o_proj(o.flatten(1, -1))           # [seq, 2048] -> [seq, 1024]
    return output
```

**Q/K Normalization** (`q_norm`, `k_norm`) is a Qwen3 innovation -- applying RMSNorm to Q and K *before* the attention dot product stabilizes training, especially at scale. This is enabled when `attention_bias=False` (the Qwen3-0.6B default).

### 2.5 Rotary Position Embeddings (RoPE)

**File:** `nanovllm/layers/rotary_embedding.py`

RoPE encodes position information by *rotating* pairs of dimensions in the Q and K vectors. Unlike absolute position embeddings (added to the input), RoPE is applied directly inside attention and naturally encodes *relative* distances between tokens.

**Initialization** -- precompute rotation angles for all positions (`rotary_embedding.py:19-35`):

```python
# For each dimension pair (0,1), (2,3), ..., (126,127):
inv_freq = 1.0 / (base ** (arange(0, 128, 2) / 128))
# base = 1,000,000 for Qwen3 (high base = slow rotation = better long-context)

# For each position 0, 1, 2, ..., max_pos:
t = arange(max_position_embeddings)
freqs = einsum("i,j -> ij", t, inv_freq)  # [max_pos, 64]
cos = freqs.cos()   # precomputed cosine table
sin = freqs.sin()   # precomputed sine table
```

**Application** -- rotate each pair of dimensions (`rotary_embedding.py:6-14`):

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = chunk(x, 2, dim=-1)       # split into pairs: [seq, heads, 64] each
    y1 = x1 * cos - x2 * sin           # 2D rotation matrix applied per pair
    y2 = x2 * cos + x1 * sin
    return cat((y1, y2), dim=-1)        # [seq, heads, 128]
```

This is a 2D rotation in each dimension pair: `(x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)`. The key insight: when computing `Q_i . K_j`, the rotation angles *subtract*, so the dot product depends only on the *relative* distance `i - j`, not the absolute positions.

**Why `rope_theta = 1,000,000`?** Higher base values mean slower rotation frequencies, which helps the model generalize to longer sequences than it was trained on (long-context extrapolation).

### 2.6 SwiGLU Feed-Forward Network

**File:** `nanovllm/models/qwen3.py:90-116` and `nanovllm/layers/activation.py`

Each decoder layer has a feed-forward network (FFN) using the **SwiGLU** architecture:

```python
# qwen3.py:112-116
def forward(self, x):
    gate_up = self.gate_up_proj(x)   # [seq, 1024] -> [seq, 3072*2=6144]
    x = self.act_fn(gate_up)          # SwiGLU: SiLU(gate) * up -> [seq, 3072]
    x = self.down_proj(x)             # [seq, 3072] -> [seq, 1024]
    return x
```

The `gate_up_proj` is a single linear layer that produces *two* outputs at once (gate and up projections, concatenated). The `SiluAndMul` activation (`activation.py:6-14`) splits and combines them:

```python
# activation.py:12-14
def forward(self, x):
    x, y = x.chunk(2, -1)    # split: gate [seq, 3072], up [seq, 3072]
    return F.silu(x) * y     # SiLU(gate) * up  (element-wise gating)
```

**SiLU** (Sigmoid Linear Unit) = `x * sigmoid(x)`. It's a smooth approximation of ReLU. The *gating* mechanism (`SiLU(gate) * up`) allows the network to learn which features to pass through -- it's been shown to outperform plain ReLU or GELU FFNs in LLMs.

### 2.7 Vocabulary Embedding and LM Head (Tied Weights)

**Files:** `nanovllm/layers/embed_head.py` and `nanovllm/models/qwen3.py:185-215`

The model starts with **token embedding** and ends with a **language model head** that shares the same weight matrix (weight tying):

```python
# qwen3.py:199-202
self.model = Qwen3Model(config)
self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
if config.tie_word_embeddings:
    self.lm_head.weight.data = self.model.embed_tokens.weight.data  # shared!
```

- **Embedding:** maps token IDs to 1024-dim vectors. Shape: `[151936, 1024]`
- **LM Head:** maps 1024-dim hidden states back to 151,936 logits (one per vocab token). Uses the *transpose* of the same weight matrix: `logits = hidden_states @ embedding_weight.T`

**Why tie weights?** The embedding matrix is the single largest parameter tensor in the model. Sharing it saves ~150M parameters and acts as a regularizer (the model must learn embeddings that are useful for both input and output).

The `ParallelLMHead` (`embed_head.py:56-66`) has a critical optimization: during **prefill**, it only computes logits for the *last token* of each sequence (since we only need to predict the next token):

```python
# embed_head.py:58-60
if context.is_prefill:
    last_indices = context.cu_seqlens_q[1:] - 1   # last position of each seq
    x = x[last_indices].contiguous()                # skip all other positions
```

### 2.8 Autoregressive Generation

LLMs generate text one token at a time. At each step:

1. Feed all tokens so far through the model
2. Get logits (a score for each of the 151,936 vocabulary tokens)
3. Sample the next token from the logits (applying temperature)
4. Append the new token and repeat

The naive approach recomputes attention over *all* previous tokens at every step. The KV cache optimization (explained in Part III) avoids this by caching intermediate results.

---

## 3. Part II -- How vLLM-Style Inference Works

### 3.1 The Big Picture: Engine, Scheduler, ModelRunner

**File:** `nanovllm/engine/llm_engine.py`

The inference engine has three core components:

```
User calls generate(prompts, sampling_params)
         |
         v
  +--------------+     schedule()     +-------------+     run()      +---------------+
  |  LLMEngine   | ----------------> |  Scheduler   | ------------> |  ModelRunner   |
  |              | <---------------- |              | <------------ |               |
  +--------------+   postprocess()   +-------------+   token_ids   +---------------+
        |                                  |                              |
        |                           BlockManager                    GPU forward pass
        |                          (KV cache mgmt)                 + sampling
        v
   Return decoded text
```

The main loop in `generate()` (`llm_engine.py:73-88`):

```python
while not self.is_finished():
    # 1. Scheduler decides which sequences to process and how (prefill/decode)
    seqs, is_prefill = self.scheduler.schedule()

    # 2. ModelRunner runs the GPU forward pass and samples new tokens
    token_ids = self.model_runner.call("run", seqs, is_prefill)

    # 3. Scheduler updates sequences with new tokens, checks stopping conditions
    self.scheduler.postprocess(seqs, token_ids)
```

This loop continues until all sequences have finished (hit EOS or max_tokens).

### 3.2 Sequences and Their Lifecycle

**File:** `nanovllm/engine/sequence.py`

Each generation request is tracked as a `Sequence` object:

```python
class Sequence:
    seq_id: int              # unique identifier
    status: SequenceStatus   # WAITING -> RUNNING -> FINISHED
    token_ids: list[int]     # all tokens (prompt + generated)
    num_prompt_tokens: int   # length of the original prompt
    num_cached_tokens: int   # tokens already in KV cache (from prefix caching)
    block_table: list[int]   # maps to physical KV cache blocks
    temperature: float       # sampling temperature
    max_tokens: int          # max tokens to generate
```

**Lifecycle:**
```
WAITING ──allocate blocks──> RUNNING ──generate tokens──> FINISHED
   ^                            |                            |
   └────── preempt (evict) ────┘                   deallocate blocks
```

A sequence can be **preempted** back to WAITING if the system runs out of KV cache memory. Its blocks are freed, and it re-enters the queue to be re-prefilled later.

### 3.3 The Two Phases: Prefill vs Decode

This is the most important concept in LLM serving:

**PREFILL (a.k.a. "prompt processing")**
- Processes *all* prompt tokens at once (in parallel)
- Compute-bound: dominated by matrix multiplications
- Fills the KV cache for all prompt positions
- Returns the *first* generated token
- Cost: proportional to `prompt_length * hidden_size`

**DECODE (a.k.a. "token generation")**
- Processes only *one new token* per sequence per step
- Memory-bound: dominated by reading the KV cache
- Appends one entry to the KV cache
- Returns one token per sequence
- Cost: proportional to `num_sequences` (not context length!)

The scheduler alternates between these phases. Prefill batches many tokens for a single (or few) sequence(s). Decode batches many sequences, each producing one token.

**In the code:**

Prefill input preparation (`model_runner.py:126-162`):
```python
# Collects ALL uncached tokens from sequences
input_ids.extend(seq[seq.num_cached_tokens:])     # e.g., 512 tokens
positions.extend(range(seq.num_cached_tokens, seqlen))
```

Decode input preparation (`model_runner.py:164-180`):
```python
# Collects ONLY the last token from each sequence
input_ids.append(seq.last_token)       # 1 token per seq
positions.append(len(seq) - 1)         # its position
```

### 3.4 The Scheduler: Continuous Batching

**File:** `nanovllm/engine/scheduler.py`

Traditional batching waits for all sequences in a batch to finish before starting new ones. **Continuous batching** (the vLLM innovation) processes new sequences as soon as any sequence finishes.

The scheduler (`scheduler.py:24-58`) implements this with a two-priority system:

```python
def schedule(self):
    # Priority 1: PREFILL -- process new requests from the waiting queue
    while self.waiting and constraints_met:
        seq = self.waiting.popleft()
        self.block_manager.allocate(seq)        # reserve KV cache blocks
        seq.status = RUNNING
        scheduled_seqs.append(seq)
    if scheduled_seqs:
        return scheduled_seqs, is_prefill=True   # prefill batch

    # Priority 2: DECODE -- generate next token for running sequences
    while self.running and constraints_met:
        seq = self.running.popleft()
        if not self.block_manager.can_append(seq):
            self.preempt(self.running.pop())      # evict if out of memory
        else:
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
    return scheduled_seqs, is_prefill=False       # decode batch
```

**Key constraints checked:**
- `num_seqs < max_num_seqs` (default: 512) -- max batch size
- `num_batched_tokens <= max_num_batched_tokens` (default: 16384) -- max tokens per step
- `block_manager.can_allocate(seq)` -- enough KV cache blocks available

**After each step**, `postprocess()` (`scheduler.py:65-71`) updates sequences:

```python
def postprocess(self, seqs, token_ids):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)
        if (token_id == self.eos) or (seq.num_completion_tokens == seq.max_tokens):
            seq.status = FINISHED
            self.block_manager.deallocate(seq)     # free KV cache blocks
            self.running.remove(seq)
```

### 3.5 The ModelRunner: Preparing Inputs and Running the Model

**File:** `nanovllm/engine/model_runner.py`

The ModelRunner translates the scheduler's sequence list into GPU tensors and runs the forward pass.

**Initialization** (`model_runner.py:17-39`):
1. Initialize NCCL for distributed communication
2. Load the Qwen3 model and weights from safetensors
3. Warm up the model (one forward pass to establish memory usage)
4. Allocate the KV cache (fill remaining GPU memory)
5. Capture CUDA graphs for decode (unless `enforce_eager=True`)

**KV cache allocation** (`model_runner.py:100-118`):

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    # Calculate how many blocks fit in remaining GPU memory (90% utilization)
    block_bytes = 2 * 28 * 256 * 8 * 128 * 2  # K+V, layers, block_size, kv_heads, head_dim, bf16
    num_blocks = int(total * 0.9 - used - peak + current) // block_bytes
    # Allocate one big tensor: [2, 28, num_blocks, 256, 8, 128]
    self.kv_cache = torch.empty(2, num_hidden_layers, num_blocks, block_size, num_kv_heads, head_dim)
```

The KV cache is a single contiguous tensor with shape `[2, 28, N, 256, 8, 128]`:
- `2` = key and value
- `28` = number of layers
- `N` = number of blocks (dynamically computed based on available GPU memory)
- `256` = tokens per block
- `8` = KV heads
- `128` = head dimension

**The run loop** (`model_runner.py:208-214`):

```python
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs)
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist()
    return token_ids
```

### 3.6 Sampling: From Logits to Tokens

**File:** `nanovllm/layers/sampler.py`

After the forward pass produces logits (151,936 scores per sequence), we need to select a token. Nano-vllm uses the **Gumbel-Softmax trick**:

```python
# sampler.py:11-15
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))  # temperature scaling
    probs = torch.softmax(logits, dim=-1)                         # convert to probabilities
    gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    sample_tokens = probs.div_(gumbel_noise).argmax(dim=-1)       # Gumbel trick
    return sample_tokens
```

**How it works:**
1. **Temperature scaling:** `logits / temperature`. Higher temperature = more random, lower = more deterministic
2. **Softmax:** converts logits to a probability distribution
3. **Gumbel trick:** `argmax(log(probs) - log(-log(U)))` where `U ~ Uniform(0,1)` is mathematically equivalent to sampling from the categorical distribution. The implementation uses `probs / Exponential(1)` which is equivalent.

**Why Gumbel over `torch.multinomial`?** The `argmax` operation is faster on GPU and plays nicely with `torch.compile`.

---

## 4. Part III -- Optimizations That Reduce Inference Time

This is where nano-vllm shines: it implements the same optimizations as the full vLLM in readable code.

### 4.1 Paged Attention and the Block Manager

**File:** `nanovllm/engine/block_manager.py`

**The Problem:** Naive KV cache allocates a contiguous chunk of memory for each sequence's maximum possible length. With 512 sequences at 4096 max length, this wastes enormous memory on partially-filled buffers.

**The Solution (Paged Attention):** Borrow the idea of virtual memory from operating systems. Divide the KV cache into fixed-size **blocks** (256 tokens each) and allocate them on demand:

```
Sequence A (300 tokens): [Block 7][Block 23]          (2 blocks, last partially filled)
Sequence B (600 tokens): [Block 1][Block 15][Block 42] (3 blocks, last partially filled)
                          ^--- blocks are NOT contiguous in memory!
```

The `BlockManager` (`block_manager.py:26-113`) tracks this mapping:

```python
class BlockManager:
    blocks: list[Block]           # all physical blocks
    free_block_ids: deque[int]    # available blocks
    used_block_ids: set[int]      # currently allocated blocks
    hash_to_block_id: dict        # prefix cache (hash -> block_id)
```

**Allocation** (`block_manager.py:59-82`): When a new sequence is scheduled for prefill, the block manager assigns blocks to cover all its tokens.

**Append** (`block_manager.py:96-113`): During decode, each sequence generates one more token. If the last block is full (256 tokens), a new block is allocated.

**Deallocation** (`block_manager.py:84-91`): When a sequence finishes, its blocks are freed. Reference counting allows shared blocks (prefix caching).

The `block_table` for each sequence maps logical block indices to physical block IDs. The attention kernel uses this table to look up the correct KV cache locations.

### 4.2 Prefix Caching

**File:** `block_manager.py:35-41, 59-82`

When multiple prompts share the same prefix (e.g., a system prompt), their KV cache entries are identical. Prefix caching detects this and **shares blocks**:

```python
# block_manager.py:36-41 -- Hash a block's token content
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # chain with previous block's hash
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

During allocation:
```python
# block_manager.py:63-81 (simplified)
for each block in sequence:
    hash = compute_hash(block_tokens, previous_block_hash)
    if hash in hash_to_block_id and tokens match:
        # CACHE HIT: reuse existing block, skip re-computation
        seq.num_cached_tokens += block_size
        existing_block.ref_count += 1
    else:
        # CACHE MISS: allocate new block
        new_block = allocate_free_block()
```

**The chain hashing** is critical: block 2's hash depends on block 1's hash, which depends on block 0's hash. This means two sequences only share a block if *all preceding blocks* also match -- exactly the semantics of a shared prefix.

**Impact on prefill:** When there's a cache hit, the model skips those tokens entirely (`model_runner.py:137`):
```python
input_ids.extend(seq[seq.num_cached_tokens:])  # only process un-cached tokens
```

### 4.3 Flash Attention

**File:** `nanovllm/layers/attention.py`

Nano-vllm uses [Flash Attention](https://github.com/Dao-AILab/flash-attention) for the actual attention computation. Flash Attention is an IO-aware attention algorithm that:

1. **Avoids materializing the full attention matrix** (N x N), which is huge for long sequences
2. **Tiles the computation** to fit in GPU SRAM (fast on-chip memory)
3. **Fuses softmax with the matmul** to avoid extra memory reads/writes

Two different Flash Attention functions are used depending on the phase:

```python
# attention.py:64-74
if context.is_prefill:
    # Variable-length attention: handles batched sequences of different lengths
    o = flash_attn_varlen_func(q, k, v,
        cu_seqlens_q=..., cu_seqlens_k=...,   # cumulative sequence lengths
        max_seqlen_q=..., max_seqlen_k=...,
        softmax_scale=self.scale,
        causal=True,                           # causal mask (no future tokens)
        block_table=context.block_tables)      # paged attention support
else:
    # Decode: single new token attends to full KV cache
    o = flash_attn_with_kvcache(q.unsqueeze(1),   # [batch, 1, heads, dim]
        k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=self.scale,
        causal=True)
```

- **`flash_attn_varlen_func`**: Handles variable-length sequences packed into a single batch (prefill). The `cu_seqlens` (cumulative sequence lengths) arrays tell it where each sequence starts and ends.
- **`flash_attn_with_kvcache`**: Optimized for the decode case where Q has length 1 but K/V are the full context from the cache.

### 4.4 CUDA Graphs

**File:** `nanovllm/engine/model_runner.py:217-251`

**The Problem:** During decode, each step processes only one token per sequence. The GPU computation is tiny, but Python/CUDA launch overhead (CPU dispatching kernels to GPU) becomes the bottleneck.

**The Solution:** Capture the entire decode forward pass as a **CUDA graph** -- a recorded sequence of GPU operations that can be replayed with near-zero CPU overhead.

```python
# model_runner.py:217-251 (simplified)
def capture_cudagraph(self):
    # Pre-capture graphs for batch sizes: [1, 2, 4, 8, 16, 32, ...]
    for bs in [1, 2, 4, 8, 16, ...]:
        graph = torch.cuda.CUDAGraph()
        # Warmup run (allocate memory)
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        # Capture the computation graph
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
        self.graphs[bs] = graph
```

**Replay** during decode (`model_runner.py:194-206`):
```python
# Find the smallest pre-captured batch size >= actual batch size
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
# Copy current inputs into the graph's pre-allocated buffers
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph_vars["slot_mapping"][:bs] = context.slot_mapping
# Replay: runs ALL GPU kernels with zero CPU dispatch overhead
graph.replay()
```

**Key details:**
- Graphs are captured for specific batch sizes: `[1, 2, 4, 8, 16, 32, ..., 512]`
- A shared memory pool (`graph.pool()`) is used across all graphs to minimize memory overhead
- CUDA graphs are NOT used for prefill (variable-length, compute-bound, not launch-bound)
- The `enforce_eager=True` flag disables CUDA graphs (useful for debugging)

### 4.5 torch.compile

Several hot-path operations are decorated with `@torch.compile`:

| Component | File | What It Compiles |
|---|---|---|
| **RoPE forward** | `rotary_embedding.py:37` | Position embedding lookup + rotation |
| **RMSNorm** | `layernorm.py:16, 28` | Both plain and fused residual variants |
| **SiLU + Mul** | `activation.py:11` | SwiGLU activation |
| **Sampler** | `sampler.py:10` | Temperature scaling + Gumbel sampling |

`torch.compile` uses the Torch Inductor backend to fuse element-wise operations, eliminate temporary tensors, and generate optimized GPU kernels. For example, the RMSNorm + residual add becomes a single fused kernel instead of 5 separate operations.

### 4.6 Tensor Parallelism

**File:** `nanovllm/layers/linear.py`

Tensor parallelism splits the model across multiple GPUs within a single node. Nano-vllm implements two parallelism patterns:

**Column Parallelism** (split output dimension):
```
          GPU 0                    GPU 1
Input x ──> [W_top] ──> y_top    [W_bot] ──> y_bot
            (half of rows)        (other half)
```
Used for: QKV projection, gate_up projection (where each GPU computes different heads/features).

**Row Parallelism** (split input dimension):
```
          GPU 0                 GPU 1
y_top ──> [W_left] ──> z_0     [W_right] ──> z_1
                          \                 /
                           all_reduce (sum)
                                 |
                                 z (full result)
```
Used for: output projection, down projection (where partial results from column-parallel layers are combined).

```python
# linear.py:149-153 -- Row parallel forward with all-reduce
def forward(self, x):
    y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
    if self.tp_size > 1:
        dist.all_reduce(y)   # sum partial results across GPUs
    return y
```

The pattern pairs naturally: a column-parallel layer's output feeds into a row-parallel layer, with a single all-reduce in between.

**Weight loading** (`linear.py:65-70, 142-147`): Each rank loads only its shard from the safetensors file:
```python
# ColumnParallel: shard along output dimension (dim 0)
shard_size = param_data.size(0)
start_idx = rank * shard_size
loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)

# RowParallel: shard along input dimension (dim 1)
loaded_weight = loaded_weight.narrow(1, start_idx, shard_size)
```

**Embeddings** (`embed_head.py:9-42`): The 151,936-token vocabulary is also sharded. Each GPU handles a subset of token IDs, masking out tokens outside its range and using an all-reduce to combine:

```python
# embed_head.py:34-42
def forward(self, x):
    mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
    x = mask * (x - self.vocab_start_idx)    # local token indices
    y = F.embedding(x, self.weight)           # lookup in local shard
    y = mask.unsqueeze(1) * y                 # zero out non-local tokens
    dist.all_reduce(y)                        # sum across GPUs
    return y
```

**Inter-process communication** (`model_runner.py:41-84`): Rank 0 coordinates execution via shared memory. It serializes method calls (name + args) with pickle, writes to shared memory, and signals worker processes via `multiprocessing.Event`.

### 4.7 Triton Kernels

**File:** `nanovllm/layers/attention.py:10-30`

The KV cache write operation uses a custom **Triton** kernel for maximum throughput:

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D):
    idx = tl.program_id(0)                                 # one program per token
    slot = tl.load(slot_mapping_ptr + idx)                  # which cache slot?
    if slot == -1: return                                   # padding token, skip
    key = tl.load(key_ptr + idx * key_stride + arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + arange(0, D))
    tl.store(k_cache_ptr + slot * D + arange(0, D), key)   # write K to cache
    tl.store(v_cache_ptr + slot * D + arange(0, D), value)  # write V to cache
```

**Why Triton?** Writing to the KV cache requires a scatter operation (each token goes to a different slot determined by paged attention). This is hard to express efficiently with PyTorch ops but trivial in Triton, which compiles to efficient GPU code. Each Triton program handles one token, and thousands run in parallel.

### 4.8 Fused Residual + LayerNorm

**File:** `nanovllm/layers/layernorm.py:28-40`

The `add_rms_forward` fuses the residual addition with RMSNorm into a single pass:

```python
@torch.compile
def add_rms_forward(self, x, residual):
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())   # residual add (in-place)
    residual = x.to(orig_dtype)             # save new residual BEFORE norm
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))     # normalize
    x = x.to(orig_dtype).mul_(self.weight)
    return x, residual
```

Without fusion, this would be: (1) add residual -> write to memory, (2) read from memory -> compute norm -> write to memory. With fusion under `@torch.compile`, the Inductor backend generates a single kernel that reads once, does everything, and writes once.

### 4.9 Merged QKV and Gate-Up Projections

**Files:** `nanovllm/layers/linear.py:76-128` and `nanovllm/models/qwen3.py:42-48, 99-103`

Instead of three separate linear layers for Q, K, and V, nano-vllm packs them into a single `QKVParallelLinear`:

```python
# One matmul instead of three:
qkv = self.qkv_proj(hidden_states)     # [seq, 1024] -> [seq, (16+8+8)*128 = 4096]
q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
```

Similarly, `gate_proj` and `up_proj` are merged into a single `MergedColumnParallelLinear`:

```python
gate_up = self.gate_up_proj(x)         # [seq, 1024] -> [seq, 3072*2 = 6144]
# split inside activation
```

**Why merge?** A single large matmul is more efficient than multiple small ones. The GPU's compute units stay better utilized, and there's only one kernel launch instead of two or three. The `packed_modules_mapping` in the model class (`qwen3.py:186-192`) tells the weight loader how to place HuggingFace's separate weight files into the merged tensors.

### 4.10 Preemption and Memory Management

**File:** `nanovllm/engine/scheduler.py:46-51, 60-63`

When the KV cache is full and a running sequence needs a new block, the scheduler must **preempt** (evict) another sequence:

```python
# scheduler.py:46-51
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())    # evict the LAST sequence (most recently added)
    else:
        self.preempt(seq)                    # evict self if nothing else to evict
        break
```

Preemption (`scheduler.py:60-63`):
```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)      # free all its KV cache blocks
    self.waiting.appendleft(seq)            # put at FRONT of waiting queue (priority)
```

The evicted sequence will be re-prefilled when there's enough memory. This means some computation is wasted, but it prevents the entire system from stalling. Preempted sequences go to the *front* of the waiting queue so they're processed before new requests.

---

## 5. Part IV -- Hands-On Exercises

### Exercise 1: Run the Example

Run `example.py` to see the basic API in action:

```bash
python example.py
```

Observe the tqdm progress bar showing prefill throughput (tok/s) and decode throughput (tok/s). Prefill should be much higher.

### Exercise 2: Benchmark with vs without CUDA Graphs

```python
from nanovllm import LLM, SamplingParams

# Without CUDA graphs (eager mode)
llm_eager = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True)
# With CUDA graphs
llm_graph = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=False)

prompts = ["Explain quantum computing in simple terms."] * 32
sp = SamplingParams(temperature=0.6, max_tokens=128)

# Compare decode throughput in the tqdm bar
outputs_eager = llm_eager.generate(prompts, sp)
outputs_graph = llm_graph.generate(prompts, sp)
```

**What to observe:** Decode throughput (tok/s) should be significantly higher with CUDA graphs enabled, especially for small batch sizes where launch overhead dominates.

### Exercise 3: Observe Prefix Caching

```python
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = "~/huggingface/Qwen3-0.6B/"
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True)

# Same system prompt, different questions
system = "You are a helpful math tutor. Explain step by step."
prompts = [
    f"{system}\nWhat is 2+2?",
    f"{system}\nWhat is the derivative of x^2?",
    f"{system}\nSolve x^2 - 4 = 0.",
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    )
    for p in prompts
]

sp = SamplingParams(temperature=0.6, max_tokens=128)
outputs = llm.generate(prompts, sp)
```

The shared system prompt prefix will be cached after the first sequence's prefill. Subsequent sequences skip re-computing attention for those tokens.

### Exercise 4: Trace the Data Flow

Add print statements to trace a single token through the system:

1. In `scheduler.py:schedule()` -- print which sequences are scheduled and whether it's prefill/decode
2. In `model_runner.py:run()` -- print `input_ids.shape`, `positions`, and `is_prefill`
3. In `attention.py:forward()` -- print `q.shape`, `k.shape`, `context.is_prefill`
4. In `sampler.py:forward()` -- print `logits.shape` and the sampled token

### Exercise 5: Measure KV Cache Utilization

Add monitoring to the block manager:

```python
# In block_manager.py, add to allocate() and deallocate():
print(f"Blocks: {len(self.used_block_ids)} used / {len(self.free_block_ids)} free")
```

Run `bench.py` with 256 requests and watch how blocks are allocated and freed as sequences start and complete. Notice preemptions when memory is tight.

### Exercise 6: Read the Code Top-Down

Recommended reading order for deep understanding:

1. `example.py` -- see how the API is used
2. `nanovllm/engine/llm_engine.py` -- the main loop
3. `nanovllm/engine/scheduler.py` -- how requests are scheduled
4. `nanovllm/engine/sequence.py` -- what a request looks like
5. `nanovllm/engine/block_manager.py` -- memory management
6. `nanovllm/engine/model_runner.py` -- GPU execution
7. `nanovllm/models/qwen3.py` -- the model architecture
8. `nanovllm/layers/attention.py` -- Flash Attention + KV cache
9. `nanovllm/layers/rotary_embedding.py` -- position encoding
10. `nanovllm/layers/linear.py` -- tensor parallelism
11. `nanovllm/layers/layernorm.py` -- RMSNorm
12. `nanovllm/layers/sampler.py` -- token sampling
13. `nanovllm/layers/embed_head.py` -- embeddings + LM head
14. `nanovllm/layers/activation.py` -- SwiGLU
15. `nanovllm/utils/loader.py` -- weight loading
16. `nanovllm/utils/context.py` -- thread-local context passing

---

## 6. Appendix: File Map

```
nano-vllm/                          ~1,200 lines total
├── example.py                      Quick-start demo (34 lines)
├── bench.py                        Throughput benchmark (47 lines)
├── pyproject.toml                  Dependencies and project metadata
│
└── nanovllm/
    ├── __init__.py                 Exports: LLM, SamplingParams
    ├── config.py                   Config dataclass (model path, batch sizes, etc.)
    ├── sampling_params.py          SamplingParams dataclass (temperature, max_tokens)
    │
    ├── engine/
    │   ├── llm_engine.py           Main engine: generate() loop, tokenizer
    │   ├── scheduler.py            Continuous batching scheduler (prefill/decode)
    │   ├── block_manager.py        Paged KV cache with prefix caching
    │   ├── sequence.py             Sequence state machine (WAITING/RUNNING/FINISHED)
    │   └── model_runner.py         GPU execution: input prep, CUDA graphs, forward pass
    │
    ├── models/
    │   └── qwen3.py                Qwen3 model (Attention, MLP, DecoderLayer, CausalLM)
    │
    ├── layers/
    │   ├── attention.py            Flash Attention + Triton KV cache store
    │   ├── rotary_embedding.py     RoPE implementation
    │   ├── linear.py               Column/Row parallel linear layers
    │   ├── layernorm.py            RMSNorm with fused residual
    │   ├── activation.py           SiLU + Mul (SwiGLU)
    │   ├── embed_head.py           Vocab-parallel embedding + LM head
    │   └── sampler.py              Gumbel-softmax token sampling
    │
    └── utils/
        ├── context.py              Thread-local inference context
        └── loader.py               Safetensors weight loader
```
