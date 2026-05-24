# TAAC-KDD Cup 2026 (ACM SIGKDD) - 0.831025

本方案基于 [流水](https://github.com/nhdzTVlxb) 开源的 [v9](https://github.com/nhdzTVlxb/TAAC-2026-Tencent-KDD) 进行改进，最终分数为 `0.831025`。整体上保留 v9 的模型框架，主要优化时间编码、NS token 构造和模型容量。这三项改动单独使用均带来约 `0.003+-0.004+` 的收益，是主要增益来源。

## 1. 时间编码替换

EDA 发现转化率与 hour 存在明显相关性，因此对时间编码进行了重新设计。相比 v9 中较分散的时间特征组合，当前方案收敛为绝对时间特征、增强序列时间 bucket 和 temporal attention bias。

```bash
--ns_token_time_feats abs
--temporal_encoding_mode enhanced
--shared_temporal_embeddings
--use_temporal_bias
--no_calendar_time
--no_time_span_buckets
--no_time_gap
```

主要变化：

- 使用样本级绝对时间连续特征，并注入 NS token。
- 使用 `enhanced` 时间编码，引入 session bucket 与 cross-day bucket。
- 保留 temporal attention bias。
- 关闭 calendar time、time span bucket 和 time gap，减少冗余时间信号。

样本级绝对时间使用周期特征编码，投影到 `d_model` 后加到 NS token：

```python
periods = np.array([3600, 86400, 604800, 2592000, 31536000], dtype=np.float32)
phases = timestamps.reshape(-1, 1).astype(np.float32) / periods.reshape(1, -1)

abs_time_feats = np.concatenate([
    np.sin(2.0 * np.pi * phases),
    np.cos(2.0 * np.pi * phases),
    seconds_in_day[:, None] / 86400.0,
    hour_of_day[:, None] / 23.0,
], axis=1)

ns_tokens = ns_tokens + F.silu(abs_time_proj(abs_time_feats)).unsqueeze(1)
```

序列侧构造 session bucket 与 cross-day bucket。前者根据相邻行为时间间隔划分 session，后者根据样本时间与历史行为时间的天级差值划分：

```python
valid = ts_padded > 0

gaps = np.zeros_like(ts_padded)
gaps[:, 1:] = np.maximum(ts_padded[:, :-1] - ts_padded[:, 1:], 0)
new_session = valid & (gaps > 30 * 60)
new_session[:, 0] = valid[:, 0]
session_bucket = np.clip(np.cumsum(new_session, axis=1), 0, 16)
session_bucket[~valid] = 0

current_day = timestamps.reshape(-1, 1) // 86400
event_day = ts_padded // 86400
day_delta = np.maximum(current_day - event_day, 0)

cross_day_bucket = np.zeros_like(ts_padded)
cross_day_bucket[(valid) & (day_delta == 0)] = 1
cross_day_bucket[(valid) & (day_delta == 1)] = 2
cross_day_bucket[(valid) & (day_delta == 2)] = 3
cross_day_bucket[(valid) & (3 <= day_delta) & (day_delta <= 6)] = 4
cross_day_bucket[(valid) & (7 <= day_delta) & (day_delta <= 13)] = 5
cross_day_bucket[(valid) & (14 <= day_delta) & (day_delta <= 29)] = 6
cross_day_bucket[(valid) & (day_delta >= 30)] = 7
```

这些 bucket 作为 embedding 注入行为 token：

```python
token_emb = token_emb + time_embedding(time_bucket)
token_emb = token_emb + session_embedding(session_bucket)
token_emb = token_emb + cross_day_embedding(cross_day_bucket)
```

## 2. Scale NS Token 和 Model Dim

在模型容量上，当前方案主要扩大 NS token 数量，并配合提升 `d_model` 与 query 数量：

```bash
--user_ns_tokens 11
--user_dense_tokens 1
--item_ns_tokens 4
--item_dense_tokens 0
--num_queries 4
--d_model 128
```

相对 v9 的变化如下：

| 配置              | v9     | best    |
| ----------------- | ------ | ------- |
| user NS tokens    | `3`  | `11`  |
| item NS tokens    | `4`  | `4`   |
| num queries       | `2`  | `4`   |
| d_model           | `64` | `128` |

实验中尝试过对多个模块进行 scale，扩大其他部分通常会造成效果下降；相对稳定有效的是增加 NS token 数量，并适度提升 `d_model`。该改动主要提升特征表达容量和多序列信息读取能力。

## 3. Tokenizer 修正

v9 使用 `rankmixer` tokenizer，会将拼接后的字段 embedding 按维度切分，可能破坏单个字段的语义完整性。当前方案改为 `fieldaware`：

```bash
--ns_tokenizer_type fieldaware
--ns_groups_json ""
```

`fieldaware` 按 fid 粒度分配 token bucket，保证字段 embedding 不被切碎，同时仍支持通过 `user_ns_tokens` 和 `item_ns_tokens` 调节 NS token 数量。

核心实现如下：

```python
all_fids = [fid for group in groups for fid in group]
token_fid_buckets = split_fids(all_fids, num_ns_tokens)

tokens = []
for bucket, proj in zip(token_fid_buckets, token_projs):
    bucket_emb = torch.cat([
        embed_fid(int_feats, fid_idx)
        for fid_idx in bucket
    ], dim=-1)
    tokens.append(F.silu(proj(bucket_emb)).unsqueeze(1))

ns_tokens = torch.cat(tokens, dim=1)
```

其中 `embed_fid` 对单值字段直接查 embedding，对多值字段做 masked average。这样既保留字段语义完整性，也保留 token 数量的可调性。

## 4. Semi-local Mask

序列编码器加入 semi-local causal mask：

```bash
--seq_semilocal_causal_mask
--seq_semilocal_window 128
```

该 mask 约束每个位置只关注历史窗口内的行为，使序列 attention 更符合时间因果关系。该改动带来小幅提升，主要作为序列建模的稳定性增强。

## 总结

相对 [v9](https://github.com/nhdzTVlxb/TAAC-2026-Tencent-KDD)，`best` 的主要收益来自三部分：时间编码替换、field-aware tokenizer 修正，以及 NS token/model dim 的容量扩展。前三项单独使用均有约 `0.004` 左右的收益；semi-local mask 提供额外小幅增益。最终分数为 `0.831025`。
