# 新增候选匹配算法对照：BM25、加权 Jaccard、字符 TF-IDF 与排序融合

## 测试目标和实现位置

复用同一批 [45 例](EXPANDED_BENCHMARK.zh-CN.md)：43 例有人工指定的对应组，2 例没有对应内容。每例候选段落与标题均已人工给定，真实 JEDEC 仍只有 JESD209-4D/4E 同一 Scope 的 2 组短片段。本轮只测试**候选组排序**；不测全文检索、自动拒判、要求含义判定或端到端 diff。

新增方法的完整实现是 [alternative_methods.py](experiments/alternative_methods.py)，逐案例分数与选中 ID 在 [alternative_methods.encoder.results.json](experiments/alternative_methods.encoder.results.json)。原有精确匹配、词面相似度和 0.50 接受门槛的实现见 [compare_baselines.py](experiments/compare_baselines.py)；encoder 的单独实验实现见 [evaluate_encoder.py](experiments/evaluate_encoder.py)。这些是可运行的**评测实现**，并非可直接读取数百 MB JSON 的生产引擎。

新增方法没有使用答案调权重；但案例和标题由人编写，且此前已查看过同一批案例，所以以下数字是开发诊断，不能当盲测泛化成绩。

## 方法与结果

| 方法 | 在脚本中的实现 | 有对应组案例的 top-1 完整组选对 | 适合的作用 |
|---|---|---:|---|
| 双向 token `SequenceMatcher` | `similarity()`，不看标题 | 38/43 | 便宜的局部词序基线 |
| 词面＋标题＋相邻段落组 | `0.7×词面 + 0.3×标题 − 组长度惩罚` | **42/43** | 目前小候选池中最稳的可解释排序 |
| **BM25** | 词项频率、文档频率、长度归一化，`k1=1.2,b=0.75` | 40/43 | 以后可用于全文候选召回；这里还不是全文 BM25 实验 |
| **IDF 加权 Jaccard** | 候选与旧段词项集合的加权交/并 | 41/43 | 对顺序变化稳健的廉价补充 |
| **字符 3/4/5-gram TF-IDF** | 字符片段向量余弦 | 41/43 | 抗轻微改写和分词差异 |
| 本地英文 text encoder | `BAAI/bge-small-en-v1.5` 向量余弦 | 41/43 | 低词面重合时的候选补充 |
| **RRF 三路融合** | 词面＋标题、BM25、字符 TF-IDF 的倒数名次合并，`k=60` | 41/43 | 不需把不同量纲分数直接相加 |
| **RRF 加 encoder** | 上述三路再加 encoder 名次 | 41/43 | 本数据未展示增益 |

单看 top-2，所有方法均为 43/43；**这不是有意义的全文 Recall@2**，因为多数案例总共只有两个左右的人工候选组。两例无对应内容被排除在 top-1 分母外：排序器总会给出第一名，不能凭第一名宣布“找到了对应内容”。原有词面 0.50 门槛在 43 个正例上只接受且选对 38 个，并在这 2 个负例上拒绝匹配；门槛并未在真实数据上校准。

### 逐例诊断

- `extra_refresh_paraphrase`：纯词面把无关候选排第一；标题增强、BM25、字符 TF-IDF 和 encoder 均排对。但词面＋标题分约 `0.46471`，按原门槛会拒绝正确候选。**排序选对与自动接受是两项不同任务。**
- `extra_duplicate_template`：错误章节的同字句使词面＋标题、BM25、加权 Jaccard、字符 TF-IDF、encoder 和两种 RRF 均选错。另一例 `repeated_boilerplate` 中，只有词面＋标题＋组方法选对。局部文本方法无法稳定解决重复模板，需要章节路径、对象、邻接段和全局冲突约束。
- `real_D-scope-single`：BM25 在两条真实 Scope 短片段中有一条选错；词面＋标题、字符 TF-IDF 与 encoder 选对。小候选池的词频统计不适合推断它在全文检索中的效果。
- `extra_zero_distractor`：不存在对应段，所有排序器仍会给无关句排第一。需要拒判与完整文档回查；encoder 余弦也不是“存在对应”的概率。

## 方案更新

**首期排序仍采用章节约束＋词面＋标题＋相邻段落组**，因为它可解释、便宜，且在现有诊断中 top-1 最高。不要据 42/43 宣称它是真实 LPDDR6 最优方法。BM25 应在实际数百 MB JSON 上作为候选检索器与全文词面扫描比较；本次小池排名不能替代检索召回评测。字符 TF-IDF、加权 Jaccard 和 RRF 当前没有显示独立增益，暂不加入默认路径。

Encoder 对大幅改写有候选价值，但这批案例里没有修复词面＋标题的唯一 top-1 错例。下一轮应在**完整章节**的真实开发集上测“词面候选并集＋encoder 候选”的 Recall@K、误配、内存和调用成本，并在冻结集确认收益。若正确候选已在池内但总选错，优先加入章节/对象约束与全局一对一匹配；若候选池根本不含正确段，再考虑 BM25 或 encoder 扩召回。匹配后仍需 token 编辑、要求字段比较和上下文/引用复核；这些排序器不能单独完成正文 diff。

## 复现

在仓库根目录运行，第一条只用 Python 标准库，第二条需要隔离安装的 FastEmbed 与缓存模型（安装见 [测试记录](ENCODER_AND_LLM_TEST.zh-CN.md)）：

```powershell
python experiments/alternative_methods.py --output experiments/alternative_methods.results.json
python experiments/alternative_methods.py --deps .test-workspace/encoder-deps --cache .test-workspace/encoder-models --output experiments/alternative_methods.encoder.results.json
```

评分只使用文本与标题，不读取 `gold_ids`；答案仅在汇总时使用。字符 TF-IDF 和 BM25 的 IDF 来自这 45 例候选组构成的小语料，生产环境需建立版本化索引并重新评测。`k1`、`b`、RRF `k` 目前是未调优的常用起点，不是 JEDEC 优化参数。
