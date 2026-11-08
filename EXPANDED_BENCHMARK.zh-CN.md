# 扩大案例集：正文匹配压力测试

## 范围与可复现材料

在原 15 个合成诊断案例之外，新增 28 个**人工编写的合成案例**，覆盖改写、数值、否定、条件、规范强度、单位等价、引用、对象、移动、重复句、拆分和删除。再加入 JESD209-4D/4E 公开 Scope 中已有的 2 组短片段，共 **45 例**。43 例存在对应组，2 例预设无对应内容。两组真实片段来自同一小节，彼此相关，**不能算作两份完整文档的独立抽样**。

数据、评分及运行入口在 [expanded_benchmark.py](experiments/expanded_benchmark.py)。[逐例 encoder 与词面结果](experiments/expanded.encoder.results.json) 和 [不调用 encoder 的结果](experiments/expanded.results.json) 均已保存。真实片段的出处及措辞见 [局部 JEDEC 实例](REAL_JEDEC_EXAMPLE.zh-CN.md)，不在这里复制全文。

候选池由人手工给出，每例只有少数候选；即使匹配全对，也不证明数百 MB 文档中能找到它们。该脚本只比较**找对应内容**，没有评测“要求是否改变”的语义分类或全局一对一对齐。合成句子是虚构技术文本，不是 JEDEC 原文。

## 已运行结果

| 子集 | 存在对应的组 | 结构＋词面 top-1 选对 | 结构＋词面按 0.50 门槛接受且选对 | Encoder top-1 选对 |
|---|---:|---:|---:|---:|
| 原 15 例合成案例 | 14 | 14 | 13 | 13 |
| 新增 28 例合成案例 | 27 | 26 | 23 | 26 |
| 真实 JEDEC Scope 短片段 | 2 | 2 | 2 | 2 |
| **合计** | **43** | **42** | **38** | **41** |

Encoder 的 top-2 包含正确对应组为 **43/43**；这同样只是在人工候选池里排序。两种方法在两例预设无对应内容上，词面 0.50 门槛都没有误接受；encoder 本身没有拒判机制，不能用 top-1 证明删除。

### 具体失败例

| 案例 | 词面最高分及结果 | Encoder 最高分及结果 | 诊断 |
|---|---|---|---|
| `extra_refresh_paraphrase` | 正确候选 `0.46471`，被 0.50 门槛拒绝 | 正确候选约 `0.87`，排第一 | 改写会被固定词面门槛漏掉 |
| `extra_voltage_paraphrase` | 正确候选约 `0.46`，被拒绝 | 正确候选约 `0.86`，排第一 | Encoder 可补候选，但余弦不是自动接受概率 |
| `extra_mode_register_paraphrase` | 正确候选约 `0.43`，被拒绝 | 正确候选约 `0.92`，排第一 | 同上 |
| `extra_duplicate_template` | 错误章节的完全相同句得 `0.70` 并被选中 | 同一错误候选余弦 `1.00`，也排第一 | 两种方法都不能仅靠局部相似度解决重复模板句 |
| `extra_zero_distractor` | 唯一无关候选约 `0.43`，未接受 | 无关候选余弦约 `0.70` | Encoder 必须有独立拒判、结构和覆盖检查 |

数值、否定、条件类在这组手工候选池里匹配成功，**不表示系统已经正确判定要求变化**。匹配分衡量“可能是同一段要求”，不能当作语义一致度。

## 当前工程决定

保留结构＋词面作为可解释的主路径；用 encoder 扩充低词面重合案例的候选，但先交给结构约束、要求字段比较或人工复核。不要在这组自编案例上调一个余弦自动接受门槛。重复模板句是当前两路的共同失败点，应增加章节路径、对象、邻接段和全局冲突约束。

真正估计业务效果，需要导入两版完整的解析 JSON，从整章或整份文件生成候选，并由专家在未筛过的输入上标注对齐组与要求变化。冻结测试集至少应单独覆盖大幅改写、重复模板、移动、拆合、删除、引用变化和否定/数值变化，并报告候选 Recall@K、组匹配、要求变更召回、误报及每千段成本。届时再比较词面、encoder 和两路并集。

## 复现

```powershell
python experiments/expanded_benchmark.py --output experiments/expanded.results.json
python experiments/expanded_benchmark.py --deps .test-workspace/encoder-deps --cache .test-workspace/encoder-models --output experiments/expanded.encoder.results.json
```

第二条命令要求本地预先安装 `fastembed==0.9.1` 到指定依赖目录，并缓存 `BAAI/bge-small-en-v1.5`；安装方法见 [encoder 测试记录](ENCODER_AND_LLM_TEST.zh-CN.md)。模型仅在本机处理这些案例。测试结果会随依赖、模型文件和脚本版本变化；提交的 JSON 是本次实际运行输出。
