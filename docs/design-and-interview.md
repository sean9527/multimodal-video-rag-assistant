# Multimodal Video RAG Assistant — 设计文档 & 面试问答

> 本文随项目推进持续更新。当前进度:**Phase 0 脚手架 ✅ · Phase 1 ingestion(转写 + 关键帧)✅ · Phase 2 embedding/index 进行中**。

---

## 0. 一句话介绍(elevator pitch)

**中文:** 一个多模态视频问答系统:上传视频后用自然语言提问,系统不仅给出答案,还能定位到视频里的具体时间点(timestamp citation)。它同时索引语音转写文本和视觉关键帧,用 CLIP 的共享图文空间做跨模态检索。

**English:** *"A multimodal video Q&A system: upload a video, ask questions in natural language, and get answers with timestamp citations pointing back to the exact moment. It indexes both the speech transcript and visual keyframes, and retrieves across both using CLIP's shared image–text embedding space."*

---

## 1. 系统架构

### 索引阶段 Ingestion(已实现)
```
video --ffmpeg--+-- audio --Whisper(faster-whisper)--> transcript segments (start/end)
                |                                         └─ chunking --> BGE-m3 text embedding --+
                └-- frames --scene detect / uniform------> keyframes (timestamp)                  +--> LanceDB
                                                            └─ OpenCLIP image embedding ----------+   (vector + metadata:
                                                                                                      timestamp / modality / video_id)
```

### 查询阶段 Query(Phase 3–4 实现)
```
question --+-- BGE-m3 embedding ------> 文本索引 top-k --+
           └-- CLIP text encoder -----> 视觉索引 top-k --+--> 融合/排序 --> prompt(含 timestamp) --> LLM(Ollama) --> 答案 + timestamp 引用
```

### 数据模型(`src/mvrag/schemas.py`)
- `TranscriptSegment(start, end, text)` — 一段带时间跨度的语音
- `Keyframe(timestamp, frame_path, scene_index)` — 一帧带时间的画面
- `TextChunk(chunk_id, start, end, text)` — 合并后的检索粒度文本块
- `IngestionResult(...)` — 一个视频的全部产物,持久化为 `data/<video_id>/ingestion.json`

**核心:** transcript 和 keyframe 都锚定在**同一条时间线(秒)**上,这是"既能答题又能定位来源"的基础。

---

## 2. 关键技术选型与"为什么"

| 环节 | 选型 | 为什么 / 权衡 |
|---|---|---|
| 转写 | **faster-whisper** | 同权重跑在 CTranslate2 上,int8/fp16 量化,比原版 whisper 快约 4×、省显存 |
| 文本 embedding | **BGE-m3** | 多语言(英/中都强),测试视频可能是中文 lecture,纯英文 embedding 会漏检 |
| 视觉 embedding | **OpenCLIP ViT-B/32** | CLIP 共享图文空间,支持"文字→画面"跨模态检索;是候选人研究方向 |
| 向量库 | **LanceDB** | 列式、原生多模态、内嵌无需起服务、持久化好、挂元数据方便;备选 FAISS(最快但要自管元数据)、Chroma(最简单) |
| 生成 LLM | **Ollama(本地)** | 免费/离线/可复现(clone 即跑,无 API key);代价是质量不如 Claude/GPT,已把生成层抽象成接口便于替换 |
| 关键帧 | **场景检测 + 均匀采样兜底** | 场景检测适合有剪辑的视频;对着讲的 lecture 检不到切换,均匀采样保证覆盖 |
| 配置 | **pydantic-settings** | 集中、类型化、可环境变量覆盖,符合 12-factor |
| 打包 | **src-layout + `[ml]` 可选依赖** | 重依赖(torch)隔离,`pip install -e .` 快,结构专业 |

---

## 3. 核心差异化(vs 普通文本 RAG)—— 面试主打

普通 RAG 只索引文本(如 PDF chatbot)。视频有**三重信息:transcript + visual frames + time alignment**。本项目:
1. **多模态索引** — transcript chunk 和 keyframe embedding 都绑到统一 timestamp 时间线。
2. **跨模态检索** — 用 CLIP 共享空间,一句文字能召回相关画面帧,即使转写里根本没提(如"什么时候出现 object movement")。
3. **timestamp citation** — 每个答案回引到来源时刻,可点击跳转。

---

## 4. 面试问答

### 4.1 RAG 基础
**Q: 什么是 RAG?为什么不直接全塞 context 或 fine-tune?**
A: RAG = 先检索相关片段,再让 LLM 基于片段生成。全塞 context:长视频转写几万 token,贵、可能超窗口、大海捞针效果差。fine-tune:知识常变(每个新视频)、贵且慢、无法溯源。RAG 让知识可插拔、可 citation。

**Q: 这个项目需要 fine-tune 任何模型吗?什么时候才需要?**
A: **不需要,而且这是刻意设计——RAG 本身就是 fine-tune 的替代方案**。知识(视频内容)在推理时通过检索注入,加一门课 = 建索引,不是重训;所有模型(Whisper / BGE / CLIP / reranker / LLM)都**预训练直接用(zero-shot)**。对"带引用的事实问答",检索 + grounding 比把知识烤进权重更好:可溯源、可更新、少幻觉。**唯一可能值得微调的是 embedding 模型 / reranker**——若评估显示本领域检索差(通用 embedding 分不开专业术语),可用领域内 `(query, 相关 chunk)` 对微调;巧的是**做 eval 用的 synthetic query 生成,同一套数据就能当微调训练对**。立场:**先用 eval 量化,确认检索是瓶颈且通用 embedding 不够,再考虑微调——data-driven,而非上来就 fine-tune**。视觉弱换 Chinese-CLIP 这类更好的预训练模型即可,不必微调。

**Q: 完整讲一下检索流程。**
A: 见 §1。索引:视频→转写+关键帧(都带时间戳)→切 chunk→双 embedding→存 LanceDB(挂时间戳元数据)。查询:问题→双路 embedding 各取 top-k→融合→拼进 prompt→LLM 答题并回引 timestamp。

**Q: 怎么切 chunk?为什么 overlap?**
A: Whisper segment 太短(5–10s),单独 embedding 上下文不足。我把连续 segment 贪心合并到约 800 字符,保留合并后的 start/end,做约 150 字符 overlap——防止一个事实正好被切在边界丢上下文。chunk 过大又会稀释语义,需权衡。

### 4.2 Embedding & 向量检索
**Q: 什么是 embedding?怎么算相似度?**
A: 把文本/图像映射成稠密向量,语义近则向量近。用 cosine similarity。我存前做 L2 归一化,这样内积=cosine,检索更快。

**Q: 为什么 BGE-m3?**
A: 多语言(英/中都强)。测试视频可能中文,纯英文 embedding 会漏检中文。它还同时支持 dense/sparse,未来做 hybrid 方便。

**Q: dense vs sparse?什么是 hybrid?**
A: dense=向量语义检索,擅长"意思近但词不同";sparse=BM25/关键词,擅长精确词、专有名词。hybrid 用 RRF 等融合两者。MVP 先 dense,hybrid 列为改进。

**Q: 词不完全匹配也能检索到吗?**
A: 能,这就是 semantic search 的价值——靠语义而非字面。问"怎么改变张量形状"能命中讲 reshape/view 的片段。

**Q: 检索排序不准会不会导致答案错?**
A: 不一定。RAG 里 LLM 读的是 top-k **全部**片段,所以关键是 **recall**(正确片段在不在 top-k 里),而不是它排第几。实测:问 "reshape",讲 `.view/.reshape` 的片段只排到第 5,但我把 top-6 都喂给 LLM,它照样在里面找到并正确引用了 (00:21-00:45)。所以排序噪声在"喂 top-k 给 LLM"的场景下影响有限;**reranking 主要用于"只取 top-1/2"或给用户展示排序时的精度**。这也是为什么我优先保证 recall(合适的 k、多语言 embedding),把 reranking 作为增强项。

**Q: reranking 是什么?bi-encoder 和 cross-encoder 有什么区别?**
A: 两阶段检索。**bi-encoder**(我的 BGE)把 query 和片段**分开**编码成向量再比 cosine——快,但"粗",它从没同时看过 query 和片段。**cross-encoder**(bge-reranker-v2-m3)把 (query, 片段)**拼在一起**送进模型直接打相关性分——准得多,但太慢没法全库跑。所以经典做法:先用 bi-encoder 便宜地召回 N 个候选(如 20),再用 cross-encoder 精排到 top-k。实测:对 "reshape" 这个 query,dense 把讲 `.view/.reshape` 的片段排到第 5,reranker 把它拉到第 1。**语料越大,reranking 收益越大**——这也是为什么它在整门课/多视频规模下尤其重要。

### 4.3 多模态 & CLIP
**Q: CLIP 为什么能用文字检索图片?**
A: CLIP 用对比学习在海量(图,文)对上训练,把图像和文本编码到**同一共享向量空间**,匹配对相互靠近。所以文字 query 与相关画面帧 cosine 高,用文字向量去比图像向量即可检索。

**Q: 怎么融合文本检索和视觉检索?**
A: 两索引各 top-k 后融合。两条流相似度分布不同,直接比大小不公平,所以归一化后加权、或用 rank-based 的 RRF。生成时把两种来源都给 LLM 综合。

**Q: 为什么算多模态而不是普通文本 RAG?**
A: 见 §3。能回答纯视觉、转写没提的问题,是纯文本 RAG 做不到的。

**Q: 多模态一定比纯文本好吗?**
A: 不一定,取决于视频类型——这是我实测得到的结论。这个 MIT PyTorch 课全程是 Colab 代码 notebook,几乎每帧都是代码,CLIP 对所有关键帧的相似度都挤在 ~0.30、区分不开,视觉模态几乎没信息量,信号全在转写里。反过来,demo 演示、实验录像这类**视觉多样**的视频,关键帧能回答转写里根本没提的问题(如"什么时候出现 object movement")。所以结论是:**多模态的价值取决于画面多样性;系统同时建两条流,但检索时应按视频/query 动态加权**——这也是 Phase 3 融合策略要考虑的。

### 4.4 视频 / 音频
**Q: 时间戳从哪来?**
A: Whisper 每个 segment 自带 start/end;关键帧记录抽帧时刻。两者同一时间线。

**Q: faster-whisper 为什么快 4×?**
A: 同权重跑在 CTranslate2 推理引擎,int8/fp16 量化 + 推理优化;原版是纯 PyTorch。

**Q: VAD 是什么,为什么用?**
A: Voice Activity Detection。开 vad_filter 跳过静音,lecture 长停顿不产生空转写/错时间戳,更准。

**Q: 关键帧为什么两种策略?**
A: 场景检测靠画面内容变化找镜头切换,适合有剪辑的;对着讲的 lecture 检不到,均匀采样(每5s)兜底保证覆盖。超上限时全片均匀降采样,不砍尾巴。

### 4.5 向量库 & 扩展性
**Q: FAISS / Chroma / LanceDB 怎么选?**
A: FAISS 最快、算法可控但要自管元数据/持久化;Chroma 最简单;LanceDB 列式、原生多模态、内嵌、持久化好。本项目多模态+挂时间戳+想简单部署→LanceDB。

**Q: ANN 是什么?**
A: Approximate Nearest Neighbor。精确检索 O(N);ANN 用 HNSW(图)、IVF(聚类倒排)等结构,牺牲一点召回换次线性检索,百万级也毫秒返回。

**Q: 扩展到万级视频/百万向量?**
A: ANN 索引 + 按 video_id 元数据过滤缩范围 + embedding 批量/缓存 + ingestion 异步化(队列)+ 必要时分片 + top 加 reranking。

**Q: 能处理完整长视频吗?一门课 30+ 节、每节 1 小时,吃得消吗?**
A: 能,这正是 RAG + 元数据 scoping 的设计目的。
- **单节 1h 讲座**:转写线性耗时(GPU 几分钟 / CPU 十几分钟);keyframe 封顶 200 张(~18s 一张);chunk ~150–200 个。无长度硬限制。
- **一门课(36 节)**:约 1.5 万向量;10 门课 ~15 万。LanceDB 轻松到百万级——这个量级 brute-force 都毫秒级,超过 ~10万–百万再建 ANN 索引(`create_index`)。存储每门课几 GB(视频文件占大头)。
- **关键洞见:生成成本与语料规模无关**——RAG 只把 top-k 喂给 LLM,回答 36 小时的课和回答一个视频一样快一样便宜。这就是检索的意义。
- **规模化真正的瓶颈**:①转写要用 GPU(几十小时视频 CPU 太慢)②语料变大 ranking 噪声增大 → reranking + hybrid 更重要 ③ingestion 走异步任务队列 ④百万级向量建 ANN 索引。
- **不切分**:完整讲座直接进,timestamp 全局正确(切段会让时间戳变成段内相对时间)。

**Q: 多门课 / 多视频,数据怎么隔离?会不会串答案?**
A: **一个共享向量库 + 元数据 scoping**,不给每门课分开建库。每行带 `course_id` / `video_id`,检索时按 `WHERE course_id=... [AND video_id=...]` 过滤,做到逻辑隔离,课程间绝不串。这样能"按整门课问"(跨该课所有视频)或"缩到单个视频"。这是向量检索里 **multi-tenancy** 的标准做法——物理分库是硬隔离,更难管、还不能跨课查,只有需要严格安全边界(如不同客户)时才用。面试问"你怎么支持多用户/多知识库"就答这套。

### 4.6 生成 & 防幻觉
**Q: 怎么防止 LLM 编造?**
A: grounding——system prompt 要求"只根据提供片段回答,不足就说不知道";把片段+时间戳放进 prompt;要求每个论断给来源 timestamp。可选:vision LLM 看关键帧二次验证视觉问题。

**Q: timestamp citation 怎么实现?**
A: 每个检索片段带 start/end;prompt 里连时间戳一起给 LLM 并要求标注来源;前端把 timestamp 做成可点击跳转到那一秒。

**Q: 为什么本地 Ollama?权衡?**
A: 免费/离线/数据不出机/可复现;代价是本地小模型质量与 vision 弱。**生成层抽象成 provider 接口**(`generation/llm.py`:`LLMClient` 接口 + Ollama/OpenAI/Claude 三个 adapter + 工厂),换云端 API 只需 `.env` 改 `MVRAG_LLM_PROVIDER` + 设 key,ingestion/检索/UI 一行不动——**依赖倒置**,高层逻辑不绑定具体 LLM。

### 4.7 工程 & 系统设计
**Q: 怎么部署?**
A: Docker + GPU 透传;模型运行时下载缓存;ingestion 长任务用 background task + 状态查询;前端 Streamlit/React;配置 pydantic-settings。

**Q: 视频处理慢,长任务怎么处理?**
A: 上传后异步(FastAPI BackgroundTasks/Celery),立即返回 job_id,前端轮询/SSE 查进度;结果持久化避免重算。

**Q: 两小时视频怎么办?**
A: 转写/抽帧分段流式;chunk 多靠 ANN+过滤;关键帧封顶+全片降采样;超长转写先粗检索再精排。

### 4.8 评估
**Q: 怎么知道检索好?**
A: 建 eval 集,算 Recall@k、MRR。我用 **synthetic 方式**:从每个 chunk 让 LLM 反向生成一个问题、该 chunk 作 ground truth,零手工标注(再手工补几条难题)。**实测(40 条 query / 6768-chunk 语料,课程内检索)**:stage-1 Recall@20 = 1.0;dense Recall@6 = 1.0、MRR = 0.86;**dense+rerank MRR = 0.955**——reranking 把 gold chunk 从平均 ~#1.16 推到 ~#1.05(约 72% → 91% 排第 1)。**recall 饱和是因为 synthetic 问题用词偏近、偏易;换说法/间接的难题会让 dense recall 掉下来,那时 reranking 对 recall 的帮助才显现**——所以这数字其实低估了 reranking。教训:用数据说话 + 知道自己指标的局限。

**Q: 怎么评估回答质量?**
A: LLM-as-judge——另一个 LLM 按 groundedness / correctness / citation 准确率打分,再人工抽查。答题与打分分两步减偏差。

### 4.9 局限 & 改进
**Q: 弱点?**
A: (1) OpenCLIP 偏英文,中文"文字→画面"弱→升级 Chinese-CLIP/multilingual-CLIP;(2) 现为 segment 级时间戳,可升 word-level;(3) 未做 reranking/hybrid;(4) 单视频,多视频需更强元数据过滤;(5) 本地 LLM 质量上限。

**Q: 更多时间会做什么?**
A: reranking(cross-encoder)、hybrid(dense+BM25)、word-level 时间戳、vision LLM 视觉验证、完整 eval pipeline、多视频库、前端 timestamp 跳转、Docker 一键部署。

### 4.10 踩过的坑(behavioral)
**Q: 最棘手的问题?**
A: faster-whisper GPU 报 `libcublas.so.12` 找不到。排查发现 pip 装了最新 **CUDA 13** 版 torch,而 faster-whisper 的 CTranslate2 后端需要 **CUDA 12** 的 cuBLAS,版本对不上。我加了 **device fallback**(GPU 失败自动回退 CPU int8,保证出结果),并提供把 torch 对齐到 CUDA 12 的方案让 whisper 用上 GPU。体现排查 CUDA/依赖问题 + 写鲁棒代码的能力。

**Q: 测试检索时发现过什么值得说的问题?**
A: 语义检索 sanity check 时,英文 query "reshape a tensor" 没能把讲 `.view/.reshape` 的片段排进 top-3(反而 "join/concatenate" 靠前);但中文 query "改变张量的形状" 却能命中 permute/reshape。两个结论:(1) **跨语言检索有效**——BGE-m3 多语言,中文 query 检回了英文内容;(2) 纯 **dense 检索在口语化 ASR 文本上区分度弱**(相似度都挤在 ~0.5),字面词可能被漏。所以光调 chunk 大小不够(那是对单 query 过拟合),真正的解法是 **hybrid(dense + BM25,BM25 精确命中 "reshape" 这个词)+ reranking(cross-encoder 重排 top-k)**——正是 Phase 3 的设计。体现用**实测驱动架构决策**,而非照搬默认值。

---

**Q: 加了 reranker 之后效果怎样?(实测)**
A: 先踩了个 bug——我在 `CrossEncoder.predict` 之外又套了一层 sigmoid(它本身已输出 0–1),把区分度压没了;用 sanity test(clean "reshape" 句=0.997、"天气"=0)定位后修掉。修完发现更有意思的事:在这段短小口语化的 ASR 上,**reranker 给所有 chunk 的分都很低(<0.15),因为没有一个 chunk 干净地回答 "reshape"**——`.view/.reshape` 那段被 chunking 和 permute 内容混在了一起。结论:**reranker 本身没问题(干净文本上 0.997),真正的瓶颈是 chunk 的连贯性;reranking 在更大候选池、更干净文本上收益才明显**,且不影响端到端(相关片段仍在 top-k 喂给 LLM)。**教训:加组件要测量、找真正瓶颈,而不是堆技术**——这也是为什么要建 eval 集用 Recall@k 量化,而不是靠眼睛看单个 query。

## 5. 英文 soundbites(可直接背)

- *"Unlike a text-only RAG, video has transcript, visual frames, and time alignment. I built a multimodal index that binds transcript chunks and keyframe embeddings to a shared timestamp timeline — so the system doesn't just answer, it locates the answer in time."*
- *"I use CLIP's shared image–text space for cross-modal retrieval: a text query pulls back visually relevant keyframes even when the transcript never mentions them."*
- *"Every answer carries a timestamp citation back to the source moment."*
- *"faster-whisper on the CTranslate2 backend gives ~4x faster transcription than vanilla Whisper via quantization."*
- *"I evaluate retrieval with Recall@k and MRR, and answer quality with an LLM-as-judge on groundedness and citation accuracy."*
