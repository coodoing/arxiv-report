### 论文信息整理

#### 标题
Photon: Federated LLM Pre-Training

#### 摘要
扩展大型语言模型（LLMs）需要大量的数据和计算资源，这些资源传统上由于分布式训练所需的高带宽而局限于数据中心。低带宽方法如联邦学习（FL）如果能有效用于预训练，则可以实现跨弱连接GPU或弱连接GPU集群的更大模型协作训练。构建强大的低带宽训练系统可以：(a) 显著降低通信基础设施成本，(b) 最小化硬件故障的影响，(c) 扩大可用GPU池，(d) 实现互联网上的协作训练，以及 (e) 根据电力价格等因素动态分配计算资源。这些进步将减少对专用数据中心的依赖，使大规模AI训练更加普及、经济且适应实时需求。为此，我们引入了Photon，这是第一个完整的联邦端到端LLM训练系统，利用跨筒仓FL实现全球规模的训练，并具有最小的通信开销。使用Photon，我们从头开始训练第一个联邦解码器LLM系列。我们展示了：(1) Photon可以在联邦方式下训练高达7B参数的模型，同时达到甚至优于集中式预训练的困惑度；(2) Photon的模型训练时间随着可用计算资源的增加而减少，实现了与集中式类似的计算时间权衡；(3) Photon通过通信量减少64倍至512倍，比基线分布式训练方法的墙钟时间提高了35%。我们的提案对数据异质性具有鲁棒性，收敛速度是之前方法（如DiLoCo）的两倍。这种令人惊讶的数据效率源于一种独特的方法，结合小客户端批量大小和极高的学习率，这得益于联邦平均对超参数的鲁棒性。因此，Photon代表了第一个经济的全球互联网范围内的LLM预训练系统。

#### 论文亮点
1. **创新的联邦学习应用**：Photon是首个完整的联邦端到端大型语言模型（LLM）训练系统，能够在低带宽环境下进行大规模预训练。
2. **显著的成本和性能优势**：通过减少通信开销，Photon不仅降低了通信基础设施成本，还提高了训练效率，相比传统的分布式训练方法，墙钟时间减少了35%，通信量减少了64倍至512倍。
3. **广泛的适用性和鲁棒性**：Photon能够适应不同的数据分布和硬件环境，对数据异质性具有鲁棒性，并且在不同计算资源条件下表现出良好的性能。
4. **高效的训练机制**：结合小批量数据和高学习率，Photon利用联邦平均的独特特性，实现了更快的收敛速度和更高的数据效率。
5. **经济性和可扩展性**：Photon使得大规模AI训练更加经济且适应全球范围内的互联网协作，降低了对专用数据中心的依赖。以下是根据您的要求整理的每篇论文的内容：

1. **标题**：Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking  
   **摘要**：尽管移动设备提供的计算能力越来越强，但DRAM带宽的改进速度要慢得多。这对于大型语言模型（LLM）的标记生成非常不利，因为其过程高度依赖内存。本文提出了一种无需预测器的动态稀疏化方法——动态输入剪枝（DIP），以及一种新颖的缓存感知掩码策略，以减少有效DRAM带宽并提高缓存命中率，从而在移动设备上提高LLM的令牌速率。DIP可以在最小微调的情况下保持准确性，并通过轻量级LoRA适配器恢复一些因稀疏化而损失的性能。  
   **论文亮点**：DIP在不同硬件设置下，在准确性和吞吐量方面均优于其他方法；在Phi-3-Medium上实现了46%的内存减少和40%的吞吐量增加。

2. **标题**：SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations  
   **摘要**：本文针对翻译型知识图谱嵌入训练提出了基于稀疏矩阵运算的方法，通过使用SpMM（稀疏-稠密矩阵乘法）核函数替换核心嵌入计算，统一了多个散射操作，减少了训练时间和内存占用。该方法不仅适用于现有的几种模型（如TransE、TransR等），还可以扩展到其他类型的模型。  
   **论文亮点**：实现了高达5.3倍的CPU加速和4.2倍的GPU加速，同时显著降低了GPU内存占用。

3. **标题**：Venn: Resource Management For Collaborative Learning Jobs  
   **摘要**：协作学习（CL）作为分布式边缘设备上的机器学习新范式，面临着资源调度难题。本文介绍了Auxo，这是一种能够有效管理异构设备资源、优化多个CL任务之间资源共享的系统。它通过解决复杂的资源竞争问题来减少平均作业完成时间。  
   **论文亮点**：相比现有最先进的CL资源管理器，Auxo可将平均作业完成时间缩短至原来的1.88倍。

4. **标题**：DiffServe: Efficiently Serving Text-to-Image Diffusion Models with Query-Aware Model Scaling  
   **摘要**：针对文本到图像生成中扩散模型服务效率低下的问题，本文提出了一个名为DiffServe的系统，它可以根据查询难度自动选择适当的模型进行处理，从而既保证了图像质量又提高了响应速度。  
   **论文亮点**：实验表明，DiffServe可以提升24%的响应质量，并且将延迟违规率降低19%-70%。

5. **标题**：Rubick: Exploiting Job Reconfigurability for Deep Learning Cluster Scheduling  
   **摘要**：为了应对大规模深度学习训练中的资源分配挑战，本文设计了Rubick集群调度系统，该系统利用作业的可重配置性来优化资源利用率和训练性能。  
   **论文亮点**：相比于现有系统，Rubick可以将平均作业完成时间和总跨度分别减少3.2倍和1.4倍。

6. **标题**：SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention  
   **摘要**：针对长上下文LLM推理过程中存在的高延迟问题，本文引入了一种新的两阶段查询引导键值过滤方法，即SampleAttention，用于动态选择重要列和斜杠条带，以满足所需的累积残差注意阈值，从而实现高效且准确的推理。  
   **论文亮点**：相较于FlashAttention2，SampleAttention可将首次生成令牌的时间缩短5.29倍。

7. **标题**：VoLUT: Efficient Volumetric streaming enhanced by LUT-based super-resolution  
   **摘要**：本文提出了一种基于查找表（LUT）的超分辨率算法VoLUT，专门用于3D体积视频流媒体传输，能够在不牺牲画质的前提下大幅降低带宽需求。  
   **论文亮点**：VoLUT可以减少70%的带宽使用，提高36.7%的质量体验，并实现8.4倍的3D超分辨率加速。

8. **标题**：Context Parallelism for Scalable Million-Token Inference  
   **摘要**：本文探讨了如何利用上下文并行性来支持百万级标记的大规模语言模型推理，特别是针对长上下文情况下的预填充延迟优化。  
   **论文亮点**：使用128个H100 GPU可以在77秒内完成1M标记的预填充，效率达到93%，浮点运算利用率63%。

9. **标题**：Supply-Chain Attacks in Machine Learning Frameworks  
   **摘要**：本文揭示了针对机器学习框架供应链攻击的新威胁，并分析了开源项目中存在的安全意识不足问题。  
   **论文亮点**：发现了ML社区与非ML社区在供应链安全性方面的相似水平，强调了加强ML特定供应链攻击防护的重要性。

10. **标题**：FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving  
    **摘要**：本文介绍了一个高效的注意力引擎FlashInfer，旨在为大型语言模型提供定制化的推理服务。它解决了KV缓存存储异构性的问题，并提供了灵活的模板设计以适应不同的应用场景。  
    **论文亮点**：相比其他LLM服务解决方案，FlashInfer能够减少29%-69%的令牌间延迟，降低28%-30%的长时间上下文推理延迟，并加快13%-17%的并行生成速度。以下是根据您的要求整理的每篇论文的内容：

1. **标题**：LeanAttention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers  
   **摘要**：基于Transformer的大规模语言模型在内存消耗和推理延迟方面面临挑战，尤其是在处理长上下文时。本文提出了一种名为LeanAttention的可扩展、硬件高效的注意力加速机制，专门用于Transformer解码阶段。通过重新设计注意力执行流程，LeanAttention能够在长上下文长度下实现显著的速度提升。  
   **论文亮点**：LeanAttention相比FlashDecoding实现了平均1.73倍的速度提升，在256k上下文长度下最高可达2.18倍。

2. **标题**：Efficient On-Device Machine Learning with a Biologically-Plausible Forward-Only Algorithm  
   **摘要**：当前深度神经网络（DNN）的训练依赖于反向传播（BP），但BP存在生物学上不可行的问题。本文提出了一种生物学合理的前向算法（Bio-FO），旨在更好地模拟人脑的学习过程并提高能效。  
   **论文亮点**：Bio-FO不仅解决了BP的生物学不合理性问题，还在多个数据集上表现出色，并且在资源受限的设备上实现了高效运行。

3. **标题**：LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention  
   **摘要**：大规模语言模型（LLM）在处理长序列时面临计算复杂度和内存占用的挑战。本文提出了LServe系统，通过统一稀疏注意力机制加速长序列LLM的推理。  
   **论文亮点**：LServe将一半的注意力头转换为几乎免费的流式头，减少了KV缓存的需求，并通过分层KV页选择策略动态修剪KV页，实现了显著的性能提升。

4. **标题**：MAS-ATTENTION: MEMORY-AWARE STREAM PROCESSING FOR ATTENTION ACCELERATION ON RESOURCE-CONSTRAINED EDGE DEVICES  
   **摘要**：尽管许多融合型注意力加速算法适用于数据中心级别的GPU，但在资源受限的边缘设备上加速注意力机制仍然具有挑战性。本文提出了一种针对内存受限边缘加速器的注意力加速方案。  
   **论文亮点**：通过多级平铺调度方案和主动缓存覆盖策略，MAS-ATTENTION在边缘计算场景中实现了高达2.75倍的速度提升和54%的能耗降低。

5. **标题**：A Bring-Your-Own-Model Approach for ML-Driven Storage Placement in Warehouse-Scale Computers  
   **摘要**：存储系统是仓库规模计算机总拥有成本的重要组成部分。本文提出了一种跨层方法，将机器学习模型从存储系统中移出并在应用程序层面运行，以优化存储放置。  
   **论文亮点**：该方法结合了小型可解释模型与自适应启发式算法，实现在不同在线环境中的高效调整，TCO节省最高可达3.47倍。

6. **标题**：MEADOW: Memory-efficient Dataflow and Data Packing for Low Power Edge LLMs  
   **摘要**：大型语言模型的计算和内存挑战促使了多种优化方法的发展。本文引入了MEADOW框架，通过新颖的数据流和权重打包技术减少低功耗边缘设备上的离片内存访问。  
   **论文亮点**：MEADOW在低功耗平台上实现了比传统GEMM方法更低的解码和预填充延迟，并提高了端到端推理效率超过40%。

7. **标题**：Self-Data Distillation for Recovering Quality in Pruned Large Language Models  
   **摘要**：大规模语言模型的压缩技术如结构化剪枝虽然减少了模型复杂度，但也可能导致性能下降。本文提出了一种自数据蒸馏微调方法来恢复剪枝后的模型质量。  
   **论文亮点**：自数据蒸馏微调方法保留了更多原始模型的知识，尤其在多步推理任务中表现优异，平均准确率提升了8%，并且在实际应用中减少了16.30%的FLOPs。

8. **标题**：ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation  
   **摘要**：强化学习从人类反馈（RLHF）是提升大型语言模型应用的关键技术。本文提出了参数重分配技术ReaL，以动态调整训练过程中的并行化策略。  
   **论文亮点**：ReaL通过自动发现高效的执行计划，实现了最高3.58倍的训练速度提升，并在长上下文场景下表现出显著优于启发式方法的性能。

9. **标题**：XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models  
   **摘要**：随着LLM代理应用的复杂性和多样性增加，结构化生成的需求也随之增长。本文提出了XGrammar引擎，用于加速上下文无关语法的执行，从而实现灵活高效的结构化生成。  
   **论文亮点**：XGrammar通过词汇表划分和持久栈等技术，使得结构化生成速度比现有解决方案快10倍以上，且在低延迟推理场景中几乎无额外开销。

10. **标题**：FedProphet: Memory-Efficient Federated Adversarial Training via Robust and Consistent Cascade Learning  
    **摘要**：联邦对抗训练（FAT）可以增强联邦学习对对抗样本的鲁棒性。本文提出了FedProphet框架，通过强凸正则化的级联学习减少了本地训练的内存需求。  
    **论文亮点**：FedProphet在保持高准确率和强鲁棒性的前提下，实现了80%的内存减少和最高10.8倍的训练时间加速。以下是根据您的要求整理的每篇论文的内容：

1. **标题**：AdaParse: An Adaptive Parallel PDF Parsing and Resource Scaling Engine  
   **摘要**：科学任务的语言模型训练依赖于从科学出版物（大多数以PDF形式发布）中提取文本。PDF解析方法从简单的启发式算法到复杂的机器学习驱动系统不等。选择“最佳”解析器取决于其计算成本和输出准确性。本文介绍了一种自适应并行PDF解析和资源扩展引擎（AdaParse），它通过数据驱动策略为每个文档分配合适的解析器。通过直接偏好优化（DPO）将科学家选择的解析结果整合到AdaParse中，使其选择过程与人类判断一致。AdaParse结合硬件需求和预测准确性来高效调度计算资源。实验表明，AdaParse在吞吐量上比现有方法提高了17倍，同时保持了相似的准确性（实际高0.2%）。  
   **论文亮点**：提出了一种自适应并行PDF解析引擎AdaParse，能够在大规模解析任务中提高效率并保持高准确性。

2. **标题**：The Hidden Bloat in Machine Learning Systems  
   **摘要**：软件膨胀指的是运行时未使用的代码和功能。对于机器学习（ML）系统而言，膨胀是技术债务的主要来源之一，导致性能下降和资源浪费。本文介绍了Negativa-ML工具，用于识别和移除ML框架中的膨胀代码，特别是GPU代码中的冗余部分。实验结果表明，ML框架在GPU和CPU代码方面都存在显著膨胀，而Negativa-ML能够减少高达75%的GPU代码大小和72%的CPU代码大小。  
   **论文亮点**：提出了Negativa-ML工具，能够有效减少ML框架中的代码膨胀，从而提升性能和减少资源浪费。

3. **标题**：Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving  
   **摘要**：键值缓存压缩技术被广泛应用于优化大型语言模型（LLM）的服务，主要通过减少内存消耗来降低计算成本。然而，现有的压缩算法在生产环境中的应用仍然有限。本文重新审视了主流的键值缓存压缩解决方案，并通过实证评估发现了两个影响计算效率的关键问题：压缩缓存可能导致更长的输出，增加端到端延迟；现有实现未能优化生产级LLM服务的吞吐性能。  
   **论文亮点**：通过对键值缓存压缩技术的深入分析，揭示了现有方法的局限性，并为未来的研究提供了改进方向。

4. **标题**：SwiftVI: Time-Efficient Planning and Learning with MDPs  
   **摘要**：马尔可夫决策过程（MDPs）在不确定环境中进行决策和学习的应用非常广泛。然而，找到最优策略需要较高的计算成本。本文提出了SwiftVI算法套件，通过优先队列组织动作集并推导备份Q值的边界来解决这一问题。实验表明，SwiftVI算法在不同模型参数下均表现出高效的性能。  
   **论文亮点**：提出了SwiftVI算法套件，能够高效地解决MDPs中的规划和学习问题，显著减少了计算成本。

5. **标题**：Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer  
   **摘要**：长上下文能力对于复杂自然语言处理任务至关重要。然而，直接在极长上下文中训练大型语言模型（LLMs）需要大量的GPU资源和内存。本文提出了完全流水线分布式的Transformer（FPDT），可以在相同硬件上将可训练的序列长度增加16倍。  
   **论文亮点**：提出了FPDT架构，能够在较少的硬件资源上训练具有超长上下文能力的LLMs，显著提高了硬件利用率。

6. **标题**：AIOpsLab: A Holistic Framework to Evaluate AI Agents for Enabling Autonomous Clouds  
   **摘要**：AI用于IT运维（AIOps）旨在自动化复杂的操作任务，如故障定位和根本原因分析。本文提出了一种名为AIOPSLAB的框架，用于评估AI代理在自主管理云系统中的表现。该框架不仅部署多样化的云环境、注入故障、生成工作负载和导出遥测数据，还提供接口用于与代理交互和评估。  
   **论文亮点**：提出了AIOPSLAB框架，为评估AI代理在云环境中的表现提供了全面的支持，推动了自主云系统的实现。

7. **标题**：Enabling Unstructured Sparse Acceleration on Structured Sparse Accelerators  
   **摘要**：利用深度神经网络（DNN）中的稀疏性是满足日益增长的计算需求的有前景领域。硬件设计师提出了结构化稀疏支持以最小化稀疏加速的开销，但这种方法灵活性有限且需要额外的模型微调。本文提出了一种近似方法，通过线性代数的分配律将任意稀疏张量转换为一系列结构化稀疏张量，从而实现无微调的稀疏加速。  
   **论文亮点**：提出了一种新的方法，使得在结构化稀疏硬件上可以加速非结构化稀疏DNN，显著提升了能效和速度。

8. **标题**：Youmu: Efficient Columnar Data Pipeline for LLM Training  
   **摘要**：大型语言模型（LLMs）的训练是极其数据密集型的，通常涉及万亿级别的token。虽然LLM数据集通常以列式格式存储，但在训练前往往需要转换为其他格式，这增加了存储和维护成本。本文提出了Youmu数据管道，可以直接将细粒度的列式数据馈入GPU，实现了成本高效的LLM训练。  
   **论文亮点**：提出了Youmu数据管道，能够在不进行数据格式转换的情况下直接使用列式数据进行LLM训练，大幅降低了存储和维护成本。

9. **标题**：HyC-LoRA: Memory Efficient LoRA Fine-tuning with Hybrid Activation Compression  
   **摘要**：低秩适应（LoRA）是一种广泛使用的轻量级微调方法，显著减少了预训练LLMs转移到下游任务时的可调权重和优化器内存。然而，过去的工作忽略了缓冲激活带来的内存开销。本文提出了HyC-LoRA，通过混合压缩框架实现了几乎2位的缓冲激活量化，进一步减少了内存消耗。  
   **论文亮点**：提出了HyC-LoRA方法，通过混合压缩机制大幅减少了LoRA微调过程中的内存消耗，同时保持了高准确性。

10. **标题**：ThunderServe: High-performance and Cost-efficient LLM Serving in Cloud Environments  
    **摘要**：在云环境中部署大型语言模型（LLMs）可以更好地应对GPU短缺问题并降低成本。然而，云环境中的网络和GPU类型的多样性带来了高性能服务的挑战。本文提出了ThunderServe系统，通过新颖的调度算法优化了异构资源和网络带宽条件下的LLM部署。  
    **论文亮点**：提出了ThunderServe系统，能够在云环境中高效部署LLMs，显著提高了吞吐量并降低了延迟，同时降低了成本。1. **标题**: APOLLO: SGD-like Memory, AdamW-level Performance  
   **摘要**: 大型语言模型（LLMs）在训练过程中需要大量的内存，尤其是在使用AdamW优化器时。这通常需要更多的高端GPU或减少批处理大小，限制了训练的可扩展性和吞吐量。本文提出了一种新的方法——APOLLO（Approximated Gradient Scaling for Memory Efficient LLM Optimization），通过近似通道级学习率缩放，使用辅助低秩优化器状态和纯随机投影来降低内存使用。实验表明，APOLLO系列在不同模型架构和任务上表现与AdamW相当甚至更好，同时显著减少了内存占用。  
   **论文亮点**: 
   - 提出了APOLLO及其极简版本APOLLO-MINI，实现了类似SGD的内存消耗但优于AdamW的性能。
   - 在8xA100-80GB设置上，APOLLO和APOLLO-Mini的吞吐量比AdamW提高了约3倍。
   - 首次使预训练LLaMA-13B模型在A100-80G上使用简单的DDP成为可能，无需其他系统级优化。

2. **标题**: GSplit: Scaling Graph Neural Network Training on Large Graphs via Split-Parallelism  
   **摘要**: 图神经网络（GNNs）在各种图分析任务中表现出色，但其训练过程存在冗余工作，特别是在多GPU上的小批量训练中，不同GPU采样的子图存在重叠。本文引入了一种混合并行的小批量训练范式——split parallelism，通过将每个小批量的采样、加载和训练拆分到多个GPU上来避免冗余工作，并提出了一种轻量级分区算法以最小化通信开销。  
   **论文亮点**: 
   - 引入了split parallelism，有效避免了多GPU训练中的冗余工作。
   - 实验表明，split parallelism在Spa上的性能优于现有的DGL、Quiver和P3等最先进的系统。

3. **标题**: ScaleFusion: Scalable Inference of Spatial-Temporal Diffusion Transformers for High-Resolution Long Video Generation  
   **摘要**: 最新的扩散模型能够在高分辨率和长时间视频生成方面取得优异效果，但由于计算成本随分辨率和持续时间呈二次增长，导致推理延迟较高。本文提出了ScaleFusion，一种用于高分辨率长视频生成的可扩展推理引擎，通过优化空间-时间注意力层的跨机器通信调度算法，实现了高效的分布式推理。  
   **论文亮点**: 
   - ScaleFusion在4台Amazon EC2 p4d.24xlarge机器（32个A100 GPU）上实现了3.6倍的强扩展性。
   - 平均速度提升了1.36倍（最高达1.58倍），显著优于现有技术。

4. **标题**: SOLA: Optimizing SLO Attainment for Large Language Model Serving with State-Aware Scheduling  
   **摘要**: 服务大型语言模型（LLMs）时，满足服务级别目标（SLOs）至关重要。现有的调度策略在不同迭代中遵循固定的调度原则，导致Time-to-First-Token（TTFT）和Time-per-Output-Token（TPOT）之间的分布偏差较大。本文提出了一种基于状态感知的调度策略，能够平衡TTFT和TPOT，并改善不同请求之间的分布，从而提高SLO达成率。  
   **论文亮点**: 
   - SOLA将SLO达成率从45.5%提升至99.4%，显著提高了请求处理能力。
   - 在给定SLO约束下，SOLA平均能处理1.04-1.27倍于现有系统的请求数量。

5. **标题**: ProtoRAIL: A Risk-cognizant Imitation Agent for Adaptive vCPU Oversubscription In the Cloud  
   **摘要**: 云系统中，虚拟CPU（vCPU）的超分配是一种常见的优化策略，但需要在成本和风险之间找到平衡。本文提出了ProtoRAIL框架，利用模仿学习和风险感知模块，动态调整vCPU超分配策略，以适应需求模式的变化。  
   **论文亮点**: 
   - ProtoRAIL在微软内部云服务中实现了90倍以上的风险降低和7%-10%的资源节省。
   - 通过学习利用模式对称性，ProtoRAIL能够适应不同的需求和粒度，优化成本和风险。

6. **标题**: Graph Learning at Scale: Characterizing and Optimizing Pre-Propagation GNNs  
   **摘要**: 图神经网络（GNNs）广泛应用于图节点嵌入学习，但随着层数增加，邻居爆炸问题导致计算和内存需求呈指数增长。本文研究了预传播GNNs（PP-GNNs），并通过优化数据加载方案和训练方法，显著提高了训练吞吐量。  
   **论文亮点**: 
   - PP-GNNs在大规模图基准测试中比基于采样的GNN快2个数量级。
   - 提出的优化方案使PP-GNN训练吞吐量提高了15倍。

7. **标题**: LAVA: Lifetime-Aware VM Allocation with Learned Distributions and Adaptation to Mispredictions  
   **摘要**: 虚拟机（VM）调度是云计算数据中心效率的关键。本文提出了一种基于生命周期预测的VM调度算法LAVA，通过重复预测和调整VM及主机的生命周期，减少了资源浪费并提高了空闲主机的数量。  
   **论文亮点**: 
   - LAVA减少了约3%的计算资源浪费和约2%的内存资源浪费。
   - 生产环境中，LAVA增加了2.3-9.2个百分点的空闲主机数量，减少了VM迁移次数。

8. **标题**: NEO: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference  
   **摘要**: 在线大型语言模型（LLMs）推理依赖于GPU加速，但由于GPU内存有限，批处理大小受限，导致GPU计算资源浪费。本文提出了NEO系统，通过将部分注意力计算和KV缓存状态卸载到CPU，有效地增加了GPU批处理大小，从而提高了推理吞吐量。  
   **论文亮点**: 
   - NEO在T4、A10G和H100 GPU上分别实现了7.5倍、26%和14%的吞吐量提升。
   - 使用更强大的CPU时，NEO在A10G GPU上实现了高达79.3%的吞吐量增益。

9. **标题**: FastTree: Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference  
   **摘要**: 树结构前缀共享在大型语言模型（LLMs）应用中很常见。本文提出了FastTree，通过优化GPU内核和运行时调度，提高了树结构LLM推理的吞吐量。  
   **论文亮点**: 
   - FastTree在SGLang基础上将吞吐量提高了2.2倍。
   - 提出了基于树结构的自适应运行时优化，有效减少了冗余内存加载和GPU张量核心利用率不足的问题。

10. **标题**: Lightweight Software Kernels and Hardware Extensions for Efficient Sparse Deep Neural Networks on Microcontrollers  
    **摘要**: 在微控制器（MCUs）上加速修剪后的深度神经网络（DNNs）具有挑战性，因为这些设备有严格的面积和功耗限制。本文提出了针对N:M修剪层的优化软件内核和硬件扩展，以加速稀疏DNN推理。  
    **论文亮点**: 
    - 优化的软件内核在1:8和1:16稀疏度下分别比密集层快2.1倍和3.4倍。
    - 轻量级ISA扩展进一步提升了1.9倍的速度，仅增加了5%的面积开销。1. **标题**: FlexInfer: Flexible LLM Inference with CPU Computations  
   **摘要**: 大型语言模型（LLMs）在多个领域表现出色，促使数据中心使用高计算成本的加速器如GPU和NPU进行模型训练和推理。然而，这些模型及其键值（KV）缓存的巨大尺寸给内存容量带来了重大挑战。虽然基于卸载的方法利用CPU内存存储模型权重和KV缓存，可以部署超过GPU内存容量的模型，但往往由于PCIe传输瓶颈导致性能下降。为了解决现有基于卸载的LLM推理在单个GPU系统中的性能限制，本文提出了FlexInfer。FlexInfer使用性能估计器动态选择每个阶段（预填充和解码）最合适的执行策略，基于硬件配置和运行时参数如序列长度和批处理大小。评估结果显示，通过为这些阶段选择最优策略，FlexInfer可将端到端延迟平均减少75%和76%，优于最先进的基于卸载的LLM推理技术FlexGen。  
   **论文亮点**: 提出了FlexInfer，一种新的基于CPU计算的LLM推理方法，能够显著减少端到端延迟，并且在不同服务器配置下表现出色。

2. **标题**: Seesaw: High-throughput LLM Inference via Model Re-sharding  
   **摘要**: 为了提高分布式大型语言模型（LLM）推理的效率，已经提出了各种并行化策略，如张量并行和管道并行。然而，LLM推理的两个阶段——预填充和解码——的不同计算特性使得单一静态并行化策略不足以有效优化这两个阶段。本文介绍了Seesaw，一个针对吞吐量任务优化的LLM推理引擎。Seesaw的关键思想是动态模型重分片，通过在阶段间动态重新配置并行化策略来最大化吞吐量。为了减少重分片开销并优化计算效率，我们采用了分层KV缓存缓冲和最小化转换调度。评估显示，Seesaw相比最先进的LLM推理引擎vLLM，吞吐量提高了1.78倍（平均1.36倍）。  
   **论文亮点**: 引入了Seesaw，一种通过动态模型重分片技术实现高吞吐量的LLM推理引擎，显著提升了推理吞吐量。

3. **标题**: Interference-aware Edge Runtime Prediction with Conformal Matrix Completion  
   **摘要**: 准确估计工作负载运行时间一直是计算机系统中的长期目标，在高效资源调配、延迟最小化和其他系统管理任务中起着关键作用。边缘系统中的运行时间预测尤为重要，因为更复杂的处理被推向边缘以寻求更好的延迟。以前的方法在数据效率或需要大量仪器方面存在问题；这些挑战在异构边缘计算环境中更为复杂，历史运行时间数据可能稀疏且难以获取。此外，边缘计算环境通常具有多租户特性，有限资源可能导致工作负载之间的干扰，进一步复杂化运行时间预测问题。本文设计了一种矩阵分解启发式方法，生成准确的干扰感知预测，并提供严格的不确定性边界。实验结果表明，该方法在24种独特设备上收集的WebAssembly运行时数据集上实现了5.2%的预测误差，比现有方法好2倍。  
   **论文亮点**: 设计了一种新的干扰感知边缘运行时间预测方法，利用矩阵分解技术提高了预测精度和效率。

4. **标题**: MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators  
   **摘要**: 高效部署混合专家（MoE）模型的一个关键方法是量化。然而，最先进的MoE模型在极端量化（如低于4位）时会遭受不可忽略的准确性损失。为此，本文引入了MiLo，一种通过低秩补偿器增强高度量化MoE的新方法。这些补偿器仅消耗少量额外内存，但显著恢复了极端量化带来的准确性损失。MiLo还识别出MoE模型在权重上的独特特征，并采用自适应秩选择策略和迭代优化以缩小准确性差距。评估显示，MiLo在各种任务上优于现有方法。  
   **论文亮点**: 提出了MiLo，一种通过低秩补偿器增强极端量化MoE模型的方法，显著提高了模型的准确性和效率。

5. **标题**: AI Metropolis: Scaling Large Language Model-based Multi-Agent Simulation with Out-of-order Execution  
   **摘要**: 随着自然语言理解和推理能力的进步，基于大型语言模型（LLMs）的代理在模拟环境中越来越多地开发出来，以执行复杂任务、与其他代理互动并展示与社会科学研究和创新游戏开发相关的新兴行为。然而，当前的多代理模拟由于虚假依赖性导致的并行性有限，经常遇到性能瓶颈。本文介绍了AI Metropolis，一种通过引入乱序执行调度来提高LLM代理模拟效率的模拟引擎。通过动态跟踪代理之间的实际依赖关系，AI Metropolis最小化虚假依赖性，增强并行性并最大化硬件利用率。评估显示，AI Metropolis相比标准并行模拟实现了1.3倍至4.15倍的速度提升。  
   **论文亮点**: 引入了AI Metropolis，一种通过乱序执行调度提高多代理模拟效率的模拟引擎，显著提升了模拟速度。

6. **标题**: COMET: Fine-grained Computation-communication Overlapping for Mixture-of-Experts  
   **摘要**: 混合专家（MoE）已被广泛用于扩展大型语言模型到万亿参数级别，同时保持固定的计算成本。然而，在分布式场景中开发大型MoE模型遇到了通信开销大的问题。MoE层的跨设备通信可能占据整个模型执行时间的47%。现有的粗粒度重叠方案引入了显著的计算效率损失，延迟隐藏效果不理想。本文提出了COMET，一种具有细粒度通信-计算重叠的优化MoE系统。通过数据依赖分析和任务重新调度，COMET实现了精确的细粒度重叠。评估显示，COMET将单个MoE层的执行加速了1.96倍，端到端执行加速了1.71倍。  
   **论文亮点**: 提出了COMET，一种通过细粒度通信-计算重叠优化MoE系统的框架，显著提高了模型执行速度。

7. **标题**: FLStore: Efficient Federated Learning Storage for non-training workloads  
   **摘要**: 联邦学习（FL）是一种隐私保护的机器学习方法，允许在多个客户端之间进行模型训练而无需集中数据收集。除了训练外，FL系统还包括大量的非训练工作负载，如调度、个性化、聚类、调试和激励。大多数现有系统依赖聚合服务器处理非训练工作负载，并使用云服务存储数据，导致高延迟和增加成本。本文提出了FLStore，一种高效的联邦学习非训练工作负载和存储的无服务器框架。FLStore统一了数据和计算平面，通过定制缓存策略减少了延迟和成本。评估显示，FLStore相比基于云对象存储的聚合服务器减少了71%的请求平均延迟和92.45%的成本。  
   **论文亮点**: 提出了FLStore，一种高效的联邦学习非训练工作负载和存储的无服务器框架，显著降低了延迟和成本。

8. **标题**: Balancing Pipeline Parallelism with Vocabulary Parallelism  
   **摘要**: 管道并行性被广泛用于扩展基于变压器的大规模语言模型的训练，许多工作致力于提高其吞吐量和内存占用。本文解决了词汇层可能导致的计算和内存使用的不平衡问题，加剧了管道气泡和内存瓶颈。为了解决这个问题，我们将词汇层均匀分布在管道设备上，并将计算分组为管道传递。我们还提出了一些算法来减少词汇层内的通信障碍。结合这些技术，我们的方法有效地平衡了计算和参数内存，同时仅增加了少量激活内存开销。评估显示，我们的方法在不同词汇量情况下提高了吞吐量5%至51%，显著减少了峰值内存使用。  
   **论文亮点**: 提出了通过词汇并行性平衡管道并行性的方法，显著提高了计算和内存平衡，改善了训练性能。

9. **标题**: PipeFill: Using GPUs During Bubbles in Pipeline-parallel LLM Training  
   **摘要**: 训练具有数十亿参数的深度神经网络（DNN）通常涉及管道并行（PP）执行。不幸的是，PP模型训练可能会低效使用GPU，特别是在大规模训练中，由于管道气泡导致的空闲GPU时间可达15-30%，甚至超过60%。为提高PP模型训练的GPU利用率，本文描述了PipeFill，它通过在管道气泡期间执行其他待处理作业来填补空闲时间。通过在8K GPU上进行大规模LLM训练，PipeFill可使整体利用率提高63%，完成相当于额外2.6K GPU的工作。  
   **论文亮点**: 提出了PipeFill，一种在管道气泡期间利用GPU时间的方法，显著提高了大规模LLM训练的GPU利用率。

10. **标题**: Radius: Range-based Gradient Sparsity for Large Foundation Model Pre-training  
    **摘要**: 本文提出了Radius，一种基于范围的梯度稀疏化算法和系统，旨在加速大型基础模型（FM）预训练的同时保持下游任务性能。Radius利用了两个关键见解：1）每次迭代中只有小部分梯度有助于模型更新，2）大梯度的空间分布随时间稳定。Radius克服了现有top-k稀疏化方法的扩展问题，因为它保持了稀疏梯度的结构，避免了后期阶段的密集通信。评估显示，Radius在预训练GPT-2.0B模型时，使用40%的稀疏度可将每步训练时间减少21%，整体预训练时间减少19%，而不影响下游任务的评估分数。  
    **论文亮点**: 提出了Radius，一种基于范围的梯度稀疏化方法，显著提高了大型基础模型预训练的速度，同时保持了下游任务性能。### 论文1
**标题**: On Distributed Larger-Than-Memory Subset Selection With Pairwise Submodular Functions  
**摘要**: 当代数据集包含数十亿个样本，使得在所有可用数据上进行训练变得不切实际。选择高质量的子集有助于减少训练成本并提高模型质量。次模性（submodularity）作为离散凸性的类比，常用于解决此类子集选择问题。然而，现有的优化次模函数的算法是顺序的，并且之前的分布式方法需要至少一台中央机器将目标子集存储在DRAM中。在十亿数据点规模下，即使是子集也可能不适合单台机器，而顺序算法的速度也慢得令人无法忍受。本文提出了一种新颖的分布式边界算法，可以证明其近似保证，无需依赖中央机器来处理目标子集。该算法迭代地设定最小和最大效用值的边界，以选择高质量的点并丢弃不重要的点。当边界设置未找到完整子集时，我们使用基于分区的多轮分布式贪婪算法来识别剩余子集。我们讨论了如何在分布式数据处理框架中实现这些算法，并对不同配置进行了实证分析。我们发现，在CIFAR-100和ImageNet上，与集中式方法相比，所选子集的质量几乎没有或完全没有损失，并且能够扩展到一个包含130亿个数据点的数据集。  
**论文亮点**: 
- 提出了一种无需中央机器的新颖分布式边界算法。
- 算法具有可证明的近似保证，适用于大规模数据集。
- 实验结果表明，该方法在大规模数据集上的表现与集中式方法相当。

### 论文2
**标题**: Lumos: Efficient Performance Modeling and Estimation for Large-scale LLM Training  
**摘要**: 在分布式环境中训练大型语言模型（LLMs）面临显著挑战，包括模型执行、部署系统和大量可配置策略的复杂性。尽管存在各种优化技术，但在实践中实现高效率仍然困难。准确的性能模型对于指导优化工作和系统级研究至关重要。本文提出了Lumos，一种基于跟踪驱动的性能建模和估计工具包，旨在准确捕捉和预测现代LLMs的执行行为。我们在配备最多512个NVIDIA H100 GPU的生产ML集群上评估了Lumos，使用各种GPT-3变体，结果表明它可以以平均仅3.3%的误差重现实行时间以及其他运行时细节，并能够从现有跟踪中估计新设置的性能，从而促进模型和部署配置的有效探索。  
**论文亮点**: 
- 提出了Lumos，一种高效的大规模LLM训练性能建模和估计工具。
- 实验表明，Lumos可以在多种模型和配置下以低误差率重现实行时间。
- 支持从现有跟踪中估计新设置的性能，便于探索新的配置。

### 论文3
**标题**: FlexAttention: A Programming Model for Generating Fused Attention Variants  
**摘要**: 过去七年中，注意力机制已成为深度学习中最重要的一类原语之一。优化注意力的主要方法是FlashAttention，它通过融合操作显著提高了运行时间和内存消耗。然而，FlashAttention的重要性及其单片性质给研究人员尝试新的注意力变体带来了“软件彩票”的问题。此外，编写高效的融合注意力内核非常困难，传统编译器方法难以应对。本文引入了FlexAttention，一种新的编译器驱动的编程模型，允许用几行简洁的PyTorch代码实现大多数注意力变体。我们展示了许多现有注意力变体（如Alibi、Document Masking、PagedAttention等）可以通过FlexAttention实现，并且性能与这些手工编写的内核相当。最后，我们展示了FlexAttention如何轻松组合注意力变体，解决了“超立方体问题”。  
**论文亮点**: 
- 提出了FlexAttention，一种新型的编译器驱动编程模型。
- 允许用几行PyTorch代码实现多种注意力变体。
- 解决了“超立方体问题”，支持注意力变体的灵活组合。

### 论文4
**标题**: Optimizing LLM Queries in Relational Data Analytics Workloads  
**摘要**: 批量数据分析已成为大型语言模型（LLMs）的一个重要应用。LLMs使用户能够执行广泛的自然语言任务，例如分类、实体提取和翻译。然而，LLM推理在计算和货币成本方面都非常昂贵。例如，NVIDIA L4 GPU运行Llama3-8B只能每秒处理6 KB的文本，处理15 GB的数据需要大约一天的时间；处理类似数量的数据在OpenAI的GPT-4上成本约为10,000美元。本文提出了一些新技术，可以显著降低关系数据分析工作负载中LLM调用的成本。我们的关键贡献是开发了有效算法，通过对输入表中的行和每行字段进行重新排序，以最大化键值（KV）缓存重用，从而优化LLM服务。该方法可以轻松应用于现有分析系统和服务平台。评估结果显示，我们的解决方案在使用Llama 3模型的各种LLM查询基准测试中，端到端延迟最高可提高3.4倍。使用OpenAI和Anthropic的前缀缓存定价模型，我们的解决方案还可以节省32%的成本。  
**论文亮点**: 
- 开发了有效算法，通过重新排序输入表中的行和字段来优化LLM服务。
- 显著降低了关系数据分析工作负载中LLM调用的成本。
- 实验结果表明，端到端延迟最高可提高3.4倍，成本节省达32%。

### 论文5
**标题**: TurboAttention: Efficient attention approximation for high throughputs llm  
**摘要**: 大型语言模型（LLM）推理需要大量的计算和内存资源，尤其是在关键的注意力机制中。虽然量化技术和加速算法（如FlashAttention）已经提高了整体推理效率，但它们针对的是不同的问题方面：量化专注于权重-激活操作，而FlashAttention则改善了执行但需要高精度格式。最近的键值（KV）缓存量化减少了内存带宽需求，但在注意力操作中仍需浮点反量化。本文提出了TurboAttention，这是一种全面的方法，实现了量化执行的注意力，同时提高了内存和计算效率。我们的解决方案引入了两个关键创新：FlashQ，一种逐头注意力量化技术，可以压缩KV缓存并实现量化执行的激活-激活乘法；以及基于稀疏性的Softmax近似（SAS），消除了在指数运算中反量化为FP32的需求。实验结果表明，TurboAttention在注意力中实现了1.2到1.8倍的速度提升，KV缓存大小减少了超过4.4倍，并在FP16基线上实现了高达2.37倍的最大吞吐量，优于最先进的量化和压缩技术。  
**论文亮点**: 
- 提出了TurboAttention，一种全面的注意力近似方法。
- 引入了FlashQ和SAS两种关键技术，分别提高了KV缓存压缩和计算效率。
- 实验结果显示，TurboAttention在多个数据集和模型上表现出色，显著提升了性能。

### 论文6
**标题**: Know Where You’re Uncertain When Planning with Multimodal Foundation Models: A Formal Framework  
**摘要**: 多模态基础模型提供了一个有前途的框架，用于通过处理感知输入生成可操作计划，以实现机器人感知和规划。然而，解决感知（感官解释）和决策（计划生成）中的不确定性仍然是确保任务可靠性的关键挑战。本文提出了一种全面的框架，用于解耦、量化和缓解这两种形式的不确定性。我们首先介绍了一种不确定性解耦框架，将感知不确定性（源自视觉理解的局限性）与决策不确定性（涉及生成计划的稳健性）区分开来。为了量化每种类型的不确定性，我们提出了针对感知和决策的独特属性的方法：使用符合性预测校准感知不确定性，并引入正式方法驱动预测（FMDP）量化决策不确定性，利用形式验证技术提供理论保证。在此基础上，我们实施了两种有针对性的干预机制：一种动态再观察高不确定性场景的主动感知过程，以增强视觉输入质量；以及一种自动细化程序，通过在高确定性数据上微调模型，提高其满足任务规范的能力。实验证实在真实世界和模拟机器人任务中，我们的不确定性解耦框架将变异性减少了多达40%，并将任务成功率提高了5%。这些改进归因于两种干预措施的综合效果，突显了不确定性解耦的重要性，促进了针对性干预，增强了自主系统的鲁棒性和可靠性。  
**论文亮点**: 
- 提出了一个全面的框架，用于解耦、量化和缓解感知和决策中的不确定性。
- 使用符合性预测和FMDP分别量化感知和决策不确定性。
- 实验结果显示，变异性减少多达40%，任务成功率提高5%。

### 论文7
**标题**: QServe:W4A8KV4 Quantization and System Co-design for Efficient LLM Serving  
**摘要**: 量化可以加速大型语言模型（LLM）推理。超越INT8量化，研究社区正在积极探索更低精度的量化方法，如INT4。然而，最先进的INT4量化技术只加速了低批量、边缘LLM推理，在大批次、云环境下的LLM服务中未能带来性能提升。我们发现，现有INT4量化方法在GPU上反量化权重或部分和时存在显著的运行时开销（20-90%）。为了解决这一挑战，我们提出了QoQ，一种W4A8KV4量化算法，具有4位权重、8位激活和4位KV缓存。QoQ代表4-8-4，来源于拉丁语quattuor-octo-quattuor。QoQ由QServe推理库实现，实现了测量速度提升。关键见解是，LLM在GPU上的服务效率受到低吞吐量CUDA核心操作的影响。基于这一见解，QServe通过渐进量化减少W4A8 GEMM的反量化开销，并开发SmoothAttention有效缓解4位KV量化带来的准确性下降。在QServe系统中，我们执行计算感知的权重重新排序，并利用寄存器级别的并行性减少反量化延迟。我们将理论上的KV4注意带来的内存节省转化为QServe中的测量速度提升。结果表明，QServe在A100上将Llama-3-8B的最大可实现服务吞吐量提高了1.2倍，在L40S上提高了1.4倍；在A100上将Qwen1.5-72B的吞吐量提高了2.4倍，在L40S上提高了3.5倍，超过了TensorRT-LLM。值得注意的是，QServe在L40S GPU上的吞吐量甚至高于TensorRT-LLM在A100上的表现。  
**论文亮点**: 
- 提出了QoQ，一种高效的W4A8KV4量化算法。
- 实现了渐进量化和SmoothAttention，显著提高了性能和效率。
- 实验结果显示，QServe在多个模型和硬件平台上显著提升了服务吞吐量。

### 论文8
**标题**: Marconi: Prefix Caching for the Era of Hybrid LLMs  
**摘要**: 混合模型结合了注意力层的能力建模与递归层的效率，已在实践中广泛应用于支持长上下文的大型语言模型服务。然而，这些模型的独特属性使诸如前缀缓存等效率优化变得复杂，后者可以跳过跨请求的冗余计算。特别是，混合模型中递归层的就地状态更新禁止回滚部分序列重叠的缓存条目，而是要求完全匹配的缓存命中；结果是每个序列产生大量（大）缓存条目，其中大多数提供的重用机会有限。我们提出了Marconi，第一个支持混合LLM高效前缀缓存的系统。Marconi的关键在于其新颖的接纳和驱逐策略，不仅考虑了最近性，还考虑了（1）不同命中场景分类中的重用可能性预测，以及（2）命中带来的计算节省相对于内存占用。在多样化的负载和混合模型中，Marconi实现了最高34.4倍的令牌命中率（71.1%或617毫秒更低的TTFT），相比最先进的前缀缓存系统。  
**论文亮点**: 
- 提出了Marconi，第一个支持混合LLM高效前缀缓存的系统。
- 引入了接纳和驱逐策略，优化了缓存条目的重用和计算节省。
- 实验结果显示，Marconi显著提高了令牌命中率和性能。

### 论文9
**标题**: Scaling Deep Learning Training with MPMD Pipeline Parallelism  
**摘要**: 我们提出了JaxPP，一个系统，用于高效扩展大型深度学习模型的训练，具有灵活的流水线并行性。我们引入了一种无缝的编程模型，允许实现用户定义的梯度累积流水线调度。JaxPP自动将任务分配到节点集群，并自动推断它们之间的通信。我们实现了MPMD运行时，用于异步执行SPMD任务。JaxPP的流水线并行实现将硬件利用率提高了1.16倍，相对于最佳的SPMD配置。  
**论文亮点**: 
- 提出了JaxPP，一个用于高效扩展深度学习训练的系统。
- 引入了无缝的编程模型和自动任务分配，简化了流水线并行实现。
- 实验结果显示，硬件利用率提高了1.16倍。

### 论文10
**标题**: TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives  
**摘要**: 大型深度学习模型在广泛的任务中取得了最先进的性能。这些模型通常需要分布式系统来进行有效的训练和推理。分布式模型执行的基本构建块是层内并行算子。最有效的方法是通过操作分解或内核融合来重叠计算与通信。虽然操作分解易于实现，但通常会导致次优性能。另一方面，将通信内核与计算内核融合需要专业知识并且容易出错。本文提出了TileLink，一种用于高效编译和生成重叠计算-通信内核的框架。TileLink由前端和后端组成。在前端，TileLink解耦通信和计算的设计空间，通过以瓦片为中心的原语连接这两部分。在后端，TileLink将这些原语转换为低级通信指令，集成通信和计算组件以实现重叠执行。实验结果表明，TileLink比非重叠基线快1.17倍至20.76倍，性能与最先进的重叠库相当。  
**论文亮点**: 
- 提出了TileLink，一种用于生成高效重叠计算-通信内核的框架。
- 解耦通信和计算设计空间，通过以瓦片为中心的原语连接两者。
- 实验结果显示，TileLink显著提高了性能，接近最先进的重叠库。