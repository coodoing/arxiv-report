### [Hard Negative Contrastive Learning for Fine-Grained Geometric Understanding in Large Multimodal Models](http://arxiv.org/abs/2505.20152v1)

Benefiting from contrastively trained visual encoders on large-scale natural
scene images, Large Multimodal Models (LMMs) have achieved remarkable
performance across various visual perception tasks. However, the inherent
limitations of contrastive learning upon summarized descriptions fundamentally
restrict the capabilities of models in meticulous reasoning, particularly in
crucial scenarios of geometric problem-solving. To enhance geometric
understanding, we propose a novel hard negative contrastive learning framework
for the vision encoder, which combines image-based contrastive learning using
generation-based hard negatives created by perturbing diagram generation code,
and text-based contrastive learning using rule-based negatives derived from
modified geometric descriptions and retrieval-based negatives selected based on
caption similarity. We train CLIP using our strong negative learning method,
namely MMCLIP (Multimodal Math CLIP), and subsequently train an LMM for
geometric problem-solving. Experiments show that our trained model, MMGeoLM,
significantly outperforms other open-source models on three geometric reasoning
benchmarks. Even with a size of 7B, it can rival powerful closed-source models
like GPT-4o. We further study the impact of different negative sample
construction methods and the number of negative samples on the geometric
reasoning performance of LMM, yielding fruitful conclusions. The code and
dataset are available at https://github.com/THU-KEG/MMGeoLM.

---


### [MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](http://arxiv.org/abs/2505.20148v1)

Spatial Planning is a crucial part in the field of spatial intelligence,
which requires the understanding and planning about object arrangements in
space perspective. AI agents with the spatial planning ability can better adapt
to various real-world applications, including robotic manipulation, automatic
assembly, urban planning etc. Recent works have attempted to construct
benchmarks for evaluating the spatial intelligence of Multimodal Large Language
Models (MLLMs). Nevertheless, these benchmarks primarily focus on spatial
reasoning based on typical Visual Question-Answering (VQA) forms, which suffers
from the gap between abstract spatial understanding and concrete task
execution. In this work, we take a step further to build a comprehensive
benchmark called MineAnyBuild, aiming to evaluate the spatial planning ability
of open-world AI agents in the Minecraft game. Specifically, MineAnyBuild
requires an agent to generate executable architecture building plans based on
the given multi-modal human instructions. It involves 4,000 curated spatial
planning tasks and also provides a paradigm for infinitely expandable data
collection by utilizing rich player-generated content. MineAnyBuild evaluates
spatial planning through four core supporting dimensions: spatial
understanding, spatial reasoning, creativity, and spatial commonsense. Based on
MineAnyBuild, we perform a comprehensive evaluation for existing MLLM-based
agents, revealing the severe limitations but enormous potential in their
spatial planning abilities. We believe our MineAnyBuild will open new avenues
for the evaluation of spatial intelligence and help promote further development
for open-world AI agents capable of spatial planning.

---


### [Named Entity Recognition in Historical Italian: The Case of Giacomo Leopardi's Zibaldone](http://arxiv.org/abs/2505.20113v1)

The increased digitization of world's textual heritage poses significant
challenges for both computer science and literary studies. Overall, there is an
urgent need of computational techniques able to adapt to the challenges of
historical texts, such as orthographic and spelling variations, fragmentary
structure and digitization errors. The rise of large language models (LLMs) has
revolutionized natural language processing, suggesting promising applications
for Named Entity Recognition (NER) on historical documents. In spite of this,
no thorough evaluation has been proposed for Italian texts. This research tries
to fill the gap by proposing a new challenging dataset for entity extraction
based on a corpus of 19th century scholarly notes, i.e. Giacomo Leopardi's
Zibaldone (1898), containing 2,899 references to people, locations and literary
works. This dataset was used to carry out reproducible experiments with both
domain-specific BERT-based models and state-of-the-art LLMs such as LLaMa3.1.
Results show that instruction-tuned models encounter multiple difficulties
handling historical humanistic texts, while fine-tuned NER models offer more
robust performance even with challenging entity types such as bibliographic
references.

---


### [Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models](http://arxiv.org/abs/2505.20087v1)

Reasoning-based language models have demonstrated strong performance across
various domains, with the most notable gains seen in mathematical and coding
tasks. Recent research has shown that reasoning also offers significant
benefits for LLM safety and guardrail applications. In this work, we conduct a
comprehensive analysis of training reasoning-based guardrail models for content
moderation, with an emphasis on generalization to custom safety policies at
inference time. Our study focuses on two key dimensions: data efficiency and
inference efficiency. On the data front, we find that reasoning-based models
exhibit strong sample efficiency, achieving competitive performance with
significantly fewer training examples than their non-reasoning counterparts.
This unlocks the potential to repurpose the remaining data for mining
high-value, difficult samples that further enhance model performance. On the
inference side, we evaluate practical trade-offs by introducing reasoning
budgets, examining the impact of reasoning length on latency and accuracy, and
exploring dual-mode training to allow runtime control over reasoning behavior.
Our findings will provide practical insights for researchers and developers to
effectively and efficiently train and deploy reasoning-based guardrails models
in real-world systems.

---


### [Curriculum-RLAIF: Curriculum Alignment with Reinforcement Learning from AI Feedback](http://arxiv.org/abs/2505.20075v1)

Reward models trained with conventional Reinforcement Learning from AI
Feedback (RLAIF) methods suffer from limited generalizability, which hinders
the alignment performance of the policy model during reinforcement learning
(RL). This challenge stems from various issues, including distribution shift,
preference label noise, and mismatches between overly challenging samples and
model capacity. In this paper, we attempt to enhance the generalizability of
reward models through a data-centric approach, driven by the insight that these
issues are inherently intertwined from the perspective of data difficulty. To
address this, we propose a novel framework, $\textit{Curriculum-RLAIF}$, which
constructs preference pairs with varying difficulty levels and produces a
curriculum that progressively incorporates preference pairs of increasing
difficulty for reward model training. Our experimental results suggest that
reward models trained with Curriculum-RLAIF achieve improved generalizability,
significantly increasing the alignment performance of the policy model by a
large margin without incurring additional inference costs compared to various
non-curriculum baselines. Detailed analysis and comparisons with alternative
approaches, including data selection via external pretrained reward models or
internal self-selection mechanisms, as well as other curriculum strategies,
further demonstrate the superiority of our approach in terms of simplicity,
efficiency, and effectiveness.

---


### [Incentivizing Reasoning from Weak Supervision](http://arxiv.org/abs/2505.20072v1)

Large language models (LLMs) have demonstrated impressive performance on
reasoning-intensive tasks, but enhancing their reasoning abilities typically
relies on either reinforcement learning (RL) with verifiable signals or
supervised fine-tuning (SFT) with high-quality long chain-of-thought (CoT)
demonstrations, both of which are expensive. In this paper, we study a novel
problem of incentivizing the reasoning capacity of LLMs without expensive
high-quality demonstrations and reinforcement learning. We investigate whether
the reasoning capabilities of LLMs can be effectively incentivized via
supervision from significantly weaker models. We further analyze when and why
such weak supervision succeeds in eliciting reasoning abilities in stronger
models. Our findings show that supervision from significantly weaker reasoners
can substantially improve student reasoning performance, recovering close to
94% of the gains of expensive RL at a fraction of the cost. Experiments across
diverse benchmarks and model architectures demonstrate that weak reasoners can
effectively incentivize reasoning in stronger student models, consistently
improving performance across a wide range of reasoning tasks. Our results
suggest that this simple weak-to-strong paradigm is a promising and
generalizable alternative to costly methods for incentivizing strong reasoning
capabilities at inference-time in LLMs. The code is publicly available at
https://github.com/yuanyige/W2SR.

---


### [SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](http://arxiv.org/abs/2505.20065v1)

As Large Language Models (LLMs) continue to advance and find applications
across a growing number of fields, ensuring the safety of LLMs has become
increasingly critical. To address safety concerns, recent studies have proposed
integrating safety constraints into Reinforcement Learning from Human Feedback
(RLHF). However, these approaches tend to be complex, as they encompass
complicated procedures in RLHF along with additional steps required by the
safety constraints. Inspired by Direct Preference Optimization (DPO), we
introduce a new algorithm called SafeDPO, which is designed to directly
optimize the safety alignment objective in a single stage of policy learning,
without requiring relaxation. SafeDPO introduces only one additional
hyperparameter to further enhance safety and requires only minor modifications
to standard DPO. As a result, it eliminates the need to fit separate reward and
cost models or to sample from the language model during fine-tuning, while
still enhancing the safety of LLMs. Finally, we demonstrate that SafeDPO
achieves competitive performance compared to state-of-the-art safety alignment
algorithms, both in terms of aligning with human preferences and improving
safety.

---


### [Learning to Select In-Context Demonstration Preferred by Large Language Model](http://arxiv.org/abs/2505.19966v1)

In-context learning (ICL) enables large language models (LLMs) to adapt to
new tasks during inference using only a few demonstrations. However, ICL
performance is highly dependent on the selection of these demonstrations.
Recent work explores retrieval-based methods for selecting query-specific
demonstrations, but these approaches often rely on surrogate objectives such as
metric learning, failing to directly optimize ICL performance. Consequently,
they struggle to identify truly beneficial demonstrations. Moreover, their
discriminative retrieval paradigm is ineffective when the candidate pool lacks
sufficient high-quality demonstrations. To address these challenges, we propose
GenICL, a novel generative preference learning framework that leverages LLM
feedback to directly optimize demonstration selection for ICL. Experiments on
19 datasets across 11 task categories demonstrate that GenICL achieves superior
performance than existing methods in selecting the most effective
demonstrations, leading to better ICL performance.

---


### [Adaptive Location Hierarchy Learning for Long-Tailed Mobility Prediction](http://arxiv.org/abs/2505.19965v1)

Human mobility prediction is crucial for applications ranging from
location-based recommendations to urban planning, which aims to forecast users'
next location visits based on historical trajectories. Despite the severe
long-tailed distribution of locations, the problem of long-tailed mobility
prediction remains largely underexplored. Existing long-tailed learning methods
primarily focus on rebalancing the skewed distribution at the data, model, or
class level, neglecting to exploit the spatiotemporal semantics of locations.
To address this gap, we propose the first plug-and-play framework for
long-tailed mobility prediction in an exploitation and exploration manner,
named \textbf{A}daptive \textbf{LO}cation \textbf{H}ier\textbf{A}rchy learning
(ALOHA). First, we construct city-tailored location hierarchy based on Large
Language Models (LLMs) by exploiting Maslow's theory of human motivation to
design Chain-of-Thought (CoT) prompts that captures spatiotemporal semantics.
Second, we optimize the location hierarchy predictions by Gumbel disturbance
and node-wise adaptive weights within the hierarchical tree structure.
Experiments on state-of-the-art models across six datasets demonstrate the
framework's consistent effectiveness and generalizability, which strikes a well
balance between head and tail locations. Weight analysis and ablation studies
reveal the optimization differences of each component for head and tail
locations. Furthermore, in-depth analyses of hierarchical distance and case
study demonstrate the effective semantic guidance from the location hierarchy.
Our code will be made publicly available.

---


### [DCG-SQL: Enhancing In-Context Learning for Text-to-SQL with Deep Contextual Schema Link Graph](http://arxiv.org/abs/2505.19956v1)

Text-to-SQL, which translates a natural language question into an SQL query,
has advanced with in-context learning of Large Language Models (LLMs). However,
existing methods show little improvement in performance compared to randomly
chosen demonstrations, and significant performance drops when smaller LLMs
(e.g., Llama 3.1-8B) are used. This indicates that these methods heavily rely
on the intrinsic capabilities of hyper-scaled LLMs, rather than effectively
retrieving useful demonstrations. In this paper, we propose a novel approach
for effectively retrieving demonstrations and generating SQL queries. We
construct a Deep Contextual Schema Link Graph, which contains key information
and semantic relationship between a question and its database schema items.
This graph-based structure enables effective representation of Text-to-SQL
samples and retrieval of useful demonstrations for in-context learning.
Experimental results on the Spider benchmark demonstrate the effectiveness of
our approach, showing consistent improvements in SQL generation performance and
efficiency across both hyper-scaled LLMs and small LLMs. Our code will be
released.

---


### [MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](http://arxiv.org/abs/2505.19955v1)

Recent advancements in AI agents have demonstrated their growing potential to
drive and support scientific discovery. In this work, we introduce MLR-Bench, a
comprehensive benchmark for evaluating AI agents on open-ended machine learning
research. MLR-Bench includes three key components: (1) 201 research tasks
sourced from NeurIPS, ICLR, and ICML workshops covering diverse ML topics; (2)
MLR-Judge, an automated evaluation framework combining LLM-based reviewers with
carefully designed review rubrics to assess research quality; and (3)
MLR-Agent, a modular agent scaffold capable of completing research tasks
through four stages: idea generation, proposal formulation, experimentation,
and paper writing. Our framework supports both stepwise assessment across these
distinct research stages, and end-to-end evaluation of the final research
paper. We then use MLR-Bench to evaluate six frontier LLMs and an advanced
coding agent, finding that while LLMs are effective at generating coherent
ideas and well-structured papers, current coding agents frequently (e.g., in
80% of the cases) produce fabricated or invalidated experimental
results--posing a major barrier to scientific reliability. We validate
MLR-Judge through human evaluation, showing high agreement with expert
reviewers, supporting its potential as a scalable tool for research evaluation.
We open-source MLR-Bench to help the community benchmark, diagnose, and improve
AI research agents toward trustworthy and transparent scientific discovery.

---


### [Can Visual Encoder Learn to See Arrows?](http://arxiv.org/abs/2505.19944v1)

The diagram is a visual representation of a relationship illustrated with
edges (lines or arrows), which is widely used in industrial and scientific
communication. Although recognizing diagrams is essential for vision language
models (VLMs) to comprehend domain-specific knowledge, recent studies reveal
that many VLMs fail to identify edges in images. We hypothesize that these
failures stem from an over-reliance on textual and positional biases,
preventing VLMs from learning explicit edge features. Based on this idea, we
empirically investigate whether the image encoder in VLMs can learn edge
representation through training on a diagram dataset in which edges are biased
neither by textual nor positional information. To this end, we conduct
contrastive learning on an artificially generated diagram--caption dataset to
train an image encoder and evaluate its diagram-related features on three
tasks: probing, image retrieval, and captioning. Our results show that the
finetuned model outperforms pretrained CLIP in all tasks and surpasses
zero-shot GPT-4o and LLaVA-Mistral in the captioning task. These findings
confirm that eliminating textual and positional biases fosters accurate edge
recognition in VLMs, offering a promising path for advancing diagram
understanding.

---


### [Subtle Risks, Critical Failures: A Framework for Diagnosing Physical Safety of LLMs for Embodied Decision Making](http://arxiv.org/abs/2505.19933v1)

Large Language Models (LLMs) are increasingly used for decision making in
embodied agents, yet existing safety evaluations often rely on coarse success
rates and domain-specific setups, making it difficult to diagnose why and where
these models fail. This obscures our understanding of embodied safety and
limits the selective deployment of LLMs in high-risk physical environments. We
introduce SAFEL, the framework for systematically evaluating the physical
safety of LLMs in embodied decision making. SAFEL assesses two key
competencies: (1) rejecting unsafe commands via the Command Refusal Test, and
(2) generating safe and executable plans via the Plan Safety Test. Critically,
the latter is decomposed into functional modules, goal interpretation,
transition modeling, action sequencing, enabling fine-grained diagnosis of
safety failures. To support this framework, we introduce EMBODYGUARD, a
PDDL-grounded benchmark containing 942 LLM-generated scenarios covering both
overtly malicious and contextually hazardous instructions. Evaluation across 13
state-of-the-art LLMs reveals that while models often reject clearly unsafe
commands, they struggle to anticipate and mitigate subtle, situational risks.
Our results highlight critical limitations in current LLMs and provide a
foundation for more targeted, modular improvements in safe embodied reasoning.

---


### [TCP: a Benchmark for Temporal Constraint-Based Planning](http://arxiv.org/abs/2505.19927v1)

Temporal reasoning and planning are essential capabilities for large language
models (LLMs), yet most existing benchmarks evaluate them in isolation and
under limited forms of complexity. To address this gap, we introduce the
Temporal Constraint-based Planning (TCP) benchmark, that jointly assesses both
capabilities. Each instance in TCP features a naturalistic dialogue around a
collaborative project, where diverse and interdependent temporal constraints
are explicitly or implicitly expressed, and models must infer an optimal
schedule that satisfies all constraints. To construct TCP, we first generate
abstract problem prototypes that are paired with realistic scenarios from
various domains and enriched into dialogues using an LLM. A human quality check
is performed on a sampled subset to confirm the reliability of our benchmark.
We evaluate state-of-the-art LLMs and find that even the strongest models
struggle with TCP, highlighting its difficulty and revealing limitations in
LLMs' temporal constraint-based planning abilities. We analyze underlying
failure cases, open source our benchmark, and hope our findings can inspire
future research.

---


### [Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles](http://arxiv.org/abs/2505.19914v1)

Large Language Models (LLMs), such as OpenAI's o1 and DeepSeek's R1, excel at
advanced reasoning tasks like math and coding via Reinforcement Learning with
Verifiable Rewards (RLVR), but still struggle with puzzles solvable by humans
without domain knowledge. We introduce Enigmata, the first comprehensive suite
tailored for improving LLMs with puzzle reasoning skills. It includes 36 tasks
across seven categories, each with 1) a generator that produces unlimited
examples with controllable difficulty and 2) a rule-based verifier for
automatic evaluation. This generator-verifier design supports scalable,
multi-task RL training, fine-grained analysis, and seamless RLVR integration.
We further propose Enigmata-Eval, a rigorous benchmark, and develop optimized
multi-task RLVR strategies. Our trained model, Qwen2.5-32B-Enigmata,
consistently surpasses o3-mini-high and o1 on the puzzle reasoning benchmarks
like Enigmata-Eval, ARC-AGI (32.8%), and ARC-AGI 2 (0.6%). It also generalizes
well to out-of-domain puzzle benchmarks and mathematical reasoning, with little
multi-tasking trade-off. When trained on larger models like Seed1.5-Thinking
(20B activated parameters and 200B total parameters), puzzle data from Enigmata
further boosts SoTA performance on advanced math and STEM reasoning tasks such
as AIME (2024-2025), BeyondAIME and GPQA (Diamond), showing nice generalization
benefits of Enigmata. This work offers a unified, controllable framework for
advancing logical reasoning in LLMs. Resources of this work can be found at
https://seed-enigmata.github.io.

---


### [APE: A Data-Centric Benchmark for Efficient LLM Adaptation in Text Summarization](http://arxiv.org/abs/2505.19912v1)

We present Adjacent Possible Exploration (APE), a simple yet effective method
for adapting large language models to specific tasks using minimal
computational resources. Unlike traditional fine-tuning that requires extensive
compute, APE iteratively fine-tunes models on small, carefully selected data
batches (200 examples), retaining only improvements. On news summarization, APE
achieves 40 percent BLEU improvement using just a T4 GPU in 60 minutes,
matching or exceeding more complex methods like LoRA while remaining
conceptually simple. Our approach is particularly valuable for researchers and
practitioners with limited computational resources. We provide open-source code
and demonstrate APE's effectiveness through both automatic metrics and human
evaluation. While inspired by evolutionary theory's "adjacent possible", APE's
core insight has a very practical application: small, iterative data
perturbations can efficiently guide LLMs toward task-specific performance
without expensive retraining.

---


### [EMAC+: Embodied Multimodal Agent for Collaborative Planning with VLM+LLM](http://arxiv.org/abs/2505.19905v1)

Although LLMs demonstrate proficiency in several text-based reasoning and
planning tasks, their implementation in robotics control is constrained by
significant deficiencies: (1) LLM agents are designed to work mainly with
textual inputs rather than visual conditions; (2) Current multimodal agents
treat LLMs as static planners, which separates their reasoning from environment
dynamics, resulting in actions that do not take domain-specific knowledge into
account; and (3) LLMs are not designed to learn from visual interactions, which
makes it harder for them to make better policies for specific domains. In this
paper, we introduce EMAC+, an Embodied Multimodal Agent that collaboratively
integrates LLM and VLM via a bidirectional training paradigm. Unlike existing
methods, EMAC+ dynamically refines high-level textual plans generated by an LLM
using real-time feedback from a VLM executing low-level visual control tasks.
We address critical limitations of previous models by enabling the LLM to
internalize visual environment dynamics directly through interactive
experience, rather than relying solely on static symbolic mappings. Extensive
experimental evaluations on ALFWorld and RT-1 benchmarks demonstrate that EMAC+
achieves superior task performance, robustness against noisy observations, and
efficient learning. We also conduct thorough ablation studies and provide
detailed analyses of success and failure cases.

---


### [ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows](http://arxiv.org/abs/2505.19897v1)

Large Language Models (LLMs) have extended their impact beyond Natural
Language Processing, substantially fostering the development of
interdisciplinary research. Recently, various LLM-based agents have been
developed to assist scientific discovery progress across multiple aspects and
domains. Among these, computer-using agents, capable of interacting with
operating systems as humans do, are paving the way to automated scientific
problem-solving and addressing routines in researchers' workflows. Recognizing
the transformative potential of these agents, we introduce ScienceBoard, which
encompasses two complementary contributions: (i) a realistic, multi-domain
environment featuring dynamic and visually rich scientific workflows with
integrated professional software, where agents can autonomously interact via
different interfaces to accelerate complex research tasks and experiments; and
(ii) a challenging benchmark of 169 high-quality, rigorously validated
real-world tasks curated by humans, spanning scientific-discovery workflows in
domains such as biochemistry, astronomy, and geoinformatics. Extensive
evaluations of agents with state-of-the-art backbones (e.g., GPT-4o, Claude
3.7, UI-TARS) show that, despite some promising results, they still fall short
of reliably assisting scientists in complex workflows, achieving only a 15%
overall success rate. In-depth analysis further provides valuable insights for
addressing current agent limitations and more effective design principles,
paving the way to build more capable agents for scientific discovery. Our code,
environment, and benchmark are at
https://qiushisun.github.io/ScienceBoard-Home/.

---


### [Large Language Models as Autonomous Spacecraft Operators in Kerbal Space Program](http://arxiv.org/abs/2505.19896v1)

Recent trends are emerging in the use of Large Language Models (LLMs) as
autonomous agents that take actions based on the content of the user text
prompts. We intend to apply these concepts to the field of Control in space,
enabling LLMs to play a significant role in the decision-making process for
autonomous satellite operations. As a first step towards this goal, we have
developed a pure LLM-based solution for the Kerbal Space Program Differential
Games (KSPDG) challenge, a public software design competition where
participants create autonomous agents for maneuvering satellites involved in
non-cooperative space operations, running on the KSP game engine. Our approach
leverages prompt engineering, few-shot prompting, and fine-tuning techniques to
create an effective LLM-based agent that ranked 2nd in the competition. To the
best of our knowledge, this work pioneers the integration of LLM agents into
space research. The project comprises several open repositories to facilitate
replication and further research. The codebase is accessible on
\href{https://github.com/ARCLab-MIT/kspdg}{GitHub}, while the trained models
and datasets are available on \href{https://huggingface.co/OhhTuRnz}{Hugging
Face}. Additionally, experiment tracking and detailed results can be reviewed
on \href{https://wandb.ai/carrusk/huggingface}{Weights \& Biases

---


### [Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging](http://arxiv.org/abs/2505.19892v1)

While foundation models update slowly due to resource-intensive training
requirements, domain-specific models evolve between updates. Model merging aims
to combine multiple expert models into a single, more capable model, thereby
reducing storage and serving costs while supporting decentralized model
development. Despite its potential, previous studies have primarily focused on
merging visual classification models or Large Language Models (LLMs) for code
and math tasks. Multimodal Large Language Models (MLLMs), which extend the
capabilities of LLMs through large-scale multimodal training, have gained
traction. However, there lacks a benchmark for model merging research that
clearly divides the tasks for MLLM training and evaluation. In this paper, (i)
we introduce the model merging benchmark for MLLMs, which includes multiple
tasks such as VQA, Geometry, Chart, OCR, and Grounding, providing both LoRA and
full fine-tuning models. Moreover, we explore how model merging can combine
different modalities (e.g., vision-language, audio-language, and video-language
models), moving toward the Omni-language model. (ii) We implement 10 model
merging algorithms on the benchmark. Furthermore, we propose a novel method
that removes noise from task vectors and robustly optimizes the merged vector
based on a loss defined over task vector interactions, achieving an average
performance gain of 2.48%. (iii) We find that model merging offers a promising
way for building improved MLLMs without requiring data training. Our results
also demonstrate that the complementarity among multiple modalities outperforms
individual modalities.

---


### [Deconstructing Obfuscation: A four-dimensional framework for evaluating Large Language Models assembly code deobfuscation capabilities](http://arxiv.org/abs/2505.19887v1)

Large language models (LLMs) have shown promise in software engineering, yet
their effectiveness for binary analysis remains unexplored. We present the
first comprehensive evaluation of commercial LLMs for assembly code
deobfuscation. Testing seven state-of-the-art models against four obfuscation
scenarios (bogus control flow, instruction substitution, control flow
flattening, and their combination), we found striking performance
variations--from autonomous deobfuscation to complete failure. We propose a
theoretical framework based on four dimensions: Reasoning Depth, Pattern
Recognition, Noise Filtering, and Context Integration, explaining these
variations. Our analysis identifies five error patterns: predicate
misinterpretation, structural mapping errors, control flow misinterpretation,
arithmetic transformation errors, and constant propagation errors, revealing
fundamental limitations in LLM code processing.We establish a three-tier
resistance model: bogus control flow (low resistance), control flow flattening
(moderate resistance), and instruction substitution/combined techniques (high
resistance). Universal failure against combined techniques demonstrates that
sophisticated obfuscation remains effective against advanced LLMs. Our findings
suggest a human-AI collaboration paradigm where LLMs reduce expertise barriers
for certain reverse engineering tasks while requiring human guidance for
complex deobfuscation. This work provides a foundation for evaluating emerging
capabilities and developing resistant obfuscation techniques.x deobfuscation.
This work provides a foundation for evaluating emerging capabilities and
developing resistant obfuscation techniques.

---


### [Beyond Specialization: Benchmarking LLMs for Transliteration of Indian Languages](http://arxiv.org/abs/2505.19851v1)

Transliteration, the process of mapping text from one script to another,
plays a crucial role in multilingual natural language processing, especially
within linguistically diverse contexts such as India. Despite significant
advancements through specialized models like IndicXlit, recent developments in
large language models suggest a potential for general-purpose models to excel
at this task without explicit task-specific training. The current work
systematically evaluates the performance of prominent LLMs, including GPT-4o,
GPT-4.5, GPT-4.1, Gemma-3-27B-it, and Mistral-Large against IndicXlit, a
state-of-the-art transliteration model, across ten major Indian languages.
Experiments utilized standard benchmarks, including Dakshina and Aksharantar
datasets, with performance assessed via Top-1 Accuracy and Character Error
Rate. Our findings reveal that while GPT family models generally outperform
other LLMs and IndicXlit for most instances. Additionally, fine-tuning GPT-4o
improves performance on specific languages notably. An extensive error analysis
and robustness testing under noisy conditions further elucidate strengths of
LLMs compared to specialized models, highlighting the efficacy of foundational
models for a wide spectrum of specialized applications with minimal overhead.

---


### [LAPA-based Dynamic Privacy Optimization for Wireless Federated Learning in Heterogeneous Environments](http://arxiv.org/abs/2505.19823v1)

Federated Learning (FL) is a distributed machine learning paradigm based on
protecting data privacy of devices, which however, can still be broken by
gradient leakage attack via parameter inversion techniques. Differential
privacy (DP) technology reduces the risk of private data leakage by adding
artificial noise to the gradients, but detrimental to the FL utility at the
same time, especially in the scenario where the data is Non-Independent
Identically Distributed (Non-IID). Based on the impact of heterogeneous data on
aggregation performance, this paper proposes a Lightweight Adaptive Privacy
Allocation (LAPA) strategy, which assigns personalized privacy budgets to
devices in each aggregation round without transmitting any additional
information beyond gradients, ensuring both privacy protection and aggregation
efficiency. Furthermore, the Deep Deterministic Policy Gradient (DDPG)
algorithm is employed to optimize the transmission power, in order to determine
the optimal timing at which the adaptively attenuated artificial noise aligns
with the communication noise, enabling an effective balance between DP and
system utility. Finally, a reliable aggregation strategy is designed by
integrating communication quality and data distribution characteristics, which
improves aggregation performance while preserving privacy. Experimental results
demonstrate that the personalized noise allocation and dynamic optimization
strategy based on LAPA proposed in this paper enhances convergence performance
while satisfying the privacy requirements of FL.

---


### [FinLoRA: Benchmarking LoRA Methods for Fine-Tuning LLMs on Financial Datasets](http://arxiv.org/abs/2505.19819v1)

Low-rank adaptation (LoRA) methods show great potential for scaling
pre-trained general-purpose Large Language Models (LLMs) to hundreds or
thousands of use scenarios. However, their efficacy in high-stakes domains like
finance is rarely explored, e.g., passing CFA exams and analyzing SEC filings.
In this paper, we present the open-source FinLoRA project that benchmarks LoRA
methods on both general and highly professional financial tasks. First, we
curated 19 datasets covering diverse financial applications; in particular, we
created four novel XBRL analysis datasets based on 150 SEC filings. Second, we
evaluated five LoRA methods and five base LLMs. Finally, we provide extensive
experimental results in terms of accuracy, F1, and BERTScore and report
computational cost in terms of time and GPU memory during fine-tuning and
inference stages. We find that LoRA methods achieved substantial performance
gains of 36\% on average over base models. Our FinLoRA project provides an
affordable and scalable approach to democratize financial intelligence to the
general public. Datasets, LoRA adapters, code, and documentation are available
at https://github.com/Open-Finance-Lab/FinLoRA

---


### [Done Is Better than Perfect: Unlocking Efficient Reasoning by Structured Multi-Turn Decomposition](http://arxiv.org/abs/2505.19788v1)

Large Reasoning Models (LRMs) are criticized for the excessively lengthy
Chain-of-Thought (CoT) to derive the final answer, suffering from high
first-token and overall latency. Typically, the CoT of LRMs mixes multiple
thinking units; each unit attempts to produce a candidate answer to the
original query. Hence, a natural idea to improve efficiency is to reduce the
unit number. Yet, the fact that the thinking units in vanilla CoT cannot be
explicitly managed renders doing so challenging. This paper introduces
Multi-Turn Decomposition (MinD) to decode conventional CoT into a sequence of
explicit, structured, and turn-wise interactions to bridge the gap. In MinD,
the model provides a multi-turn response to the query, where each turn embraces
a thinking unit and yields a corresponding answer. The subsequent turns can
reflect, verify, revise, or explore alternative approaches to both the thinking
and answer parts of earlier ones. This not only makes the answer delivered more
swiftly, but also enables explicit controls over the iterative reasoning
process (i.e., users may halt or continue at any turn). We follow a supervised
fine-tuning (SFT) then reinforcement learning (RL) paradigm to realize MinD. We
first rephrase the outputs of an LRM into multi-turn formats by prompting
another LLM, and then tune the LRM with such data. Observing that the tuned
model tends to consume even more tokens than the original one (probably due to
that the multi-turn formats introduce additional answer tokens), we advocate
leveraging RL algorithms like GRPO to prioritize correct outputs with fewer
turns. Trained on the MATH dataset using R1-Distill models, MinD can achieve up
to ~70% reduction in both output token usage and time to first token (TTFT),
while maintaining competitive performance on reasoning benchmarks such as
MATH-500, AIME24, AMC23, and GPQA-Diamond.

---


### [MedDreamer: Model-Based Reinforcement Learning with Latent Imagination on Complex EHRs for Clinical Decision Support](http://arxiv.org/abs/2505.19785v1)

Timely and personalized treatment decisions are essential across a wide range
of healthcare settings where patient responses vary significantly and evolve
over time. Clinical data used to support these decisions are often irregularly
sampled, sparse, and noisy. Existing decision support systems commonly rely on
discretization and imputation, which can distort critical temporal dynamics and
degrade decision quality. Moreover, they often overlook the clinical
significance of irregular recording frequencies, filtering out patterns in how
and when data is collected. Reinforcement Learning (RL) is a natural fit for
clinical decision-making, enabling sequential, long-term optimization in
dynamic, uncertain environments. However, most existing treatment
recommendation systems are model-free and trained solely on offline data,
making them sample-inefficient, sensitive to data quality, and poorly
generalizable across tasks or cohorts. To address these limitations, we propose
MedDreamer, a two-phase model-based RL framework for personalized treatment
recommendation. MedDreamer uses a world model with an Adaptive Feature
Integration (AFI) module to effectively model irregular, sparse clinical data.
Through latent imagination, it simulates plausible patient trajectories to
enhance learning, refining its policy using a mix of real and imagined
experiences. This enables learning policies that go beyond suboptimal
historical decisions while remaining grounded in clinical data. To our
knowledge, this is the first application of latent imagination to irregular
healthcare data. Evaluations on sepsis and mechanical ventilation (MV)
treatment using two large-scale EHR datasets show that MedDreamer outperforms
both model-free and model-based baselines in clinical outcomes and off-policy
metrics.

---


### [Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification](http://arxiv.org/abs/2505.19776v1)

Political biases encoded by LLMs might have detrimental effects on downstream
applications. Existing bias analysis methods rely on small-size intermediate
tasks (questionnaire answering or political content generation) and rely on the
LLMs themselves for analysis, thus propagating bias. We propose a new approach
leveraging the observation that LLM sentiment predictions vary with the target
entity in the same sentence. We define an entropy-based inconsistency metric to
encode this prediction variability. We insert 1319 demographically and
politically diverse politician names in 450 political sentences and predict
target-oriented sentiment using seven models in six widely spoken languages. We
observe inconsistencies in all tested combinations and aggregate them in a
statistically robust analysis at different granularity levels. We observe
positive and negative bias toward left and far-right politicians and positive
correlations between politicians with similar alignment. Bias intensity is
higher for Western languages than for others. Larger models exhibit stronger
and more consistent biases and reduce discrepancies between similar languages.
We partially mitigate LLM unreliability in target-oriented sentiment
classification (TSC) by replacing politician names with fictional but plausible
counterparts.

---


### [TeViR: Text-to-Video Reward with Diffusion Models for Efficient Reinforcement Learning](http://arxiv.org/abs/2505.19769v1)

Developing scalable and generalizable reward engineering for reinforcement
learning (RL) is crucial for creating general-purpose agents, especially in the
challenging domain of robotic manipulation. While recent advances in reward
engineering with Vision-Language Models (VLMs) have shown promise, their sparse
reward nature significantly limits sample efficiency. This paper introduces
TeViR, a novel method that leverages a pre-trained text-to-video diffusion
model to generate dense rewards by comparing the predicted image sequence with
current observations. Experimental results across 11 complex robotic tasks
demonstrate that TeViR outperforms traditional methods leveraging sparse
rewards and other state-of-the-art (SOTA) methods, achieving better sample
efficiency and performance without ground truth environmental rewards. TeViR's
ability to efficiently guide agents in complex environments highlights its
potential to advance reinforcement learning applications in robotic
manipulation.

---


### [CIDRe: A Reference-Free Multi-Aspect Criterion for Code Comment Quality Measurement](http://arxiv.org/abs/2505.19757v1)

Effective generation of structured code comments requires robust quality
metrics for dataset curation, yet existing approaches (SIDE, MIDQ, STASIS)
suffer from limited code-comment analysis. We propose CIDRe, a
language-agnostic reference-free quality criterion combining four synergistic
aspects: (1) relevance (code-comment semantic alignment), (2) informativeness
(functional coverage), (3) completeness (presence of all structure sections),
and (4) description length (detail sufficiency). We validate our criterion on a
manually annotated dataset. Experiments demonstrate CIDRe's superiority over
existing metrics, achieving improvement in cross-entropy evaluation. When
applied to filter comments, the models finetuned on CIDRe-filtered data show
statistically significant quality gains in GPT-4o-mini assessments.

---


### [NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](http://arxiv.org/abs/2505.19754v1)

The increasing number of academic papers poses significant challenges for
researchers to efficiently acquire key details. While retrieval augmented
generation (RAG) shows great promise in large language model (LLM) based
automated question answering, previous works often isolate neural and symbolic
retrieval despite their complementary strengths. Moreover, conventional
single-view chunking neglects the rich structure and layout of PDFs, e.g.,
sections and tables. In this work, we propose NeuSym-RAG, a hybrid neural
symbolic retrieval framework which combines both paradigms in an interactive
process. By leveraging multi-view chunking and schema-based parsing, NeuSym-RAG
organizes semi-structured PDF content into both the relational database and
vectorstore, enabling LLM agents to iteratively gather context until sufficient
to generate answers. Experiments on three full PDF-based QA datasets, including
a self-annotated one AIRQA-REAL, show that NeuSym-RAG stably defeats both the
vector-based RAG and various structured baselines, highlighting its capacity to
unify both retrieval schemes and utilize multiple views. Code and data are
publicly available at https://github.com/X-LANCE/NeuSym-RAG.

---


### [MT$^{3}$: Scaling MLLM-based Text Image Machine Translation via Multi-Task Reinforcement Learning](http://arxiv.org/abs/2505.19714v1)

Text Image Machine Translation (TIMT)-the task of translating textual content
embedded in images-is critical for applications in accessibility, cross-lingual
information access, and real-world document understanding. However, TIMT
remains a complex challenge due to the need for accurate optical character
recognition (OCR), robust visual-text reasoning, and high-quality translation,
often requiring cascading multi-stage pipelines. Recent advances in large-scale
Reinforcement Learning (RL) have improved reasoning in Large Language Models
(LLMs) and Multimodal LLMs (MLLMs), but their application to end-to-end TIMT is
still underexplored. To bridge this gap, we introduce MT$^{3}$, the first
framework to apply Multi-Task RL to MLLMs for end-to-end TIMT. MT$^{3}$ adopts
a multi-task optimization paradigm targeting three key sub-skills: text
recognition, context-aware reasoning, and translation. It is trained using a
novel multi-mixed reward mechanism that adapts rule-based RL strategies to
TIMT's intricacies, offering fine-grained, non-binary feedback across tasks.
Furthermore, to facilitate the evaluation of TIMT in authentic cross-cultural
and real-world social media contexts, we introduced XHSPost, the first social
media TIMT benchmark. Our MT$^{3}$-7B-Zero achieves state-of-the-art results on
the latest in-domain MIT-10M benchmark, outperforming strong baselines such as
Qwen2.5-VL-72B and InternVL2.5-78B by notable margins across multiple metrics.
Additionally, the model shows strong generalization to out-of-distribution
language pairs and datasets. In-depth analyses reveal how multi-task synergy,
reinforcement learning initialization, curriculum design, and reward
formulation contribute to advancing MLLM-driven TIMT.

---


### [Error Typing for Smarter Rewards: Improving Process Reward Models with Error-Aware Hierarchical Supervision](http://arxiv.org/abs/2505.19706v1)

Large Language Models (LLMs) are prone to hallucination, especially during
multi-hop and reasoning-intensive tasks such as mathematical problem solving.
While Outcome Reward Models verify only final answers, Process Reward Models
(PRMs) score each intermediate step to steer generation toward coherent
solutions. We introduce PathFinder-PRM, a novel hierarchical, error-aware
discriminative PRM that first classifies math and consistency errors at each
step, then combines these fine-grained signals to estimate step correctness. To
train PathFinder-PRM, we construct a 400K-sample dataset by enriching the
human-annotated PRM800K corpus and RLHFlow Mistral traces with
three-dimensional step-level labels. On PRMBench, PathFinder-PRM achieves a new
state-of-the-art PRMScore of 67.7, outperforming the prior best (65.5) while
using 3 times less data. When applied to reward guided greedy search, our model
yields prm@8 48.3, a +1.5 point gain over the strongest baseline. These results
demonstrate that decoupled error detection and reward estimation not only boost
fine-grained error detection but also substantially improve end-to-end,
reward-guided mathematical reasoning with greater data efficiency.

---


### [Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](http://arxiv.org/abs/2505.19700v1)

The widespread adoption of large language models (LLMs) across industries has
increased the demand for high-quality and customizable outputs. However,
traditional alignment methods often require retraining large pretrained models,
making it difficult to quickly adapt and optimize LLMs for diverse
applications. To address this limitation, we propose a novel \textit{Residual
Alignment Model} (\textit{RAM}) that formalizes the alignment process as a type
of importance sampling. In this framework, the unaligned upstream model serves
as the proposal distribution, while the alignment process is framed as
secondary sampling based on an autoregressive alignment module that acts as an
estimator of the importance weights. This design enables a natural detachment
of the alignment module from the target aligned model, improving flexibility
and scalability. Based on this model, we derive an efficient sequence-level
training strategy for the alignment module, which operates independently of the
proposal module. Additionally, we develop a resampling algorithm with iterative
token-level decoding to address the common first-token latency issue in
comparable methods. Experimental evaluations on two leading open-source LLMs
across diverse tasks, including instruction following, domain adaptation, and
preference optimization, demonstrate that our approach consistently outperforms
baseline models.

---


### [Beyond Safe Answers: A Benchmark for Evaluating True Risk Awareness in Large Reasoning Models](http://arxiv.org/abs/2505.19690v1)

Despite the remarkable proficiency of \textit{Large Reasoning Models} (LRMs)
in handling complex reasoning tasks, their reliability in safety-critical
scenarios remains uncertain. Existing evaluations primarily assess
response-level safety, neglecting a critical issue we identify as
\textbf{\textit{Superficial Safety Alignment} (SSA)} -- a phenomenon where
models produce superficially safe outputs while internal reasoning processes
fail to genuinely detect and mitigate underlying risks, resulting in
inconsistent safety behaviors across multiple sampling attempts. To
systematically investigate SSA, we introduce \textbf{Beyond Safe Answers (BSA)}
bench, a novel benchmark comprising 2,000 challenging instances organized into
three distinct SSA scenario types and spanning nine risk categories, each
meticulously annotated with risk rationales. Evaluations of 19 state-of-the-art
LRMs demonstrate the difficulty of this benchmark, with top-performing models
achieving only 38.0\% accuracy in correctly identifying risk rationales. We
further explore the efficacy of safety rules, specialized fine-tuning on safety
reasoning data, and diverse decoding strategies in mitigating SSA. Our work
provides a comprehensive assessment tool for evaluating and improving safety
reasoning fidelity in LRMs, advancing the development of genuinely risk-aware
and reliably safe AI systems.

---


### [Large Language Models for Planning: A Comprehensive and Systematic Survey](http://arxiv.org/abs/2505.19683v1)

Planning represents a fundamental capability of intelligent agents, requiring
comprehensive environmental understanding, rigorous logical reasoning, and
effective sequential decision-making. While Large Language Models (LLMs) have
demonstrated remarkable performance on certain planning tasks, their broader
application in this domain warrants systematic investigation. This paper
presents a comprehensive review of LLM-based planning. Specifically, this
survey is structured as follows: First, we establish the theoretical
foundations by introducing essential definitions and categories about automated
planning. Next, we provide a detailed taxonomy and analysis of contemporary
LLM-based planning methodologies, categorizing them into three principal
approaches: 1) External Module Augmented Methods that combine LLMs with
additional components for planning, 2) Finetuning-based Methods that involve
using trajectory data and feedback signals to adjust LLMs in order to improve
their planning abilities, and 3) Searching-based Methods that break down
complex tasks into simpler components, navigate the planning space, or enhance
decoding strategies to find the best solutions. Subsequently, we systematically
summarize existing evaluation frameworks, including benchmark datasets,
evaluation metrics and performance comparisons between representative planning
methods. Finally, we discuss the underlying mechanisms enabling LLM-based
planning and outline promising research directions for this rapidly evolving
field. We hope this survey will serve as a valuable resource to inspire
innovation and drive progress in this field.

---


### [Large Language Models' Reasoning Stalls: An Investigation into the Capabilities of Frontier Models](http://arxiv.org/abs/2505.19676v1)

Empirical methods to examine the capability of Large Language Models (LLMs)
to use Automated Theorem Prover (ATP) reasoning strategies are studied. We
evaluate the performance of State of the Art models from December 2023 and
August 2024 on PRONTOQA steamroller reasoning problems. For that, we develop
methods for assessing LLM response accuracy and correct answer correlation.
  Our results show that progress in improving LLM reasoning abilities has
stalled over the nine month period. By tracking completion tokens, we show that
almost all improvement in reasoning ability since GPT-4 was released can be
attributed to either hidden system prompts or the training of models to
automatically use generic Chain of Thought prompting strategies. Among the ATP
reasoning strategies tried, we found that current frontier LLMs are best able
to follow the bottom-up (also known as forward-chaining) strategy. A low
positive correlation was found between an LLM response containing correct
reasoning and arriving at the correct conclusion.

---


### [Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement](http://arxiv.org/abs/2505.19675v1)

The traditional process of creating labeled datasets is labor-intensive and
expensive. Recent breakthroughs in open-source large language models (LLMs)
have opened up a new avenue in generating labeled datasets automatically for
various natural language processing (NLP) tasks, providing an alternative to
such an expensive annotation process. However, the reliability of such
auto-generated labels remains a significant concern due to inherent
inaccuracies. When learning from noisy labels, the model's generalization is
likely to be harmed as it is prone to overfit to those label noises. While
previous studies in learning from noisy labels mainly focus on synthetic noise
and real-world noise, LLM-generated label noise receives less attention. In
this paper, we propose SiDyP: Simplex Label Diffusion with Dynamic Prior to
calibrate the classifier's prediction, thus enhancing its robustness towards
LLM-generated noisy labels. SiDyP retrieves potential true label candidates by
neighborhood label distribution in text embedding space and iteratively refines
noisy candidates using a simplex diffusion model. Our framework can increase
the performance of the BERT classifier fine-tuned on both zero-shot and
few-shot LLM-generated noisy label datasets by an average of 7.21% and 7.30%
respectively. We demonstrate the effectiveness of SiDyP by conducting extensive
benchmarking for different LLMs over a variety of NLP tasks. Our code is
available on Github.

---


### [Automated evaluation of children's speech fluency for low-resource languages](http://arxiv.org/abs/2505.19671v1)

Assessment of children's speaking fluency in education is well researched for
majority languages, but remains highly challenging for low resource languages.
This paper proposes a system to automatically assess fluency by combining a
fine-tuned multilingual ASR model, an objective metrics extraction stage, and a
generative pre-trained transformer (GPT) network. The objective metrics include
phonetic and word error rates, speech rate, and speech-pause duration ratio.
These are interpreted by a GPT-based classifier guided by a small set of
human-evaluated ground truth examples, to score fluency. We evaluate the
proposed system on a dataset of children's speech in two low-resource
languages, Tamil and Malay and compare the classification performance against
Random Forest and XGBoost, as well as using ChatGPT-4o to predict fluency
directly from speech input. Results demonstrate that the proposed approach
achieves significantly higher accuracy than multimodal GPT or other methods.

---


### [LeCoDe: A Benchmark Dataset for Interactive Legal Consultation Dialogue Evaluation](http://arxiv.org/abs/2505.19667v1)

Legal consultation is essential for safeguarding individual rights and
ensuring access to justice, yet remains costly and inaccessible to many
individuals due to the shortage of professionals. While recent advances in
Large Language Models (LLMs) offer a promising path toward scalable, low-cost
legal assistance, current systems fall short in handling the interactive and
knowledge-intensive nature of real-world consultations. To address these
challenges, we introduce LeCoDe, a real-world multi-turn benchmark dataset
comprising 3,696 legal consultation dialogues with 110,008 dialogue turns,
designed to evaluate and improve LLMs' legal consultation capability. With
LeCoDe, we innovatively collect live-streamed consultations from short-video
platforms, providing authentic multi-turn legal consultation dialogues. The
rigorous annotation by legal experts further enhances the dataset with
professional insights and expertise. Furthermore, we propose a comprehensive
evaluation framework that assesses LLMs' consultation capabilities in terms of
(1) clarification capability and (2) professional advice quality. This unified
framework incorporates 12 metrics across two dimensions. Through extensive
experiments on various general and domain-specific LLMs, our results reveal
significant challenges in this task, with even state-of-the-art models like
GPT-4 achieving only 39.8% recall for clarification and 59% overall score for
advice quality, highlighting the complexity of professional consultation
scenarios. Based on these findings, we further explore several strategies to
enhance LLMs' legal consultation abilities. Our benchmark contributes to
advancing research in legal domain dialogue systems, particularly in simulating
more real-world user-expert interactions.

---


### [FieldWorkArena: Agentic AI Benchmark for Real Field Work Tasks](http://arxiv.org/abs/2505.19662v1)

This paper proposes FieldWorkArena, a benchmark for agentic AI targeting
real-world field work. With the recent increase in demand for agentic AI, they
are required to monitor and report safety and health incidents, as well as
manufacturing-related incidents, that may occur in real-world work
environments. Existing agentic AI benchmarks have been limited to evaluating
web tasks and are insufficient for evaluating agents in real-world work
environments, where complexity increases significantly. In this paper, we
define a new action space that agentic AI should possess for real world work
environment benchmarks and improve the evaluation function from previous
methods to assess the performance of agentic AI in diverse real-world tasks.
The dataset consists of videos captured on-site and documents actually used in
factories and warehouses, and tasks were created based on interviews with
on-site workers and managers. Evaluation results confirmed that performance
evaluation considering the characteristics of Multimodal LLM (MLLM) such as
GPT-4o is feasible. Additionally, the effectiveness and limitations of the
proposed new evaluation method were identified. The complete dataset
(HuggingFace) and evaluation program (GitHub) can be downloaded from the
following website:
https://en-documents.research.global.fujitsu.com/fieldworkarena/.

---


### [Large Language Models in Code Co-generation for Safe Autonomous Vehicles](http://arxiv.org/abs/2505.19658v1)

Software engineers in various industrial domains are already using Large
Language Models (LLMs) to accelerate the process of implementing parts of
software systems. When considering its potential use for ADAS or AD systems in
the automotive context, there is a need to systematically assess this new
setup: LLMs entail a well-documented set of risks for safety-related systems'
development due to their stochastic nature. To reduce the effort for code
reviewers to evaluate LLM-generated code, we propose an evaluation pipeline to
conduct sanity-checks on the generated code. We compare the performance of six
state-of-the-art LLMs (CodeLlama, CodeGemma, DeepSeek-r1, DeepSeek-Coders,
Mistral, and GPT-4) on four safety-related programming tasks. Additionally, we
qualitatively analyse the most frequent faults generated by these LLMs,
creating a failure-mode catalogue to support human reviewers. Finally, the
limitations and capabilities of LLMs in code generation, and the use of the
proposed pipeline in the existing process, are discussed.

---


### [Token-Importance Guided Direct Preference Optimization](http://arxiv.org/abs/2505.19653v1)

Ensuring that large language models (LLMs) generate outputs aligned with
human preferences is important for safe and effective AI interactions. While
Direct Preference Optimization (DPO) employs an implicit reward function to
optimize the policy model, however, it and its related variants overlook the
differential importance of individual tokens and are sensitive to judgment
noise in preference datasets during generation. Although recent methods attempt
to assess the important weight of tokens via probability prediction or
simplistic weighting schemes, these evaluation methods are prone to biases and
still cannot fully address these issues. To solve this problem, we propose the
Token-Importance Guided Direct Preference Optimization (TI-DPO), which
introduces two key innovations: the gradient-based token-importance weights
that dynamically prioritize critical tokens, and a triple loss that explicitly
guides model outputs to approach human-preferred responses and stay away from
non-preferred responses. Experimental results show that TI-DPO achieves higher
accuracy and stronger generative diversity, providing more stable and
computationally efficient solutions compared with DPO and other RLHF methods.

---


### [MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE](http://arxiv.org/abs/2505.19645v1)

Large Language Models (LLMs) have achieved remarkable success across many
applications, with Mixture of Experts (MoE) models demonstrating great
potential. Compared to traditional dense models, MoEs achieve better
performance with less computation. Speculative decoding (SD) is a widely used
technique to accelerate LLM inference without accuracy loss, but it has been
considered efficient only for dense models. In this work, we first demonstrate
that, under medium batch sizes, MoE surprisingly benefits more from SD than
dense models. Furthermore, as MoE becomes sparser -- the prevailing trend in
MoE designs -- the batch size range where SD acceleration is expected to be
effective becomes broader. To quantitatively understand tradeoffs involved in
SD, we develop a reliable modeling based on theoretical analyses. While current
SD research primarily focuses on improving acceptance rates of algorithms,
changes in workload and model architecture can still lead to degraded SD
acceleration even with high acceptance rates. To address this limitation, we
introduce a new metric 'target efficiency' that characterizes these effects,
thus helping researchers identify system bottlenecks and understand SD
acceleration more comprehensively. For scenarios like private serving, this
work unveils a new perspective to speed up MoE inference, where existing
solutions struggle. Experiments on different GPUs show up to 2.29x speedup for
Qwen2-57B-A14B at medium batch sizes and validate our theoretical predictions.

---


### [SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond](http://arxiv.org/abs/2505.19641v1)

Recent advances such as OpenAI-o1 and DeepSeek R1 have demonstrated the
potential of Reinforcement Learning (RL) to enhance reasoning abilities in
Large Language Models (LLMs). While open-source replication efforts have
primarily focused on mathematical and coding domains, methods and resources for
developing general reasoning capabilities remain underexplored. This gap is
partly due to the challenge of collecting diverse and verifiable reasoning data
suitable for RL. We hypothesize that logical reasoning is critical for
developing general reasoning capabilities, as logic forms a fundamental
building block of reasoning. In this work, we present SynLogic, a data
synthesis framework and dataset that generates diverse logical reasoning data
at scale, encompassing 35 diverse logical reasoning tasks. The SynLogic
approach enables controlled synthesis of data with adjustable difficulty and
quantity. Importantly, all examples can be verified by simple rules, making
them ideally suited for RL with verifiable rewards. In our experiments, we
validate the effectiveness of RL training on the SynLogic dataset based on 7B
and 32B models. SynLogic leads to state-of-the-art logical reasoning
performance among open-source datasets, surpassing DeepSeek-R1-Distill-Qwen-32B
by 6 points on BBEH. Furthermore, mixing SynLogic data with mathematical and
coding tasks improves the training efficiency of these domains and
significantly enhances reasoning generalization. Notably, our mixed training
model outperforms DeepSeek-R1-Zero-Qwen-32B across multiple benchmarks. These
findings position SynLogic as a valuable resource for advancing the broader
reasoning capabilities of LLMs. We open-source both the data synthesis pipeline
and the SynLogic dataset at https://github.com/MiniMax-AI/SynLogic.

---


### [Benchmarking Large Multimodal Models for Ophthalmic Visual Question Answering with OphthalWeChat](http://arxiv.org/abs/2505.19624v1)

Purpose: To develop a bilingual multimodal visual question answering (VQA)
benchmark for evaluating VLMs in ophthalmology. Methods: Ophthalmic image posts
and associated captions published between January 1, 2016, and December 31,
2024, were collected from WeChat Official Accounts. Based on these captions,
bilingual question-answer (QA) pairs in Chinese and English were generated
using GPT-4o-mini. QA pairs were categorized into six subsets by question type
and language: binary (Binary_CN, Binary_EN), single-choice (Single-choice_CN,
Single-choice_EN), and open-ended (Open-ended_CN, Open-ended_EN). The benchmark
was used to evaluate the performance of three VLMs: GPT-4o, Gemini 2.0 Flash,
and Qwen2.5-VL-72B-Instruct. Results: The final OphthalWeChat dataset included
3,469 images and 30,120 QA pairs across 9 ophthalmic subspecialties, 548
conditions, 29 imaging modalities, and 68 modality combinations. Gemini 2.0
Flash achieved the highest overall accuracy (0.548), outperforming GPT-4o
(0.522, P < 0.001) and Qwen2.5-VL-72B-Instruct (0.514, P < 0.001). It also led
in both Chinese (0.546) and English subsets (0.550). Subset-specific
performance showed Gemini 2.0 Flash excelled in Binary_CN (0.687),
Single-choice_CN (0.666), and Single-choice_EN (0.646), while GPT-4o ranked
highest in Binary_EN (0.717), Open-ended_CN (BLEU-1: 0.301; BERTScore: 0.382),
and Open-ended_EN (BLEU-1: 0.183; BERTScore: 0.240). Conclusions: This study
presents the first bilingual VQA benchmark for ophthalmology, distinguished by
its real-world context and inclusion of multiple examinations per patient. The
dataset reflects authentic clinical decision-making scenarios and enables
quantitative evaluation of VLMs, supporting the development of accurate,
specialized, and trustworthy AI systems for eye care.

---


### [Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models](http://arxiv.org/abs/2505.19616v1)

Multimodal Large Language Models (MLLMs) have demonstrated impressive
capabilities across tasks, yet they often exhibit difficulty in distinguishing
task-relevant from irrelevant signals, particularly in tasks like Visual
Question Answering (VQA), which can lead to susceptibility to misleading or
spurious inputs. We refer to this broader limitation as the Cross-Modality
Competency Problem: the model's inability to fairly evaluate all modalities.
This vulnerability becomes more evident in modality-specific tasks such as
image classification or pure text question answering, where models are expected
to rely solely on one modality. In such tasks, spurious information from
irrelevant modalities often leads to significant performance degradation. We
refer to this failure as Modality Interference, which serves as a concrete and
measurable instance of the cross-modality competency problem. We further design
a perturbation-based causal diagnostic experiment to verify and quantify this
problem. To mitigate modality interference, we propose a novel framework to
fine-tune MLLMs, including perturbation-based data augmentations with both
heuristic perturbations and adversarial perturbations via Projected Gradient
Descent (PGD), and a consistency regularization strategy applied to model
outputs with original and perturbed inputs. Experiments on multiple benchmark
datasets (image-heavy, text-heavy, and VQA tasks) and multiple model families
with different scales demonstrate significant improvements in robustness and
cross-modality competency, indicating our method's effectiveness in boosting
unimodal reasoning ability while enhancing performance on multimodal tasks.

---


### [Energy-based Preference Optimization for Test-time Adaptation](http://arxiv.org/abs/2505.19607v1)

Test-Time Adaptation (TTA) enhances model robustness by enabling adaptation
to target distributions that differ from training distributions, improving
real-world generalizability. Existing TTA approaches focus on adjusting the
conditional distribution; however these methods often depend on uncertain
predictions in the absence of label information, leading to unreliable
performance. Energy-based frameworks suggest a promising alternative to address
distribution shifts without relying on uncertain predictions, instead computing
the marginal distribution of target data. However, they involve the critical
challenge of requiring extensive SGLD sampling, which is impractical for
test-time scenarios requiring immediate adaptation. In this work, we propose
Energy-based Preference Optimization for Test-time Adaptation (EPOTTA), which
is based on a sampling free strategy. We first parameterize the target model
using a pretrained model and residual energy function, enabling marginal
likelihood maximization of target data without sampling. Building on the
observation that the parameterization is mathematically equivalent to DPO
objective, we then directly adapt the model to a target distribution without
explicitly training the residual. Our experiments verify that EPOTTA is
well-calibrated and performant while achieving computational efficiency.

---


### [Preference Optimization by Estimating the Ratio of the Data Distribution](http://arxiv.org/abs/2505.19601v1)

Direct preference optimization (DPO) is widely used as a simple and stable
method for aligning large language models (LLMs) with human preferences. This
paper investigates a generalized DPO loss that enables a policy model to match
the target policy from a likelihood ratio estimation perspective. The ratio of
the target policy provides a unique identification of the policy distribution
without relying on reward models or partition functions. This allows the
generalized loss to retain both simplicity and theoretical guarantees, which
prior work such as $f$-PO fails to achieve simultaneously. We propose Bregman
preference optimization (BPO), a generalized framework for ratio matching that
provides a family of objective functions achieving target policy optimality.
BPO subsumes DPO as a special case and offers tractable forms for all
instances, allowing implementation with a few lines of code. We further develop
scaled Basu's power divergence (SBA), a gradient scaling method that can be
used for BPO instances. The BPO framework complements other DPO variants and is
applicable to target policies defined by these variants. In experiments, unlike
other probabilistic loss extensions such as $f$-DPO or $f$-PO, which exhibit a
trade-off between generation fidelity and diversity, instances of BPO improve
both win rate and entropy compared with DPO. When applied to
Llama-3-Instruct-8B, BPO achieves state-of-the-art performance among Llama-3-8B
backbones, with a 55.9\% length-controlled win rate on AlpacaEval2.

---


### [MSD-LLM: Predicting Ship Detention in Port State Control Inspections with Large Language Model](http://arxiv.org/abs/2505.19568v1)

Maritime transportation is the backbone of global trade, making ship
inspection essential for ensuring maritime safety and environmental protection.
Port State Control (PSC), conducted by national ports, enforces compliance with
safety regulations, with ship detention being the most severe consequence,
impacting both ship schedules and company reputations. Traditional machine
learning methods for ship detention prediction are limited by the capacity of
representation learning and thus suffer from low accuracy. Meanwhile,
autoencoder-based deep learning approaches face challenges due to the severe
data imbalance in learning historical PSC detention records. To address these
limitations, we propose Maritime Ship Detention with Large Language Models
(MSD-LLM), integrating a dual robust subspace recovery (DSR) layer-based
autoencoder with a progressive learning pipeline to handle imbalanced data and
extract meaningful PSC representations. Then, a large language model groups and
ranks features to identify likely detention cases, enabling dynamic
thresholding for flexible detention predictions. Extensive evaluations on
31,707 PSC inspection records from the Asia-Pacific region show that MSD-LLM
outperforms state-of-the-art methods more than 12\% on Area Under the Curve
(AUC) for Singapore ports. Additionally, it demonstrates robustness to
real-world challenges, making it adaptable to diverse maritime risk assessment
scenarios.

---


### [LLM-Agent-Controller: A Universal Multi-Agent Large Language Model System as a Control Engineer](http://arxiv.org/abs/2505.19567v1)

This study presents the LLM-Agent-Controller, a multi-agent large language
model (LLM) system developed to address a wide range of problems in control
engineering (Control Theory). The system integrates a central controller agent
with multiple specialized auxiliary agents, responsible for tasks such as
controller design, model representation, control analysis, time-domain
response, and simulation. A supervisor oversees high-level decision-making and
workflow coordination, enhancing the system's reliability and efficiency. The
LLM-Agent-Controller incorporates advanced capabilities, including
Retrieval-Augmented Generation (RAG), Chain-of-Thought reasoning,
self-criticism and correction, efficient memory handling, and user-friendly
natural language communication. It is designed to function without requiring
users to have prior knowledge of Control Theory, enabling them to input
problems in plain language and receive complete, real-time solutions. To
evaluate the system, we propose new performance metrics assessing both
individual agents and the system as a whole. We test five categories of Control
Theory problems and benchmark performance across three advanced LLMs.
Additionally, we conduct a comprehensive qualitative conversational analysis
covering all key services. Results show that the LLM-Agent-Controller
successfully solved 83% of general tasks, with individual agents achieving an
average success rate of 87%. Performance improved with more advanced LLMs. This
research demonstrates the potential of multi-agent LLM architectures to solve
complex, domain-specific problems. By integrating specialized agents,
supervisory control, and advanced reasoning, the LLM-Agent-Controller offers a
scalable, robust, and accessible solution framework that can be extended to
various technical domains.

---


### [Automated Text-to-Table for Reasoning-Intensive Table QA: Pipeline Design and Benchmarking Insights](http://arxiv.org/abs/2505.19563v1)

Reasoning with tabular data holds increasing importance in modern
applications, yet comprehensive evaluation methodologies for
reasoning-intensive Table Question Answering (QA) tasks remain nascent.
Existing research is constrained by two primary bottlenecks: 1) Reliance on
costly manually annotated real-world data, which is difficult to cover complex
reasoning scenarios; 2) The heterogeneity of table structures hinders
systematic analysis of the intrinsic mechanisms behind the underperformance of
LLMs, especially in reasoning-intensive tasks. To address these issues, we
propose an automated generation pipeline AutoT2T that transforms mathematical
word problems into table-based reasoning tasks, eliminating the need for manual
annotation. The pipeline can generate multiple variants of a table for the same
reasoning problem, including noisy versions to support robustness evaluation.
Based on this, we construct a new benchmark TabularGSM, which systematically
spans a range of table complexities and trap problems. Experimental analyses
through AutoT2T and TabularGSM reveal that the tight coupling between reasoning
and retrieval or identification processes is a key factor underlying the
failure of LLMs in complex Table QA tasks. This highlights the necessity for
models to develop synergistic reasoning capabilities in order to perform
effectively in complex Table QA tasks.

---


### [AMQA: An Adversarial Dataset for Benchmarking Bias of LLMs in Medicine and Healthcare](http://arxiv.org/abs/2505.19562v1)

Large language models (LLMs) are reaching expert-level accuracy on medical
diagnosis questions, yet their mistakes and the biases behind them pose
life-critical risks. Bias linked to race, sex, and socioeconomic status is
already well known, but a consistent and automatic testbed for measuring it is
missing. To fill this gap, this paper presents AMQA -- an Adversarial Medical
Question-Answering dataset -- built for automated, large-scale bias evaluation
of LLMs in medical QA. AMQA includes 4,806 medical QA pairs sourced from the
United States Medical Licensing Examination (USMLE) dataset, generated using a
multi-agent framework to create diverse adversarial descriptions and question
pairs. Using AMQA, we benchmark five representative LLMs and find surprisingly
substantial disparities: even GPT-4.1, the least biased model tested, answers
privileged-group questions over 10 percentage points more accurately than
unprivileged ones. Compared with the existing benchmark CPV, AMQA reveals 15%
larger accuracy gaps on average between privileged and unprivileged groups. Our
dataset and code are publicly available at https://github.com/XY-Showing/AMQA
to support reproducible research and advance trustworthy, bias-aware medical
AI.

---


### [Minimalist Softmax Attention Provably Learns Constrained Boolean Functions](http://arxiv.org/abs/2505.19531v1)

We study the computational limits of learning $k$-bit Boolean functions
(specifically, $\mathrm{AND}$, $\mathrm{OR}$, and their noisy variants), using
a minimalist single-head softmax-attention mechanism, where $k=\Theta(d)$
relevant bits are selected from $d$ inputs. We show that these simple
$\mathrm{AND}$ and $\mathrm{OR}$ functions are unsolvable with a single-head
softmax-attention mechanism alone. However, with teacher forcing, the same
minimalist attention is capable of solving them. These findings offer two key
insights: Architecturally, solving these Boolean tasks requires only minimalist
attention, without deep Transformer blocks or FFNs. Methodologically, one
gradient descent update with supervision suffices and replaces the multi-step
Chain-of-Thought (CoT) reasoning scheme of [Kim and Suzuki, ICLR 2025] for
solving Boolean problems. Together, the bounds expose a fundamental gap between
what this minimal architecture achieves under ideal supervision and what is
provably impossible under standard training.

---


### [SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback](http://arxiv.org/abs/2505.19514v1)

Prompt quality plays a critical role in the performance of large language
models (LLMs), motivating a growing body of work on prompt optimization. Most
existing methods optimize prompts over a fixed dataset, assuming static input
distributions and offering limited support for iterative improvement. We
introduce SIPDO (Self-Improving Prompts through Data-Augmented Optimization), a
closed-loop framework for prompt learning that integrates synthetic data
generation into the optimization process. SIPDO couples a synthetic data
generator with a prompt optimizer, where the generator produces new examples
that reveal current prompt weaknesses and the optimizer incrementally refines
the prompt in response. This feedback-driven loop enables systematic
improvement of prompt performance without assuming access to external
supervision or new tasks. Experiments across question answering and reasoning
benchmarks show that SIPDO outperforms standard prompt tuning methods,
highlighting the value of integrating data synthesis into prompt learning
workflows.

---


### [Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models](http://arxiv.org/abs/2505.19509v1)

Large Multimodal Models(LMMs) face notable challenges when encountering
multimodal knowledge conflicts, particularly under retrieval-augmented
generation(RAG) frameworks where the contextual information from external
sources may contradict the model's internal parametric knowledge, leading to
unreliable outputs. However, existing benchmarks fail to reflect such realistic
conflict scenarios. Most focus solely on intra-memory conflicts, while
context-memory and inter-context conflicts remain largely investigated.
Furthermore, commonly used factual knowledge-based evaluations are often
overlooked, and existing datasets lack a thorough investigation into conflict
detection capabilities. To bridge this gap, we propose MMKC-Bench, a benchmark
designed to evaluate factual knowledge conflicts in both context-memory and
inter-context scenarios. MMKC-Bench encompasses three types of multimodal
knowledge conflicts and includes 1,573 knowledge instances and 3,381 images
across 23 broad types, collected through automated pipelines with human
verification. We evaluate three representative series of LMMs on both model
behavior analysis and conflict detection tasks. Our findings show that while
current LMMs are capable of recognizing knowledge conflicts, they tend to favor
internal parametric knowledge over external evidence. We hope MMKC-Bench will
foster further research in multimodal knowledge conflict and enhance the
development of multimodal RAG systems. The source code is available at
https://github.com/MLLMKCBENCH/MLLMKC.

---


### [CODE-DITING: A Reasoning-Based Metric for Functional Alignment in Code Evaluation](http://arxiv.org/abs/2505.19502v1)

Trustworthy evaluation methods for code snippets play a crucial role in
neural code generation. Traditional methods, which either rely on reference
solutions or require executable test cases, have inherent limitation in
flexibility and scalability. The recent LLM-as-Judge methodology offers a
promising alternative by directly evaluating functional consistency between the
problem description and the generated code. To systematically understand the
landscape of these LLM-as-Judge methods, we conduct a comprehensive empirical
study across three diverse datasets. Our investigation reveals the pros and
cons of two categories of LLM-as-Judge methods: the methods based on general
foundation models can achieve good performance but require complex prompts and
lack explainability, while the methods based on reasoning foundation models
provide better explainability with simpler prompts but demand substantial
computational resources due to their large parameter sizes. To address these
limitations, we propose CODE-DITING, a novel code evaluation method that
balances accuracy, efficiency and explainability. We develop a data
distillation framework that effectively transfers reasoning capabilities from
DeepSeek-R1671B to our CODE-DITING 1.5B and 7B models, significantly enhancing
evaluation explainability and reducing the computational cost. With the
majority vote strategy in the inference process, CODE-DITING 1.5B outperforms
all models with the same magnitude of parameters and achieves performance which
would normally exhibit in a model with 5 times of parameter scale. CODE-DITING
7B surpasses GPT-4o and DeepSeek-V3 671B, even though it only uses 1% of the
parameter volume of these large models. Further experiments show that
CODEDITING is robust to preference leakage and can serve as a promising
alternative for code evaluation.

---


### [Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](http://arxiv.org/abs/2505.19490v1)

Designing complex computer-aided design (CAD) models is often time-consuming
due to challenges such as computational inefficiency and the difficulty of
generating precise models. We propose a novel language-guided framework for
industrial design automation to address these issues, integrating large
language models (LLMs) with computer-automated design (CAutoD).Through this
framework, CAD models are automatically generated from parameters and
appearance descriptions, supporting the automation of design tasks during the
detailed CAD design phase. Our approach introduces three key innovations: (1) a
semi-automated data annotation pipeline that leverages LLMs and vision-language
large models (VLLMs) to generate high-quality parameters and appearance
descriptions; (2) a Transformer-based CAD generator (TCADGen) that predicts
modeling sequences via dual-channel feature aggregation; (3) an enhanced CAD
modeling generation model, called CADLLM, that is designed to refine the
generated sequences by incorporating the confidence scores from TCADGen.
Experimental results demonstrate that the proposed approach outperforms
traditional methods in both accuracy and efficiency, providing a powerful tool
for automating industrial workflows and generating complex CAD models from
textual prompts. The code is available at
https://jianxliao.github.io/cadllm-page/

---


### [Origin Tracer: A Method for Detecting LoRA Fine-Tuning Origins in LLMs](http://arxiv.org/abs/2505.19466v1)

As large language models (LLMs) continue to advance, their deployment often
involves fine-tuning to enhance performance on specific downstream tasks.
However, this customization is sometimes accompanied by misleading claims about
the origins, raising significant concerns about transparency and trust within
the open-source community. Existing model verification techniques typically
assess functional, representational, and weight similarities. However, these
approaches often struggle against obfuscation techniques, such as permutations
and scaling transformations. To address this limitation, we propose a novel
detection method Origin-Tracer that rigorously determines whether a model has
been fine-tuned from a specified base model. This method includes the ability
to extract the LoRA rank utilized during the fine-tuning process, providing a
more robust verification framework. This framework is the first to provide a
formalized approach specifically aimed at pinpointing the sources of model
fine-tuning. We empirically validated our method on thirty-one diverse
open-source models under conditions that simulate real-world obfuscation
scenarios. We empirically analyze the effectiveness of our framework and
finally, discuss its limitations. The results demonstrate the effectiveness of
our approach and indicate its potential to establish new benchmarks for model
verification.

---


### [BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs](http://arxiv.org/abs/2505.19457v1)

Large language models excel in general tasks, yet assessing their reliability
in logic-heavy, precision-critical domains like finance, law, and healthcare
remains challenging. To address this, we introduce BizFinBench, the first
benchmark specifically designed to evaluate LLMs in real-world financial
applications. BizFinBench consists of 6,781 well-annotated queries in Chinese,
spanning five dimensions: numerical calculation, reasoning, information
extraction, prediction recognition, and knowledge-based question answering,
grouped into nine fine-grained categories. The benchmark includes both
objective and subjective metrics. We also introduce IteraJudge, a novel LLM
evaluation method that reduces bias when LLMs serve as evaluators in objective
metrics. We benchmark 25 models, including both proprietary and open-source
systems. Extensive experiments show that no model dominates across all tasks.
Our evaluation reveals distinct capability patterns: (1) In Numerical
Calculation, Claude-3.5-Sonnet (63.18) and DeepSeek-R1 (64.04) lead, while
smaller models like Qwen2.5-VL-3B (15.92) lag significantly; (2) In Reasoning,
proprietary models dominate (ChatGPT-o3: 83.58, Gemini-2.0-Flash: 81.15), with
open-source models trailing by up to 19.49 points; (3) In Information
Extraction, the performance spread is the largest, with DeepSeek-R1 scoring
71.46, while Qwen3-1.7B scores 11.23; (4) In Prediction Recognition,
performance variance is minimal, with top models scoring between 39.16 and
50.00. We find that while current LLMs handle routine finance queries
competently, they struggle with complex scenarios requiring cross-concept
reasoning. BizFinBench offers a rigorous, business-aligned benchmark for future
research. The code and dataset are available at
https://github.com/HiThink-Research/BizFinBench.

---


### [Style2Code: A Style-Controllable Code Generation Framework with Dual-Modal Contrastive Representation Learning](http://arxiv.org/abs/2505.19442v1)

Controllable code generation, the ability to synthesize code that follows a
specified style while maintaining functionality, remains a challenging task. We
propose a two-stage training framework combining contrastive learning and
conditional decoding to enable flexible style control. The first stage aligns
code style representations with semantic and structural features. In the second
stage, we fine-tune a language model (e.g., Flan-T5) conditioned on the learned
style vector to guide generation. Our method supports style interpolation and
user personalization via lightweight mixing. Compared to prior work, our
unified framework offers improved stylistic control without sacrificing code
correctness. This is among the first approaches to combine contrastive
alignment with conditional decoding for style-guided code generation.

---


### [Deriving Strategic Market Insights with Large Language Models: A Benchmark for Forward Counterfactual Generation](http://arxiv.org/abs/2505.19430v1)

Counterfactual reasoning typically involves considering alternatives to
actual events. While often applied to understand past events, a distinct
form-forward counterfactual reasoning-focuses on anticipating plausible future
developments. This type of reasoning is invaluable in dynamic financial
markets, where anticipating market developments can powerfully unveil potential
risks and opportunities for stakeholders, guiding their decision-making.
However, performing this at scale is challenging due to the cognitive demands
involved, underscoring the need for automated solutions. Large Language Models
(LLMs) offer promise, but remain unexplored for this application. To address
this gap, we introduce a novel benchmark, Fin-Force-FINancial FORward
Counterfactual Evaluation. By curating financial news headlines and providing
structured evaluation, Fin-Force supports LLM based forward counterfactual
generation. This paves the way for scalable and automated solutions for
exploring and anticipating future market developments, thereby providing
structured insights for decision-making. Through experiments on Fin-Force, we
evaluate state-of-the-art LLMs and counterfactual generation methods, analyzing
their limitations and proposing insights for future research.

---


### [The Role of Diversity in In-Context Learning for Large Language Models](http://arxiv.org/abs/2505.19426v1)

In-context learning (ICL) is a crucial capability of current large language
models (LLMs), where the selection of examples plays a key role in performance.
While most existing approaches focus on selecting the most similar examples to
the query, the impact of diversity in example selection remains underexplored.
We systematically investigate the role of diversity in in-context example
selection through experiments across a range of tasks, from sentiment
classification to more challenging math and code problems. Experiments on
Llama-3.1, Gemma-2, and Mistral-v0.3 families of models show that
diversity-aware selection methods improve performance, particularly on complex
tasks like math and code, and enhance robustness to out-of-distribution
queries. To support these findings, we introduce a theoretical framework that
explains the benefits of incorporating diversity in in-context example
selection.

---


### [Surrogate-Assisted Evolutionary Reinforcement Learning Based on Autoencoder and Hyperbolic Neural Network](http://arxiv.org/abs/2505.19423v1)

Evolutionary Reinforcement Learning (ERL), training the Reinforcement
Learning (RL) policies with Evolutionary Algorithms (EAs), have demonstrated
enhanced exploration capabilities and greater robustness than using traditional
policy gradient. However, ERL suffers from the high computational costs and low
search efficiency, as EAs require evaluating numerous candidate policies with
expensive simulations, many of which are ineffective and do not contribute
meaningfully to the training. One intuitive way to reduce the ineffective
evaluations is to adopt the surrogates. Unfortunately, existing ERL policies
are often modeled as deep neural networks (DNNs) and thus naturally represented
as high-dimensional vectors containing millions of weights, which makes the
building of effective surrogates for ERL policies extremely challenging. This
paper proposes a novel surrogate-assisted ERL that integrates Autoencoders (AE)
and Hyperbolic Neural Networks (HNN). Specifically, AE compresses
high-dimensional policies into low-dimensional representations while extracting
key features as the inputs for the surrogate. HNN, functioning as a
classification-based surrogate model, can learn complex nonlinear relationships
from sampled data and enable more accurate pre-selection of the sampled
policies without real evaluations. The experiments on 10 Atari and 4 Mujoco
games have verified that the proposed method outperforms previous approaches
significantly. The search trajectories guided by AE and HNN are also visually
demonstrated to be more effective, in terms of both exploration and
convergence. This paper not only presents the first learnable policy embedding
and surrogate-modeling modules for high-dimensional ERL policies, but also
empirically reveals when and why they can be successful.

---


### [Unveiling the Compositional Ability Gap in Vision-Language Reasoning Model](http://arxiv.org/abs/2505.19406v1)

While large language models (LLMs) demonstrate strong reasoning capabilities
utilizing reinforcement learning (RL) with verifiable reward, whether large
vision-language models (VLMs) can directly inherit such capabilities through
similar post-training strategies remains underexplored. In this work, we
conduct a systematic compositional probing study to evaluate whether current
VLMs trained with RL or other post-training strategies can compose capabilities
across modalities or tasks under out-of-distribution conditions. We design a
suite of diagnostic tasks that train models on unimodal tasks or isolated
reasoning skills, and evaluate them on multimodal, compositional variants
requiring skill integration. Through comparisons between supervised fine-tuning
(SFT) and RL-trained models, we identify three key findings: (1) RL-trained
models consistently outperform SFT on compositional generalization,
demonstrating better integration of learned skills; (2) although VLMs achieve
strong performance on individual tasks, they struggle to generalize
compositionally under cross-modal and cross-task scenario, revealing a
significant gap in current training strategies; (3) enforcing models to
explicitly describe visual content before reasoning (e.g.,
caption-before-thinking), along with rewarding progressive vision-to-text
grounding, yields notable gains. It highlights two essential ingredients for
improving compositionality in VLMs: visual-to-text alignment and accurate
visual grounding. Our findings shed light on the current limitations of
RL-based reasoning VLM training and provide actionable insights toward building
models that reason compositionally across modalities and tasks.

---


### [VADER: A Human-Evaluated Benchmark for Vulnerability Assessment, Detection, Explanation, and Remediation](http://arxiv.org/abs/2505.19395v1)

Ensuring that large language models (LLMs) can effectively assess, detect,
explain, and remediate software vulnerabilities is critical for building robust
and secure software systems. We introduce VADER, a human-evaluated benchmark
designed explicitly to assess LLM performance across four key
vulnerability-handling dimensions: assessment, detection, explanation, and
remediation. VADER comprises 174 real-world software vulnerabilities, each
carefully curated from GitHub repositories and annotated by security experts.
For each vulnerability case, models are tasked with identifying the flaw,
classifying it using Common Weakness Enumeration (CWE), explaining its
underlying cause, proposing a patch, and formulating a test plan. Using a
one-shot prompting strategy, we benchmark six state-of-the-art LLMs (Claude 3.7
Sonnet, Gemini 2.5 Pro, GPT-4.1, GPT-4.5, Grok 3 Beta, and o3) on VADER, and
human security experts evaluated each response according to a rigorous scoring
rubric emphasizing remediation (quality of the code fix, 50%), explanation
(20%), and classification and test plan (30%) according to a standardized
rubric. Our results show that current state-of-the-art LLMs achieve only
moderate success on VADER - OpenAI's o3 attained 54.7% accuracy overall, with
others in the 49-54% range, indicating ample room for improvement. Notably,
remediation quality is strongly correlated (Pearson r > 0.97) with accurate
classification and test plans, suggesting that models that effectively
categorize vulnerabilities also tend to fix them well. VADER's comprehensive
dataset, detailed evaluation rubrics, scoring tools, and visualized results
with confidence intervals are publicly released, providing the community with
an interpretable, reproducible benchmark to advance vulnerability-aware LLMs.
All code and data are available at: https://github.com/AfterQuery/vader

---


### [CaseEdit: Enhancing Localized Commonsense Reasoning via Null-Space Constrained Knowledge Editing in Small Parameter Language Models](http://arxiv.org/abs/2505.19383v1)

Large language models (LLMs) exhibit strong performance on factual recall and
general reasoning but struggle to adapt to user-specific, commonsense
knowledge, a challenge particularly acute in small-parameter settings where
computational efficiency is prioritized. We introduce CaseEdit, a new dataset
and generation pipeline for evaluating localized, personalized commonsense
knowledge editing in small LLMs to address this. Built upon the ATOMIC20/20
commonsense graph, CaseEdit uses a multi-stage inference process to generate
both typical and atypical contextual edits for household objects, paired with
targeted evaluation questions across four axes: reliability, generalization,
locality, and portability. We evaluate established knowledge editing methods
using CaseEdit and demonstrate that AlphaEdit, a technique employing null-space
projection to minimize interference with unrelated knowledge, consistently
outperforms other methods when applied to an LLaMA 3.2 3B model, even in
scalability tests, showing minimal ripple effects. Our results indicate that
using CaseEdit with effective editing techniques like AlphaEdit allows small
models to internalize high-quality, context-sensitive common-sense knowledge,
paving the way for lightweight, personalized assistants.

---


### [Adaptive Deep Reasoning: Triggering Deep Thinking When Needed](http://arxiv.org/abs/2505.20101v1)

Large language models (LLMs) have shown impressive capabilities in handling
complex tasks through long-chain reasoning. However, the extensive reasoning
steps involved can significantly increase computational costs, posing
challenges for real-world deployment. Recent efforts have focused on optimizing
reasoning efficiency by shortening the Chain-of-Thought (CoT) reasoning
processes through various approaches, such as length-aware prompt engineering,
supervised fine-tuning on CoT data with variable lengths, and reinforcement
learning with length penalties. Although these methods effectively reduce
reasoning length, they still necessitate an initial reasoning phase. More
recent approaches have attempted to integrate long-chain and short-chain
reasoning abilities into a single model, yet they still rely on manual control
to toggle between short and long CoT.In this work, we propose a novel approach
that autonomously switches between short and long reasoning chains based on
problem complexity. Our method begins with supervised fine-tuning of the base
model to equip both long-chain and short-chain reasoning abilities. We then
employ reinforcement learning to further balance short and long CoT generation
while maintaining accuracy through two key strategies: first, integrating
reinforcement learning with a long-short adaptive group-wise reward strategy to
assess prompt complexity and provide corresponding rewards; second,
implementing a logit-based reasoning mode switching loss to optimize the
model's initial token choice, thereby guiding the selection of the reasoning
type.Evaluations on mathematical datasets demonstrate that our model can
dynamically switch between long-chain and short-chain reasoning modes without
substantially sacrificing performance. This advancement enhances the
practicality of reasoning in large language models for real-world applications.

---


### [REARANK: Reasoning Re-ranking Agent via Reinforcement Learning](http://arxiv.org/abs/2505.20046v1)

We present REARANK, a large language model (LLM)-based listwise reasoning
reranking agent. REARANK explicitly reasons before reranking, significantly
improving both performance and interpretability. Leveraging reinforcement
learning and data augmentation, REARANK achieves substantial improvements over
baseline models across popular information retrieval benchmarks, notably
requiring only 179 annotated samples. Built on top of Qwen2.5-7B, our
REARANK-7B demonstrates performance comparable to GPT-4 on both in-domain and
out-of-domain benchmarks and even surpasses GPT-4 on reasoning-intensive BRIGHT
benchmarks. These results underscore the effectiveness of our approach and
highlight how reinforcement learning can enhance LLM reasoning capabilities in
reranking.

---


### [Training LLM-Based Agents with Synthetic Self-Reflected Trajectories and Partial Masking](http://arxiv.org/abs/2505.20023v1)

Autonomous agents, which perceive environments and take actions to achieve
goals, have become increasingly feasible with the advancements in large
language models (LLMs). However, current powerful agents often depend on
sophisticated prompt engineering combined with closed-source LLMs like GPT-4.
Although training open-source LLMs using expert trajectories from teacher
models has yielded some improvements in agent capabilities, this approach still
faces limitations such as performance plateauing and error propagation. To
mitigate these challenges, we propose STeP, a novel method for improving
LLM-based agent training. We synthesize self-reflected trajectories that
include reflections and corrections of error steps, which enhance the
effectiveness of LLM agents in learning from teacher models, enabling them to
become agents capable of self-reflecting and correcting. We also introduce
partial masking strategy that prevents the LLM from internalizing incorrect or
suboptimal steps. Experiments demonstrate that our method improves agent
performance across three representative tasks: ALFWorld, WebShop, and SciWorld.
For the open-source model LLaMA2-7B-Chat, when trained using self-reflected
trajectories constructed with Qwen1.5-110B-Chat as the teacher model, it
achieves comprehensive improvements with less training data compared to agents
trained exclusively on expert trajectories.

---


### [TTPA: Token-level Tool-use Preference Alignment Training Framework with Fine-grained Evaluation](http://arxiv.org/abs/2505.20016v1)

Existing tool-learning methods usually rely on supervised fine-tuning, they
often overlook fine-grained optimization of internal tool call details, leading
to limitations in preference alignment and error discrimination. To overcome
these challenges, we propose Token-level Tool-use Preference Alignment Training
Framework (TTPA), a training paradigm for constructing token-level tool-use
preference datasets that align LLMs with fine-grained preferences using a novel
error-oriented scoring mechanism. TTPA first introduces reversed dataset
construction, a method for creating high-quality, multi-turn tool-use datasets
by reversing the generation flow. Additionally, we propose Token-level
Preference Sampling (TPS) to capture fine-grained preferences by modeling
token-level differences during generation. To address biases in scoring, we
introduce the Error-oriented Scoring Mechanism (ESM), which quantifies
tool-call errors and can be used as a training signal. Extensive experiments on
three diverse benchmark datasets demonstrate that TTPA significantly improves
tool-using performance while showing strong generalization ability across
models and datasets.

---


### [Does Rationale Quality Matter? Enhancing Mental Disorder Detection via Selective Reasoning Distillation](http://arxiv.org/abs/2505.20014v1)

The detection of mental health problems from social media and the
interpretation of these results have been extensively explored. Research has
shown that incorporating clinical symptom information into a model enhances
domain expertise, improving its detection and interpretation performance. While
large language models (LLMs) are shown to be effective for generating
explanatory rationales in mental health detection, their substantially large
parameter size and high computational cost limit their practicality. Reasoning
distillation transfers this ability to smaller language models (SLMs), but
inconsistencies in the relevance and domain alignment of LLM-generated
rationales pose a challenge. This paper investigates how rationale quality
impacts SLM performance in mental health detection and explanation generation.
We hypothesize that ensuring high-quality and domain-relevant rationales
enhances the distillation. To this end, we propose a framework that selects
rationales based on their alignment with expert clinical reasoning. Experiments
show that our quality-focused approach significantly enhances SLM performance
in both mental disorder detection and rationale generation. This work
highlights the importance of rationale quality and offers an insightful
framework for knowledge transfer in mental health applications.

---


### [Mixture of LoRA Experts for Low-Resourced Multi-Accent Automatic Speech Recognition](http://arxiv.org/abs/2505.20006v1)

We aim to improve the robustness of Automatic Speech Recognition (ASR)
systems against non-native speech, particularly in low-resourced multi-accent
settings. We introduce Mixture of Accent-Specific LoRAs (MAS-LoRA), a
fine-tuning method that leverages a mixture of Low-Rank Adaptation (LoRA)
experts, each specialized in a specific accent. This method can be used when
the accent is known or unknown at inference time, without the need to fine-tune
the model again. Our experiments, conducted using Whisper on the L2-ARCTIC
corpus, demonstrate significant improvements in Word Error Rate compared to
regular LoRA and full fine-tuning when the accent is unknown. When the accent
is known, the results further improve. Furthermore, MAS-LoRA shows less
catastrophic forgetting than the other fine-tuning methods. To the best of our
knowledge, this is the first use of a mixture of LoRA experts for non-native
multi-accent ASR.

---


### [How Well Do Large Reasoning Models Translate? A Comprehensive Evaluation for Multi-Domain Machine Translation](http://arxiv.org/abs/2505.19987v1)

Large language models (LLMs) have demonstrated strong performance in
general-purpose machine translation, but their effectiveness in complex,
domain-sensitive translation tasks remains underexplored. Recent advancements
in Large Reasoning Models (LRMs), raise the question of whether structured
reasoning can enhance translation quality across diverse domains. In this work,
we compare the performance of LRMs with traditional LLMs across 15
representative domains and four translation directions. Our evaluation
considers various factors, including task difficulty, input length, and
terminology density. We use a combination of automatic metrics and an enhanced
MQM-based evaluation hierarchy to assess translation quality. Our findings show
that LRMs consistently outperform traditional LLMs in semantically complex
domains, especially in long-text and high-difficulty translation scenarios.
Moreover, domain-adaptive prompting strategies further improve performance by
better leveraging the reasoning capabilities of LRMs. These results highlight
the potential of structured reasoning in MDMT tasks and provide valuable
insights for optimizing translation systems in domain-sensitive contexts.

---


### [Conversational Lexicography: Querying Lexicographic Data on Knowledge Graphs with SPARQL through Natural Language](http://arxiv.org/abs/2505.19971v1)

Knowledge graphs offer an excellent solution for representing the
lexical-semantic structures of lexicographic data. However, working with the
SPARQL query language represents a considerable hurdle for many non-expert
users who could benefit from the advantages of this technology. This paper
addresses the challenge of creating natural language interfaces for
lexicographic data retrieval on knowledge graphs such as Wikidata. We develop a
multidimensional taxonomy capturing the complexity of Wikidata's lexicographic
data ontology module through four dimensions and create a template-based
dataset with over 1.2 million mappings from natural language utterances to
SPARQL queries. Our experiments with GPT-2 (124M), Phi-1.5 (1.3B), and
GPT-3.5-Turbo reveal significant differences in model capabilities. While all
models perform well on familiar patterns, only GPT-3.5-Turbo demonstrates
meaningful generalization capabilities, suggesting that model size and diverse
pre-training are crucial for adaptability in this domain. However, significant
challenges remain in achieving robust generalization, handling diverse
linguistic data, and developing scalable solutions that can accommodate the
full complexity of lexicographic knowledge representation.

---


### [CP-Router: An Uncertainty-Aware Router Between LLM and LRM](http://arxiv.org/abs/2505.19970v1)

Recent advances in Large Reasoning Models (LRMs) have significantly improved
long-chain reasoning capabilities over Large Language Models (LLMs). However,
LRMs often produce unnecessarily lengthy outputs even for simple queries,
leading to inefficiencies or even accuracy degradation compared to LLMs. To
overcome this, we propose CP-Router, a training-free and model-agnostic routing
framework that dynamically selects between an LLM and an LRM, demonstrated with
multiple-choice question answering (MCQA) prompts. The routing decision is
guided by the prediction uncertainty estimates derived via Conformal Prediction
(CP), which provides rigorous coverage guarantees. To further refine the
uncertainty differentiation across inputs, we introduce Full and Binary Entropy
(FBE), a novel entropy-based criterion that adaptively selects the appropriate
CP threshold. Experiments across diverse MCQA benchmarks, including
mathematics, logical reasoning, and Chinese chemistry, demonstrate that
CP-Router efficiently reduces token usage while maintaining or even improving
accuracy compared to using LRM alone. We also extend CP-Router to diverse model
pairings and open-ended QA, where it continues to demonstrate strong
performance, validating its generality and robustness.

---


### [ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs](http://arxiv.org/abs/2505.19937v1)

Large Language Models (LLMs) are widely used in Spoken Language Understanding
(SLU). Recent SLU models process audio directly by adapting speech input into
LLMs for better multimodal learning. A key consideration for these models is
the cross-modal alignment between text and audio modalities, which is a
telltale sign as to whether or not LLM is able to associate semantic meaning to
audio segments. While various methods exist for fusing these modalities, there
is no standard metric to evaluate alignment quality in LLMs. In this work, we
propose a new metric, ALAS (Automatic Latent Alignment Score). Our study
examines the correlation between audio and text representations across
transformer layers, for two different tasks (Spoken Question Answering and
Emotion Recognition). We showcase that our metric behaves as expected across
different layers and different tasks.

---


### [REA-RL: Reflection-Aware Online Reinforcement Learning for Efficient Large Reasoning Models](http://arxiv.org/abs/2505.19862v1)

Large Reasoning Models (LRMs) demonstrate strong performance in complex tasks
but often face the challenge of overthinking, leading to substantially high
inference costs. Existing approaches synthesize shorter reasoning responses for
LRMs to learn, but are inefficient for online usage due to the time-consuming
data generation and filtering processes. Meanwhile, online reinforcement
learning mainly adopts a length reward to encourage short reasoning responses,
but tends to lose the reflection ability and harm the performance. To address
these issues, we propose REA-RL, which introduces a small reflection model for
efficient scaling in online training, offering both parallel sampling and
sequential revision. Besides, a reflection reward is designed to further
prevent LRMs from favoring short yet non-reflective responses. Experiments show
that both methods maintain or enhance performance while significantly improving
inference efficiency. Their combination achieves a good balance between
performance and efficiency, reducing inference costs by 35% without
compromising performance. Further analysis demonstrates that our methods are
effective by maintaining reflection frequency for hard problems while
appropriately reducing it for simpler ones without losing reflection ability.
Codes are available at https://github.com/hexuandeng/REA-RL.

---


### [Improving Multilingual Math Reasoning for African Languages](http://arxiv.org/abs/2505.19848v1)

Researchers working on low-resource languages face persistent challenges due
to limited data availability and restricted access to computational resources.
Although most large language models (LLMs) are predominantly trained in
high-resource languages, adapting them to low-resource contexts, particularly
African languages, requires specialized techniques. Several strategies have
emerged for adapting models to low-resource languages in todays LLM landscape,
defined by multi-stage pre-training and post-training paradigms. However, the
most effective approaches remain uncertain. This work systematically
investigates which adaptation strategies yield the best performance when
extending existing LLMs to African languages. We conduct extensive experiments
and ablation studies to evaluate different combinations of data types
(translated versus synthetically generated), training stages (pre-training
versus post-training), and other model adaptation configurations. Our
experiments focuses on mathematical reasoning tasks, using the Llama 3.1 model
family as our base model.

---


### [The Avengers: A Simple Recipe for Uniting Smaller Language Models to Challenge Proprietary Giants](http://arxiv.org/abs/2505.19797v1)

As proprietary giants increasingly dominate the race for ever-larger language
models, a pressing question arises for the open-source community: can smaller
models remain competitive across a broad range of tasks? In this paper, we
present the Avengers--a simple recipe that effectively leverages the collective
intelligence of open-source, smaller language models. Our framework is built
upon four lightweight operations: (i) embedding: encode queries using a text
embedding model; (ii) clustering: group queries based on their semantic
similarity; (iii) scoring: scores each model's performance within each cluster;
and (iv) voting: improve outputs via repeated sampling and voting. At inference
time, each query is embedded and assigned to its nearest cluster. The
top-performing model(s) within that cluster are selected to generate the
response using the Self-Consistency or its multi-model variant. Remarkably,
with 10 open-source models (~7B parameters each), the Avengers collectively
outperforms GPT-4.1 on 10 out of 15 datasets (spanning mathematics, code,
logic, knowledge, and affective tasks). In particular, it surpasses GPT-4.1 on
mathematics tasks by 18.21% and on code tasks by 7.46%. Furthermore, the
Avengers delivers superior out-of-distribution generalization, and remains
robust across various embedding models, clustering algorithms, ensemble
strategies, and values of its sole parameter--the number of clusters. We have
open-sourced the code on GitHub: https://github.com/ZhangYiqun018/Avengers

---


### [Understanding the Performance Gap in Preference Learning: A Dichotomy of RLHF and DPO](http://arxiv.org/abs/2505.19770v1)

We present a fine-grained theoretical analysis of the performance gap between
reinforcement learning from human feedback (RLHF) and direct preference
optimization (DPO) under a representation gap. Our study decomposes this gap
into two sources: an explicit representation gap under exact optimization and
an implicit representation gap under finite samples. In the exact optimization
setting, we characterize how the relative capacities of the reward and policy
model classes influence the final policy qualities. We show that RLHF, DPO, or
online DPO can outperform one another depending on the type of model
mis-specifications. Notably, online DPO can outperform both RLHF and standard
DPO when the reward and policy model classes are isomorphic and both
mis-specified. In the approximate optimization setting, we provide a concrete
construction where the ground-truth reward is implicitly sparse and show that
RLHF requires significantly fewer samples than DPO to recover an effective
reward model -- highlighting a statistical advantage of two-stage learning.
Together, these results provide a comprehensive understanding of the
performance gap between RLHF and DPO under various settings, and offer
practical insights into when each method is preferred.

---


### [Token-level Accept or Reject: A Micro Alignment Approach for Large Language Models](http://arxiv.org/abs/2505.19743v1)

With the rapid development of Large Language Models (LLMs), aligning these
models with human preferences and values is critical to ensuring ethical and
safe applications. However, existing alignment techniques such as RLHF or DPO
often require direct fine-tuning on LLMs with billions of parameters, resulting
in substantial computational costs and inefficiencies. To address this, we
propose Micro token-level Accept-Reject Aligning (MARA) approach designed to
operate independently of the language models. MARA simplifies the alignment
process by decomposing sentence-level preference learning into token-level
binary classification, where a compact three-layer fully-connected network
determines whether candidate tokens are "Accepted" or "Rejected" as part of the
response. Extensive experiments across seven different LLMs and three
open-source datasets show that MARA achieves significant improvements in
alignment performance while reducing computational costs.

---


### [Interleaved Reasoning for Large Language Models via Reinforcement Learning](http://arxiv.org/abs/2505.19640v1)

Long chain-of-thought (CoT) significantly enhances large language models'
(LLM) reasoning capabilities. However, the extensive reasoning traces lead to
inefficiencies and an increased time-to-first-token (TTFT). We propose a novel
training paradigm that uses reinforcement learning (RL) to guide reasoning LLMs
to interleave thinking and answering for multi-hop questions. We observe that
models inherently possess the ability to perform interleaved reasoning, which
can be further enhanced through RL. We introduce a simple yet effective
rule-based reward to incentivize correct intermediate steps, which guides the
policy model toward correct reasoning paths by leveraging intermediate signals
generated during interleaved reasoning. Extensive experiments conducted across
five diverse datasets and three RL algorithms (PPO, GRPO, and REINFORCE++)
demonstrate consistent improvements over traditional think-answer reasoning,
without requiring external tools. Specifically, our approach reduces TTFT by
over 80% on average and improves up to 19.3% in Pass@1 accuracy. Furthermore,
our method, trained solely on question answering and logical reasoning
datasets, exhibits strong generalization ability to complex reasoning datasets
such as MATH, GPQA, and MMLU. Additionally, we conduct in-depth analysis to
reveal several valuable insights into conditional reward modeling.

---


### [DoctorAgent-RL: A Multi-Agent Collaborative Reinforcement Learning System for Multi-Turn Clinical Dialogue](http://arxiv.org/abs/2505.19630v1)

Large language models (LLMs) have demonstrated excellent capabilities in the
field of biomedical question answering, but their application in real-world
clinical consultations still faces core challenges. Existing systems rely on a
one-way information transmission mode where patients must fully describe their
symptoms in a single round, leading to nonspecific diagnostic recommendations
when complaints are vague. Traditional multi-turn dialogue methods based on
supervised learning are constrained by static data-driven paradigms, lacking
generalizability and struggling to intelligently extract key clinical
information. To address these limitations, we propose DoctorAgent-RL, a
reinforcement learning (RL)-based multi-agent collaborative framework that
models medical consultations as a dynamic decision-making process under
uncertainty. The doctor agent continuously optimizes its questioning strategy
within the RL framework through multi-turn interactions with the patient agent,
dynamically adjusting its information-gathering path based on comprehensive
rewards from the Consultation Evaluator. This RL fine-tuning mechanism enables
LLMs to autonomously develop interaction strategies aligned with clinical
reasoning logic, rather than superficially imitating patterns in existing
dialogue data. Notably, we constructed MTMedDialog, the first English
multi-turn medical consultation dataset capable of simulating patient
interactions. Experiments demonstrate that DoctorAgent-RL outperforms existing
models in both multi-turn reasoning capability and final diagnostic
performance, demonstrating practical value in assisting clinical consultations.
https://github.com/JarvisUSTC/DoctorAgent-RL

---


### [HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices](http://arxiv.org/abs/2505.19628v1)

Large language models (LLMs) have the potential to revolutionize smart home
assistants by enhancing their ability to accurately understand user needs and
respond appropriately, which is extremely beneficial for building a smarter
home environment. While recent studies have explored integrating LLMs into
smart home systems, they primarily focus on handling straightforward, valid
single-device operation instructions. However, real-world scenarios are far
more complex and often involve users issuing invalid instructions or
controlling multiple devices simultaneously. These have two main challenges:
LLMs must accurately identify and rectify errors in user instructions and
execute multiple user instructions perfectly. To address these challenges and
advance the development of LLM-based smart home assistants, we introduce
HomeBench, the first smart home dataset with valid and invalid instructions
across single and multiple devices in this paper. We have experimental results
on 13 distinct LLMs; e.g., GPT-4o achieves only a 0.0% success rate in the
scenario of invalid multi-device instructions, revealing that the existing
state-of-the-art LLMs still cannot perform well in this situation even with the
help of in-context learning, retrieval-augmented generation, and fine-tuning.
Our code and dataset are publicly available at
https://github.com/BITHLP/HomeBench.

---


### [Evaluating Robustness of Large Audio Language Models to Audio Injection: An Empirical Study](http://arxiv.org/abs/2505.19598v1)

Large Audio-Language Models (LALMs) are increasingly deployed in real-world
applications, yet their robustness against malicious audio injection attacks
remains underexplored. This study systematically evaluates five leading LALMs
across four attack scenarios: Audio Interference Attack, Instruction Following
Attack, Context Injection Attack, and Judgment Hijacking Attack. Using metrics
like Defense Success Rate, Context Robustness Score, and Judgment Robustness
Index, their vulnerabilities and resilience were quantitatively assessed.
Experimental results reveal significant performance disparities among models;
no single model consistently outperforms others across all attack types. The
position of malicious content critically influences attack effectiveness,
particularly when placed at the beginning of sequences. A negative correlation
between instruction-following capability and robustness suggests models
adhering strictly to instructions may be more susceptible, contrasting with
greater resistance by safety-aligned models. Additionally, system prompts show
mixed effectiveness, indicating the need for tailored strategies. This work
introduces a benchmark framework and highlights the importance of integrating
robustness into training pipelines. Findings emphasize developing multi-modal
defenses and architectural designs that decouple capability from susceptibility
for secure LALMs deployment.

---


### [Learning to Reason without External Rewards](http://arxiv.org/abs/2505.19590v1)

Training large language models (LLMs) for complex reasoning via Reinforcement
Learning with Verifiable Rewards (RLVR) is effective but limited by reliance on
costly, domain-specific supervision. We explore Reinforcement Learning from
Internal Feedback (RLIF), a framework that enables LLMs to learn from intrinsic
signals without external rewards or labeled data. We propose Intuitor, an RLIF
method that uses a model's own confidence, termed self-certainty, as its sole
reward signal. Intuitor replaces external rewards in Group Relative Policy
Optimization (GRPO) with self-certainty scores, enabling fully unsupervised
learning. Experiments demonstrate that Intuitor matches GRPO's performance on
mathematical benchmarks while achieving superior generalization to
out-of-domain tasks like code generation, without requiring gold solutions or
test cases. Our findings show that intrinsic model signals can drive effective
learning across domains, offering a scalable alternative to RLVR for autonomous
AI systems where verifiable rewards are unavailable. Code is available at
https://github.com/sunblaze-ucb/Intuitor

---


### [TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization](http://arxiv.org/abs/2505.19586v1)

The Key-Value (KV) cache in generative large language models (LLMs)
introduces substantial memory overhead. Existing works mitigate this burden by
offloading or compressing the KV cache. However, loading the entire cache
incurs significant latency due to PCIe bandwidth bottlenecks in CPU-GPU
communication, while aggressive compression causes notable performance
degradation. We identify that certain layers in the LLM need to maintain global
information and are unsuitable for selective loading. In contrast, other layers
primarily focus on a few tokens with dominant activations that potentially
incur substantial quantization error. This observation leads to a key insight
that loading dominant tokens and quantizing all tokens can complement each
other. Building on this insight, we propose a hybrid compression method,
TailorKV, which seamlessly integrates quantization and offloading. TailorKV
develops an inference framework along with a hardware-friendly implementation
that leverages these complementary characteristics. Extensive long-context
evaluations exhibit that TailorKV achieves nearly lossless performance under
aggressive compression settings, outperforming the state-of-the-art.
Particularly, the Llama-3.1-8B with 128k context can be served within a single
RTX 3090 GPU, reaching 82 ms per token during decoding.

---


### [Bias in Political Dialogue: Tagging U.S. Presidential Debates with an Extended DAMSL Framework](http://arxiv.org/abs/2505.19515v1)

We present a critical discourse analysis of the 2024 U.S. presidential
debates, examining Donald Trump's rhetorical strategies in his interactions
with Joe Biden and Kamala Harris. We introduce a novel annotation framework,
BEADS (Bias Enriched Annotation for Dialogue Structure), which systematically
extends the DAMSL framework to capture bias driven and adversarial discourse
features in political communication. BEADS includes a domain and language
agnostic set of tags that model ideological framing, emotional appeals, and
confrontational tactics. Our methodology compares detailed human annotation
with zero shot ChatGPT assisted tagging on verified transcripts from the Trump
and Biden (19,219 words) and Trump and Harris (18,123 words) debates. Our
analysis shows that Trump consistently dominated in key categories: Challenge
and Adversarial Exchanges, Selective Emphasis, Appeal to Fear, Political Bias,
and Perceived Dismissiveness. These findings underscore his use of emotionally
charged and adversarial rhetoric to control the narrative and influence
audience perception. In this work, we establish BEADS as a scalable and
reproducible framework for critical discourse analysis across languages,
domains, and political contexts.

---


### [Anveshana: A New Benchmark Dataset for Cross-Lingual Information Retrieval On English Queries and Sanskrit Documents](http://arxiv.org/abs/2505.19494v1)

The study presents a comprehensive benchmark for retrieving Sanskrit
documents using English queries, focusing on the chapters of the
Srimadbhagavatam. It employs a tripartite approach: Direct Retrieval (DR),
Translation-based Retrieval (DT), and Query Translation (QT), utilizing shared
embedding spaces and advanced translation methods to enhance retrieval systems
in a RAG framework. The study fine-tunes state-of-the-art models for Sanskrit's
linguistic nuances, evaluating models such as BM25, REPLUG, mDPR, ColBERT,
Contriever, and GPT-2. It adapts summarization techniques for Sanskrit
documents to improve QA processing. Evaluation shows DT methods outperform DR
and QT in handling the cross-lingual challenges of ancient texts, improving
accessibility and understanding. A dataset of 3,400 English-Sanskrit
query-document pairs underpins the study, aiming to preserve Sanskrit
scriptures and share their philosophical importance widely. Our dataset is
publicly available at https://huggingface.co/datasets/manojbalaji1/anveshana

---


### [Continuous Self-Improvement of Large Language Models by Test-time Training with Verifier-Driven Sample Selection](http://arxiv.org/abs/2505.19475v1)

Learning to adapt pretrained language models to unlabeled,
out-of-distribution data is a critical challenge, as models often falter on
structurally novel reasoning tasks even while excelling within their training
distribution. We introduce a new framework called VDS-TTT - Verifier-Driven
Sample Selection for Test-Time Training to efficiently address this. We use a
learned verifier to score a pool of generated responses and select only from
high ranking pseudo-labeled examples for fine-tuned adaptation. Specifically,
for each input query our LLM generates N candidate answers; the verifier
assigns a reliability score to each, and the response with the highest
confidence and above a fixed threshold is paired with its query for test-time
training. We fine-tune only low-rank LoRA adapter parameters, ensuring
adaptation efficiency and fast convergence. Our proposed self-supervised
framework is the first to synthesize verifier driven test-time training data
for continuous self-improvement of the model. Experiments across three diverse
benchmarks and three state-of-the-art LLMs demonstrate that VDS-TTT yields up
to a 32.29% relative improvement over the base model and a 6.66% gain compared
to verifier-based methods without test-time training, highlighting its
effectiveness and efficiency for on-the-fly large language model adaptation.

---


### [Surrogate Signals from Format and Length: Reinforcement Learning for Solving Mathematical Problems without Ground Truth Answers](http://arxiv.org/abs/2505.19439v1)

Large Language Models have achieved remarkable success in natural language
processing tasks, with Reinforcement Learning playing a key role in adapting
them to specific applications. However, obtaining ground truth answers for
training LLMs in mathematical problem-solving is often challenging, costly, and
sometimes unfeasible. This research delves into the utilization of format and
length as surrogate signals to train LLMs for mathematical problem-solving,
bypassing the need for traditional ground truth answers.Our study shows that a
reward function centered on format correctness alone can yield performance
improvements comparable to the standard GRPO algorithm in early phases.
Recognizing the limitations of format-only rewards in the later phases, we
incorporate length-based rewards. The resulting GRPO approach, leveraging
format-length surrogate signals, not only matches but surpasses the performance
of the standard GRPO algorithm relying on ground truth answers in certain
scenarios, achieving 40.0\% accuracy on AIME2024 with a 7B base model. Through
systematic exploration and experimentation, this research not only offers a
practical solution for training LLMs to solve mathematical problems and
reducing the dependence on extensive ground truth data collection, but also
reveals the essence of why our label-free approach succeeds: base model is like
an excellent student who has already mastered mathematical and logical
reasoning skills, but performs poorly on the test paper, it simply needs to
develop good answering habits to achieve outstanding results in exams , in
other words, to unlock the capabilities it already possesses.

---


### [Rhapsody: A Dataset for Highlight Detection in Podcasts](http://arxiv.org/abs/2505.19429v1)

Podcasts have become daily companions for half a billion users. Given the
enormous amount of podcast content available, highlights provide a valuable
signal that helps viewers get the gist of an episode and decide if they want to
invest in listening to it in its entirety. However, identifying highlights
automatically is challenging due to the unstructured and long-form nature of
the content. We introduce Rhapsody, a dataset of 13K podcast episodes paired
with segment-level highlight scores derived from YouTube's 'most replayed'
feature. We frame the podcast highlight detection as a segment-level binary
classification task. We explore various baseline approaches, including
zero-shot prompting of language models and lightweight finetuned language
models using segment-level classification heads. Our experimental results
indicate that even state-of-the-art language models like GPT-4o and Gemini
struggle with this task, while models finetuned with in-domain data
significantly outperform their zero-shot performance. The finetuned model
benefits from leveraging both speech signal features and transcripts. These
findings highlight the challenges for fine-grained information access in
long-form spoken media.

---


### [Frictional Agent Alignment Framework: Slow Down and Don't Break Things](http://arxiv.org/abs/2505.19428v1)

AI support of collaborative interactions entails mediating potential
misalignment between interlocutor beliefs. Common preference alignment methods
like DPO excel in static settings, but struggle in dynamic collaborative tasks
where the explicit signals of interlocutor beliefs are sparse and skewed. We
propose the Frictional Agent Alignment Framework (FAAF), to generate precise,
context-aware "friction" that prompts for deliberation and re-examination of
existing evidence. FAAF's two-player objective decouples from data skew: a
frictive-state policy identifies belief misalignments, while an intervention
policy crafts collaborator-preferred responses. We derive an analytical
solution to this objective, enabling training a single policy via a simple
supervised loss. Experiments on three benchmarks show FAAF outperforms
competitors in producing concise, interpretable friction and in OOD
generalization. By aligning LLMs to act as adaptive "thought partners" -- not
passive responders -- FAAF advances scalable, dynamic human-AI collaboration.
Our code and data can be found at https://github.com/csu-signal/FAAF_ACL.

---


### [Self-Reflective Planning with Knowledge Graphs: Enhancing LLM Reasoning Reliability for Question Answering](http://arxiv.org/abs/2505.19410v1)

Recently, large language models (LLMs) have demonstrated remarkable
capabilities in natural language processing tasks, yet they remain prone to
hallucinations when reasoning with insufficient internal knowledge. While
integrating LLMs with knowledge graphs (KGs) provides access to structured,
verifiable information, existing approaches often generate incomplete or
factually inconsistent reasoning paths. To this end, we propose Self-Reflective
Planning (SRP), a framework that synergizes LLMs with KGs through iterative,
reference-guided reasoning. Specifically, given a question and topic entities,
SRP first searches for references to guide planning and reflection. In the
planning process, it checks initial relations and generates a reasoning path.
After retrieving knowledge from KGs through a reasoning path, it implements
iterative reflection by judging the retrieval result and editing the reasoning
path until the answer is correctly retrieved. Extensive experiments on three
public datasets demonstrate that SRP surpasses various strong baselines and
further underscore its reliable reasoning ability.

---


### [CoTGuard: Using Chain-of-Thought Triggering for Copyright Protection in Multi-Agent LLM Systems](http://arxiv.org/abs/2505.19405v1)

As large language models (LLMs) evolve into autonomous agents capable of
collaborative reasoning and task execution, multi-agent LLM systems have
emerged as a powerful paradigm for solving complex problems. However, these
systems pose new challenges for copyright protection, particularly when
sensitive or copyrighted content is inadvertently recalled through inter-agent
communication and reasoning. Existing protection techniques primarily focus on
detecting content in final outputs, overlooking the richer, more revealing
reasoning processes within the agents themselves. In this paper, we introduce
CoTGuard, a novel framework for copyright protection that leverages
trigger-based detection within Chain-of-Thought (CoT) reasoning. Specifically,
we can activate specific CoT segments and monitor intermediate reasoning steps
for unauthorized content reproduction by embedding specific trigger queries
into agent prompts. This approach enables fine-grained, interpretable detection
of copyright violations in collaborative agent scenarios. We evaluate CoTGuard
on various benchmarks in extensive experiments and show that it effectively
uncovers content leakage with minimal interference to task performance. Our
findings suggest that reasoning-level monitoring offers a promising direction
for safeguarding intellectual property in LLM-based agent systems.

---


### [Multimodal Reasoning Agent for Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2505.19952v1)

Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images
given a compositional query, consisting of a reference image and a modifying
text-without relying on annotated training data. Existing approaches often
generate a synthetic target text using large language models (LLMs) to serve as
an intermediate anchor between the compositional query and the target image.
Models are then trained to align the compositional query with the generated
text, and separately align images with their corresponding texts using
contrastive learning. However, this reliance on intermediate text introduces
error propagation, as inaccuracies in query-to-text and text-to-image mappings
accumulate, ultimately degrading retrieval performance. To address these
problems, we propose a novel framework by employing a Multimodal Reasoning
Agent (MRA) for ZS-CIR. MRA eliminates the dependence on textual intermediaries
by directly constructing triplets, <reference image, modification text, target
image>, using only unlabeled image data. By training on these synthetic
triplets, our model learns to capture the relationships between compositional
queries and candidate images directly. Extensive experiments on three standard
CIR benchmarks demonstrate the effectiveness of our approach. On the FashionIQ
dataset, our method improves Average R@10 by at least 7.5\% over existing
baselines; on CIRR, it boosts R@1 by 9.6\%; and on CIRCO, it increases mAP@5 by
9.5\%.

---


### [Attention! You Vision Language Model Could Be Maliciously Manipulated](http://arxiv.org/abs/2505.19911v1)

Large Vision-Language Models (VLMs) have achieved remarkable success in
understanding complex real-world scenarios and supporting data-driven
decision-making processes. However, VLMs exhibit significant vulnerability
against adversarial examples, either text or image, which can lead to various
adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In
this work, we empirically and theoretically demonstrate that VLMs are
particularly susceptible to image-based adversarial examples, where
imperceptible perturbations can precisely manipulate each output token. To this
end, we propose a novel attack called Vision-language model Manipulation Attack
(VMA), which integrates first-order and second-order momentum optimization
techniques with a differentiable transformation mechanism to effectively
optimize the adversarial perturbation. Notably, VMA can be a double-edged
sword: it can be leveraged to implement various attacks, such as jailbreaking,
hijacking, privacy breaches, Denial-of-Service, and the generation of sponge
examples, etc, while simultaneously enabling the injection of watermarks for
copyright protection. Extensive empirical evaluations substantiate the efficacy
and generalizability of VMA across diverse scenarios and datasets.

---


### [Vad-R1: Towards Video Anomaly Reasoning via Perception-to-Cognition Chain-of-Thought](http://arxiv.org/abs/2505.19877v1)

Recent advancements in reasoning capability of Multimodal Large Language
Models (MLLMs) demonstrate its effectiveness in tackling complex visual tasks.
However, existing MLLM-based Video Anomaly Detection (VAD) methods remain
limited to shallow anomaly descriptions without deep reasoning. In this paper,
we propose a new task named Video Anomaly Reasoning (VAR), which aims to enable
deep analysis and understanding of anomalies in the video by requiring MLLMs to
think explicitly before answering. To this end, we propose Vad-R1, an
end-to-end MLLM-based framework for VAR. Specifically, we design a
Perception-to-Cognition Chain-of-Thought (P2C-CoT) that simulates the human
process of recognizing anomalies, guiding the MLLM to reason anomaly
step-by-step. Based on the structured P2C-CoT, we construct Vad-Reasoning, a
dedicated dataset for VAR. Furthermore, we propose an improved reinforcement
learning algorithm AVA-GRPO, which explicitly incentivizes the anomaly
reasoning capability of MLLMs through a self-verification mechanism with
limited annotations. Experimental results demonstrate that Vad-R1 achieves
superior performance, outperforming both open-source and proprietary models on
VAD and VAR tasks. Codes and datasets will be released at
https://github.com/wbfwonderful/Vad-R1.

---


### [MLLM-Guided VLM Fine-Tuning with Joint Inference for Zero-Shot Composed Image Retrieval](http://arxiv.org/abs/2505.19707v1)

Existing Zero-Shot Composed Image Retrieval (ZS-CIR) methods typically train
adapters that convert reference images into pseudo-text tokens, which are
concatenated with the modifying text and processed by frozen text encoders in
pretrained VLMs or LLMs. While this design leverages the strengths of large
pretrained models, it only supervises the adapter to produce encoder-compatible
tokens that loosely preserve visual semantics. Crucially, it does not directly
optimize the composed query representation to capture the full intent of the
composition or to align with the target semantics, thereby limiting retrieval
performance, particularly in cases involving fine-grained or complex visual
transformations. To address this problem, we propose MLLM-Guided VLM
Fine-Tuning with Joint Inference (MVFT-JI), a novel approach that leverages a
pretrained multimodal large language model (MLLM) to construct two
complementary training tasks using only unlabeled images: target text retrieval
taskand text-to-image retrieval task. By jointly optimizing these tasks, our
method enables the VLM to inherently acquire robust compositional retrieval
capabilities, supported by the provided theoretical justifications and
empirical validation. Furthermore, during inference, we further prompt the MLLM
to generate target texts from composed queries and compute retrieval scores by
integrating similarities between (i) the composed query and candidate images,
and (ii) the MLLM-generated target text and candidate images. This strategy
effectively combines the VLM's semantic alignment strengths with the MLLM's
reasoning capabilities.

---


### [Knowledge-Aligned Counterfactual-Enhancement Diffusion Perception for Unsupervised Cross-Domain Visual Emotion Recognition](http://arxiv.org/abs/2505.19694v1)

Visual Emotion Recognition (VER) is a critical yet challenging task aimed at
inferring emotional states of individuals based on visual cues. However,
existing works focus on single domains, e.g., realistic images or stickers,
limiting VER models' cross-domain generalizability. To fill this gap, we
introduce an Unsupervised Cross-Domain Visual Emotion Recognition (UCDVER)
task, which aims to generalize visual emotion recognition from the source
domain (e.g., realistic images) to the low-resource target domain (e.g.,
stickers) in an unsupervised manner. Compared to the conventional unsupervised
domain adaptation problems, UCDVER presents two key challenges: a significant
emotional expression variability and an affective distribution shift. To
mitigate these issues, we propose the Knowledge-aligned
Counterfactual-enhancement Diffusion Perception (KCDP) framework. Specifically,
KCDP leverages a VLM to align emotional representations in a shared knowledge
space and guides diffusion models for improved visual affective perception.
Furthermore, a Counterfactual-Enhanced Language-image Emotional Alignment
(CLIEA) method generates high-quality pseudo-labels for the target domain.
Extensive experiments demonstrate that our model surpasses SOTA models in both
perceptibility and generalization, e.g., gaining 12% improvements over the SOTA
VER model TGCA-PVT. The project page is at https://yinwen2019.github.io/ucdver.

---


### [VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models](http://arxiv.org/abs/2505.19684v1)

The emergence of Multimodal Large Language Models (MLRMs) has enabled
sophisticated visual reasoning capabilities by integrating reinforcement
learning and Chain-of-Thought (CoT) supervision. However, while these enhanced
reasoning capabilities improve performance, they also introduce new and
underexplored safety risks. In this work, we systematically investigate the
security implications of advanced visual reasoning in MLRMs. Our analysis
reveals a fundamental trade-off: as visual reasoning improves, models become
more vulnerable to jailbreak attacks. Motivated by this critical finding, we
introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework
that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA
combines targeted visual attention masking with a two-stage reasoning induction
strategy to precisely control harmful outputs. Extensive experiments
demonstrate VisCRA's significant effectiveness, achieving high attack success
rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking,
68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical
insight: the very capability that empowers MLRMs -- their visual reasoning --
can also serve as an attack vector, posing significant security risks.

---


### [JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models](http://arxiv.org/abs/2505.19610v1)

Vision-Language Models (VLMs) exhibit impressive performance, yet the
integration of powerful vision encoders has significantly broadened their
attack surface, rendering them increasingly susceptible to jailbreak attacks.
However, lacking well-defined attack objectives, existing jailbreak methods
often struggle with gradient-based strategies prone to local optima and lacking
precise directional guidance, and typically decouple visual and textual
modalities, thereby limiting their effectiveness by neglecting crucial
cross-modal interactions. Inspired by the Eliciting Latent Knowledge (ELK)
framework, we posit that VLMs encode safety-relevant information within their
internal fusion-layer representations, revealing an implicit safety decision
boundary in the latent space. This motivates exploiting boundary to steer model
behavior. Accordingly, we propose JailBound, a novel latent space jailbreak
framework comprising two stages: (1) Safety Boundary Probing, which addresses
the guidance issue by approximating decision boundary within fusion layer's
latent space, thereby identifying optimal perturbation directions towards the
target region; and (2) Safety Boundary Crossing, which overcomes the
limitations of decoupled approaches by jointly optimizing adversarial
perturbations across both image and text inputs. This latter stage employs an
innovative mechanism to steer the model's internal state towards
policy-violating outputs while maintaining cross-modal semantic consistency.
Extensive experiments on six diverse VLMs demonstrate JailBound's efficacy,
achieves 94.32% white-box and 67.28% black-box attack success averagely, which
are 6.17% and 21.13% higher than SOTA methods, respectively. Our findings
expose a overlooked safety risk in VLMs and highlight the urgent need for more
robust defenses. Warning: This paper contains potentially sensitive, harmful
and offensive content.

---


### [VTBench: Comprehensive Benchmark Suite Towards Real-World Virtual Try-on Models](http://arxiv.org/abs/2505.19571v1)

While virtual try-on has achieved significant progress, evaluating these
models towards real-world scenarios remains a challenge. A comprehensive
benchmark is essential for three key reasons:(1) Current metrics inadequately
reflect human perception, particularly in unpaired try-on settings;(2)Most
existing test sets are limited to indoor scenarios, lacking complexity for
real-world evaluation; and (3) An ideal system should guide future advancements
in virtual try-on generation. To address these needs, we introduce VTBench, a
hierarchical benchmark suite that systematically decomposes virtual image
try-on into hierarchical, disentangled dimensions, each equipped with tailored
test sets and evaluation criteria. VTBench exhibits three key advantages:1)
Multi-Dimensional Evaluation Framework: The benchmark encompasses five critical
dimensions for virtual try-on generation (e.g., overall image quality, texture
preservation, complex background consistency, cross-category size adaptability,
and hand-occlusion handling). Granular evaluation metrics of corresponding test
sets pinpoint model capabilities and limitations across diverse, challenging
scenarios.2) Human Alignment: Human preference annotations are provided for
each test set, ensuring the benchmark's alignment with perceptual quality
across all evaluation dimensions. (3) Valuable Insights: Beyond standard indoor
settings, we analyze model performance variations across dimensions and
investigate the disparity between indoor and real-world try-on scenarios. To
foster the field of virtual try-on towards challenging real-world scenario,
VTBench will be open-sourced, including all test sets, evaluation protocols,
generated results, and human annotations.

---


### [TDVE-Assessor: Benchmarking and Evaluating the Quality of Text-Driven Video Editing with LMMs](http://arxiv.org/abs/2505.19535v1)

Text-driven video editing is rapidly advancing, yet its rigorous evaluation
remains challenging due to the absence of dedicated video quality assessment
(VQA) models capable of discerning the nuances of editing quality. To address
this critical gap, we introduce TDVE-DB, a large-scale benchmark dataset for
text-driven video editing. TDVE-DB consists of 3,857 edited videos generated
from 12 diverse models across 8 editing categories, and is annotated with
173,565 human subjective ratings along three crucial dimensions, i.e., edited
video quality, editing alignment, and structural consistency. Based on TDVE-DB,
we first conduct a comprehensive evaluation for the 12 state-of-the-art editing
models revealing the strengths and weaknesses of current video techniques, and
then benchmark existing VQA methods in the context of text-driven video editing
evaluation. Building on these insights, we propose TDVE-Assessor, a novel VQA
model specifically designed for text-driven video editing assessment.
TDVE-Assessor integrates both spatial and temporal video features into a large
language model (LLM) for rich contextual understanding to provide comprehensive
quality assessment. Extensive experiments demonstrate that TDVE-Assessor
substantially outperforms existing VQA models on TDVE-DB across all three
evaluation dimensions, setting a new state-of-the-art. Both TDVE-DB and
TDVE-Assessor will be released upon the publication.

---


### [A Contrastive Learning Foundation Model Based on Perfectly Aligned Sample Pairs for Remote Sensing Images](http://arxiv.org/abs/2505.19447v1)

Self-Supervised Learning (SSL) enables us to pre-train foundation models
without costly labeled data. Among SSL methods, Contrastive Learning (CL)
methods are better at obtaining accurate semantic representations in noise
interference. However, due to the significant domain gap, while CL methods have
achieved great success in many computer vision tasks, they still require
specific adaptation for Remote Sensing (RS) images. To this end, we present a
novel self-supervised method called PerA, which produces all-purpose RS
features through semantically Perfectly Aligned sample pairs. Specifically,
PerA obtains features from sampled views by applying spatially disjoint masks
to augmented images rather than random cropping. With disjoint masks, we divide
patches from different views into different parts that are semantically aligned
but inconsistent in appearance. Our framework provides high-quality features by
ensuring consistency between teacher and student and predicting learnable mask
tokens. Compared to previous contrastive methods, our method demonstrates
higher memory efficiency and can be trained with larger batches due to its
sparse inputs. We also collect an unlabeled pre-training dataset, which
contains about 5 million RS images. We conducted experiments on multiple
downstream task datasets and achieved performance comparable to previous
state-of-the-art methods with a limited model scale, which verified the
superiority of our method. We hope this work will contribute to practical
remote sensing interpretation works.

---


### [MMIG-Bench: Towards Comprehensive and Explainable Evaluation of Multi-Modal Image Generation Models](http://arxiv.org/abs/2505.19415v1)

Recent multimodal image generators such as GPT-4o, Gemini 2.0 Flash, and
Gemini 2.5 Pro excel at following complex instructions, editing images and
maintaining concept consistency. However, they are still evaluated by disjoint
toolkits: text-to-image (T2I) benchmarks that lacks multi-modal conditioning,
and customized image generation benchmarks that overlook compositional
semantics and common knowledge. We propose MMIG-Bench, a comprehensive
Multi-Modal Image Generation Benchmark that unifies these tasks by pairing
4,850 richly annotated text prompts with 1,750 multi-view reference images
across 380 subjects, spanning humans, animals, objects, and artistic styles.
MMIG-Bench is equipped with a three-level evaluation framework: (1) low-level
metrics for visual artifacts and identity preservation of objects; (2) novel
Aspect Matching Score (AMS): a VQA-based mid-level metric that delivers
fine-grained prompt-image alignment and shows strong correlation with human
judgments; and (3) high-level metrics for aesthetics and human preference.
Using MMIG-Bench, we benchmark 17 state-of-the-art models, including Gemini 2.5
Pro, FLUX, DreamBooth, and IP-Adapter, and validate our metrics with 32k human
ratings, yielding in-depth insights into architecture and data design. We will
release the dataset and evaluation code to foster rigorous, unified evaluation
and accelerate future innovations in multi-modal image generation.

---


### [Beyond Simple Concatenation: Fairly Assessing PLM Architectures for Multi-Chain Protein-Protein Interactions Prediction](http://arxiv.org/abs/2505.20036v1)

Protein-protein interactions (PPIs) are fundamental to numerous cellular
processes, and their characterization is vital for understanding disease
mechanisms and guiding drug discovery. While protein language models (PLMs)
have demonstrated remarkable success in predicting protein structure and
function, their application to sequence-based PPI binding affinity prediction
remains relatively underexplored. This gap is often attributed to the scarcity
of high-quality, rigorously refined datasets and the reliance on simple
strategies for concatenating protein representations. In this work, we address
these limitations. First, we introduce a meticulously curated version of the
PPB-Affinity dataset of a total of 8,207 unique protein-protein interaction
entries, by resolving annotation inconsistencies and duplicate entries for
multi-chain protein interactions. This dataset incorporates a stringent, less
than or equal to 30%, sequence identity threshold to ensure robust splitting
into training, validation, and test sets, minimizing data leakage. Second, we
propose and systematically evaluate four architectures for adapting PLMs to PPI
binding affinity prediction: embeddings concatenation (EC), sequences
concatenation (SC), hierarchical pooling (HP), and pooled attention addition
(PAD). These architectures were assessed using two training methods: full
fine-tuning and a lightweight approach employing ConvBERT heads over frozen PLM
features. Our comprehensive experiments across multiple leading PLMs (ProtT5,
ESM2, Ankh, Ankh2, and ESM3) demonstrated that the HP and PAD architectures
consistently outperform conventional concatenation methods, achieving up to 12%
increase in terms of Spearman correlation. These results highlight the
necessity of sophisticated architectural designs to fully exploit the
capabilities of PLMs for nuanced PPI binding affinity prediction.

---


### [Regret Analysis of Average-Reward Unichain MDPs via an Actor-Critic Approach](http://arxiv.org/abs/2505.19986v1)

Actor-Critic methods are widely used for their scalability, yet existing
theoretical guarantees for infinite-horizon average-reward Markov Decision
Processes (MDPs) often rely on restrictive ergodicity assumptions. We propose
NAC-B, a Natural Actor-Critic with Batching, that achieves order-optimal regret
of $\tilde{O}(\sqrt{T})$ in infinite-horizon average-reward MDPs under the
unichain assumption, which permits both transient states and periodicity. This
assumption is among the weakest under which the classic policy gradient theorem
remains valid for average-reward settings. NAC-B employs function approximation
for both the actor and the critic, enabling scalability to problems with large
state and action spaces. The use of batching in our algorithm helps mitigate
potential periodicity in the MDP and reduces stochasticity in gradient
estimates, and our analysis formalizes these benefits through the introduction
of the constants $C_{\text{hit}}$ and $C_{\text{tar}}$, which characterize the
rate at which empirical averages over Markovian samples converge to the
stationary distribution.

---


### [Which Data Attributes Stimulate Math and Code Reasoning? An Investigation via Influence Functions](http://arxiv.org/abs/2505.19949v1)

Large language models (LLMs) have demonstrated remarkable reasoning
capabilities in math and coding, often bolstered by post-training on the
chain-of-thoughts (CoTs) generated by stronger models. However, existing
strategies for curating such training data predominantly rely on heuristics,
limiting generalizability and failing to capture subtleties underlying in data.
To address these limitations, we leverage influence functions to systematically
attribute LLMs' reasoning ability on math and coding to individual training
examples, sequences, and tokens, enabling deeper insights into effective data
characteristics. Our Influence-based Reasoning Attribution (Infra) uncovers
nontrivial cross-domain effects across math and coding tasks: high-difficulty
math examples improve both math and code reasoning, while low-difficulty code
tasks most effectively benefit code reasoning. Based on these findings, we
introduce a simple yet effective dataset reweighting strategy by flipping task
difficulty, which doubles AIME24 accuracy from 10\% to 20\% and boosts
LiveCodeBench accuracy from 33.8\% to 35.3\% for Qwen2.5-7B-Instruct. Moreover,
our fine-grained attribution reveals that the sequence-level exploratory
behaviors enhance reasoning performance in both math and code, and the
token-level influence patterns are distinct for math and code reasoning: the
former prefers natural language logic connectors and the latter emphasizes
structural syntax.

---


### [One Surrogate to Fool Them All: Universal, Transferable, and Targeted Adversarial Attacks with CLIP](http://arxiv.org/abs/2505.19840v1)

Deep Neural Networks (DNNs) have achieved widespread success yet remain prone
to adversarial attacks. Typically, such attacks either involve frequent queries
to the target model or rely on surrogate models closely mirroring the target
model -- often trained with subsets of the target model's training data -- to
achieve high attack success rates through transferability. However, in
realistic scenarios where training data is inaccessible and excessive queries
can raise alarms, crafting adversarial examples becomes more challenging. In
this paper, we present UnivIntruder, a novel attack framework that relies
solely on a single, publicly available CLIP model and publicly available
datasets. By using textual concepts, UnivIntruder generates universal,
transferable, and targeted adversarial perturbations that mislead DNNs into
misclassifying inputs into adversary-specified classes defined by textual
concepts.
  Our extensive experiments show that our approach achieves an Attack Success
Rate (ASR) of up to 85% on ImageNet and over 99% on CIFAR-10, significantly
outperforming existing transfer-based methods. Additionally, we reveal
real-world vulnerabilities, showing that even without querying target models,
UnivIntruder compromises image search engines like Google and Baidu with ASR
rates up to 84%, and vision language models like GPT-4 and Claude-3.5 with ASR
rates up to 80%. These findings underscore the practicality of our attack in
scenarios where traditional avenues are blocked, highlighting the need to
reevaluate security paradigms in AI applications.

---


### [Multi-Agent Reinforcement Learning in Cybersecurity: From Fundamentals to Applications](http://arxiv.org/abs/2505.19837v1)

Multi-Agent Reinforcement Learning (MARL) has shown great potential as an
adaptive solution for addressing modern cybersecurity challenges. MARL enables
decentralized, adaptive, and collaborative defense strategies and provides an
automated mechanism to combat dynamic, coordinated, and sophisticated threats.
This survey investigates the current state of research in MARL applications for
automated cyber defense (ACD), focusing on intruder detection and lateral
movement containment. Additionally, it examines the role of Autonomous
Intelligent Cyber-defense Agents (AICA) and Cyber Gyms in training and
validating MARL agents. Finally, the paper outlines existing challenges, such
as scalability and adversarial robustness, and proposes future research
directions. This also discusses how MARL integrates in AICA to provide
adaptive, scalable, and dynamic solutions to counter the increasingly
sophisticated landscape of cyber threats. It highlights the transformative
potential of MARL in areas like intrusion detection and lateral movement
containment, and underscores the value of Cyber Gyms for training and
validation of AICA.

---


### [What Can RL Bring to VLA Generalization? An Empirical Study](http://arxiv.org/abs/2505.19789v1)

Large Vision-Language Action (VLA) models have shown significant potential
for embodied AI. However, their predominant training via supervised fine-tuning
(SFT) limits generalization due to susceptibility to compounding errors under
distribution shifts. Reinforcement learning (RL) offers a path to overcome
these limitations by optimizing for task objectives via trial-and-error, yet a
systematic understanding of its specific generalization benefits for VLAs
compared to SFT is lacking. To address this, our study introduces a
comprehensive benchmark for evaluating VLA generalization and systematically
investigates the impact of RL fine-tuning across diverse visual, semantic, and
execution dimensions. Our extensive experiments reveal that RL fine-tuning,
particularly with PPO, significantly enhances generalization in semantic
understanding and execution robustness over SFT, while maintaining comparable
visual robustness. We identify PPO as a more effective RL algorithm for VLAs
than LLM-derived methods like DPO and GRPO. We also develop a simple recipe for
efficient PPO training on VLAs, and demonstrate its practical utility for
improving VLA generalization. The project page is at https://rlvla.github.io

---


### [Accelerating Nash Learning from Human Feedback via Mirror Prox](http://arxiv.org/abs/2505.19731v1)

Traditional Reinforcement Learning from Human Feedback (RLHF) often relies on
reward models, frequently assuming preference structures like the Bradley-Terry
model, which may not accurately capture the complexities of real human
preferences (e.g., intransitivity). Nash Learning from Human Feedback (NLHF)
offers a more direct alternative by framing the problem as finding a Nash
equilibrium of a game defined by these preferences. In this work, we introduce
Nash Mirror Prox ($\mathtt{Nash-MP}$), an online NLHF algorithm that leverages
the Mirror Prox optimization scheme to achieve fast and stable convergence to
the Nash equilibrium. Our theoretical analysis establishes that Nash-MP
exhibits last-iterate linear convergence towards the $\beta$-regularized Nash
equilibrium. Specifically, we prove that the KL-divergence to the optimal
policy decreases at a rate of order $(1+2\beta)^{-N/2}$, where $N$ is a number
of preference queries. We further demonstrate last-iterate linear convergence
for the exploitability gap and uniformly for the span semi-norm of
log-probabilities, with all these rates being independent of the size of the
action space. Furthermore, we propose and analyze an approximate version of
Nash-MP where proximal steps are estimated using stochastic policy gradients,
making the algorithm closer to applications. Finally, we detail a practical
implementation strategy for fine-tuning large language models and present
experiments that demonstrate its competitive performance and compatibility with
existing methods.

---


### [ExAnte: A Benchmark for Ex-Ante Inference in Large Language Models](http://arxiv.org/abs/2505.19533v1)

Large language models (LLMs) face significant challenges in ex-ante
reasoning, where analysis, inference, or predictions must be made without
access to information from future events. Even with explicit prompts enforcing
temporal cutoffs, LLMs often generate outputs influenced by internalized
knowledge of events beyond the specified cutoff. This paper introduces a novel
task and benchmark designed to evaluate the ability of LLMs to reason while
adhering to such temporal constraints. The benchmark includes a variety of
tasks: stock prediction, Wikipedia event prediction, scientific publication
prediction, and Question Answering (QA), designed to assess factual knowledge
under temporal cutoff constraints. We use leakage rate to quantify models'
reliance on future information beyond cutoff timestamps. Experimental results
reveal that LLMs struggle to consistently adhere to temporal cutoffs across
common prompting strategies and tasks, demonstrating persistent challenges in
ex-ante reasoning. This benchmark provides a potential evaluation framework to
advance the development of LLMs' temporal reasoning ability for time-sensitive
applications.

---


### [Fox in the Henhouse: Supply-Chain Backdoor Attacks Against Reinforcement Learning](http://arxiv.org/abs/2505.19532v1)

The current state-of-the-art backdoor attacks against Reinforcement Learning
(RL) rely upon unrealistically permissive access models, that assume the
attacker can read (or even write) the victim's policy parameters, observations,
or rewards. In this work, we question whether such a strong assumption is
required to launch backdoor attacks against RL. To answer this question, we
propose the \underline{S}upply-\underline{C}h\underline{a}in
\underline{B}ackdoor (SCAB) attack, which targets a common RL workflow:
training agents using external agents that are provided separately or embedded
within the environment. In contrast to prior works, our attack only relies on
legitimate interactions of the RL agent with the supplied agents. Despite this
limited access model, by poisoning a mere $3\%$ of training experiences, our
attack can successfully activate over $90\%$ of triggered actions, reducing the
average episodic return by $80\%$ for the victim. Our novel attack demonstrates
that RL attacks are likely to become a reality under untrusted RL training
supply-chains.

---


### [VLMLight: Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning](http://arxiv.org/abs/2505.19486v1)

Traffic signal control (TSC) is a core challenge in urban mobility, where
real-time decisions must balance efficiency and safety. Existing methods -
ranging from rule-based heuristics to reinforcement learning (RL) - often
struggle to generalize to complex, dynamic, and safety-critical scenarios. We
introduce VLMLight, a novel TSC framework that integrates vision-language
meta-control with dual-branch reasoning. At the core of VLMLight is the first
image-based traffic simulator that enables multi-view visual perception at
intersections, allowing policies to reason over rich cues such as vehicle type,
motion, and spatial density. A large language model (LLM) serves as a
safety-prioritized meta-controller, selecting between a fast RL policy for
routine traffic and a structured reasoning branch for critical cases. In the
latter, multiple LLM agents collaborate to assess traffic phases, prioritize
emergency vehicles, and verify rule compliance. Experiments show that VLMLight
reduces waiting times for emergency vehicles by up to 65% over RL-only systems,
while preserving real-time performance in standard conditions with less than 1%
degradation. VLMLight offers a scalable, interpretable, and safety-aware
solution for next-generation traffic signal control.

---


### [Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression](http://arxiv.org/abs/2505.19433v1)

Post-training compression reduces the computational and memory costs of large
language models (LLMs), enabling resource-efficient deployment. However,
existing compression benchmarks only focus on language modeling (e.g.,
perplexity) and natural language understanding tasks (e.g., GLUE accuracy),
ignoring the agentic capabilities - workflow, tool use/function call,
long-context understanding and real-world application. We introduce the Agent
Compression Benchmark (ACBench), the first comprehensive benchmark for
evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1)
12 tasks across 4 capabilities (e.g., WorfBench for workflow generation,
Needle-in-Haystack for long-context retrieval), (2) quantization (GPTQ, AWQ)
and pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B),
standard (Qwen2.5 7B-32B), and distilled reasoning LLMs (DeepSeek-R1-Distill).
Our experiments reveal compression tradeoffs: 4-bit quantization preserves
workflow generation and tool use (1%-3% drop) but degrades real-world
application accuracy by 10%-15%. We introduce ERank, Top-k Ranking Correlation
and Energy to systematize analysis. ACBench provides actionable insights for
optimizing LLM compression in agentic scenarios. The code can be found in
https://github.com/pprp/ACBench.

---


### [Alignment of large language models with constrained learning](http://arxiv.org/abs/2505.19387v1)

We study the problem of computing an optimal large language model (LLM)
policy for a constrained alignment problem, where the goal is to maximize a
primary reward objective while satisfying constraints on secondary utilities.
Despite the popularity of Lagrangian-based LLM policy search in constrained
alignment, iterative primal-dual methods often fail to converge, and
non-iterative dual-based methods do not achieve optimality in the LLM parameter
space. To address these challenges, we employ Lagrangian duality to develop an
iterative dual-based alignment method that alternates between updating the LLM
policy via Lagrangian maximization and updating the dual variable via dual
descent. In theory, we characterize the primal-dual gap between the primal
value in the distribution space and the dual value in the LLM parameter space.
We further quantify the optimality gap of the learned LLM policies at
near-optimal dual variables with respect to both the objective and the
constraint functions. These results prove that dual-based alignment methods can
find an optimal constrained LLM policy, up to an LLM parametrization gap. We
demonstrate the effectiveness and merits of our approach through extensive
experiments conducted on the PKU-SafeRLHF dataset.

---


### [Uncertainty-Aware Safety-Critical Decision and Control for Autonomous Vehicles at Unsignalized Intersections](http://arxiv.org/abs/2505.19939v1)

Reinforcement learning (RL) has demonstrated potential in autonomous driving
(AD) decision tasks. However, applying RL to urban AD, particularly in
intersection scenarios, still faces significant challenges. The lack of safety
constraints makes RL vulnerable to risks. Additionally, cognitive limitations
and environmental randomness can lead to unreliable decisions in
safety-critical scenarios. Therefore, it is essential to quantify confidence in
RL decisions to improve safety. This paper proposes an Uncertainty-aware
Safety-Critical Decision and Control (USDC) framework, which generates a
risk-averse policy by constructing a risk-aware ensemble distributional RL,
while estimating uncertainty to quantify the policy's reliability.
Subsequently, a high-order control barrier function (HOCBF) is employed as a
safety filter to minimize intervention policy while dynamically enhancing
constraints based on uncertainty. The ensemble critics evaluate both HOCBF and
RL policies, embedding uncertainty to achieve dynamic switching between safe
and flexible strategies, thereby balancing safety and efficiency. Simulation
tests on unsignalized intersections in multiple tasks indicate that USDC can
improve safety while maintaining traffic efficiency compared to baselines.

---


### [Integrating emotional intelligence, memory architecture, and gestures to achieve empathetic humanoid robot interaction in an educational setting](http://arxiv.org/abs/2505.19803v1)

This study investigates the integration of individual human traits into an
empathetically adaptive educational robot tutor system designed to improve
student engagement and learning outcomes with corresponding Engagement Vector
measurement. While prior research in the field of Human-Robot Interaction (HRI)
has examined the integration of the traits, such as emotional intelligence,
memory-driven personalization, and non-verbal communication, by themselves,
they have thus-far neglected to consider their synchronized integration into a
cohesive, operational education framework. To address this gap, we customize a
Multi-Modal Large Language Model (LLaMa 3.2 from Meta) deployed with modules
for human-like traits (emotion, memory and gestures) into an AI-Agent
framework. This constitutes to the robot's intelligent core mimicing the human
emotional system, memory architecture and gesture control to allow the robot to
behave more empathetically while recognizing and responding appropriately to
the student's emotional state. It can also recall the student's past learning
record and adapt its style of interaction accordingly. This allows the robot
tutor to react to the student in a more sympathetic manner by delivering
personalized verbal feedback synchronized with relevant gestures. Our study
investigates the extent of this effect through the introduction of Engagement
Vector Model which can be a surveyor's pole for judging the quality of HRI
experience. Quantitative and qualitative results demonstrate that such an
empathetic responsive approach significantly improves student engagement and
learning outcomes compared with a baseline humanoid robot without these
human-like traits. This indicates that robot tutors with empathetic
capabilities can create a more supportive, interactive learning experience that
ultimately leads to better outcomes for the student.

---


### [GPU acceleration of non-equilibrium Green's function calculation using OpenACC and CUDA FORTRAN](http://arxiv.org/abs/2505.19467v1)

The numerical solution of the Kadanoff-Baym nonlinear integro-differential
equations, which yields the non-equilibrium Green's functions (NEGFs) of
quantum many-body systems, poses significant computational challenges due to
its high computational complexity. In this work, we present efficient
implementations of a numerical method for solving these equations on
distributed-memory architectures, including many-core CPUs and multi-GPU
systems. For CPU-based platforms, we adopt a hybrid MPI/OpenMP programming
model to exploit both inter-node and intra-node parallelism. On GPU-accelerated
systems, we implement the method using two distinct approaches: MPI/OpenACC and
MPI/CUDA FORTRAN. Several optimization strategies are employed to enhance GPU
performance, including techniques to maximize computational resource
utilization and minimize the overhead associated with kernel launches and
memory management. Although OpenACC is easy to use, CUDA FORTRAN provides more
advanced features for configuring and managing multiple levels of concurrency,
while also simplifying memory allocation and data movement between host and
device. This flexibility translates into significant performance improvements.
We compare the performance of the three implementations and demonstrate that
the GPU-based approaches achieve substantial speedups over CPU-based
implementations. Furthermore, both CPU and GPU versions exhibit excellent
strong and weak scaling, confirming the scalability and efficiency of our
approach for large-scale NEGF computations.

---


