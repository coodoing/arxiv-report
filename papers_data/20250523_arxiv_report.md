### [Let Androids Dream of Electric Sheep: A Human-like Image Implication Understanding and Reasoning Framework](http://arxiv.org/abs/2505.17019v1)

Metaphorical comprehension in images remains a critical challenge for AI
systems, as existing models struggle to grasp the nuanced cultural, emotional,
and contextual implications embedded in visual content. While multimodal large
language models (MLLMs) excel in basic Visual Question Answer (VQA) tasks, they
struggle with a fundamental limitation on image implication tasks: contextual
gaps that obscure the relationships between different visual elements and their
abstract meanings. Inspired by the human cognitive process, we propose Let
Androids Dream (LAD), a novel framework for image implication understanding and
reasoning. LAD addresses contextual missing through the three-stage framework:
(1) Perception: converting visual information into rich and multi-level textual
representations, (2) Search: iteratively searching and integrating cross-domain
knowledge to resolve ambiguity, and (3) Reasoning: generating context-alignment
image implication via explicit reasoning. Our framework with the lightweight
GPT-4o-mini model achieves SOTA performance compared to 15+ MLLMs on English
image implication benchmark and a huge improvement on Chinese benchmark,
performing comparable with the GPT-4o model on Multiple-Choice Question (MCQ)
and outperforms 36.7% on Open-Style Question (OSQ). Additionally, our work
provides new insights into how AI can more effectively interpret image
implications, advancing the field of vision-language reasoning and human-AI
interaction. Our project is publicly available at
https://github.com/MING-ZCH/Let-Androids-Dream-of-Electric-Sheep.

---


### [Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO](http://arxiv.org/abs/2505.17017v1)

Recent advancements underscore the significant role of Reinforcement Learning
(RL) in enhancing the Chain-of-Thought (CoT) reasoning capabilities of large
language models (LLMs). Two prominent RL algorithms, Direct Preference
Optimization (DPO) and Group Relative Policy Optimization (GRPO), are central
to these developments, showcasing different pros and cons. Autoregressive image
generation, also interpretable as a sequential CoT reasoning process, presents
unique challenges distinct from LLM-based CoT reasoning. These encompass
ensuring text-image consistency, improving image aesthetic quality, and
designing sophisticated reward models, rather than relying on simpler
rule-based rewards. While recent efforts have extended RL to this domain, these
explorations typically lack an in-depth analysis of the domain-specific
challenges and the characteristics of different RL strategies. To bridge this
gap, we provide the first comprehensive investigation of the GRPO and DPO
algorithms in autoregressive image generation, evaluating their in-domain
performance and out-of-domain generalization, while scrutinizing the impact of
different reward models on their respective capabilities. Our findings reveal
that GRPO and DPO exhibit distinct advantages, and crucially, that reward
models possessing stronger intrinsic generalization capabilities potentially
enhance the generalization potential of the applied RL algorithms. Furthermore,
we systematically explore three prevalent scaling strategies to enhance both
their in-domain and out-of-domain proficiency, deriving unique insights into
efficiently scaling performance for each paradigm. We hope our study paves a
new path for inspiring future work on developing more effective RL algorithms
to achieve robust CoT reasoning in the realm of autoregressive image
generation. Code is released at
https://github.com/ZiyuGuo99/Image-Generation-CoT

---


### [R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning](http://arxiv.org/abs/2505.17005v1)

Large Language Models (LLMs) are powerful but prone to hallucinations due to
static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting
external information, but current methods often are costly, generalize poorly,
or ignore the internal knowledge of the model. In this paper, we introduce
R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage
both internal and external knowledge sources. R1-Searcher++ employs a two-stage
training strategy: an initial SFT Cold-start phase for preliminary format
learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses
outcome-supervision to encourage exploration, incorporates a reward mechanism
for internal knowledge utilization, and integrates a memorization mechanism to
continuously assimilate retrieved information, thereby enriching the model's
internal knowledge. By leveraging internal knowledge and external search
engine, the model continuously improves its capabilities, enabling efficient
retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++
outperforms previous RAG and reasoning methods and achieves efficient
retrieval. The code is available at
https://github.com/RUCAIBox/R1-Searcher-plus.

---


### [Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design](http://arxiv.org/abs/2505.16979v1)

Single-agent LLMs hit hard limits--finite context, role overload, and brittle
domain transfer. Conventional multi-agent fixes soften those edges yet expose
fresh pains: ill-posed decompositions, fuzzy contracts, and verification
overhead that blunts the gains. We therefore present Know-The-Ropes (KtR), a
framework that converts domain priors into an algorithmic blueprint hierarchy,
in which tasks are recursively split into typed, controller-mediated subtasks,
each solved zero-shot or with the lightest viable boost (e.g.,
chain-of-thought, micro-tune, self-check). Grounded in the No-Free-Lunch
theorem, KtR trades the chase for a universal prompt for disciplined
decomposition. On the Knapsack problem (3-8 items), three GPT-4o-mini agents
raise accuracy from 3% zero-shot to 95% on size-5 instances after patching a
single bottleneck agent. On the tougher Task-Assignment problem (6-15 jobs), a
six-agent o3-mini blueprint hits 100% up to size 10 and 84% on sizes 13-15,
versus 11% zero-shot. Algorithm-aware decomposition plus targeted augmentation
thus turns modest models into reliable collaborators--no ever-larger monoliths
required.

---


### [CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark](http://arxiv.org/abs/2505.16968v1)

We introduce \texttt{CASS}, the first large-scale dataset and model suite for
cross-architecture GPU code transpilation, targeting both source-level
(CUDA~$\leftrightarrow$~HIP) and assembly-level (Nvidia
SASS~$\leftrightarrow$~AMD RDNA3) translation. The dataset comprises 70k
verified code pairs across host and device, addressing a critical gap in
low-level GPU code portability. Leveraging this resource, we train the
\texttt{CASS} family of domain-specific language models, achieving 95\% source
translation accuracy and 37.5\% assembly translation accuracy, substantially
outperforming commercial baselines such as GPT-4o, Claude, and Hipify. Our
generated code matches native performance in over 85\% of test cases,
preserving runtime and memory behavior. To support rigorous evaluation, we
introduce \texttt{CASS-Bench}, a curated benchmark spanning 16 GPU domains with
ground-truth execution. All data, models, and evaluation tools are released as
open source to foster progress in GPU compiler tooling, binary compatibility,
and LLM-guided hardware translation. Dataset and benchmark are on
\href{https://huggingface.co/datasets/MBZUAI/cass}{\textcolor{blue}{HuggingFace}},
with code at
\href{https://github.com/GustavoStahl/CASS}{\textcolor{blue}{GitHub}}.

---


### [Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval](http://arxiv.org/abs/2505.16967v1)

Training robust retrieval and reranker models typically relies on large-scale
retrieval datasets; for example, the BGE collection contains 1.6 million
query-passage pairs sourced from various data sources. However, we find that
certain datasets can negatively impact model effectiveness -- pruning 8 out of
15 datasets from the BGE collection reduces the training set size by
2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a
deeper examination of training data quality, with a particular focus on "false
negatives", where relevant passages are incorrectly labeled as irrelevant. We
propose a simple, cost-effective approach using cascading LLM prompts to
identify and relabel hard negatives. Experimental results show that relabeling
false negatives with true positives improves both E5 (base) and Qwen2.5-7B
retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot
AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on
the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the
cascading design is further supported by human annotation results, where we
find judgment by GPT-4o shows much higher agreement with humans than
GPT-4o-mini.

---


### [The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm](http://arxiv.org/abs/2505.16932v1)

Computing the polar decomposition and the related matrix sign function, has
been a well-studied problem in numerical analysis for decades. More recently,
it has emerged as an important subroutine in deep learning, particularly within
the Muon optimization framework. However, the requirements in this setting
differ significantly from those of traditional numerical analysis. In deep
learning, methods must be highly efficient and GPU-compatible, but high
accuracy is often unnecessary. As a result, classical algorithms like
Newton-Schulz (which suffers from slow initial convergence) and methods based
on rational functions (which rely on QR decompositions or matrix inverses) are
poorly suited to this context. In this work, we introduce Polar Express, a
GPU-friendly algorithm for computing the polar decomposition. Like classical
polynomial methods such as Newton-Schulz, our approach uses only matrix-matrix
multiplications, making it GPU-compatible. Motivated by earlier work of Chen &
Chow and Nakatsukasa & Freund, Polar Express adapts the polynomial update rule
at each iteration by solving a minimax optimization problem, and we prove that
it enjoys a strong worst-case optimality guarantee. This property ensures both
rapid early convergence and fast asymptotic convergence. We also address
finite-precision issues, making it stable in bfloat16 in practice. We apply
Polar Express within the Muon optimization framework and show consistent
improvements in validation loss on large-scale models such as GPT-2,
outperforming recent alternatives across a range of learning rates.

---


### [Efficient Online RL Fine Tuning with Offline Pre-trained Policy Only](http://arxiv.org/abs/2505.16856v1)

Improving the performance of pre-trained policies through online
reinforcement learning (RL) is a critical yet challenging topic. Existing
online RL fine-tuning methods require continued training with offline
pretrained Q-functions for stability and performance. However, these offline
pretrained Q-functions commonly underestimate state-action pairs beyond the
offline dataset due to the conservatism in most offline RL methods, which
hinders further exploration when transitioning from the offline to the online
setting. Additionally, this requirement limits their applicability in scenarios
where only pre-trained policies are available but pre-trained Q-functions are
absent, such as in imitation learning (IL) pre-training. To address these
challenges, we propose a method for efficient online RL fine-tuning using
solely the offline pre-trained policy, eliminating reliance on pre-trained
Q-functions. We introduce PORL (Policy-Only Reinforcement Learning
Fine-Tuning), which rapidly initializes the Q-function from scratch during the
online phase to avoid detrimental pessimism. Our method not only achieves
competitive performance with advanced offline-to-online RL algorithms and
online RL approaches that leverage data or policies prior, but also pioneers a
new path for directly fine-tuning behavior cloning (BC) policies.

---


### [Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models](http://arxiv.org/abs/2505.16854v1)

Reinforcement Learning (RL) has proven to be an effective post-training
strategy for enhancing reasoning in vision-language models (VLMs). Group
Relative Policy Optimization (GRPO) is a recent prominent method that
encourages models to generate complete reasoning traces before answering,
leading to increased token usage and computational cost. Inspired by the
human-like thinking process-where people skip reasoning for easy questions but
think carefully when needed-we explore how to enable VLMs to first decide when
reasoning is necessary. To realize this, we propose TON, a two-stage training
strategy: (i) a supervised fine-tuning (SFT) stage with a simple yet effective
'thought dropout' operation, where reasoning traces are randomly replaced with
empty thoughts. This introduces a think-or-not format that serves as a cold
start for selective reasoning; (ii) a GRPO stage that enables the model to
freely explore when to think or not, while maximizing task-aware outcome
rewards. Experimental results show that TON can reduce the completion length by
up to 90% compared to vanilla GRPO, without sacrificing performance or even
improving it. Further evaluations across diverse vision-language tasks-covering
a range of reasoning difficulties under both 3B and 7B models-consistently
reveal that the model progressively learns to bypass unnecessary reasoning
steps as training advances. These findings shed light on the path toward
human-like reasoning patterns in reinforcement learning approaches. Our code is
available at https://github.com/kokolerk/TON.

---


### [Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning](http://arxiv.org/abs/2505.16836v1)

The rapid spread of multimodal misinformation on social media has raised
growing concerns, while research on video misinformation detection remains
limited due to the lack of large-scale, diverse datasets. Existing methods
often overfit to rigid templates and lack deep reasoning over deceptive
content. To address these challenges, we introduce FakeVV, a large-scale
benchmark comprising over 100,000 video-text pairs with fine-grained,
interpretable annotations. In addition, we further propose Fact-R1, a novel
framework that integrates deep reasoning with collaborative rule-based
reinforcement learning. Fact-R1 is trained through a three-stage process: (1)
misinformation long-Chain-of-Thought (CoT) instruction tuning, (2) preference
alignment via Direct Preference Optimization (DPO), and (3) Group Relative
Policy Optimization (GRPO) using a novel verifiable reward function. This
enables Fact-R1 to exhibit emergent reasoning behaviors comparable to those
observed in advanced text-based reinforcement learning systems, but in the more
complex multimodal misinformation setting. Our work establishes a new paradigm
for misinformation detection, bridging large-scale video understanding,
reasoning-guided alignment, and interpretable verification.

---


### [SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis](http://arxiv.org/abs/2505.16834v1)

Retrieval-augmented generation (RAG) systems have advanced large language
models (LLMs) in complex deep search scenarios requiring multi-step reasoning
and iterative information retrieval. However, existing approaches face critical
limitations that lack high-quality training trajectories or suffer from the
distributional mismatches in simulated environments and prohibitive
computational costs for real-world deployment. This paper introduces
SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap
through strategic data engineering rather than complex training paradigms. Our
approach synthesizes high-quality training data by simulating realistic user
interactions in live web search environments, coupled with a multi-criteria
curation strategy that optimizes the diversity and quality of input and output
side. Experiments on five benchmarks across diverse domains demonstrate that
SFT on only 871 curated samples yields significant improvements over RL-based
baselines. Our work establishes SFT as a viable pathway by systematically
addressing the data-scarce bottleneck, offering practical insights for
efficient deep search systems. Our code is available at
https://github.com/RUCAIBox/SimpleDeepSearcher.

---


### [GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](http://arxiv.org/abs/2505.16827v1)

GUI automation faces critical challenges in dynamic environments. MLLMs
suffer from two key issues: misinterpreting UI components and outdated
knowledge. Traditional fine-tuning methods are costly for app-specific
knowledge updates. We propose GUI-explorer, a training-free GUI agent that
incorporates two fundamental mechanisms: (1) Autonomous Exploration of
Function-aware Trajectory. To comprehensively cover all application
functionalities, we design a Function-aware Task Goal Generator that
automatically constructs exploration goals by analyzing GUI structural
information (e.g., screenshots and activity hierarchies). This enables
systematic exploration to collect diverse trajectories. (2) Unsupervised Mining
of Transition-aware Knowledge. To establish precise screen-operation logic, we
develop a Transition-aware Knowledge Extractor that extracts effective
screen-operation logic through unsupervised analysis the state transition of
structured interaction triples (observation, action, outcome). This eliminates
the need for human involvement in knowledge extraction. With a task success
rate of 53.7% on SPA-Bench and 47.4% on AndroidWorld, GUI-explorer shows
significant improvements over SOTA agents. It requires no parameter updates for
new apps. GUI-explorer is open-sourced and publicly available at
https://github.com/JiuTian-VL/GUI-explorer.

---


### [Data-Driven Breakthroughs and Future Directions in AI Infrastructure: A Comprehensive Review](http://arxiv.org/abs/2505.16771v1)

This paper presents a comprehensive synthesis of major breakthroughs in
artificial intelligence (AI) over the past fifteen years, integrating
historical, theoretical, and technological perspectives. It identifies key
inflection points in AI' s evolution by tracing the convergence of
computational resources, data access, and algorithmic innovation. The analysis
highlights how researchers enabled GPU based model training, triggered a data
centric shift with ImageNet, simplified architectures through the Transformer,
and expanded modeling capabilities with the GPT series. Rather than treating
these advances as isolated milestones, the paper frames them as indicators of
deeper paradigm shifts. By applying concepts from statistical learning theory
such as sample complexity and data efficiency, the paper explains how
researchers translated breakthroughs into scalable solutions and why the field
must now embrace data centric approaches. In response to rising privacy
concerns and tightening regulations, the paper evaluates emerging solutions
like federated learning, privacy enhancing technologies (PETs), and the data
site paradigm, which reframe data access and security. In cases where real
world data remains inaccessible, the paper also assesses the utility and
constraints of mock and synthetic data generation. By aligning technical
insights with evolving data infrastructure, this study offers strategic
guidance for future AI research and policy development.

---


### [TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning](http://arxiv.org/abs/2505.16743v1)

Large Language Models (LLMs) present significant computational and memory
challenges due to their extensive size, making pruning essential for their
efficient deployment. Existing one-shot pruning methods often apply uniform
sparsity constraints across layers or within each layer, resulting in
suboptimal performance, especially at high sparsity ratios. This work
introduces TRIM (Targeted Row-wise Iterative Metric-driven pruning), a novel
approach that applies varying sparsity ratios to individual output dimensions
(rows) within each layer. TRIM employs an iterative adjustment process guided
by quality metrics to optimize dimension-wise sparsity allocation, focusing on
reducing variance in quality retention across outputs to preserve critical
information. TRIM can be seamlessly integrated with existing layer-wise pruning
strategies. Our evaluations on perplexity and zero-shot tasks across diverse
LLM families (Qwen2.5, LLaMA-2, and OPT) and sparsity levels demonstrate that
TRIM achieves new state-of-the-art results and enhances stability. For
instance, at 80% sparsity, TRIM reduces perplexity by 48% for Qwen2.5-14B and
over 90% for OPT-13B compared to baseline methods. We conclude that
fine-grained, dimension-wise sparsity adaptation is crucial for pushing the
limits of extreme LLM compression. Code available at:
https://github.com/flobk/TRIM

---


### [Training Long-Context LLMs Efficiently via Chunk-wise Optimization](http://arxiv.org/abs/2505.16710v1)

While long-context large language models (LLMs) exhibit remarkable document
processing capabilities, their prohibitively high training costs often hinder
customized applications. To mitigate this issue, we propose \textit{Sequential
Chunk-wise Optimization} (SeCO), a memory-efficient training paradigm that
partitions lengthy inputs into manageable chunks. Each chunk independently
constructs its computational graph and performs localized backpropagation,
ensuring that only one chunk's forward activations are stored in memory.
Building on SeCO, we further introduce \textit{Sparse Chunk-wise Optimization}
(SpaCO), which reduces computational overhead by selectively propagating
gradients to specific chunks and incorporates a carefully designed compensation
factor to ensure unbiased gradient estimation. SpaCO decouples the
computational cost of backpropagation from the context length, enabling
training time to gradually converge to inference time as sequences become
longer. Implemented as lightweight training wrappers, both SeCO and SpaCO offer
substantial practical benefits. For example, when fine-tuning an 8B model with
LoRA on a single RTX 3090 GPU, SeCO expands maximum sequence length from 1K to
16K tokens, while SpaCO demonstrates accelerated training speed -- achieving up
to 3x faster than SeCO under the same experimental setup. These innovations
provide new insights into optimizing long-context models, making them more
accessible for practical applications. We have open-sourced the code at
\href{https://github.com/wenhaoli-xmu/seco}{here}.

---


### [Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence](http://arxiv.org/abs/2505.16694v1)

Transformer-based language models exhibit In-Context Learning (ICL), where
predictions are made adaptively based on context. While prior work links
induction heads to ICL through a sudden jump in accuracy, this can only account
for ICL when the answer is included within the context. However, an important
property of practical ICL in large language models is the ability to meta-learn
how to solve tasks from context, rather than just copying answers from context;
how such an ability is obtained during training is largely unexplored. In this
paper, we experimentally clarify how such meta-learning ability is acquired by
analyzing the dynamics of the model's circuit during training. Specifically, we
extend the copy task from previous research into an In-Context Meta Learning
setting, where models must infer a task from examples to answer queries.
Interestingly, in this setting, we find that there are multiple phases in the
process of acquiring such abilities, and that a unique circuit emerges in each
phase, contrasting with the single-phases change in induction heads. The
emergence of such circuits can be related to several phenomena known in large
language models, and our analysis lead to a deeper understanding of the source
of the transformer's ICL ability.

---


### [Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](http://arxiv.org/abs/2505.16690v1)

Post-training of large language models is essential for adapting pre-trained
language models (PLMs) to align with human preferences and downstream tasks.
While PLMs typically exhibit well-calibrated confidence, post-trained language
models (PoLMs) often suffer from over-confidence, assigning high confidence to
both correct and incorrect outputs, which can undermine reliability in critical
applications. A major obstacle in calibrating PoLMs is the scarcity of labeled
data for individual downstream tasks. To address this, we propose
Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to
optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence
calibration. Our method is motivated by the under-confidence issue caused by
prediction disagreement between the PLM and PoLM while aligning their
confidence via temperature scaling. Theoretically, the PLM's confidence
underestimates PoLM's prediction accuracy on disagreement examples, causing a
larger $\tau$ and producing under-confident predictions. DACA mitigates this by
selectively using only agreement examples for calibration, effectively
decoupling the influence of disagreement. In this manner, our method avoids an
overly large $\tau$ in temperature scaling caused by disagreement examples,
improving calibration performance. Extensive experiments demonstrate the
effectiveness of our method, improving the average ECE of open-sourced and
API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks.

---


### [R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO](http://arxiv.org/abs/2505.16673v1)

In this work, we aim to incentivize the reasoning ability of Multimodal Large
Language Models (MLLMs) via reinforcement learning (RL) and develop an
effective approach that mitigates the sparse reward and advantage vanishing
issues during RL. To this end, we propose Share-GRPO, a novel RL approach that
tackle these issues by exploring and sharing diverse reasoning trajectories
over expanded question space. Specifically, Share-GRPO first expands the
question space for a given question via data transformation techniques, and
then encourages MLLM to effectively explore diverse reasoning trajectories over
the expanded question space and shares the discovered reasoning trajectories
across the expanded questions during RL. In addition, Share-GRPO also shares
reward information during advantage computation, which estimates solution
advantages hierarchically across and within question variants, allowing more
accurate estimation of relative advantages and improving the stability of
policy training. Extensive evaluations over six widely-used reasoning
benchmarks showcase the superior performance of our method. Code will be
available at https://github.com/HJYao00/R1-ShareVL.

---


### [BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models](http://arxiv.org/abs/2505.16670v1)

Large language models (LLMs) have shown impressive capabilities across a wide
range of applications, but their ever-increasing size and resource demands make
them vulnerable to inference cost attacks, where attackers induce victim LLMs
to generate the longest possible output content. In this paper, we revisit
existing inference cost attacks and reveal that these methods can hardly
produce large-scale malicious effects since they are self-targeting, where
attackers are also the users and therefore have to execute attacks solely
through the inputs, whose generated content will be charged by LLMs and can
only directly influence themselves. Motivated by these findings, this paper
introduces a new type of inference cost attacks (dubbed 'bit-flip inference
cost attack') that target the victim model itself rather than its inputs.
Specifically, we design a simple yet effective method (dubbed 'BitHydra') to
effectively flip critical bits of model parameters. This process is guided by a
loss function designed to suppress <EOS> token's probability with an efficient
critical bit search algorithm, thus explicitly defining the attack objective
and enabling effective optimization. We evaluate our method on 11 LLMs ranging
from 1.5B to 14B parameters under both int8 and float16 settings. Experimental
results demonstrate that with just 4 search samples and as few as 3 bit flips,
BitHydra can force 100% of test prompts to reach the maximum generation length
(e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its
efficiency, scalability, and strong transferability across unseen inputs.

---


### [Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models](http://arxiv.org/abs/2505.16647v1)

We investigate fine-tuning Vision-Language Models (VLMs) for multi-task
medical image understanding, focusing on detection, localization, and counting
of findings in medical images. Our objective is to evaluate whether
instruction-tuned VLMs can simultaneously improve these tasks, with the goal of
enhancing diagnostic accuracy and efficiency. Using MedMultiPoints, a
multimodal dataset with annotations from endoscopy (polyps and instruments) and
microscopy (sperm cells), we reformulate each task into instruction-based
prompts suitable for vision-language reasoning. We fine-tune
Qwen2.5-VL-7B-Instruct using Low-Rank Adaptation (LoRA) across multiple task
combinations. Results show that multi-task training improves robustness and
accuracy. For example, it reduces the Count Mean Absolute Error (MAE) and
increases Matching Accuracy in the Counting + Pointing task. However,
trade-offs emerge, such as more zero-case point predictions, indicating reduced
reliability in edge cases despite overall performance gains. Our study
highlights the potential of adapting general-purpose VLMs to specialized
medical tasks via prompt-driven fine-tuning. This approach mirrors clinical
workflows, where radiologists simultaneously localize, count, and describe
findings - demonstrating how VLMs can learn composite diagnostic reasoning
patterns. The model produces interpretable, structured outputs, offering a
promising step toward explainable and versatile medical AI. Code, model
weights, and scripts will be released for reproducibility at
https://github.com/simula/PointDetectCount.

---


### [From Evaluation to Defense: Advancing Safety in Video Large Language Models](http://arxiv.org/abs/2505.16643v1)

While the safety risks of image-based large language models have been
extensively studied, their video-based counterparts (Video LLMs) remain
critically under-examined. To systematically study this problem, we introduce
\textbf{VideoSafetyBench (VSB-77k) - the first large-scale, culturally diverse
benchmark for Video LLM safety}, which compromises 77,646 video-query pairs and
spans 19 principal risk categories across 10 language communities. \textit{We
reveal that integrating video modality degrades safety performance by an
average of 42.3\%, exposing systemic risks in multimodal attack exploitation.}
To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage
framework achieving unprecedented safety gains through two innovations: (1)
Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens
into visual and textual sequences, enabling explicit harm perception across
modalities via multitask objectives. (2) Then, Safety-Guided GRPO enhances
defensive reasoning through dynamic policy optimization with rule-based rewards
derived from dual-modality verification. These components synergize to shift
safety alignment from passive harm recognition to active reasoning. The
resulting framework achieves a 65.1\% improvement on VSB-Eval-HH, and improves
by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard,
and FigStep, respectively. \textit{Our codes are available in the supplementary
materials.} \textcolor{red}{Warning: This paper contains examples of harmful
language and videos, and reader discretion is recommended.}

---


### [BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization](http://arxiv.org/abs/2505.16640v1)

Vision-Language-Action (VLA) models have advanced robotic control by enabling
end-to-end decision-making directly from multimodal inputs. However, their
tightly coupled architectures expose novel security vulnerabilities. Unlike
traditional adversarial perturbations, backdoor attacks represent a stealthier,
persistent, and practically significant threat-particularly under the emerging
Training-as-a-Service paradigm-but remain largely unexplored in the context of
VLA models. To address this gap, we propose BadVLA, a backdoor attack method
based on Objective-Decoupled Optimization, which for the first time exposes the
backdoor vulnerabilities of VLA models. Specifically, it consists of a
two-stage process: (1) explicit feature-space separation to isolate trigger
representations from benign inputs, and (2) conditional control deviations that
activate only in the presence of the trigger, while preserving clean-task
performance. Empirical results on multiple VLA benchmarks demonstrate that
BadVLA consistently achieves near-100% attack success rates with minimal impact
on clean task accuracy. Further analyses confirm its robustness against common
input perturbations, task transfers, and model fine-tuning, underscoring
critical security vulnerabilities in current VLA deployments. Our work offers
the first systematic investigation of backdoor vulnerabilities in VLA models,
highlighting an urgent need for secure and trustworthy embodied model design
practices. We have released the project page at
https://badvla-project.github.io/.

---


### [SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation](http://arxiv.org/abs/2505.16637v1)

Large language models (LLMs) have recently demonstrated remarkable
capabilities in machine translation (MT). However, most advanced MT-specific
LLMs heavily rely on external supervision signals during training, such as
human-annotated reference data or trained reward models (RMs), which are often
expensive to obtain and challenging to scale. To overcome this limitation, we
propose a Simple Self-Rewarding (SSR) Reinforcement Learning (RL) framework for
MT that is reference-free, fully online, and relies solely on self-judging
rewards. Training with SSR using 13K monolingual examples and Qwen-2.5-7B as
the backbone, our model SSR-Zero-7B outperforms existing MT-specific LLMs,
e.g., TowerInstruct-13B and GemmaX-28-9B, as well as larger general LLMs like
Qwen2.5-32B-Instruct in English $\leftrightarrow$ Chinese translation tasks
from WMT23, WMT24, and Flores200 benchmarks. Furthermore, by augmenting SSR
with external supervision from COMET, our strongest model, SSR-X-Zero-7B,
achieves state-of-the-art performance in English $\leftrightarrow$ Chinese
translation, surpassing all existing open-source models under 72B parameters
and even outperforming closed-source models, e.g., GPT-4o and Gemini 1.5 Pro.
Our analysis highlights the effectiveness of the self-rewarding mechanism
compared to the external LLM-as-a-judge approach in MT and demonstrates its
complementary benefits when combined with trained RMs. Our findings provide
valuable insight into the potential of self-improving RL methods. We have
publicly released our code, data and models.

---


### [Bridging the Dynamic Perception Gap: Training-Free Draft Chain-of-Thought for Dynamic Multimodal Spatial Reasoning](http://arxiv.org/abs/2505.16579v1)

While chains-of-thought (CoT) have advanced complex reasoning in multimodal
large language models (MLLMs), existing methods remain confined to text or
static visual domains, often faltering in dynamic spatial reasoning tasks. To
bridge this gap, we present GRASSLAND, a novel maze navigation benchmark
designed to evaluate dynamic spatial reasoning. Our experiments show that
augmenting textual reasoning chains with dynamic visual drafts, overlaid on
input images, significantly outperforms conventional approaches, offering new
insights into spatial reasoning in evolving environments. To generalize this
capability, we propose D2R (Dynamic Draft-Augmented Reasoning), a training-free
framework that seamlessly integrates textual CoT with corresponding visual
drafts into MLLMs. Extensive evaluations demonstrate that D2R consistently
enhances performance across diverse tasks, establishing a robust baseline for
dynamic spatial reasoning without requiring model fine-tuning. Project is open
at https://github.com/Cratileo/D2R.

---


### [Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning](http://arxiv.org/abs/2505.16483v1)

Teaching large language models (LLMs) to be faithful in the provided context
is crucial for building reliable information-seeking systems. Therefore, we
propose a systematic framework, CANOE, to improve the faithfulness of LLMs in
both short-form and long-form generation tasks without human annotations.
Specifically, we first synthesize short-form question-answering (QA) data with
four diverse tasks to construct high-quality and easily verifiable training
data without human annotation. Also, we propose Dual-GRPO, a rule-based
reinforcement learning method that includes three tailored rule-based rewards
derived from synthesized short-form QA data, while simultaneously optimizing
both short-form and long-form response generation. Notably, Dual-GRPO
eliminates the need to manually label preference data to train reward models
and avoids over-optimizing short-form generation when relying only on the
synthesized short-form QA data. Experimental results show that CANOE greatly
improves the faithfulness of LLMs across 11 different downstream tasks, even
outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1.

---


### [MMMR: Benchmarking Massive Multi-Modal Reasoning Tasks](http://arxiv.org/abs/2505.16459v1)

Recent advances in Multi-Modal Large Language Models (MLLMs) have enabled
unified processing of language, vision, and structured inputs, opening the door
to complex tasks such as logical deduction, spatial reasoning, and scientific
analysis. Despite their promise, the reasoning capabilities of MLLMs,
particularly those augmented with intermediate thinking traces (MLLMs-T),
remain poorly understood and lack standardized evaluation benchmarks. Existing
work focuses primarily on perception or final answer correctness, offering
limited insight into how models reason or fail across modalities. To address
this gap, we introduce the MMMR, a new benchmark designed to rigorously
evaluate multi-modal reasoning with explicit thinking. The MMMR comprises 1) a
high-difficulty dataset of 1,083 questions spanning six diverse reasoning types
with symbolic depth and multi-hop demands and 2) a modular Reasoning Trace
Evaluation Pipeline (RTEP) for assessing reasoning quality beyond accuracy
through metrics like relevance, consistency, and structured error annotations.
Empirical results show that MLLMs-T overall outperform non-thinking
counterparts, but even top models like Claude-3.7-Sonnet and Gemini-2.5 Pro
suffer from reasoning pathologies such as inconsistency and overthinking. This
benchmark reveals persistent gaps between accuracy and reasoning quality and
provides an actionable evaluation pipeline for future model development.
Overall, the MMMR offers a scalable foundation for evaluating, comparing, and
improving the next generation of multi-modal reasoning systems.

---


### [Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models](http://arxiv.org/abs/2505.16416v1)

Rotary Position Embedding (RoPE) is a widely adopted technique for encoding
relative positional information in large language models (LLMs). However, when
extended to large vision-language models (LVLMs), its variants introduce
unintended cross-modal positional biases. Specifically, they enforce relative
positional dependencies between text token indices and image tokens, causing
spurious alignments. This issue arises because image tokens representing the
same content but located at different spatial positions are assigned distinct
positional biases, leading to inconsistent cross-modal associations. To address
this, we propose Per-Token Distance (PTD) - a simple yet effective metric for
quantifying the independence of positional encodings across modalities.
Informed by this analysis, we introduce Circle-RoPE, a novel encoding scheme
that maps image token indices onto a circular trajectory orthogonal to the
linear path of text token indices, forming a cone-like structure. This
configuration ensures that each text token maintains an equal distance to all
image tokens, reducing artificial cross-modal biases while preserving
intra-image spatial information. To further enhance performance, we propose a
staggered layer strategy that applies different RoPE variants across layers.
This design leverages the complementary strengths of each RoPE variant, thereby
enhancing the model's overall performance. Our experimental results demonstrate
that our method effectively preserves spatial information from images while
reducing relative positional bias, offering a more robust and flexible
positional encoding framework for LVLMs. The code is available at
[https://github.com/lose4578/CircleRoPE](https://github.com/lose4578/CircleRoPE).

---


### [Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](http://arxiv.org/abs/2505.16415v1)

Retrieval-Augmented Generation (RAG) leverages large language models (LLMs)
combined with external contexts to enhance the accuracy and reliability of
generated responses. However, reliably attributing generated content to
specific context segments, context attribution, remains challenging due to the
computationally intensive nature of current methods, which often require
extensive fine-tuning or human annotation. In this work, we introduce a novel
Jensen-Shannon Divergence driven method to Attribute Response to Context
(ARC-JSD), enabling efficient and accurate identification of essential context
sentences without additional fine-tuning or surrogate modelling. Evaluations on
a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using
instruction-tuned LLMs in different scales demonstrate superior accuracy and
significant computational efficiency improvements compared to the previous
surrogate-based method. Furthermore, our mechanistic analysis reveals specific
attention heads and multilayer perceptron (MLP) layers responsible for context
attribution, providing valuable insights into the internal workings of RAG
models.

---


### [Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning](http://arxiv.org/abs/2505.16410v1)

Recently, large language models (LLMs) have shown remarkable reasoning
capabilities via large-scale reinforcement learning (RL). However, leveraging
the RL algorithm to empower effective multi-tool collaborative reasoning in
LLMs remains an open challenge. In this paper, we introduce Tool-Star, an
RL-based framework designed to empower LLMs to autonomously invoke multiple
external tools during stepwise reasoning. Tool-Star integrates six types of
tools and incorporates systematic designs in both data synthesis and training.
To address the scarcity of tool-use data, we propose a general tool-integrated
reasoning data synthesis pipeline, which combines tool-integrated prompting
with hint-based sampling to automatically and scalably generate tool-use
trajectories. A subsequent quality normalization and difficulty-aware
classification process filters out low-quality samples and organizes the
dataset from easy to hard. Furthermore, we propose a two-stage training
framework to enhance multi-tool collaborative reasoning by: (1) cold-start
fine-tuning, which guides LLMs to explore reasoning patterns via
tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with
hierarchical reward design, which reinforces reward understanding and promotes
effective tool collaboration. Experimental analyses on over 10 challenging
reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star.
The code is available at https://github.com/dongguanting/Tool-Star.

---


### [Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)](http://arxiv.org/abs/2505.16394v1)

Reinforcement Learning (RL) can mitigate the causal confusion and
distribution shift inherent to imitation learning (IL). However, applying RL to
end-to-end autonomous driving (E2E-AD) remains an open problem for its training
difficulty, and IL is still the mainstream paradigm in both academia and
industry. Recently Model-based Reinforcement Learning (MBRL) have demonstrated
promising results in neural planning; however, these methods typically require
privileged information as input rather than raw sensor data. We fill this gap
by designing Raw2Drive, a dual-stream MBRL approach. Initially, we efficiently
train an auxiliary privileged world model paired with a neural planner that
uses privileged information as input. Subsequently, we introduce a raw sensor
world model trained via our proposed Guidance Mechanism, which ensures
consistency between the raw sensor world model and the privileged world model
during rollouts. Finally, the raw sensor world model combines the prior
knowledge embedded in the heads of the privileged world model to effectively
guide the training of the raw sensor policy. Raw2Drive is so far the only RL
based end-to-end method on CARLA Leaderboard 2.0, and Bench2Drive and it
achieves state-of-the-art performance.

---


### [VL-SAFE: Vision-Language Guided Safety-Aware Reinforcement Learning with World Models for Autonomous Driving](http://arxiv.org/abs/2505.16377v1)

Reinforcement learning (RL)-based autonomous driving policy learning faces
critical limitations such as low sample efficiency and poor generalization; its
reliance on online interactions and trial-and-error learning is especially
unacceptable in safety-critical scenarios. Existing methods including safe RL
often fail to capture the true semantic meaning of "safety" in complex driving
contexts, leading to either overly conservative driving behavior or constraint
violations. To address these challenges, we propose VL-SAFE, a world
model-based safe RL framework with Vision-Language model
(VLM)-as-safety-guidance paradigm, designed for offline safe policy learning.
Specifically, we construct offline datasets containing data collected by expert
agents and labeled with safety scores derived from VLMs. A world model is
trained to generate imagined rollouts together with safety estimations,
allowing the agent to perform safe planning without interacting with the real
environment. Based on these imagined trajectories and safety evaluations,
actor-critic learning is conducted under VLM-based safety guidance to optimize
the driving policy more safely and efficiently. Extensive evaluations
demonstrate that VL-SAFE achieves superior sample efficiency, generalization,
safety, and overall performance compared to existing baselines. To the best of
our knowledge, this is the first work that introduces a VLM-guided world
model-based approach for safe autonomous driving. The demo video and code can
be accessed at: https://ys-qu.github.io/vlsafe-website/

---


### [SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning](http://arxiv.org/abs/2505.16368v1)

How to design reinforcement learning (RL) tasks that effectively unleash the
reasoning capability of large language models (LLMs) remains an open question.
Existing RL tasks (e.g., math, programming, and constructing reasoning tasks)
suffer from three key limitations: (1) Scalability. They rely heavily on human
annotation or expensive LLM synthesis to generate sufficient training data. (2)
Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3)
Controllable Difficulty. Most tasks lack fine-grained difficulty control,
making it hard to train LLMs to develop reasoning ability from easy to hard.
  To address these limitations, we propose Saturn, a SAT-based RL framework
that uses Boolean Satisfiability (SAT) problems to train and evaluate LLM
reasoning. Saturn enables scalable task construction, rule-based verification,
and precise difficulty control. Saturn designs a curriculum learning pipeline
that continuously improves LLMs' reasoning capability by constructing SAT tasks
of increasing difficulty and training LLMs from easy to hard. To ensure stable
training, we design a principled mechanism to control difficulty transitions.
  We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying
difficulty. It supports the evaluation of how LLM reasoning changes with
problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain
Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT
problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of
+14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B
and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g.,
AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in
constructing RL tasks, Saturn achieves further improvements of +8.8%. We
release the source code, data, and models to support future research.

---


### [AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training](http://arxiv.org/abs/2505.16363v1)

We introduce AdamS, a simple yet effective alternative to Adam for large
language model (LLM) pretraining and post-training. By leveraging a novel
denominator, i.e., the root of weighted sum of squares of the momentum and the
current gradient, AdamS eliminates the need for second-moment estimates. Hence,
AdamS is efficient, matching the memory and compute footprint of SGD with
momentum while delivering superior optimization performance. Moreover, AdamS is
easy to adopt: it can directly inherit hyperparameters of AdamW, and is
entirely model-agnostic, integrating seamlessly into existing pipelines without
modifications to optimizer APIs or architectures. The motivation behind AdamS
stems from the observed $(L_0, L_1)$ smoothness properties in transformer
objectives, where local smoothness is governed by gradient magnitudes that can
be further approximated by momentum magnitudes. We establish rigorous
theoretical convergence guarantees and provide practical guidelines for
hyperparameter selection. Empirically, AdamS demonstrates strong performance in
various tasks, including pre-training runs on GPT-2 and Llama2 (up to 13B
parameters) and reinforcement learning in post-training regimes. With its
efficiency, simplicity, and theoretical grounding, AdamS stands as a compelling
alternative to existing optimizers.

---


### [FPQVAR: Floating Point Quantization for Visual Autoregressive Model with FPGA Hardware Co-design](http://arxiv.org/abs/2505.16335v1)

Visual autoregressive (VAR) modeling has marked a paradigm shift in image
generation from next-token prediction to next-scale prediction. VAR predicts a
set of tokens at each step from coarse to fine scale, leading to better image
quality and faster inference speed compared to existing diffusion models.
However, the large parameter size and computation cost hinder its deployment on
edge devices. To reduce the memory and computation cost, we propose FPQVAR, an
efficient post-training floating-point (FP) quantization framework for VAR
featuring algorithm and hardware co-design. At the algorithm level, we first
identify the challenges of quantizing VAR. To address them, we propose Dual
Format Quantization for the highly imbalanced input activation. We further
propose Group-wise Hadamard Transformation and GHT-Aware Learnable
Transformation to address the time-varying outlier channels. At the hardware
level, we design the first low-bit FP quantizer and multiplier with lookup
tables on FPGA and propose the first FPGA-based VAR accelerator featuring
low-bit FP computation and an elaborate two-level pipeline. Extensive
experiments show that compared to the state-of-the-art quantization method, our
proposed FPQVAR significantly improves Fr\'echet Inception Distance (FID) from
10.83 to 3.58, Inception Score (IS) from 175.9 to 241.5 under 4-bit
quantization. FPQVAR also significantly improves the performance of 6-bit
quantized VAR, bringing it on par with the FP16 model. Our accelerator on
AMD-Xilinx VCK190 FPGA achieves a throughput of 1.1 image/s, which is 3.1x
higher than the integer-based accelerator. It also demonstrates 3.6x and 2.8x
higher energy efficiency compared to the integer-based accelerator and GPU
baseline, respectively.

---


### [Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning](http://arxiv.org/abs/2505.16315v1)

Large reasoning models (LRMs) have demonstrated strong performance on complex
reasoning tasks, but often suffer from overthinking, generating redundant
content regardless of task difficulty. Inspired by the dual process theory in
cognitive science, we propose Adaptive Cognition Policy Optimization (ACPO), a
reinforcement learning framework that enables LRMs to achieve efficient
reasoning through adaptive cognitive allocation and dynamic system switch. ACPO
incorporates two key components: (1) introducing system-aware reasoning tokens
to explicitly represent the thinking modes thereby making the model's cognitive
process transparent, and (2) integrating online difficulty estimation and token
length budget to guide adaptive system switch and reasoning during
reinforcement learning. To this end, we propose a two-stage training strategy.
The first stage begins with supervised fine-tuning to cold start the model,
enabling it to generate reasoning paths with explicit thinking modes. In the
second stage, we apply ACPO to further enhance adaptive system switch for
difficulty-aware reasoning. Experimental results demonstrate that ACPO
effectively reduces redundant reasoning while adaptively adjusting cognitive
allocation based on task complexity, achieving efficient hybrid reasoning.

---


### [PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models](http://arxiv.org/abs/2505.16307v1)

Prompt optimization offers a practical and broadly applicable alternative to
fine-tuning for improving large language model (LLM) performance. However,
existing methods often rely on costly output generation, self-critiquing
abilities, or human-annotated preferences, which limit their scalability,
especially for smaller or non-instruction-tuned models. We introduce PMPO
(Probabilistic Metric Prompt Optimization), a unified framework that refines
prompts using token-level cross-entropy loss as a direct, lightweight
evaluation signal. PMPO identifies low-quality prompt segments by masking and
measuring their impact on loss, then rewrites and selects improved variants by
minimizing loss over positive and negative examples. Unlike prior methods, it
requires no output sampling or human evaluation during optimization, relying
only on forward passes and log-likelihoods. PMPO supports both supervised and
preference-based tasks through a closely aligned loss-based evaluation
strategy. Experiments show that PMPO consistently outperforms prior methods
across model sizes and tasks: it achieves the highest average accuracy on BBH,
performs strongly on GSM8K and AQUA-RAT, and improves AlpacaEval 2.0 win rates
by over 19 points. These results highlight PMPO's effectiveness, efficiency,
and broad applicability.

---


### [Multimodal Generative AI for Story Point Estimation in Software Development](http://arxiv.org/abs/2505.16290v1)

This research explores the application of Multimodal Generative AI to enhance
story point estimation in Agile software development. By integrating text,
image, and categorical data using advanced models like BERT, CNN, and XGBoost,
our approach surpasses the limitations of traditional single-modal estimation
methods. The results demonstrate strong accuracy for simpler story points,
while also highlighting challenges in more complex categories due to data
imbalance. This study further explores the impact of categorical data,
particularly severity, on the estimation process, emphasizing its influence on
model performance. Our findings emphasize the transformative potential of
multimodal data integration in refining AI-driven project management, paving
the way for more precise, adaptable, and domain-specific AI capabilities.
Additionally, this work outlines future directions for addressing data
variability and enhancing the robustness of AI in Agile methodologies.

---


### [DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](http://arxiv.org/abs/2505.16278v1)

End-to-end autonomous driving (E2E-AD) demands effective processing of
multi-view sensory data and robust handling of diverse and complex driving
scenarios, particularly rare maneuvers such as aggressive turns. Recent success
of Mixture-of-Experts (MoE) architecture in Large Language Models (LLMs)
demonstrates that specialization of parameters enables strong scalability. In
this work, we propose DriveMoE, a novel MoE-based E2E-AD framework, with a
Scene-Specialized Vision MoE and a Skill-Specialized Action MoE. DriveMoE is
built upon our $\pi_0$ Vision-Language-Action (VLA) baseline (originally from
the embodied AI field), called Drive-$\pi_0$. Specifically, we add Vision MoE
to Drive-$\pi_0$ by training a router to select relevant cameras according to
the driving context dynamically. This design mirrors human driving cognition,
where drivers selectively attend to crucial visual cues rather than
exhaustively processing all visual information. In addition, we add Action MoE
by training another router to activate specialized expert modules for different
driving behaviors. Through explicit behavioral specialization, DriveMoE is able
to handle diverse scenarios without suffering from modes averaging like
existing models. In Bench2Drive closed-loop evaluation experiments, DriveMoE
achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness
of combining vision and action MoE in autonomous driving tasks. We will release
our code and models of DriveMoE and Drive-$\pi_0$.

---


### [LIFEBench: Evaluating Length Instruction Following in Large Language Models](http://arxiv.org/abs/2505.16234v1)

While large language models (LLMs) can solve PhD-level reasoning problems
over long context inputs, they still struggle with a seemingly simpler task:
following explicit length instructions-e.g., write a 10,000-word novel.
Additionally, models often generate far too short outputs, terminate
prematurely, or even refuse the request. Existing benchmarks focus primarily on
evaluating generations quality, but often overlook whether the generations meet
length constraints. To this end, we introduce Length Instruction Following
Evaluation Benchmark (LIFEBench) to comprehensively evaluate LLMs' ability to
follow length instructions across diverse tasks and a wide range of specified
lengths. LIFEBench consists of 10,800 instances across 4 task categories in
both English and Chinese, covering length constraints ranging from 16 to 8192
words. We evaluate 26 widely-used LLMs and find that most models reasonably
follow short-length instructions but deteriorate sharply beyond a certain
threshold. Surprisingly, almost all models fail to reach the vendor-claimed
maximum output lengths in practice, as further confirmed by our evaluations
extending up to 32K words. Even long-context LLMs, despite their extended
input-output windows, counterintuitively fail to improve length-instructions
following. Notably, Reasoning LLMs outperform even specialized long-text
generation models, achieving state-of-the-art length following. Overall,
LIFEBench uncovers fundamental limitations in current LLMs' length instructions
following ability, offering critical insights for future progress.

---


### [Explain Less, Understand More: Jargon Detection via Personalized Parameter-Efficient Fine-tuning](http://arxiv.org/abs/2505.16227v1)

Personalizing jargon detection and explanation is essential for making
technical documents accessible to readers with diverse disciplinary
backgrounds. However, tailoring models to individual users typically requires
substantial annotation efforts and computational resources due to user-specific
finetuning. To address this, we present a systematic study of personalized
jargon detection, focusing on methods that are both efficient and scalable for
real-world deployment. We explore two personalization strategies: (1)
lightweight fine-tuning using Low-Rank Adaptation (LoRA) on open-source models,
and (2) personalized prompting, which tailors model behavior at inference time
without retaining. To reflect realistic constraints, we also investigate hybrid
approaches that combine limited annotated data with unsupervised user
background signals. Our personalized LoRA model outperforms GPT-4 by 21.4% in
F1 score and exceeds the best performing oracle baseline by 8.3%. Remarkably,
our method achieves comparable performance using only 10% of the annotated
training data, demonstrating its practicality for resource-constrained
settings. Our study offers the first work to systematically explore efficient,
low-resource personalization of jargon detection using open-source language
models, offering a practical path toward scalable, user-adaptive NLP system.

---


### [MAPLE: Many-Shot Adaptive Pseudo-Labeling for In-Context Learning](http://arxiv.org/abs/2505.16225v1)

In-Context Learning (ICL) empowers Large Language Models (LLMs) to tackle
diverse tasks by incorporating multiple input-output examples, known as
demonstrations, into the input of LLMs. More recently, advancements in the
expanded context windows of LLMs have led to many-shot ICL, which uses hundreds
of demonstrations and outperforms few-shot ICL, which relies on fewer examples.
However, this approach is often hindered by the high cost of obtaining large
amounts of labeled data. To address this challenge, we propose Many-Shot
Adaptive Pseudo-LabEling, namely MAPLE, a novel influence-based many-shot ICL
framework that utilizes pseudo-labeled samples to compensate for the lack of
label information. We first identify a subset of impactful unlabeled samples
and perform pseudo-labeling on them by querying LLMs. These pseudo-labeled
samples are then adaptively selected and tailored to each test query as input
to improve the performance of many-shot ICL, without significant labeling
costs. Extensive experiments on real-world datasets demonstrate the
effectiveness of our framework, showcasing its ability to enhance LLM
adaptability and performance with limited labeled data.

---


### [SpecMaskFoley: Steering Pretrained Spectral Masked Generative Transformer Toward Synchronized Video-to-audio Synthesis via ControlNet](http://arxiv.org/abs/2505.16195v1)

Foley synthesis aims to synthesize high-quality audio that is both
semantically and temporally aligned with video frames. Given its broad
application in creative industries, the task has gained increasing attention in
the research community. To avoid the non-trivial task of training audio
generative models from scratch, adapting pretrained audio generative models for
video-synchronized foley synthesis presents an attractive direction.
ControlNet, a method for adding fine-grained controls to pretrained generative
models, has been applied to foley synthesis, but its use has been limited to
handcrafted human-readable temporal conditions. In contrast, from-scratch
models achieved success by leveraging high-dimensional deep features extracted
using pretrained video encoders. We have observed a performance gap between
ControlNet-based and from-scratch foley models. To narrow this gap, we propose
SpecMaskFoley, a method that steers the pretrained SpecMaskGIT model toward
video-synchronized foley synthesis via ControlNet. To unlock the potential of a
single ControlNet branch, we resolve the discrepancy between the temporal video
features and the time-frequency nature of the pretrained SpecMaskGIT via a
frequency-aware temporal feature aligner, eliminating the need for complicated
conditioning mechanisms widely used in prior arts. Evaluations on a common
foley synthesis benchmark demonstrate that SpecMaskFoley could even outperform
strong from-scratch baselines, substantially advancing the development of
ControlNet-based foley synthesis models. Demo page:
https://zzaudio.github.io/SpecMaskFoley_Demo/

---


### [VLM-R$^3$: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought](http://arxiv.org/abs/2505.16192v1)

Recently, reasoning-based MLLMs have achieved a degree of success in
generating long-form textual reasoning chains. However, they still struggle
with complex tasks that necessitate dynamic and iterative focusing on and
revisiting of visual regions to achieve precise grounding of textual reasoning
in visual evidence. We introduce \textbf{VLM-R$^3$} (\textbf{V}isual
\textbf{L}anguage \textbf{M}odel with \textbf{R}egion \textbf{R}ecognition and
\textbf{R}easoning), a framework that equips an MLLM with the ability to (i)
decide \emph{when} additional visual evidence is needed, (ii) determine
\emph{where} to ground within the image, and (iii) seamlessly weave the
relevant sub-image content back into an interleaved chain-of-thought. The core
of our method is \textbf{Region-Conditioned Reinforcement Policy Optimization
(R-GRPO)}, a training paradigm that rewards the model for selecting informative
regions, formulating appropriate transformations (e.g.\ crop, zoom), and
integrating the resulting visual context into subsequent reasoning steps. To
bootstrap this policy, we compile a modest but carefully curated Visuo-Lingual
Interleaved Rationale (VLIR) corpus that provides step-level supervision on
region selection and textual justification. Extensive experiments on MathVista,
ScienceQA, and other benchmarks show that VLM-R$^3$ sets a new state of the art
in zero-shot and few-shot settings, with the largest gains appearing on
questions demanding subtle spatial reasoning or fine-grained visual cue
extraction.

---


### [SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning](http://arxiv.org/abs/2505.16186v1)

Large Reasoning Models (LRMs) introduce a new generation paradigm of
explicitly reasoning before answering, leading to remarkable improvements in
complex tasks. However, they pose great safety risks against harmful queries
and adversarial attacks. While recent mainstream safety efforts on LRMs,
supervised fine-tuning (SFT), improve safety performance, we find that
SFT-aligned models struggle to generalize to unseen jailbreak prompts. After
thorough investigation of LRMs' generation, we identify a safety aha moment
that can activate safety reasoning and lead to a safe response. This aha moment
typically appears in the `key sentence', which follows models' query
understanding process and can indicate whether the model will proceed safely.
Based on these insights, we propose SafeKey, including two complementary
objectives to better activate the safety aha moment in the key sentence: (1) a
Dual-Path Safety Head to enhance the safety signal in the model's internal
representations before the key sentence, and (2) a Query-Mask Modeling
objective to improve the models' attention on its query understanding, which
has important safety hints. Experiments across multiple safety benchmarks
demonstrate that our methods significantly improve safety generalization to a
wide range of jailbreak attacks and out-of-distribution harmful prompts,
lowering the average harmfulness rate by 9.6\%, while maintaining general
abilities. Our analysis reveals how SafeKey enhances safety by reshaping
internal attention and improving the quality of hidden representations.

---


### [Understanding Generative AI Capabilities in Everyday Image Editing Tasks](http://arxiv.org/abs/2505.16181v1)

Generative AI (GenAI) holds significant promise for automating everyday image
editing tasks, especially following the recent release of GPT-4o on March 25,
2025. However, what subjects do people most often want edited? What kinds of
editing actions do they want to perform (e.g., removing or stylizing the
subject)? Do people prefer precise edits with predictable outcomes or highly
creative ones? By understanding the characteristics of real-world requests and
the corresponding edits made by freelance photo-editing wizards, can we draw
lessons for improving AI-based editors and determine which types of requests
can currently be handled successfully by AI editors? In this paper, we present
a unique study addressing these questions by analyzing 83k requests from the
past 12 years (2013-2025) on the Reddit community, which collected 305k
PSR-wizard edits. According to human ratings, approximately only 33% of
requests can be fulfilled by the best AI editors (including GPT-4o,
Gemini-2.0-Flash, SeedEdit). Interestingly, AI editors perform worse on
low-creativity requests that require precise editing than on more open-ended
tasks. They often struggle to preserve the identity of people and animals, and
frequently make non-requested touch-ups. On the other side of the table, VLM
judges (e.g., o1) perform differently from human judges and may prefer AI edits
more than human edits. Code and qualitative examples are available at:
https://psrdataset.github.io

---


### [Dynamic Sampling that Adapts: Iterative DPO for Self-Aware Mathematical Reasoning](http://arxiv.org/abs/2505.16176v1)

In the realm of data selection for reasoning tasks, existing approaches
predominantly rely on externally predefined static metrics such as difficulty
and diversity, which are often designed for supervised fine-tuning (SFT) and
lack adaptability to continuous training processes. A critical limitation of
these methods is their inability to dynamically align with the evolving
capabilities of models during online training, a gap that becomes increasingly
pronounced with the rise of dynamic training paradigms and online reinforcement
learning (RL) frameworks (e.g., R1 models). To address this, we introduce
SAI-DPO, an algorithm that dynamically selects training data by continuously
assessing a model's stage-specific reasoning abilities across different
training phases. By integrating real-time model performance feedback, SAI-DPO
adaptively adapts data selection to the evolving strengths and weaknesses of
the model, thus enhancing both data utilization efficiency and final task
performance. Extensive experiments on three state-of-the-art models and eight
mathematical reasoning benchmarks, including challenging competition-level
datasets (e.g., AIME24 and AMC23), demonstrate that SAI-DPO achieves an average
performance boost of up to 21.3 percentage points, with particularly notable
improvements of 10 and 15 points on AIME24 and AMC23, respectively. These
results highlight the superiority of dynamic, model-adaptive data selection
over static, externally defined strategies in advancing reasoning.

---


### [QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design](http://arxiv.org/abs/2505.16175v1)

Long-video understanding has emerged as a crucial capability in real-world
applications such as video surveillance, meeting summarization, educational
lecture analysis, and sports broadcasting. However, it remains computationally
prohibitive for VideoLLMs, primarily due to two bottlenecks: 1) sequential
video decoding, the process of converting the raw bit stream to RGB frames can
take up to a minute for hour-long video inputs, and 2) costly prefilling of up
to several million tokens for LLM inference, resulting in high latency and
memory use. To address these challenges, we propose QuickVideo, a
system-algorithm co-design that substantially accelerates long-video
understanding to support real-time downstream applications. It comprises three
key innovations: QuickDecoder, a parallelized CPU-based video decoder that
achieves 2-3 times speedup by splitting videos into keyframe-aligned intervals
processed concurrently; QuickPrefill, a memory-efficient prefilling method
using KV-cache pruning to support more frames with less GPU memory; and an
overlapping scheme that overlaps CPU video decoding with GPU inference.
Together, these components infernece time reduce by a minute on long video
inputs, enabling scalable, high-quality video understanding even on limited
hardware. Experiments show that QuickVideo generalizes across durations and
sampling rates, making long video processing feasible in practice.

---


### [Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation](http://arxiv.org/abs/2505.16146v1)

Large vision-language models (LVLMs) have achieved remarkable performance on
multimodal tasks such as visual question answering (VQA) and image captioning.
However, they still suffer from hallucinations, generating text inconsistent
with visual input, posing significant risks in real-world applications.
Existing approaches to address this issue focus on incorporating external
knowledge bases, alignment training, or decoding strategies, all of which
require substantial computational cost and time. Recent works try to explore
more efficient alternatives by adjusting LVLMs' internal representations.
Although promising, these methods may cause hallucinations to be insufficiently
suppressed or lead to excessive interventions that negatively affect normal
semantics. In this work, we leverage sparse autoencoders (SAEs) to identify
semantic directions closely associated with either hallucinations or actuality,
realizing more precise and direct hallucination-related representations. Our
analysis demonstrates that interventions along the faithful direction we
identified can mitigate hallucinations, while those along the hallucinatory
direction can exacerbate them. Building on these insights, we propose Steering
LVLMs via SAE Latent Directions (SSL), a training-free method based on
SAE-derived latent directions to mitigate hallucinations in LVLMs. Extensive
experiments demonstrate that SSL significantly outperforms existing decoding
approaches in mitigating hallucinations, while maintaining transferability
across different model architectures with negligible additional time overhead.

---


### [Can AI Read Between The Lines? Benchmarking LLMs On Financial Nuance](http://arxiv.org/abs/2505.16090v1)

As of 2025, Generative Artificial Intelligence (GenAI) has become a central
tool for productivity across industries. Beyond text generation, GenAI now
plays a critical role in coding, data analysis, and research workflows. As
large language models (LLMs) continue to evolve, it is essential to assess the
reliability and accuracy of their outputs, especially in specialized,
high-stakes domains like finance. Most modern LLMs transform text into
numerical vectors, which are used in operations such as cosine similarity
searches to generate responses. However, this abstraction process can lead to
misinterpretation of emotional tone, particularly in nuanced financial
contexts. While LLMs generally excel at identifying sentiment in everyday
language, these models often struggle with the nuanced, strategically ambiguous
language found in earnings call transcripts. Financial disclosures frequently
embed sentiment in hedged statements, forward-looking language, and
industry-specific jargon, making it difficult even for human analysts to
interpret consistently, let alone AI models. This paper presents findings from
the Santa Clara Microsoft Practicum Project, led by Professor Charlie
Goldenberg, which benchmarks the performance of Microsoft's Copilot, OpenAI's
ChatGPT, Google's Gemini, and traditional machine learning models for sentiment
analysis of financial text. Using Microsoft earnings call transcripts, the
analysis assesses how well LLM-derived sentiment correlates with market
sentiment and stock movements and evaluates the accuracy of model outputs.
Prompt engineering techniques are also examined to improve sentiment analysis
results. Visualizations of sentiment consistency are developed to evaluate
alignment between tone and stock performance, with sentiment trends analyzed
across Microsoft's lines of business to determine which segments exert the
greatest influence.

---


### [DecoupledESC: Enhancing Emotional Support Generation via Strategy-Response Decoupled Preference Optimization](http://arxiv.org/abs/2505.16995v1)

Recent advances in Emotional Support Conversation (ESC) have improved
emotional support generation by fine-tuning Large Language Models (LLMs) via
Supervised Fine-Tuning (SFT). However, common psychological errors still
persist. While Direct Preference Optimization (DPO) shows promise in reducing
such errors through pairwise preference learning, its effectiveness in ESC
tasks is limited by two key challenges: (1) Entangled data structure: Existing
ESC data inherently entangles psychological strategies and response content,
making it difficult to construct high-quality preference pairs; and (2)
Optimization ambiguity: Applying vanilla DPO to such entangled pairwise data
leads to ambiguous training objectives. To address these issues, we introduce
Inferential Preference Mining (IPM) to construct high-quality preference data,
forming the IPM-PrefDial dataset. Building upon this data, we propose a
Decoupled ESC framework inspired by Gross's Extended Process Model of Emotion
Regulation, which decomposes the ESC task into two sequential subtasks:
strategy planning and empathic response generation. Each was trained via SFT
and subsequently enhanced by DPO to align with the psychological preference.
Extensive experiments demonstrate that our Decoupled ESC framework outperforms
joint optimization baselines, reducing preference bias and improving response
quality.

---


### [SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development](http://arxiv.org/abs/2505.16975v1)

Large Language Models (LLMs) have shown strong capability in diverse software
engineering tasks, e.g. code completion, bug fixing, and document generation.
However, feature-driven development (FDD), a highly prevalent real-world task
that involves developing new functionalities for large, existing codebases,
remains underexplored. We therefore introduce SWE-Dev, the first large-scale
dataset (with 14,000 training and 500 test samples) designed to evaluate and
train autonomous coding systems on real-world feature development tasks. To
ensure verifiable and diverse training, SWE-Dev uniquely provides all instances
with a runnable environment and its developer-authored executable unit tests.
This collection not only provides high-quality data for Supervised Fine-Tuning
(SFT), but also enables Reinforcement Learning (RL) by delivering accurate
reward signals from executable unit tests. Our extensive evaluations on
SWE-Dev, covering 17 chatbot LLMs, 10 reasoning models, and 10 Multi-Agent
Systems (MAS), reveal that FDD is a profoundly challenging frontier for current
AI (e.g., Claude-3.7-Sonnet achieves only 22.45\% Pass@3 on the hard test
split). Crucially, we demonstrate that SWE-Dev serves as an effective platform
for model improvement: fine-tuning on training set enabled a 7B model
comparable to GPT-4o on \textit{hard} split, underscoring the value of its
high-quality training data. Code is available here
\href{https://github.com/justLittleWhite/SWE-Dev}{https://github.com/justLittleWhite/SWE-Dev}.

---


### [VeriFastScore: Speeding up long-form factuality evaluation](http://arxiv.org/abs/2505.16973v1)

Metrics like FactScore and VeriScore that evaluate long-form factuality
operate by decomposing an input response into atomic claims and then
individually verifying each claim. While effective and interpretable, these
methods incur numerous LLM calls and can take upwards of 100 seconds to
evaluate a single response, limiting their practicality in large-scale
evaluation and training scenarios. To address this, we propose VeriFastScore,
which leverages synthetic data to fine-tune Llama3.1 8B for simultaneously
extracting and verifying all verifiable claims within a given text based on
evidence from Google Search. We show that this task cannot be solved via
few-shot prompting with closed LLMs due to its complexity: the model receives
~4K tokens of evidence on average and needs to concurrently decompose claims,
judge their verifiability, and verify them against noisy evidence. However, our
fine-tuned VeriFastScore model demonstrates strong correlation with the
original VeriScore pipeline at both the example level (r=0.80) and system level
(r=0.94) while achieving an overall speedup of 6.6x (9.9x excluding evidence
retrieval) over VeriScore. To facilitate future factuality research, we
publicly release our VeriFastScore model and synthetic datasets.

---


### [MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning](http://arxiv.org/abs/2505.16964v1)

Existing medical VQA benchmarks mostly focus on single-image analysis, yet
clinicians almost always compare a series of images before reaching a
diagnosis. To better approximate this workflow, we introduce MedFrameQA -- the
first benchmark that explicitly evaluates multi-image reasoning in medical VQA.
To build MedFrameQA both at scale and in high-quality, we develop 1) an
automated pipeline that extracts temporally coherent frames from medical videos
and constructs VQA items whose content evolves logically across images, and 2)
a multiple-stage filtering strategy, including model-based and manual review,
to preserve data clarity, difficulty, and medical relevance. The resulting
dataset comprises 2,851 VQA pairs (gathered from 9,237 high-quality frames in
3,420 videos), covering nine human body systems and 43 organs; every question
is accompanied by two to five images. We comprehensively benchmark ten advanced
Multimodal LLMs -- both proprietary and open source, with and without explicit
reasoning modules -- on MedFrameQA. The evaluation challengingly reveals that
all models perform poorly, with most accuracies below 50%, and accuracy
fluctuates as the number of images per question increases. Error analysis
further shows that models frequently ignore salient findings, mis-aggregate
evidence across images, and propagate early mistakes through their reasoning
chains; results also vary substantially across body systems, organs, and
modalities. We hope this work can catalyze research on clinically grounded,
multi-image reasoning and accelerate progress toward more capable diagnostic AI
systems.

---


### [LLaDA-V: Large Language Diffusion Models with Visual Instruction Tuning](http://arxiv.org/abs/2505.16933v1)

In this work, we introduce LLaDA-V, a purely diffusion-based Multimodal Large
Language Model (MLLM) that integrates visual instruction tuning with masked
diffusion models, representing a departure from the autoregressive paradigms
dominant in current multimodal approaches. Built upon LLaDA, a representative
large language diffusion model, LLaDA-V incorporates a vision encoder and MLP
connector that projects visual features into the language embedding space,
enabling effective multimodal alignment. Our empirical investigation reveals
several intriguing results: First, LLaDA-V demonstrates promising multimodal
performance despite its language model being weaker on purely textual tasks
than counterparts like LLaMA3-8B and Qwen2-7B. When trained on the same
instruction data, LLaDA-V is highly competitive to LLaMA3-V across multimodal
tasks with better data scalability. It also narrows the performance gap to
Qwen2-VL, suggesting the effectiveness of its architecture for multimodal
tasks. Second, LLaDA-V achieves state-of-the-art performance in multimodal
understanding compared to existing hybrid autoregressive-diffusion and purely
diffusion-based MLLMs. Our findings suggest that large language diffusion
models show promise in multimodal contexts and warrant further investigation in
future research. Project page and codes:
https://ml-gsai.github.io/LLaDA-V-demo/.

---


### [Shadows in the Attention: Contextual Perturbation and Representation Drift in the Dynamics of Hallucination in LLMs](http://arxiv.org/abs/2505.16894v1)

Hallucinations -- plausible yet erroneous outputs -- remain a critical
barrier to reliable deployment of large language models (LLMs). We present the
first systematic study linking hallucination incidence to internal-state drift
induced by incremental context injection. Using TruthfulQA, we construct two
16-round "titration" tracks per question: one appends relevant but partially
flawed snippets, the other injects deliberately misleading content. Across six
open-source LLMs, we track overt hallucination rates with a tri-perspective
detector and covert dynamics via cosine, entropy, JS and Spearman drifts of
hidden states and attention maps. Results reveal (1) monotonic growth of
hallucination frequency and representation drift that plateaus after 5--7
rounds; (2) relevant context drives deeper semantic assimilation, producing
high-confidence "self-consistent" hallucinations, whereas irrelevant context
induces topic-drift errors anchored by attention re-routing; and (3)
convergence of JS-Drift ($\sim0.69$) and Spearman-Drift ($\sim0$) marks an
"attention-locking" threshold beyond which hallucinations solidify and become
resistant to correction. Correlation analyses expose a seesaw between
assimilation capacity and attention diffusion, clarifying size-dependent error
modes. These findings supply empirical foundations for intrinsic hallucination
prediction and context-aware mitigation mechanisms.

---


### [MPO: Multilingual Safety Alignment via Reward Gap Optimization](http://arxiv.org/abs/2505.16869v1)

Large language models (LLMs) have become increasingly central to AI
applications worldwide, necessitating robust multilingual safety alignment to
ensure secure deployment across diverse linguistic contexts. Existing
preference learning methods for safety alignment, such as RLHF and DPO, are
primarily monolingual and struggle with noisy multilingual data. To address
these limitations, we introduce Multilingual reward gaP Optimization (MPO), a
novel approach that leverages the well-aligned safety capabilities of the
dominant language (English) to improve safety alignment across multiple
languages. MPO directly minimizes the reward gap difference between the
dominant language and target languages, effectively transferring safety
capabilities while preserving the original strengths of the dominant language.
Extensive experiments on three LLMs, LLaMA-3.1, Gemma-2 and Qwen2.5, validate
MPO's efficacy in multilingual safety alignment without degrading general
multilingual utility.

---


### [ATR-Bench: A Federated Learning Benchmark for Adaptation, Trust, and Reasoning](http://arxiv.org/abs/2505.16850v1)

Federated Learning (FL) has emerged as a promising paradigm for collaborative
model training while preserving data privacy across decentralized participants.
As FL adoption grows, numerous techniques have been proposed to tackle its
practical challenges. However, the lack of standardized evaluation across key
dimensions hampers systematic progress and fair comparison of FL methods. In
this work, we introduce ATR-Bench, a unified framework for analyzing federated
learning through three foundational dimensions: Adaptation, Trust, and
Reasoning. We provide an in-depth examination of the conceptual foundations,
task formulations, and open research challenges associated with each theme. We
have extensively benchmarked representative methods and datasets for adaptation
to heterogeneous clients and trustworthiness in adversarial or unreliable
environments. Due to the lack of reliable metrics and models for reasoning in
FL, we only provide literature-driven insights for this dimension. ATR-Bench
lays the groundwork for a systematic and holistic evaluation of federated
learning with real-world relevance. We will make our complete codebase publicly
accessible and a curated repository that continuously tracks new developments
and research in the FL literature.

---


### [R1-Compress: Long Chain-of-Thought Compression via Chunk Compression and Search](http://arxiv.org/abs/2505.16838v1)

Chain-of-Thought (CoT) reasoning enhances large language models (LLMs) by
enabling step-by-step problem-solving, yet its extension to Long-CoT introduces
substantial computational overhead due to increased token length. Existing
compression approaches -- instance-level and token-level -- either sacrifice
essential local reasoning signals like reflection or yield incoherent outputs.
To address these limitations, we propose R1-Compress, a two-stage chunk-level
compression framework that preserves both local information and coherence. Our
method segments Long-CoT into manageable chunks, applies LLM-driven inner-chunk
compression, and employs an inter-chunk search mechanism to select the short
and coherent sequence. Experiments on Qwen2.5-Instruct models across MATH500,
AIME24, and GPQA-Diamond demonstrate that R1-Compress significantly reduces
token usage while maintaining comparable reasoning accuracy. On MATH500,
R1-Compress achieves an accuracy of 92.4%, with only a 0.6% drop compared to
the Long-CoT baseline, while reducing token usage by about 20%. Source code
will be available at https://github.com/w-yibo/R1-Compress

---


### [Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning](http://arxiv.org/abs/2505.16782v1)

Large Language Models (LLMs) have achieved impressive performance on complex
reasoning tasks with Chain-of-Thought (CoT) prompting. However, conventional
CoT relies on reasoning steps explicitly verbalized in natural language,
introducing inefficiencies and limiting its applicability to abstract
reasoning. To address this, there has been growing research interest in latent
CoT reasoning, where inference occurs within latent spaces. By decoupling
reasoning from language, latent reasoning promises richer cognitive
representations and more flexible, faster inference. Researchers have explored
various directions in this promising field, including training methodologies,
structural innovations, and internal reasoning mechanisms. This paper presents
a comprehensive overview and analysis of this reasoning paradigm. We begin by
proposing a unified taxonomy from four perspectives: token-wise strategies,
internal mechanisms, analysis, and applications. We then provide in-depth
discussions and comparative analyses of representative methods, highlighting
their design patterns, strengths, and open challenges. We aim to provide a
structured foundation for advancing this emerging direction in LLM reasoning.
The relevant papers will be regularly updated at
https://github.com/EIT-NLP/Awesome-Latent-CoT.

---


### [A Japanese Language Model and Three New Evaluation Benchmarks for Pharmaceutical NLP](http://arxiv.org/abs/2505.16661v1)

We present a Japanese domain-specific language model for the pharmaceutical
field, developed through continual pretraining on 2 billion Japanese
pharmaceutical tokens and 8 billion English biomedical tokens. To enable
rigorous evaluation, we introduce three new benchmarks: YakugakuQA, based on
national pharmacist licensing exams; NayoseQA, which tests cross-lingual
synonym and terminology normalization; and SogoCheck, a novel task designed to
assess consistency reasoning between paired statements. We evaluate our model
against both open-source medical LLMs and commercial models, including GPT-4o.
Results show that our domain-specific model outperforms existing open models
and achieves competitive performance with commercial ones, particularly on
terminology-heavy and knowledge-based tasks. Interestingly, even GPT-4o
performs poorly on SogoCheck, suggesting that cross-sentence consistency
reasoning remains an open challenge. Our benchmark suite offers a broader
diagnostic lens for pharmaceutical NLP, covering factual recall, lexical
variation, and logical consistency. This work demonstrates the feasibility of
building practical, secure, and cost-effective language models for Japanese
domain-specific applications, and provides reusable evaluation resources for
future research in pharmaceutical and healthcare NLP. Our model, codes, and
datasets are released at https://github.com/EQUES-Inc/pharma-LLM-eval.

---


### [What Media Frames Reveal About Stance: A Dataset and Study about Memes in Climate Change Discourse](http://arxiv.org/abs/2505.16592v1)

Media framing refers to the emphasis on specific aspects of perceived reality
to shape how an issue is defined and understood. Its primary purpose is to
shape public perceptions often in alignment with the authors' opinions and
stances. However, the interaction between stance and media frame remains
largely unexplored. In this work, we apply an interdisciplinary approach to
conceptualize and computationally explore this interaction with internet memes
on climate change. We curate CLIMATEMEMES, the first dataset of climate-change
memes annotated with both stance and media frames, inspired by research in
communication science. CLIMATEMEMES includes 1,184 memes sourced from 47
subreddits, enabling analysis of frame prominence over time and communities,
and sheds light on the framing preferences of different stance holders. We
propose two meme understanding tasks: stance detection and media frame
detection. We evaluate LLaVA-NeXT and Molmo in various setups, and report the
corresponding results on their LLM backbone. Human captions consistently
enhance performance. Synthetic captions and human-corrected OCR also help
occasionally. Our findings highlight that VLMs perform well on stance, but
struggle on frames, where LLMs outperform VLMs. Finally, we analyze VLMs'
limitations in handling nuanced frames and stance expressions on climate change
internet memes.

---


### [Evaluating Large Language Model with Knowledge Oriented Language Specific Simple Question Answering](http://arxiv.org/abs/2505.16591v1)

We introduce KoLasSimpleQA, the first benchmark evaluating the multilingual
factual ability of Large Language Models (LLMs). Inspired by existing research,
we created the question set with features such as single knowledge point
coverage, absolute objectivity, unique answers, and temporal stability. These
questions enable efficient evaluation using the LLM-as-judge paradigm, testing
both the LLMs' factual memory and self-awareness ("know what they don't know").
KoLasSimpleQA expands existing research in two key dimensions: (1) Breadth
(Multilingual Coverage): It includes 9 languages, supporting global
applicability evaluation. (2) Depth (Dual Domain Design): It covers both the
general domain (global facts) and the language-specific domain (such as
history, culture, and regional traditions) for a comprehensive assessment of
multilingual capabilities. We evaluated mainstream LLMs, including traditional
LLM and emerging Large Reasoning Models. Results show significant performance
differences between the two domains, particularly in performance metrics,
ranking, calibration, and robustness. This highlights the need for targeted
evaluation and optimization in multilingual contexts. We hope KoLasSimpleQA
will help the research community better identify LLM capability boundaries in
multilingual contexts and provide guidance for model optimization. We will
release KoLasSimpleQA at https://github.com/opendatalab/KoLasSimpleQA .

---


### [URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](http://arxiv.org/abs/2505.16570v1)

Large Language Models (LLMs) are commonly pretrained on vast corpora of text
without utilizing contextual metadata such as source, quality, or topic,
leading to a context-free learning paradigm. While recent studies suggest that
adding metadata like URL information as context (i.e., auxiliary inputs not
used in the loss calculation) can improve training efficiency and downstream
performance, they offer limited understanding of which types of metadata are
truly effective and under what conditions. In this work, we conduct a
systematic evaluation and find that not all metadata types contribute equally.
Only URL context speeds up training, whereas quality scores and topic/format
domain information offer no clear benefit. Furthermore, the improved downstream
performances of URL conditioning emerge only when longer prompts are used at
inference time. In addition, we demonstrate that context-aware pretraining
enables more controllable generation than context-free pretraining, in a
classifier-free guidance fashion. Although topic and format metadata do not
accelerate training, they are effective for steering outputs, offering
human-interpretable control over generation.

---


### [Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains](http://arxiv.org/abs/2505.16552v1)

Large Language Models (LLMs) achieve superior performance through
Chain-of-Thought (CoT) reasoning, but these token-level reasoning chains are
computationally expensive and inefficient. In this paper, we introduce
Compressed Latent Reasoning (CoLaR), a novel framework that dynamically
compresses reasoning processes in latent space through a two-stage training
approach. First, during supervised fine-tuning, CoLaR extends beyond next-token
prediction by incorporating an auxiliary next compressed embedding prediction
objective. This process merges embeddings of consecutive tokens using a
compression factor randomly sampled from a predefined range, and trains a
specialized latent head to predict distributions of subsequent compressed
embeddings. Second, we enhance CoLaR through reinforcement learning (RL) that
leverages the latent head's non-deterministic nature to explore diverse
reasoning paths and exploit more compact ones. This approach enables CoLaR to:
i) perform reasoning at a dense latent level (i.e., silently), substantially
reducing reasoning chain length, and ii) dynamically adjust reasoning speed at
inference time by simply prompting the desired compression factor. Extensive
experiments across four mathematical reasoning datasets demonstrate that CoLaR
achieves 14.1% higher accuracy than latent-based baseline methods at comparable
compression ratios, and reduces reasoning chain length by 53.3% with only 4.8%
performance degradation compared to explicit CoT method. Moreover, when applied
to more challenging mathematical reasoning tasks, our RL-enhanced CoLaR
demonstrates performance gains of up to 5.4% while dramatically reducing latent
reasoning chain length by 82.8%. The code and models will be released upon
acceptance.

---


### [Benchmarking Retrieval-Augmented Multimomal Generation for Document Question Answering](http://arxiv.org/abs/2505.16470v1)

Document Visual Question Answering (DocVQA) faces dual challenges in
processing lengthy multimodal documents (text, images, tables) and performing
cross-modal reasoning. Current document retrieval-augmented generation (DocRAG)
methods remain limited by their text-centric approaches, frequently missing
critical visual information. The field also lacks robust benchmarks for
assessing multimodal evidence selection and integration. We introduce MMDocRAG,
a comprehensive benchmark featuring 4,055 expert-annotated QA pairs with
multi-page, cross-modal evidence chains. Our framework introduces innovative
metrics for evaluating multimodal quote selection and enables answers that
interleave text with relevant visual elements. Through large-scale experiments
with 60 VLM/LLM models and 14 retrieval systems, we identify persistent
challenges in multimodal evidence retrieval, selection, and integration.Key
findings reveal advanced proprietary LVMs show superior performance than
open-sourced alternatives. Also, they show moderate advantages using multimodal
inputs over text-only inputs, while open-source alternatives show significant
performance degradation. Notably, fine-tuned LLMs achieve substantial
improvements when using detailed image descriptions. MMDocRAG establishes a
rigorous testing ground and provides actionable insights for developing more
robust multimodal DocVQA systems. Our benchmark and code are available at
https://mmdocrag.github.io/MMDocRAG/.

---


### [WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](http://arxiv.org/abs/2505.16421v1)

While reinforcement learning (RL) has demonstrated remarkable success in
enhancing large language models (LLMs), it has primarily focused on single-turn
tasks such as solving math problems. Training effective web agents for
multi-turn interactions remains challenging due to the complexity of
long-horizon decision-making across dynamic web interfaces. In this work, we
present WebAgent-R1, a simple yet effective end-to-end multi-turn RL framework
for training web agents. It learns directly from online interactions with web
environments by asynchronously generating diverse trajectories, entirely guided
by binary rewards depending on task success. Experiments on the WebArena-Lite
benchmark demonstrate the effectiveness of WebAgent-R1, boosting the task
success rate of Qwen-2.5-3B from 6.1% to 33.9% and Llama-3.1-8B from 8.5% to
44.8%, significantly outperforming existing state-of-the-art methods and strong
proprietary models such as OpenAI o3. In-depth analyses reveal the
effectiveness of the thinking-based prompting strategy and test-time scaling
through increased interactions for web tasks. We further investigate different
RL initialization policies by introducing two variants, namely WebAgent-R1-Zero
and WebAgent-R1-CoT, which highlight the importance of the warm-up training
stage (i.e., behavior cloning) and provide insights on incorporating long
chain-of-thought (CoT) reasoning in web agents.

---


### [Embodied Agents Meet Personalization: Exploring Memory Utilization for Personalized Assistance](http://arxiv.org/abs/2505.16348v1)

Embodied agents empowered by large language models (LLMs) have shown strong
performance in household object rearrangement tasks. However, these tasks
primarily focus on single-turn interactions with simplified instructions, which
do not truly reflect the challenges of providing meaningful assistance to
users. To provide personalized assistance, embodied agents must understand the
unique semantics that users assign to the physical world (e.g., favorite cup,
breakfast routine) by leveraging prior interaction history to interpret
dynamic, real-world instructions. Yet, the effectiveness of embodied agents in
utilizing memory for personalized assistance remains largely underexplored. To
address this gap, we present MEMENTO, a personalized embodied agent evaluation
framework designed to comprehensively assess memory utilization capabilities to
provide personalized assistance. Our framework consists of a two-stage memory
evaluation process design that enables quantifying the impact of memory
utilization on task performance. This process enables the evaluation of agents'
understanding of personalized knowledge in object rearrangement tasks by
focusing on its role in goal interpretation: (1) the ability to identify target
objects based on personal meaning (object semantics), and (2) the ability to
infer object-location configurations from consistent user patterns, such as
routines (user patterns). Our experiments across various LLMs reveal
significant limitations in memory utilization, with even frontier models like
GPT-4o experiencing a 30.5% performance drop when required to reference
multiple memories, particularly in tasks involving user patterns. These
findings, along with our detailed analyses and case studies, provide valuable
insights for future research in developing more effective personalized embodied
agents. Project website: https://connoriginal.github.io/MEMENTO

---


### [Augmenting LLM Reasoning with Dynamic Notes Writing for Complex QA](http://arxiv.org/abs/2505.16293v1)

Iterative RAG for multi-hop question answering faces challenges with lengthy
contexts and the buildup of irrelevant information. This hinders a model's
capacity to process and reason over retrieved content and limits performance.
While recent methods focus on compressing retrieved information, they are
either restricted to single-round RAG, require finetuning or lack scalability
in iterative RAG. To address these challenges, we propose Notes Writing, a
method that generates concise and relevant notes from retrieved documents at
each step, thereby reducing noise and retaining only essential information.
This indirectly increases the effective context length of Large Language Models
(LLMs), enabling them to reason and plan more effectively while processing
larger volumes of input text. Notes Writing is framework agnostic and can be
integrated with different iterative RAG methods. We demonstrate its
effectiveness with three iterative RAG methods, across two models and four
evaluation datasets. Notes writing yields an average improvement of 15.6
percentage points overall, with minimal increase in output tokens.

---


### [HiMATE: A Hierarchical Multi-Agent Framework for Machine Translation Evaluation](http://arxiv.org/abs/2505.16281v1)

The advancement of Large Language Models (LLMs) enables flexible and
interpretable automatic evaluations. In the field of machine translation
evaluation, utilizing LLMs with translation error annotations based on
Multidimensional Quality Metrics (MQM) yields more human-aligned judgments.
However, current LLM-based evaluation methods still face challenges in
accurately identifying error spans and assessing their severity. In this paper,
we propose HiMATE, a Hierarchical Multi-Agent Framework for Machine Translation
Evaluation. We argue that existing approaches inadequately exploit the
fine-grained structural and semantic information within the MQM hierarchy. To
address this, we develop a hierarchical multi-agent system grounded in the MQM
error typology, enabling granular evaluation of subtype errors. Two key
strategies are incorporated to further mitigate systemic hallucinations within
the framework: the utilization of the model's self-reflection capability and
the facilitation of agent discussion involving asymmetric information.
Empirically, HiMATE outperforms competitive baselines across different datasets
in conducting human-aligned evaluations. Further analyses underscore its
significant advantage in error span detection and severity assessment,
achieving an average F1-score improvement of 89% over the best-performing
baseline. We make our code and data publicly available at
https://anonymous.4open.science/r/HiMATE-Anony.

---


### [Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers](http://arxiv.org/abs/2505.16241v1)

Recently, Large Reasoning Models (LRMs) have demonstrated superior logical
capabilities compared to traditional Large Language Models (LLMs), gaining
significant attention. Despite their impressive performance, the potential for
stronger reasoning abilities to introduce more severe security vulnerabilities
remains largely underexplored. Existing jailbreak methods often struggle to
balance effectiveness with robustness against adaptive safety mechanisms. In
this work, we propose SEAL, a novel jailbreak attack that targets LRMs through
an adaptive encryption pipeline designed to override their reasoning processes
and evade potential adaptive alignment. Specifically, SEAL introduces a stacked
encryption approach that combines multiple ciphers to overwhelm the models
reasoning capabilities, effectively bypassing built-in safety mechanisms. To
further prevent LRMs from developing countermeasures, we incorporate two
dynamic strategies - random and adaptive - that adjust the cipher length,
order, and combination. Extensive experiments on real-world reasoning models,
including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the
effectiveness of our approach. Notably, SEAL achieves an attack success rate of
80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant
margin of 27.2%. Warning: This paper contains examples of inappropriate,
offensive, and harmful content.

---


### [Align-GRAG: Reasoning-Guided Dual Alignment for Graph Retrieval-Augmented Generation](http://arxiv.org/abs/2505.16237v1)

Large language models (LLMs) have demonstrated remarkable capabilities, but
still struggle with issues like hallucinations and outdated information.
Retrieval-augmented generation (RAG) addresses these issues by grounding LLM
outputs in external knowledge with an Information Retrieval (IR) system.
Building on this foundation, graph-based RAG systems go a step further by
retrieving subgraphs, which preserve the relationships between knowledge
entities and provide more comprehensive context. However, graph RAG faces two
challenges: (1) Retrieving relevant information introduces irrelevant nodes
(especially in dense graph databases, where retrieval usually extends to
adjacent nodes), and leads to overly lengthy inputs that hinder efficiency; (2)
The representation gap between graph and language during generation with LLMs
limits the ability to fully leverage graph structures for enhanced
understanding. To address these limitations, we propose Align-GRAG, a novel
reasoning-guided dual alignment framework in post-retrieval phrase. It first
formulates a subgraph by retrieving nodes and edges. Then an Aligner is
proposed to jointly optimizes a graph encoder with LLM-summarized reasoning. It
achieves dual alignment of graph node and representation by leveraging KL
divergence loss and contrastive loss, facilitating efficient pruning of
irrelevant knowledge and establishing a unified semantic space. The Generator
integrates the aligned graph data with LLM to produce coherent and accurate
answers. Experiments on GraphQA benchmark across three tasks (including common
sense reasoning, scene graph understanding, and knowledge graph reasoning)
validate the effectiveness of our method. The code will be available upon
accepted.

---


### [MuseRAG: Idea Originality Scoring At Scale](http://arxiv.org/abs/2505.16232v1)

An objective, face-valid way to assess the originality of creative ideas is
to measure how rare each idea is within a population -- an approach long used
in creativity research but difficult to automate at scale. Tabulating response
frequencies via manual bucketing of idea rephrasings is labor-intensive,
error-prone, and brittle under large corpora. We introduce a fully automated,
psychometrically validated pipeline for frequency-based originality scoring.
Our method, MuseRAG, combines large language models (LLMs) with an externally
orchestrated retrieval-augmented generation (RAG) framework. Given a new idea,
the system retrieves semantically similar prior idea buckets and zero-shot
prompts the LLM to judge whether the new idea belongs to an existing bucket or
forms a new one. The resulting buckets enable computation of frequency-based
originality metrics. Across five datasets (N=1143, n_ideas=16294), MuseRAG
matches human annotators in idea clustering structure and resolution (AMI =
0.59) and in participant-level scoring (r = 0.89) -- while exhibiting strong
convergent and external validity. Our work enables intent-sensitive,
human-aligned originality scoring at scale to aid creativity research.

---


### [Large Language Models based ASR Error Correction for Child Conversations](http://arxiv.org/abs/2505.16212v1)

Automatic Speech Recognition (ASR) has recently shown remarkable progress,
but accurately transcribing children's speech remains a significant challenge.
Recent developments in Large Language Models (LLMs) have shown promise in
improving ASR transcriptions. However, their applications in child speech
including conversational scenarios are underexplored. In this study, we explore
the use of LLMs in correcting ASR errors for conversational child speech. We
demonstrate the promises and challenges of LLMs through experiments on two
children's conversational speech datasets with both zero-shot and fine-tuned
ASR outputs. We find that while LLMs are helpful in correcting zero-shot ASR
outputs and fine-tuned CTC-based ASR outputs, it remains challenging for LLMs
to improve ASR performance when incorporating contextual information or when
using fine-tuned autoregressive ASR (e.g., Whisper) outputs.

---


### [An Empirical Study on Configuring In-Context Learning Demonstrations for Unleashing MLLMs' Sentimental Perception Capability](http://arxiv.org/abs/2505.16193v1)

The advancements in Multimodal Large Language Models (MLLMs) have enabled
various multimodal tasks to be addressed under a zero-shot paradigm. This
paradigm sidesteps the cost of model fine-tuning, emerging as a dominant trend
in practical application. Nevertheless, Multimodal Sentiment Analysis (MSA), a
pivotal challenge in the quest for general artificial intelligence, fails to
accommodate this convenience. The zero-shot paradigm exhibits undesirable
performance on MSA, casting doubt on whether MLLMs can perceive sentiments as
competent as supervised models. By extending the zero-shot paradigm to
In-Context Learning (ICL) and conducting an in-depth study on configuring
demonstrations, we validate that MLLMs indeed possess such capability.
Specifically, three key factors that cover demonstrations' retrieval,
presentation, and distribution are comprehensively investigated and optimized.
A sentimental predictive bias inherent in MLLMs is also discovered and later
effectively counteracted. By complementing each other, the devised strategies
for three factors result in average accuracy improvements of 15.9% on six MSA
datasets against the zero-shot paradigm and 11.2% against the random ICL
baseline.

---


### [Can LLMs Simulate Human Behavioral Variability? A Case Study in the Phonemic Fluency Task](http://arxiv.org/abs/2505.16164v1)

Large language models (LLMs) are increasingly explored as substitutes for
human participants in cognitive tasks, but their ability to simulate human
behavioral variability remains unclear. This study examines whether LLMs can
approximate individual differences in the phonemic fluency task, where
participants generate words beginning with a target letter. We evaluated 34
model configurations, varying prompt specificity, sampling temperature, and
model type, and compared outputs to responses from 106 human participants.
While some configurations, especially Claude 3.7 Sonnet, matched human averages
and lexical preferences, none reproduced the scope of human variability. LLM
outputs were consistently less diverse and structurally rigid, and LLM
ensembles failed to increase diversity. Network analyses further revealed
fundamental differences in retrieval structure between humans and models. These
results highlight key limitations in using LLMs to simulate human cognition and
behavior.

---


### [Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning](http://arxiv.org/abs/2505.16142v1)

Distilling reasoning paths from teacher to student models via supervised
fine-tuning (SFT) provides a shortcut for improving the reasoning ability of
smaller Large Language Models (LLMs). However, the reasoning paths generated by
teacher models often reflect only surface-level traces of their underlying
authentic reasoning. Insights from cognitive neuroscience suggest that
authentic reasoning involves a complex interweaving between meta-reasoning
(which selects appropriate sub-problems from multiple candidates) and solving
(which addresses the sub-problem). This implies authentic reasoning has an
implicit multi-branch structure. Supervised fine-tuning collapses this rich
structure into a flat sequence of token prediction in the teacher's reasoning
path, preventing effective distillation of this structure to students. To
address this limitation, we propose RLKD, a reinforcement learning (RL)-based
distillation framework guided by a novel Generative Structure Reward Model
(GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving
steps and computes rewards to measure structural alignment between student and
teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to
internalize the teacher's implicit multi-branch reasoning structure rather than
merely mimicking fixed output paths. Experiments show RLKD surpasses standard
SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime,
unlocking greater student reasoning potential than SFT-based distillation.

---


### [Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning](http://arxiv.org/abs/2505.16128v1)

Despite LLMs' explicit alignment against demographic stereotypes, they have
been shown to exhibit biases under various social contexts. In this work, we
find that LLMs exhibit concerning biases in how they associate solution
veracity with demographics. Through experiments across five human value-aligned
LLMs on mathematics, coding, commonsense, and writing problems, we reveal two
forms of such veracity biases: Attribution Bias, where models
disproportionately attribute correct solutions to certain demographic groups,
and Evaluation Bias, where models' assessment of identical solutions varies
based on perceived demographic authorship. Our results show pervasive biases:
LLMs consistently attribute fewer correct solutions and more incorrect ones to
African-American groups in math and coding, while Asian authorships are least
preferred in writing evaluation. In additional studies, we show LLMs
automatically assign racially stereotypical colors to demographic groups in
visualization code, suggesting these biases are deeply embedded in models'
reasoning processes. Our findings indicate that demographic bias extends beyond
surface-level stereotypes and social context provocations, raising concerns
about LLMs' deployment in educational and evaluation settings.

---


### [Hierarchical Safety Realignment: Lightweight Restoration of Safety in Pruned Large Vision-Language Models](http://arxiv.org/abs/2505.16104v1)

With the increasing size of Large Vision-Language Models (LVLMs), network
pruning techniques aimed at compressing models for deployment in
resource-constrained environments have garnered significant attention. However,
we observe that pruning often leads to a degradation in safety performance. To
address this issue, we present a novel and lightweight approach, termed
Hierarchical Safety Realignment (HSR). HSR operates by first quantifying the
contribution of each attention head to safety, identifying the most critical
ones, and then selectively restoring neurons directly within these attention
heads that play a pivotal role in maintaining safety. This process
hierarchically realigns the safety of pruned LVLMs, progressing from the
attention head level to the neuron level. We validate HSR across various models
and pruning strategies, consistently achieving notable improvements in safety
performance. To our knowledge, this is the first work explicitly focused on
restoring safety in LVLMs post-pruning.

---


### [Continually Self-Improving Language Models for Bariatric Surgery Question--Answering](http://arxiv.org/abs/2505.16102v1)

While bariatric and metabolic surgery (MBS) is considered the gold standard
treatment for severe and morbid obesity, its therapeutic efficacy hinges upon
active and longitudinal engagement with multidisciplinary providers, including
surgeons, dietitians/nutritionists, psychologists, and endocrinologists. This
engagement spans the entire patient journey, from preoperative preparation to
long-term postoperative management. However, this process is often hindered by
numerous healthcare disparities, such as logistical and access barriers, which
impair easy patient access to timely, evidence-based, clinician-endorsed
information. To address these gaps, we introduce bRAGgen, a novel adaptive
retrieval-augmented generation (RAG)-based model that autonomously integrates
real-time medical evidence when response confidence dips below dynamic
thresholds. This self-updating architecture ensures that responses remain
current and accurate, reducing the risk of misinformation. Additionally, we
present bRAGq, a curated dataset of 1,302 bariatric surgery--related questions,
validated by an expert bariatric surgeon. bRAGq constitutes the first
large-scale, domain-specific benchmark for comprehensive MBS care. In a
two-phase evaluation, bRAGgen is benchmarked against state-of-the-art models
using both large language model (LLM)--based metrics and expert surgeon review.
Across all evaluation dimensions, bRAGgen demonstrates substantially superior
performance in generating clinically accurate and relevant responses.

---


### [ARB: A Comprehensive Arabic Multimodal Reasoning Benchmark](http://arxiv.org/abs/2505.17021v1)

As Large Multimodal Models (LMMs) become more capable, there is growing
interest in evaluating their reasoning processes alongside their final outputs.
However, most benchmarks remain focused on English, overlooking languages with
rich linguistic and cultural contexts, such as Arabic. To address this gap, we
introduce the Comprehensive Arabic Multimodal Reasoning Benchmark (ARB), the
first benchmark designed to evaluate step-by-step reasoning in Arabic across
both textual and visual modalities. ARB spans 11 diverse domains, including
visual reasoning, document understanding, OCR, scientific analysis, and
cultural interpretation. It comprises 1,356 multimodal samples paired with
5,119 human-curated reasoning steps and corresponding actions. We evaluated 12
state-of-the-art open- and closed-source LMMs and found persistent challenges
in coherence, faithfulness, and cultural grounding. ARB offers a structured
framework for diagnosing multimodal reasoning in underrepresented languages and
marks a critical step toward inclusive, transparent, and culturally aware AI
systems. We release the benchmark, rubric, and evaluation suit to support
future research and reproducibility. Code available at:
https://github.com/mbzuai-oryx/ARB

---


### [SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](http://arxiv.org/abs/2505.17018v1)

Recent advances have shown success in eliciting strong reasoning abilities in
multimodal large language models (MLLMs) through rule-based reinforcement
learning (RL) with outcome rewards. However, this paradigm typically lacks
supervision over the thinking process leading to the final outcome.As a result,
the model may learn sub-optimal reasoning strategies, which can hinder its
generalization ability. In light of this, we propose SophiaVL-R1, as an attempt
to add reward signals for the thinking process in this paradigm. To achieve
this, we first train a thinking reward model that evaluates the quality of the
entire thinking process. Given that the thinking reward may be unreliable for
certain samples due to reward hacking, we propose the Trust-GRPO method, which
assigns a trustworthiness weight to the thinking reward during training. This
weight is computed based on the thinking reward comparison of responses leading
to correct answers versus incorrect answers, helping to mitigate the impact of
potentially unreliable thinking rewards. Moreover, we design an annealing
training strategy that gradually reduces the thinking reward over time,
allowing the model to rely more on the accurate rule-based outcome reward in
later training stages. Experiments show that our SophiaVL-R1 surpasses a series
of reasoning MLLMs on various benchmarks (e.g., MathVisita, MMMU),
demonstrating strong reasoning and generalization capabilities. Notably, our
SophiaVL-R1-7B even outperforms LLaVA-OneVision-72B on most benchmarks, despite
the latter having 10 times more parameters. All code, models, and datasets are
made publicly available at https://github.com/kxfan2002/SophiaVL-R1.

---


### [CrossLMM: Decoupling Long Video Sequences from LMMs via Dual Cross-Attention Mechanisms](http://arxiv.org/abs/2505.17020v1)

The advent of Large Multimodal Models (LMMs) has significantly enhanced Large
Language Models (LLMs) to process and interpret diverse data modalities (e.g.,
image and video). However, as input complexity increases, particularly with
long video sequences, the number of required tokens has grown significantly,
leading to quadratically computational costs. This has made the efficient
compression of video tokens in LMMs, while maintaining performance integrity, a
pressing research challenge. In this paper, we introduce CrossLMM, decoupling
long video sequences from LMMs via a dual cross-attention mechanism, which
substantially reduces visual token quantity with minimal performance
degradation. Specifically, we first implement a significant token reduction
from pretrained visual encoders through a pooling methodology. Then, within LLM
layers, we employ a visual-to-visual cross-attention mechanism, wherein the
pooled visual tokens function as queries against the original visual token set.
This module enables more efficient token utilization while retaining
fine-grained informational fidelity. In addition, we introduce a text-to-visual
cross-attention mechanism, for which the text tokens are enhanced through
interaction with the original visual tokens, enriching the visual comprehension
of the text tokens. Comprehensive empirical evaluation demonstrates that our
approach achieves comparable or superior performance across diverse video-based
LMM benchmarks, despite utilizing substantially fewer computational resources.

---


### [OpenSeg-R: Improving Open-Vocabulary Segmentation via Step-by-Step Visual Reasoning](http://arxiv.org/abs/2505.16974v1)

Open-Vocabulary Segmentation (OVS) has drawn increasing attention for its
capacity to generalize segmentation beyond predefined categories. However,
existing methods typically predict segmentation masks with simple forward
inference, lacking explicit reasoning and interpretability. This makes it
challenging for OVS model to distinguish similar categories in open-world
settings due to the lack of contextual understanding and discriminative visual
cues. To address this limitation, we propose a step-by-step visual reasoning
framework for open-vocabulary segmentation, named OpenSeg-R. The proposed
OpenSeg-R leverages Large Multimodal Models (LMMs) to perform hierarchical
visual reasoning before segmentation. Specifically, we generate both generic
and image-specific reasoning for each image, forming structured triplets that
explain the visual reason for objects in a coarse-to-fine manner. Based on
these reasoning steps, we can compose detailed description prompts, and feed
them to the segmentor to produce more accurate segmentation masks. To the best
of our knowledge, OpenSeg-R is the first framework to introduce explicit
step-by-step visual reasoning into OVS. Experimental results demonstrate that
OpenSeg-R significantly outperforms state-of-the-art methods on open-vocabulary
semantic segmentation across five benchmark datasets. Moreover, it achieves
consistent gains across all metrics on open-vocabulary panoptic segmentation.
Qualitative results further highlight the effectiveness of our reasoning-guided
framework in improving both segmentation precision and interpretability. Our
code is publicly available at https://github.com/Hanzy1996/OpenSeg-R.

---


### [RBench-V: A Primary Assessment for Visual Reasoning Models with Multi-modal Outputs](http://arxiv.org/abs/2505.16770v1)

The rapid advancement of native multi-modal models and omni-models,
exemplified by GPT-4o, Gemini, and o3, with their capability to process and
generate content across modalities such as text and images, marks a significant
milestone in the evolution of intelligence. Systematic evaluation of their
multi-modal output capabilities in visual thinking processes (also known as
multi-modal chain of thought, M-CoT) becomes critically important. However,
existing benchmarks for evaluating multi-modal models primarily focus on
assessing multi-modal inputs and text-only reasoning while neglecting the
importance of reasoning through multi-modal outputs. In this paper, we present
a benchmark, dubbed RBench-V, designed to assess models' vision-indispensable
reasoning abilities. To construct RBench-V, we carefully hand-pick 803
questions covering math, physics, counting, and games. Unlike previous
benchmarks that typically specify certain input modalities, RBench-V presents
problems centered on multi-modal outputs, which require image manipulation such
as generating novel images and constructing auxiliary lines to support the
reasoning process. We evaluate numerous open- and closed-source models on
RBench-V, including o3, Gemini 2.5 Pro, Qwen2.5-VL, etc. Even the
best-performing model, o3, achieves only 25.8% accuracy on RBench-V, far below
the human score of 82.3%, highlighting that current models struggle to leverage
multi-modal reasoning. Data and code are available at
https://evalmodels.github.io/rbenchv

---


### [Mesh-RFT: Enhancing Mesh Generation via Fine-grained Reinforcement Fine-Tuning](http://arxiv.org/abs/2505.16761v1)

Existing pretrained models for 3D mesh generation often suffer from data
biases and produce low-quality results, while global reinforcement learning
(RL) methods rely on object-level rewards that struggle to capture local
structure details. To address these challenges, we present \textbf{Mesh-RFT}, a
novel fine-grained reinforcement fine-tuning framework that employs Masked
Direct Preference Optimization (M-DPO) to enable localized refinement via
quality-aware face masking. To facilitate efficient quality evaluation, we
introduce an objective topology-aware scoring system to evaluate geometric
integrity and topological regularity at both object and face levels through two
metrics: Boundary Edge Ratio (BER) and Topology Score (TS). By integrating
these metrics into a fine-grained RL strategy, Mesh-RFT becomes the first
method to optimize mesh quality at the granularity of individual faces,
resolving localized errors while preserving global coherence. Experiment
results show that our M-DPO approach reduces Hausdorff Distance (HD) by 24.6\%
and improves Topology Score (TS) by 3.8\% over pre-trained models, while
outperforming global DPO methods with a 17.4\% HD reduction and 4.9\% TS gain.
These results demonstrate Mesh-RFT's ability to improve geometric integrity and
topological regularity, achieving new state-of-the-art performance in
production-ready mesh generation. Project Page:
\href{https://hitcslj.github.io/mesh-rft/}{this https URL}.

---


### [Representation Discrepancy Bridging Method for Remote Sensing Image-Text Retrieval](http://arxiv.org/abs/2505.16756v1)

Remote Sensing Image-Text Retrieval (RSITR) plays a critical role in
geographic information interpretation, disaster monitoring, and urban planning
by establishing semantic associations between image and textual descriptions.
Existing Parameter-Efficient Fine-Tuning (PEFT) methods for Vision-and-Language
Pre-training (VLP) models typically adopt symmetric adapter structures for
exploring cross-modal correlations. However, the strong discriminative nature
of text modality may dominate the optimization process and inhibits image
representation learning. The nonnegligible imbalanced cross-modal optimization
remains a bottleneck to enhancing the model performance. To address this issue,
this study proposes a Representation Discrepancy Bridging (RDB) method for the
RSITR task. On the one hand, a Cross-Modal Asymmetric Adapter (CMAA) is
designed to enable modality-specific optimization and improve feature
alignment. The CMAA comprises a Visual Enhancement Adapter (VEA) and a Text
Semantic Adapter (TSA). VEA mines fine-grained image features by Differential
Attention (DA) mechanism, while TSA identifies key textual semantics through
Hierarchical Attention (HA) mechanism. On the other hand, this study extends
the traditional single-task retrieval framework to a dual-task optimization
framework and develops a Dual-Task Consistency Loss (DTCL). The DTCL improves
cross-modal alignment robustness through an adaptive weighted combination of
cross-modal, classification, and exponential moving average consistency
constraints. Experiments on RSICD and RSITMD datasets show that the proposed
RDB method achieves a 6%-11% improvement in mR metrics compared to
state-of-the-art PEFT methods and a 1.15%-2% improvement over the full
fine-tuned GeoRSCLIP model.

---


### [Zero-Shot Anomaly Detection in Battery Thermal Images Using Visual Question Answering with Prior Knowledge](http://arxiv.org/abs/2505.16674v1)

Batteries are essential for various applications, including electric vehicles
and renewable energy storage, making safety and efficiency critical concerns.
Anomaly detection in battery thermal images helps identify failures early, but
traditional deep learning methods require extensive labeled data, which is
difficult to obtain, especially for anomalies due to safety risks and high data
collection costs. To overcome this, we explore zero-shot anomaly detection
using Visual Question Answering (VQA) models, which leverage pretrained
knowledge and textbased prompts to generalize across vision tasks. By
incorporating prior knowledge of normal battery thermal behavior, we design
prompts to detect anomalies without battery-specific training data. We evaluate
three VQA models (ChatGPT-4o, LLaVa-13b, and BLIP-2) analyzing their robustness
to prompt variations, repeated trials, and qualitative outputs. Despite the
lack of finetuning on battery data, our approach demonstrates competitive
performance compared to state-of-the-art models that are trained with the
battery data. Our findings highlight the potential of VQA-based zero-shot
learning for battery anomaly detection and suggest future directions for
improving its effectiveness.

---


### [ManipLVM-R1: Reinforcement Learning for Reasoning in Embodied Manipulation with Large Vision-Language Models](http://arxiv.org/abs/2505.16517v1)

Large Vision-Language Models (LVLMs) have recently advanced robotic
manipulation by leveraging vision for scene perception and language for
instruction following. However, existing methods rely heavily on costly
human-annotated training datasets, which limits their generalization and causes
them to struggle in out-of-domain (OOD) scenarios, reducing real-world
adaptability. To address these challenges, we propose ManipLVM-R1, a novel
reinforcement learning framework that replaces traditional supervision with
Reinforcement Learning using Verifiable Rewards (RLVR). By directly optimizing
for task-aligned outcomes, our method enhances generalization and physical
reasoning while removing the dependence on costly annotations. Specifically, we
design two rule-based reward functions targeting key robotic manipulation
subtasks: an Affordance Perception Reward to enhance localization of
interaction regions, and a Trajectory Match Reward to ensure the physical
plausibility of action paths. These rewards provide immediate feedback and
impose spatial-logical constraints, encouraging the model to go beyond shallow
pattern matching and instead learn deeper, more systematic reasoning about
physical interactions.

---


### [Panoptic Captioning: Seeking An Equivalency Bridge for Image and Text](http://arxiv.org/abs/2505.16334v1)

This work introduces panoptic captioning, a novel task striving to seek the
minimum text equivalence of images. We take the first step towards panoptic
captioning by formulating it as a task of generating a comprehensive textual
description for an image, which encapsulates all entities, their respective
locations and attributes, relationships among entities, as well as global image
state.Through an extensive evaluation, our work reveals that state-of-the-art
Multi-modal Large Language Models (MLLMs) have limited performance in solving
panoptic captioning. To address this, we propose an effective data engine named
PancapEngine to produce high-quality data and a novel method named PancapChain
to improve panoptic captioning. Specifically, our PancapEngine first detects
diverse categories of entities in images by an elaborate detection suite, and
then generates required panoptic captions using entity-aware prompts.
Additionally, our PancapChain explicitly decouples the challenging panoptic
captioning task into multiple stages and generates panoptic captions step by
step. More importantly, we contribute a comprehensive metric named PancapScore
and a human-curated test set for reliable model evaluation.Experiments show
that our PancapChain-13B model can beat state-of-the-art open-source MLLMs like
InternVL-2.5-78B and even surpass proprietary models like GPT-4o and
Gemini-2.0-Pro, demonstrating the effectiveness of our data engine and method.
Project page: https://visual-ai.github.io/pancap/

---


### [ARPO:End-to-End Policy Optimization for GUI Agents with Experience Replay](http://arxiv.org/abs/2505.16282v1)

Training large language models (LLMs) as interactive agents for controlling
graphical user interfaces (GUIs) presents a unique challenge to optimize
long-horizon action sequences with multimodal feedback from complex
environments. While recent works have advanced multi-turn reinforcement
learning (RL) for reasoning and tool-using capabilities in LLMs, their
application to GUI-based agents remains relatively underexplored due to the
difficulty of sparse rewards, delayed feedback, and high rollout costs. In this
paper, we investigate end-to-end policy optimization for vision-language-based
GUI agents with the aim of improving performance on complex, long-horizon
computer tasks. We propose Agentic Replay Policy Optimization (ARPO), an
end-to-end RL approach that augments Group Relative Policy Optimization (GRPO)
with a replay buffer to reuse the successful experience across training
iterations. To further stabilize the training process, we propose a task
selection strategy that filters tasks based on baseline agent performance,
allowing the agent to focus on learning from informative interactions.
Additionally, we compare ARPO with offline preference optimization approaches,
highlighting the advantages of policy-based methods in GUI environments.
Experiments on the OSWorld benchmark demonstrate that ARPO achieves competitive
results, establishing a new performance baseline for LLM-based GUI agents
trained via reinforcement learning. Our findings underscore the effectiveness
of reinforcement learning for training multi-turn, vision-language GUI agents
capable of managing complex real-world UI interactions. Codes and
models:https://github.com/dvlab-research/ARPO.git.

---


### [Training-Free Reasoning and Reflection in MLLMs](http://arxiv.org/abs/2505.16151v1)

Recent advances in Reasoning LLMs (e.g., DeepSeek-R1 and OpenAI-o1) have
showcased impressive reasoning capabilities via reinforcement learning.
However, extending these capabilities to Multimodal LLMs (MLLMs) is hampered by
the prohibitive costs of retraining and the scarcity of high-quality,
verifiable multimodal reasoning datasets. This paper introduces FRANK Model, a
training-FRee ANd r1-liKe MLLM that imbues off-the-shelf MLLMs with reasoning
and reflection abilities, without any gradient updates or extra supervision.
Our key insight is to decouple perception and reasoning across MLLM decoder
layers. Specifically, we observe that compared to the deeper decoder layers,
the shallow decoder layers allocate more attention to visual tokens, while the
deeper decoder layers concentrate on textual semantics. This observation
motivates a hierarchical weight merging approach that combines a
visual-pretrained MLLM with a reasoning-specialized LLM. To this end, we
propose a layer-wise, Taylor-derived closed-form fusion mechanism that
integrates reasoning capacity into deep decoder layers while preserving visual
grounding in shallow decoder layers. Extensive experiments on challenging
multimodal reasoning benchmarks demonstrate the effectiveness of our approach.
On the MMMU benchmark, our model FRANK-38B achieves an accuracy of 69.2,
outperforming the strongest baseline InternVL2.5-38B by +5.3, and even
surpasses the proprietary GPT-4o model. Our project homepage is at:
http://iip.whu.edu.cn/frank/index.html

---


### [Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks](http://arxiv.org/abs/2505.16901v1)

Recent advances in Large Language Models (LLMs) have shown promise in
function-level code generation, yet repository-level software engineering tasks
remain challenging. Current solutions predominantly rely on proprietary LLM
agents, which introduce unpredictability and limit accessibility, raising
concerns about data privacy and model customization. This paper investigates
whether open-source LLMs can effectively address repository-level tasks without
requiring agent-based approaches. We demonstrate this is possible by enabling
LLMs to comprehend functions and files within codebases through their semantic
information and structural dependencies. To this end, we introduce Code Graph
Models (CGMs), which integrate repository code graph structures into the LLM's
attention mechanism and map node attributes to the LLM's input space using a
specialized adapter. When combined with an agentless graph RAG framework, our
approach achieves a 43.00% resolution rate on the SWE-bench Lite benchmark
using the open-source Qwen2.5-72B model. This performance ranks first among
open weight models, second among methods with open-source systems, and eighth
overall, surpassing the previous best open-source model-based method by 12.33%.

---


### [LLM-Based Emulation of the Radio Resource Control Layer: Towards AI-Native RAN Protocols](http://arxiv.org/abs/2505.16821v1)

Integrating large AI models (LAMs) into 6G mobile networks promises to
redefine protocol design and control-plane intelligence by enabling autonomous,
cognitive network operations. While industry concepts, such as ETSI's
Experiential Networked Intelligence (ENI), envision LAM-driven agents for
adaptive network slicing and intent-based management, practical implementations
still face challenges in protocol literacy and real-world deployment. This
paper presents an end-to-end demonstration of a LAM that generates
standards-compliant, ASN.1-encoded Radio Resource Control (RRC) messages as
part of control-plane procedures inside a gNB. We treat RRC messaging as a
domain-specific language and fine-tune a decoder-only transformer model (LLaMA
class) using parameter-efficient Low-Rank Adaptation (LoRA) on RRC messages
linearized to retain their ASN.1 syntactic structure before standard byte-pair
encoding tokenization. This enables combinatorial generalization over RRC
protocol states while minimizing training overhead. On 30k field-test
request-response pairs, our 8 B model achieves a median cosine similarity of
0.97 with ground-truth messages on an edge GPU -- a 61 % relative gain over a
zero-shot LLaMA-3 8B baseline -- indicating substantially improved structural
and semantic RRC fidelity. Overall, our results show that LAMs, when augmented
with Radio Access Network (RAN)-specific reasoning, can directly orchestrate
control-plane procedures, representing a stepping stone toward the AI-native
air-interface paradigm. Beyond RRC emulation, this work lays the groundwork for
future AI-native wireless standards.

---


### [Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models](http://arxiv.org/abs/2505.16446v1)

Multimodal large language models (MLLMs) enable powerful cross-modal
reasoning capabilities. However, the expanded input space introduces new attack
surfaces. Previous jailbreak attacks often inject malicious instructions from
text into less aligned modalities, such as vision. As MLLMs increasingly
incorporate cross-modal consistency and alignment mechanisms, such explicit
attacks become easier to detect and block. In this work, we propose a novel
implicit jailbreak framework termed IJA that stealthily embeds malicious
instructions into images via least significant bit steganography and couples
them with seemingly benign, image-related textual prompts. To further enhance
attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes
generated by a surrogate model and introduce a template optimization module
that iteratively refines both the prompt and embedding based on model feedback.
On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack
success rates of over 90% using an average of only 3 queries.

---


### [Performance Guaranteed Poisoning Attacks in Federated Learning: A Sliding Mode Approach](http://arxiv.org/abs/2505.16403v1)

Manipulation of local training data and local updates, i.e., the poisoning
attack, is the main threat arising from the collaborative nature of the
federated learning (FL) paradigm. Most existing poisoning attacks aim to
manipulate local data/models in a way that causes denial-of-service (DoS)
issues. In this paper, we introduce a novel attack method, named Federated
Learning Sliding Attack (FedSA) scheme, aiming at precisely introducing the
extent of poisoning in a subtle controlled manner. It operates with a
predefined objective, such as reducing global model's prediction accuracy by
10\%. FedSA integrates robust nonlinear control-Sliding Mode Control (SMC)
theory with model poisoning attacks. It can manipulate the updates from
malicious clients to drive the global model towards a compromised state,
achieving this at a controlled and inconspicuous rate. Additionally, leveraging
the robust control properties of FedSA allows precise control over the
convergence bounds, enabling the attacker to set the global accuracy of the
poisoned model to any desired level. Experimental results demonstrate that
FedSA can accurately achieve a predefined global accuracy with fewer malicious
clients while maintaining a high level of stealth and adjustable learning
rates.

---


### [Divide-Fuse-Conquer: Eliciting "Aha Moments" in Multi-Scenario Games](http://arxiv.org/abs/2505.16401v1)

Large language models (LLMs) have been observed to suddenly exhibit advanced
reasoning abilities during reinforcement learning (RL), resembling an ``aha
moment'' triggered by simple outcome-based rewards. While RL has proven
effective in eliciting such breakthroughs in tasks involving mathematics,
coding, and vision, it faces significant challenges in multi-scenario games.
The diversity of game rules, interaction modes, and environmental complexities
often leads to policies that perform well in one scenario but fail to
generalize to others. Simply combining multiple scenarios during training
introduces additional challenges, such as training instability and poor
performance. To overcome these challenges, we propose Divide-Fuse-Conquer, a
framework designed to enhance generalization in multi-scenario RL. This
approach starts by heuristically grouping games based on characteristics such
as rules and difficulties. Specialized models are then trained for each group
to excel at games in the group is what we refer to as the divide step. Next, we
fuse model parameters from different groups as a new model, and continue
training it for multiple groups, until the scenarios in all groups are
conquered. Experiments across 18 TextArena games show that Qwen2.5-32B-Align
trained with the Divide-Fuse-Conquer strategy reaches a performance level
comparable to Claude3.5, achieving 7 wins and 4 draws. We hope our approach can
inspire future research on using reinforcement learning to improve the
generalization of LLMs.

---


### [Understanding Differential Transformer Unchains Pretrained Self-Attentions](http://arxiv.org/abs/2505.16333v1)

Differential Transformer has recently gained significant attention for its
impressive empirical performance, often attributed to its ability to perform
noise canceled attention. However, precisely how differential attention
achieves its empirical benefits remains poorly understood. Moreover,
Differential Transformer architecture demands large-scale training from
scratch, hindering utilization of open pretrained weights. In this work, we
conduct an in-depth investigation of Differential Transformer, uncovering three
key factors behind its success: (1) enhanced expressivity via negative
attention, (2) reduced redundancy among attention heads, and (3) improved
learning dynamics. Based on these findings, we propose DEX, a novel method to
efficiently integrate the advantages of differential attention into pretrained
language models. By reusing the softmax attention scores and adding a
lightweight differential operation on the output value matrix, DEX effectively
incorporates the key advantages of differential attention while remaining
lightweight in both training and inference. Evaluations confirm that DEX
substantially improves the pretrained LLMs across diverse benchmarks, achieving
significant performance gains with minimal adaptation data (< 0.01\%).

---


### [ChemMLLM: Chemical Multimodal Large Language Model](http://arxiv.org/abs/2505.16326v1)

Multimodal large language models (MLLMs) have made impressive progress in
many applications in recent years. However, chemical MLLMs that can handle
cross-modal understanding and generation remain underexplored. To fill this
gap, in this paper, we propose ChemMLLM, a unified chemical multimodal large
language model for molecule understanding and generation. Also, we design five
multimodal tasks across text, molecular SMILES strings, and image, and curate
the datasets. We benchmark ChemMLLM against a range of general leading MLLMs
and Chemical LLMs on these tasks. Experimental results show that ChemMLLM
achieves superior performance across all evaluated tasks. For example, in
molecule image optimization task, ChemMLLM outperforms the best baseline
(GPT-4o) by 118.9\% (4.27 vs 1.95 property improvement). The code is publicly
available at https://github.com/bbsbz/ChemMLLM.git.

---


### [Offline Guarded Safe Reinforcement Learning for Medical Treatment Optimization Strategies](http://arxiv.org/abs/2505.16242v1)

When applying offline reinforcement learning (RL) in healthcare scenarios,
the out-of-distribution (OOD) issues pose significant risks, as inappropriate
generalization beyond clinical expertise can result in potentially harmful
recommendations. While existing methods like conservative Q-learning (CQL)
attempt to address the OOD issue, their effectiveness is limited by only
constraining action selection by suppressing uncertain actions. This
action-only regularization imitates clinician actions that prioritize
short-term rewards, but it fails to regulate downstream state trajectories,
thereby limiting the discovery of improved long-term treatment strategies. To
safely improve policy beyond clinician recommendations while ensuring that
state-action trajectories remain in-distribution, we propose \textit{Offline
Guarded Safe Reinforcement Learning} ($\mathsf{OGSRL}$), a theoretically
grounded model-based offline RL framework. $\mathsf{OGSRL}$ introduces a novel
dual constraint mechanism for improving policy with reliability and safety.
First, the OOD guardian is established to specify clinically validated regions
for safe policy exploration. By constraining optimization within these regions,
it enables the reliable exploration of treatment strategies that outperform
clinician behavior by leveraging the full patient state history, without
drifting into unsupported state-action trajectories. Second, we introduce a
safety cost constraint that encodes medical knowledge about physiological
safety boundaries, providing domain-specific safeguards even in areas where
training data might contain potentially unsafe interventions. Notably, we
provide theoretical guarantees on safety and near-optimality: policies that
satisfy these constraints remain in safe and reliable regions and achieve
performance close to the best possible policy supported by the data.

---


### [Plan and Budget: Effective and Efficient Test-Time Scaling on Large Language Model Reasoning](http://arxiv.org/abs/2505.16122v1)

Large Language Models (LLMs) have achieved remarkable success in complex
reasoning tasks, but their inference remains computationally inefficient. We
observe a common failure mode in many prevalent LLMs, overthinking, where
models generate verbose and tangential reasoning traces even for simple
queries. Recent works have tried to mitigate this by enforcing fixed token
budgets, however, this can lead to underthinking, especially on harder
problems. Through empirical analysis, we identify that this inefficiency often
stems from unclear problem-solving strategies. To formalize this, we develop a
theoretical model, BBAM (Bayesian Budget Allocation Model), which models
reasoning as a sequence of sub-questions with varying uncertainty, and
introduce the $E^3$ metric to capture the trade-off between correctness and
computation efficiency. Building on theoretical results from BBAM, we propose
Plan-and-Budget, a model-agnostic, test-time framework that decomposes complex
queries into sub-questions and allocates token budgets based on estimated
complexity using adaptive scheduling. Plan-and-Budget improves reasoning
efficiency across a range of tasks and models, achieving up to +70% accuracy
gains, -39% token reduction, and +187.5% improvement in $E^3$. Notably, it
elevates a smaller model (DS-Qwen-32B) to match the efficiency of a larger
model (DS-LLaMA-70B)-demonstrating Plan-and-Budget's ability to close
performance gaps without retraining. Our code is available at
anonymous.4open.science/r/P-and-B-6513/.

---


### [Tools in the Loop: Quantifying Uncertainty of LLM Question Answering Systems That Use Tools](http://arxiv.org/abs/2505.16113v1)

Modern Large Language Models (LLMs) often require external tools, such as
machine learning classifiers or knowledge retrieval systems, to provide
accurate answers in domains where their pre-trained knowledge is insufficient.
This integration of LLMs with external tools expands their utility but also
introduces a critical challenge: determining the trustworthiness of responses
generated by the combined system. In high-stakes applications, such as medical
decision-making, it is essential to assess the uncertainty of both the LLM's
generated text and the tool's output to ensure the reliability of the final
response. However, existing uncertainty quantification methods do not account
for the tool-calling scenario, where both the LLM and external tool contribute
to the overall system's uncertainty. In this work, we present a novel framework
for modeling tool-calling LLMs that quantifies uncertainty by jointly
considering the predictive uncertainty of the LLM and the external tool. We
extend previous methods for uncertainty quantification over token sequences to
this setting and propose efficient approximations that make uncertainty
computation practical for real-world applications. We evaluate our framework on
two new synthetic QA datasets, derived from well-known machine learning
datasets, which require tool-calling for accurate answers. Additionally, we
apply our method to retrieval-augmented generation (RAG) systems and conduct a
proof-of-concept experiment demonstrating the effectiveness of our uncertainty
metrics in scenarios where external information retrieval is needed. Our
results show that the framework is effective in enhancing trust in LLM-based
systems, especially in cases where the LLM's internal knowledge is insufficient
and external tools are required.

---


### [Reinforcement Learning for Stock Transactions](http://arxiv.org/abs/2505.16099v1)

Much research has been done to analyze the stock market. After all, if one
can determine a pattern in the chaotic frenzy of transactions, then they could
make a hefty profit from capitalizing on these insights. As such, the goal of
our project was to apply reinforcement learning (RL) to determine the best time
to buy a stock within a given time frame. With only a few adjustments, our
model can be extended to identify the best time to sell a stock as well. In
order to use the format of free, real-world data to train the model, we define
our own Markov Decision Process (MDP) problem. These two papers [5] [6] helped
us in formulating the state space and the reward system of our MDP problem. We
train a series of agents using Q-Learning, Q-Learning with linear function
approximation, and deep Q-Learning. In addition, we try to predict the stock
prices using machine learning regression and classification models. We then
compare our agents to see if they converge on a policy, and if so, which one
learned the best policy to maximize profit on the stock market.

---


### [Tactile-based Reinforcement Learning for Adaptive Grasping under Observation Uncertainties](http://arxiv.org/abs/2505.16167v1)

Robotic manipulation in industrial scenarios such as construction commonly
faces uncertain observations in which the state of the manipulating object may
not be accurately captured due to occlusions and partial observables. For
example, object status estimation during pipe assembly, rebar installation, and
electrical installation can be impacted by observation errors. Traditional
vision-based grasping methods often struggle to ensure robust stability and
adaptability. To address this challenge, this paper proposes a tactile
simulator that enables a tactile-based adaptive grasping method to enhance
grasping robustness. This approach leverages tactile feedback combined with the
Proximal Policy Optimization (PPO) reinforcement learning algorithm to
dynamically adjust the grasping posture, allowing adaptation to varying
grasping conditions under inaccurate object state estimations. Simulation
results demonstrate that the proposed method effectively adapts grasping
postures, thereby improving the success rate and stability of grasping tasks.

---


### [Cosmos: A CXL-Based Full In-Memory System for Approximate Nearest Neighbor Search](http://arxiv.org/abs/2505.16096v1)

Retrieval-Augmented Generation (RAG) is crucial for improving the quality of
large language models by injecting proper contexts extracted from external
sources. RAG requires high-throughput, low-latency Approximate Nearest Neighbor
Search (ANNS) over billion-scale vector databases. Conventional DRAM/SSD
solutions face capacity/latency limits, whereas specialized hardware or RDMA
clusters lack flexibility or incur network overhead. We present Cosmos,
integrating general-purpose cores within CXL memory devices for full ANNS
offload and introducing rank-level parallel distance computation to maximize
memory bandwidth. We also propose an adjacency-aware data placement that
balances search loads across CXL devices based on inter-cluster proximity.
Evaluations on SIFT1B and DEEP1B traces show that Cosmos achieves up to 6.72x
higher throughput than the baseline CXL system and 2.35x over a
state-of-the-art CXL-based solution, demonstrating scalability for RAG
pipelines.

---


