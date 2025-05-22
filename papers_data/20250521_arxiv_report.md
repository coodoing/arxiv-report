### [Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](http://arxiv.org/abs/2505.14684v1)

Large language models (LLMs) have achieved remarkable progress on
mathemati-cal tasks through Chain-of-Thought (CoT) reasoning. However, existing
mathematical CoT datasets often suffer from Thought Leaps due to experts
omitting intermediate steps, which negatively impacts model learning and
generalization. We propose the CoT Thought Leap Bridge Task, which aims to
automatically detect leaps and generate missing intermediate reasoning steps to
restore the completeness and coherence of CoT. To facilitate this, we
constructed a specialized training dataset called ScaleQM+, based on the
structured ScaleQuestMath dataset, and trained CoT-Bridge to bridge thought
leaps. Through comprehensive experiments on mathematical reasoning benchmarks,
we demonstrate that models fine-tuned on bridged datasets consistently
outperform those trained on original datasets, with improvements of up to
+5.87% on NuminaMath. Our approach effectively enhances distilled data (+3.02%)
and provides better starting points for reinforcement learning (+3.1%),
functioning as a plug-and-play module compatible with existing optimization
techniques. Furthermore, CoT-Bridge demonstrate improved generalization to
out-of-domain logical reasoning tasks, confirming that enhancing reasoning
completeness yields broadly applicable benefits.

---


### [SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](http://arxiv.org/abs/2505.14667v1)

Large Reasoning Models (LRMs) have become powerful tools for complex problem
solving, but their structured reasoning pathways can lead to unsafe outputs
when exposed to harmful prompts. Existing safety alignment methods reduce
harmful outputs but can degrade reasoning depth, leading to significant
trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated
jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight
alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at
the start of their reasoning, in response to harmful prompts, while leaving the
rest of the reasoning process unsupervised. Empirical results across multiple
benchmarks indicate that SAFEPATH effectively reduces harmful outputs while
maintaining reasoning performance. Specifically, SAFEPATH reduces harmful
responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the
DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than
Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot
variant that requires no fine-tuning. In addition, we provide a comprehensive
analysis of how existing methods in LLMs generalize, or fail, when applied to
reasoning-centric models, revealing critical gaps and new directions for safer
AI.

---


### [Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning](http://arxiv.org/abs/2505.14656v1)

While LLMs excel at open-ended reasoning, they often struggle with
cost-sensitive planning, either treating all actions as having equal cost or
failing to stay within strict budgets. In this paper, we introduce
Cost-Augmented Monte Carlo Tree Search (CATS), a novel approach that brings
explicit cost-awareness into LLM-guided planning. Tight cost constraints push
the planner to quickly identify infeasible solutions, while looser constraints
encourage optimization for minimal cost. We benchmark top LLMs such as GPT-4.1,
Claude-3.7-Sonnet, and DeepSeek-R1, against our CATS planner to evaluate their
performance in cost-sensitive scenarios. Our experiments suggest that raw LLMs
such as GPT-4.1 often falter under tight budgets, whereas CATS consistently
delivers strong performance, achieving higher task success rates and better
cost efficiency. CATS provides an effective solution for budget-aware
decision-making by combining the reasoning power of LLMs with structured
search.

---


### [CAD-Coder: An Open-Source Vision-Language Model for Computer-Aided Design Code Generation](http://arxiv.org/abs/2505.14646v1)

Efficient creation of accurate and editable 3D CAD models is critical in
engineering design, significantly impacting cost and time-to-market in product
innovation. Current manual workflows remain highly time-consuming and demand
extensive user expertise. While recent developments in AI-driven CAD generation
show promise, existing models are limited by incomplete representations of CAD
operations, inability to generalize to real-world images, and low output
accuracy. This paper introduces CAD-Coder, an open-source Vision-Language Model
(VLM) explicitly fine-tuned to generate editable CAD code (CadQuery Python)
directly from visual input. Leveraging a novel dataset that we
created--GenCAD-Code, consisting of over 163k CAD-model image and code
pairs--CAD-Coder outperforms state-of-the-art VLM baselines such as GPT-4.5 and
Qwen2.5-VL-72B, achieving a 100% valid syntax rate and the highest accuracy in
3D solid similarity. Notably, our VLM demonstrates some signs of
generalizability, successfully generating CAD code from real-world images and
executing CAD operations unseen during fine-tuning. The performance and
adaptability of CAD-Coder highlights the potential of VLMs fine-tuned on code
to streamline CAD workflows for engineers and designers. CAD-Coder is publicly
available at: https://github.com/anniedoris/CAD-Coder.

---


### [KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](http://arxiv.org/abs/2505.14629v1)

Recent advances in large language models (LLMs) and the abundance of food
data have resulted in studies to improve food understanding using LLMs. Despite
several recommendation systems utilizing LLMs and Knowledge Graphs (KGs), there
has been limited research on integrating food related KGs with LLMs. We
introduce KERL, a unified system that leverages food KGs and LLMs to provide
personalized food recommendations and generates recipes with associated
micro-nutritional information. Given a natural language question, KERL extracts
entities, retrieves subgraphs from the KG, which are then fed into the LLM as
context to select the recipes that satisfy the constraints. Next, our system
generates the cooking steps and nutritional information for each recipe. To
evaluate our approach, we also develop a benchmark dataset by curating recipe
related questions, combined with constraints and personal preferences. Through
extensive experiments, we show that our proposed KG-augmented LLM significantly
outperforms existing approaches, offering a complete and coherent solution for
food recommendation, recipe generation, and nutritional analysis. Our code and
benchmark datasets are publicly available at
https://github.com/mohbattharani/KERL.

---


### [Debating for Better Reasoning: An Unsupervised Multimodal Approach](http://arxiv.org/abs/2505.14627v1)

As Large Language Models (LLMs) gain expertise across diverse domains and
modalities, scalable oversight becomes increasingly challenging, particularly
when their capabilities may surpass human evaluators. Debate has emerged as a
promising mechanism for enabling such oversight. In this work, we extend the
debate paradigm to a multimodal setting, exploring its potential for weaker
models to supervise and enhance the performance of stronger models. We focus on
visual question answering (VQA), where two "sighted" expert vision-language
models debate an answer, while a "blind" (text-only) judge adjudicates based
solely on the quality of the arguments. In our framework, the experts defend
only answers aligned with their beliefs, thereby obviating the need for
explicit role-playing and concentrating the debate on instances of expert
disagreement. Experiments on several multimodal tasks demonstrate that the
debate framework consistently outperforms individual expert models. Moreover,
judgments from weaker LLMs can help instill reasoning capabilities in
vision-language models through finetuning.

---


### [TinyV: Reducing False Negatives in Verification Improves RL for LLM Reasoning](http://arxiv.org/abs/2505.14625v1)

Reinforcement Learning (RL) has become a powerful tool for enhancing the
reasoning abilities of large language models (LLMs) by optimizing their
policies with reward signals. Yet, RL's success relies on the reliability of
rewards, which are provided by verifiers. In this paper, we expose and analyze
a widespread problem--false negatives--where verifiers wrongly reject correct
model outputs. Our in-depth study of the Big-Math-RL-Verified dataset reveals
that over 38% of model-generated responses suffer from false negatives, where
the verifier fails to recognize correct answers. We show, both empirically and
theoretically, that these false negatives severely impair RL training by
depriving the model of informative gradient signals and slowing convergence. To
mitigate this, we propose tinyV, a lightweight LLM-based verifier that augments
existing rule-based methods, which dynamically identifies potential false
negatives and recovers valid responses to produce more accurate reward
estimates. Across multiple math-reasoning benchmarks, integrating TinyV boosts
pass rates by up to 10% and accelerates convergence relative to the baseline.
Our findings highlight the critical importance of addressing verifier false
negatives and offer a practical approach to improve RL-based fine-tuning of
LLMs. Our code is available at https://github.com/uw-nsl/TinyV.

---


### [KIPPO: Koopman-Inspired Proximal Policy Optimization](http://arxiv.org/abs/2505.14566v1)

Reinforcement Learning (RL) has made significant strides in various domains,
and policy gradient methods like Proximal Policy Optimization (PPO) have gained
popularity due to their balance in performance, training stability, and
computational efficiency. These methods directly optimize policies through
gradient-based updates. However, developing effective control policies for
environments with complex and non-linear dynamics remains a challenge. High
variance in gradient estimates and non-convex optimization landscapes often
lead to unstable learning trajectories. Koopman Operator Theory has emerged as
a powerful framework for studying non-linear systems through an
infinite-dimensional linear operator that acts on a higher-dimensional space of
measurement functions. In contrast with their non-linear counterparts, linear
systems are simpler, more predictable, and easier to analyze. In this paper, we
present Koopman-Inspired Proximal Policy Optimization (KIPPO), which learns an
approximately linear latent-space representation of the underlying system's
dynamics while retaining essential features for effective policy learning. This
is achieved through a Koopman-approximation auxiliary network that can be added
to the baseline policy optimization algorithms without altering the
architecture of the core policy or value function. Extensive experimental
results demonstrate consistent improvements over the PPO baseline with 6-60%
increased performance while reducing variability by up to 91% when evaluated on
various continuous control tasks.

---


### [SSPS: Self-Supervised Positive Sampling for Robust Self-Supervised Speaker Verification](http://arxiv.org/abs/2505.14561v1)

Self-Supervised Learning (SSL) has led to considerable progress in Speaker
Verification (SV). The standard framework uses same-utterance positive sampling
and data-augmentation to generate anchor-positive pairs of the same speaker.
This is a major limitation, as this strategy primarily encodes channel
information from the recording condition, shared by the anchor and positive. We
propose a new positive sampling technique to address this bottleneck:
Self-Supervised Positive Sampling (SSPS). For a given anchor, SSPS aims to find
an appropriate positive, i.e., of the same speaker identity but a different
recording condition, in the latent space using clustering assignments and a
memory queue of positive embeddings. SSPS improves SV performance for both
SimCLR and DINO, reaching 2.57% and 2.53% EER, outperforming SOTA SSL methods
on VoxCeleb1-O. In particular, SimCLR-SSPS achieves a 58% EER reduction by
lowering intra-speaker variance, providing comparable performance to DINO-SSPS.

---


### [KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation](http://arxiv.org/abs/2505.14552v1)

Recent advancements in large language models (LLMs) underscore the need for
more comprehensive evaluation methods to accurately assess their reasoning
capabilities. Existing benchmarks are often domain-specific and thus cannot
fully capture an LLM's general reasoning potential. To address this limitation,
we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic
evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over
fifty games in either textual or visual formats and supports interactive,
multi-turn assessments with reinforcement learning scenarios. Using KORGym, we
conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent
reasoning patterns within model families and demonstrating the superior
performance of closed-source models. Further analysis examines the effects of
modality, reasoning strategies, reinforcement learning techniques, and response
length on model performance. We expect KORGym to become a valuable resource for
advancing LLM reasoning research and developing evaluation methodologies suited
to complex, interactive environments.

---


### [Multi-agent Reinforcement Learning vs. Fixed-Time Control for Traffic Signal Optimization: A Simulation Study](http://arxiv.org/abs/2505.14544v1)

Urban traffic congestion, particularly at intersections, significantly
impacts travel time, fuel consumption, and emissions. Traditional fixed-time
signal control systems often lack the adaptability to manage dynamic traffic
patterns effectively. This study explores the application of multi-agent
reinforcement learning (MARL) to optimize traffic signal coordination across
multiple intersections within a simulated environment. Utilizing Pygame, a
simulation was developed to model a network of interconnected intersections
with randomly generated vehicle flows to reflect realistic traffic variability.
A decentralized MARL controller was implemented, in which each traffic signal
operates as an autonomous agent, making decisions based on local observations
and information from neighboring agents. Performance was evaluated against a
baseline fixed-time controller using metrics such as average vehicle wait time
and overall throughput. The MARL approach demonstrated statistically
significant improvements, including reduced average waiting times and improved
throughput. These findings suggest that MARL-based dynamic control strategies
hold substantial promise for improving urban traffic management efficiency.
More research is recommended to address scalability and real-world
implementation challenges.

---


### [Energy-Efficient Deep Reinforcement Learning with Spiking Transformers](http://arxiv.org/abs/2505.14533v1)

Agent-based Transformers have been widely adopted in recent reinforcement
learning advances due to their demonstrated ability to solve complex tasks.
However, the high computational complexity of Transformers often results in
significant energy consumption, limiting their deployment in real-world
autonomous systems. Spiking neural networks (SNNs), with their biologically
inspired structure, offer an energy-efficient alternative for machine learning.
In this paper, a novel Spike-Transformer Reinforcement Learning (STRL)
algorithm that combines the energy efficiency of SNNs with the powerful
decision-making capabilities of reinforcement learning is developed.
Specifically, an SNN using multi-step Leaky Integrate-and-Fire (LIF) neurons
and attention mechanisms capable of processing spatio-temporal patterns over
multiple time steps is designed. The architecture is further enhanced with
state, action, and reward encodings to create a Transformer-like structure
optimized for reinforcement learning tasks. Comprehensive numerical experiments
conducted on state-of-the-art benchmarks demonstrate that the proposed SNN
Transformer achieves significantly improved policy performance compared to
conventional agent-based Transformers. With both enhanced energy efficiency and
policy optimality, this work highlights a promising direction for deploying
bio-inspired, low-cost machine learning models in complex real-world
decision-making scenarios.

---


### [NavBench: A Unified Robotics Benchmark for Reinforcement Learning-Based Autonomous Navigation](http://arxiv.org/abs/2505.14526v1)

Autonomous robots must navigate and operate in diverse environments, from
terrestrial and aquatic settings to aerial and space domains. While
Reinforcement Learning (RL) has shown promise in training policies for specific
autonomous robots, existing benchmarks are often constrained to unique
platforms, limiting generalization and fair comparisons across different
mobility systems. In this paper, we present NavBench, a multi-domain benchmark
for training and evaluating RL-based navigation policies across diverse robotic
platforms and operational environments. Built on IsaacLab, our framework
standardizes task definitions, enabling different robots to tackle various
navigation challenges without the need for ad-hoc task redesigns or custom
evaluation metrics. Our benchmark addresses three key challenges: (1) Unified
cross-medium benchmarking, enabling direct evaluation of diverse actuation
methods (thrusters, wheels, water-based propulsion) in realistic environments;
(2) Scalable and modular design, facilitating seamless robot-task
interchangeability and reproducible training pipelines; and (3) Robust
sim-to-real validation, demonstrated through successful policy transfer to
multiple real-world robots, including a satellite robotic simulator, an
unmanned surface vessel, and a wheeled ground vehicle. By ensuring consistency
between simulation and real-world deployment, NavBench simplifies the
development of adaptable RL-based navigation strategies. Its modular design
allows researchers to easily integrate custom robots and tasks by following the
framework's predefined templates, making it accessible for a wide range of
applications. Our code is publicly available at NavBench.

---


### [Guarded Query Routing for Large Language Models](http://arxiv.org/abs/2505.14524v1)

Query routing, the task to route user queries to different large language
model (LLM) endpoints, can be considered as a text classification problem.
However, out-of-distribution queries must be handled properly, as those could
be questions about unrelated domains, queries in other languages, or even
contain unsafe text. Here, we thus study a \emph{guarded} query routing
problem, for which we first introduce the Guarded Query Routing Benchmark
(GQR-Bench), which covers three exemplary target domains (law, finance, and
healthcare), and seven datasets to test robustness against out-of-distribution
queries. We then use GQR-Bench to contrast the effectiveness and efficiency of
LLM-based routing mechanisms (GPT-4o-mini, Llama-3.2-3B, and Llama-3.1-8B),
standard LLM-based guardrail approaches (LlamaGuard and NVIDIA NeMo
Guardrails), continuous bag-of-words classifiers (WideMLP, fastText), and
traditional machine learning models (SVM, XGBoost). Our results show that
WideMLP, enhanced with out-of-domain detection capabilities, yields the best
trade-off between accuracy (88\%) and speed (<4ms). The embedding-based
fastText excels at speed (<1ms) with acceptable accuracy (80\%), whereas LLMs
yield the highest accuracy (91\%) but are comparatively slow (62ms for local
Llama-3.1:8B and 669ms for remote GPT-4o-mini calls). Our findings challenge
the automatic reliance on LLMs for (guarded) query routing and provide concrete
recommendations for practical applications. GQR-Bench will be released as a
Python package -- \texttt{gqr}.

---


### [Creative Preference Optimization](http://arxiv.org/abs/2505.14442v1)

While Large Language Models (LLMs) have demonstrated impressive performance
across natural language generation tasks, their ability to generate truly
creative content-characterized by novelty, diversity, surprise, and
quality-remains limited. Existing methods for enhancing LLM creativity often
focus narrowly on diversity or specific tasks, failing to address creativity's
multifaceted nature in a generalizable way. In this work, we propose Creative
Preference Optimization (CrPO), a novel alignment method that injects signals
from multiple creativity dimensions into the preference optimization objective
in a modular fashion. We train and evaluate creativity-augmented versions of
several models using CrPO and MuCE, a new large-scale human preference dataset
spanning over 200,000 human-generated responses and ratings from more than 30
psychological creativity assessments. Our models outperform strong baselines,
including GPT-4o, on both automated and human evaluations, producing more
novel, diverse, and surprising generations while maintaining high output
quality. Additional evaluations on NoveltyBench further confirm the
generalizability of our approach. Together, our results demonstrate that
directly optimizing for creativity within preference frameworks is a promising
direction for advancing the creative capabilities of LLMs without compromising
output quality.

---


### [Neural Incompatibility: The Unbridgeable Gap of Cross-Scale Parametric Knowledge Transfer in Large Language Models](http://arxiv.org/abs/2505.14436v1)

Large Language Models (LLMs) offer a transparent brain with accessible
parameters that encode extensive knowledge, which can be analyzed, located and
transferred. Consequently, a key research challenge is to transcend traditional
knowledge transfer paradigms rooted in symbolic language and achieve genuine
Parametric Knowledge Transfer (PKT). Significantly, exploring effective methods
for transferring knowledge across LLMs of different scales through parameters
presents an intriguing and valuable research direction. In this paper, we first
demonstrate $\textbf{Alignment}$ in parametric space is the fundamental
prerequisite to achieve successful cross-scale PKT. We redefine the previously
explored knowledge transfer as Post-Align PKT (PostPKT), which utilizes
extracted parameters for LoRA initialization and requires subsequent fine-tune
for alignment. Hence, to reduce cost for further fine-tuning, we introduce a
novel Pre-Align PKT (PrePKT) paradigm and propose a solution called
$\textbf{LaTen}$
($\textbf{L}$oc$\textbf{a}$te-$\textbf{T}$h$\textbf{e}$n-Alig$\textbf{n}$) that
aligns the parametric spaces of LLMs across scales only using several training
steps without following training. Comprehensive experiments on four benchmarks
demonstrate that both PostPKT and PrePKT face challenges in achieving
consistently stable transfer. Through in-depth analysis, we identify
$\textbf{Neural Incompatibility}$ as the ethological and parametric structural
differences between LLMs of varying scales, presenting fundamental challenges
to achieving effective PKT. These findings provide fresh insights into the
parametric architectures of LLMs and highlight promising directions for future
research on efficient PKT. Our code is available at
https://github.com/Trae1ounG/Neural_Incompatibility.

---


### [Choosing a Model, Shaping a Future: Comparing LLM Perspectives on Sustainability and its Relationship with AI](http://arxiv.org/abs/2505.14435v1)

As organizations increasingly rely on AI systems for decision support in
sustainability contexts, it becomes critical to understand the inherent biases
and perspectives embedded in Large Language Models (LLMs). This study
systematically investigates how five state-of-the-art LLMs -- Claude, DeepSeek,
GPT, LLaMA, and Mistral - conceptualize sustainability and its relationship
with AI. We administered validated, psychometric sustainability-related
questionnaires - each 100 times per model -- to capture response patterns and
variability. Our findings revealed significant inter-model differences: For
example, GPT exhibited skepticism about the compatibility of AI and
sustainability, whereas LLaMA demonstrated extreme techno-optimism with perfect
scores for several Sustainable Development Goals (SDGs). Models also diverged
in attributing institutional responsibility for AI and sustainability
integration, a results that holds implications for technology governance
approaches. Our results demonstrate that model selection could substantially
influence organizational sustainability strategies, highlighting the need for
awareness of model-specific biases when deploying LLMs for
sustainability-related decision-making.

---


### [Interpretable Neural System Dynamics: Combining Deep Learning with System Dynamics Modeling to Support Critical Applications](http://arxiv.org/abs/2505.14428v1)

The objective of this proposal is to bridge the gap between Deep Learning
(DL) and System Dynamics (SD) by developing an interpretable neural system
dynamics framework. While DL excels at learning complex models and making
accurate predictions, it lacks interpretability and causal reliability.
Traditional SD approaches, on the other hand, provide transparency and causal
insights but are limited in scalability and require extensive domain knowledge.
To overcome these limitations, this project introduces a Neural System Dynamics
pipeline, integrating Concept-Based Interpretability, Mechanistic
Interpretability, and Causal Machine Learning. This framework combines the
predictive power of DL with the interpretability of traditional SD models,
resulting in both causal reliability and scalability. The efficacy of the
proposed pipeline will be validated through real-world applications of the
EU-funded AutoMoTIF project, which is focused on autonomous multimodal
transportation systems. The long-term goal is to collect actionable insights
that support the integration of explainability and safety in autonomous
systems.

---


### [PRL: Prompts from Reinforcement Learning](http://arxiv.org/abs/2505.14412v1)

Effective prompt engineering remains a central challenge in fully harnessing
the capabilities of LLMs. While well-designed prompts can dramatically enhance
performance, crafting them typically demands expert intuition and a nuanced
understanding of the task. Moreover, the most impactful prompts often hinge on
subtle semantic cues, ones that may elude human perception but are crucial for
guiding LLM behavior. In this paper, we introduce PRL (Prompts from
Reinforcement Learning), a novel RL-based approach for automatic prompt
generation. Unlike previous methods, PRL can produce novel few-shot examples
that were not seen during training. Our approach achieves state-of-the-art
performance across a range of benchmarks, including text classification,
simplification, and summarization. On the classification task, it surpasses
prior methods by 2.58% over APE and 1.00% over EvoPrompt. Additionally, it
improves the average ROUGE scores on the summarization task by 4.32 over APE
and by 2.12 over EvoPrompt and the SARI score on simplification by 6.93 over
APE and by 6.01 over EvoPrompt. Our code is available at
https://github.com/Batorskq/prl .

---


### [Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning](http://arxiv.org/abs/2505.14403v1)

Recent advances in reasoning language models have witnessed a paradigm shift
from short to long CoT pattern. Given the substantial computational cost of
rollouts in long CoT models, maximizing the utility of fixed training datasets
becomes crucial. Our analysis reveals that negative responses contain valuable
components such as self-reflection and error-correction steps, yet primary
existing methods either completely discard negative samples (RFT) or apply
equal penalization across all tokens (RL), failing to leverage these potential
learning signals. In light of this, we propose Behavior Constrained Policy
Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline
RL framework that encompasses three stages: 1) sample segmentation, 2)
consensus-based step correctness assessment combining LLM and PRM judgers, and
3) policy optimization with NSA designed to effectively mine positive steps
within negative samples. Experimental results show that BCPG-NSA outperforms
baselines on several challenging math/coding reasoning benchmarks using the
same training dataset, achieving improved sample efficiency and demonstrating
robustness and scalability when extended to multiple iterations.

---


### [SCAN: Semantic Document Layout Analysis for Textual and Visual Retrieval-Augmented Generation](http://arxiv.org/abs/2505.14381v1)

With the increasing adoption of Large Language Models (LLMs) and
Vision-Language Models (VLMs), rich document analysis technologies for
applications like Retrieval-Augmented Generation (RAG) and visual RAG are
gaining significant attention. Recent research indicates that using VLMs can
achieve better RAG performance, but processing rich documents still remains a
challenge since a single page contains large amounts of information. In this
paper, we present SCAN (\textbf{S}emanti\textbf{C} Document Layout
\textbf{AN}alysis), a novel approach enhancing both textual and visual
Retrieval-Augmented Generation (RAG) systems working with visually rich
documents. It is a VLM-friendly approach that identifies document components
with appropriate semantic granularity, balancing context preservation with
processing efficiency. SCAN uses a coarse-grained semantic approach that
divides documents into coherent regions covering continuous components. We
trained the SCAN model by fine-tuning object detection models with
sophisticated annotation datasets. Our experimental results across English and
Japanese datasets demonstrate that applying SCAN improves end-to-end textual
RAG performance by up to 9.0\% and visual RAG performance by up to 6.4\%,
outperforming conventional approaches and even commercial document processing
solutions.

---


### [Towards Embodied Cognition in Robots via Spatially Grounded Synthetic Worlds](http://arxiv.org/abs/2505.14366v1)

We present a conceptual framework for training Vision-Language Models (VLMs)
to perform Visual Perspective Taking (VPT), a core capability for embodied
cognition essential for Human-Robot Interaction (HRI). As a first step toward
this goal, we introduce a synthetic dataset, generated in NVIDIA Omniverse,
that enables supervised learning for spatial reasoning tasks. Each instance
includes an RGB image, a natural language description, and a ground-truth 4X4
transformation matrix representing object pose. We focus on inferring Z-axis
distance as a foundational skill, with future extensions targeting full 6
Degrees Of Freedom (DOFs) reasoning. The dataset is publicly available to
support further research. This work serves as a foundational step toward
embodied AI systems capable of spatial understanding in interactive human-robot
scenarios.

---


### [SafetyNet: Detecting Harmful Outputs in LLMs by Modeling and Monitoring Deceptive Behaviors](http://arxiv.org/abs/2505.14300v1)

High-risk industries like nuclear and aviation use real-time monitoring to
detect dangerous system conditions. Similarly, Large Language Models (LLMs)
need monitoring safeguards. We propose a real-time framework to predict harmful
AI outputs before they occur by using an unsupervised approach that treats
normal behavior as the baseline and harmful outputs as outliers. Our study
focuses specifically on backdoor-triggered responses -- where specific input
phrases activate hidden vulnerabilities causing the model to generate unsafe
content like violence, pornography, or hate speech. We address two key
challenges: (1) identifying true causal indicators rather than surface
correlations, and (2) preventing advanced models from deception -- deliberately
evading monitoring systems. Hence, we approach this problem from an
unsupervised lens by drawing parallels to human deception: just as humans
exhibit physical indicators while lying, we investigate whether LLMs display
distinct internal behavioral signatures when generating harmful content. Our
study addresses two critical challenges: 1) designing monitoring systems that
capture true causal indicators rather than superficial correlations; and
2)preventing intentional evasion by increasingly capable "Future models''. Our
findings show that models can produce harmful content through causal mechanisms
and can become deceptive by: (a) alternating between linear and non-linear
representations, and (b) modifying feature relationships. To counter this, we
developed Safety-Net -- a multi-detector framework that monitors different
representation dimensions, successfully detecting harmful behavior even when
information is shifted across representational spaces to evade individual
monitors. Our evaluation shows 96% accuracy in detecting harmful cases using
our unsupervised ensemble approach.

---


### [Think-J: Learning to Think for Generative LLM-as-a-Judge](http://arxiv.org/abs/2505.14268v1)

LLM-as-a-Judge refers to the automatic modeling of preferences for responses
generated by Large Language Models (LLMs), which is of significant importance
for both LLM evaluation and reward modeling. Although generative LLMs have made
substantial progress in various tasks, their performance as LLM-Judge still
falls short of expectations. In this work, we propose Think-J, which improves
generative LLM-as-a-Judge by learning how to think. We first utilized a small
amount of curated data to develop the model with initial judgment thinking
capabilities. Subsequently, we optimize the judgment thinking traces based on
reinforcement learning (RL). We propose two methods for judgment thinking
optimization, based on offline and online RL, respectively. The offline RL
requires training a critic model to construct positive and negative examples
for learning. The online method defines rule-based reward as feedback for
optimization. Experimental results showed that our approach can significantly
enhance the evaluation capability of generative LLM-Judge, surpassing both
generative and classifier-based LLM-Judge without requiring extra human
annotations.

---


### [Visual Agentic Reinforcement Fine-Tuning](http://arxiv.org/abs/2505.14246v1)

A key trend in Large Reasoning Models (e.g., OpenAI's o3) is the native
agentic ability to use external tools such as web browsers for searching and
writing/executing code for image manipulation to think with images. In the
open-source research community, while significant progress has been made in
language-only agentic abilities such as function calling and tool integration,
the development of multi-modal agentic capabilities that involve truly thinking
with images, and their corresponding benchmarks, are still less explored. This
work highlights the effectiveness of Visual Agentic Reinforcement Fine-Tuning
(Visual-ARFT) for enabling flexible and adaptive reasoning abilities for Large
Vision-Language Models (LVLMs). With Visual-ARFT, open-source LVLMs gain the
ability to browse websites for real-time information updates and write code to
manipulate and analyze input images through cropping, rotation, and other image
processing techniques. We also present a Multi-modal Agentic Tool Bench (MAT)
with two settings (MAT-Search and MAT-Coding) designed to evaluate LVLMs'
agentic search and coding abilities. Our experimental results demonstrate that
Visual-ARFT outperforms its baseline by +18.6% F1 / +13.0% EM on MAT-Coding and
+10.3% F1 / +8.7% EM on MAT-Search, ultimately surpassing GPT-4o. Visual-ARFT
also achieves +29.3 F1% / +25.9% EM gains on existing multi-hop QA benchmarks
such as 2Wiki and HotpotQA, demonstrating strong generalization capabilities.
Our findings suggest that Visual-ARFT offers a promising path toward building
robust and generalizable multimodal agents.

---


### [ABBA: Highly Expressive Hadamard Product Adaptation for Large Language Models](http://arxiv.org/abs/2505.14238v1)

Large Language Models have demonstrated strong performance across a wide
range of tasks, but adapting them efficiently to new domains remains a key
challenge. Parameter-Efficient Fine-Tuning (PEFT) methods address this by
introducing lightweight, trainable modules while keeping most pre-trained
weights fixed. The prevailing approach, LoRA, models updates using a low-rank
decomposition, but its expressivity is inherently constrained by the rank.
Recent methods like HiRA aim to increase expressivity by incorporating a
Hadamard product with the frozen weights, but still rely on the structure of
the pre-trained model. We introduce ABBA, a new PEFT architecture that
reparameterizes the update as a Hadamard product of two independently learnable
low-rank matrices. In contrast to prior work, ABBA fully decouples the update
from the pre-trained weights, enabling both components to be optimized freely.
This leads to significantly higher expressivity under the same parameter
budget. We formally analyze ABBA's expressive capacity and validate its
advantages through matrix reconstruction experiments. Empirically, ABBA
achieves state-of-the-art results on arithmetic and commonsense reasoning
benchmarks, consistently outperforming existing PEFT methods by a significant
margin across multiple models. Our code is publicly available at:
https://github.com/CERT-Lab/abba.

---


### [Automatic Dataset Generation for Knowledge Intensive Question Answering Tasks](http://arxiv.org/abs/2505.14212v1)

A question-answering (QA) system is to search suitable answers within a
knowledge base. Current QA systems struggle with queries requiring complex
reasoning or real-time knowledge integration. They are often supplemented with
retrieval techniques on a data source such as Retrieval-Augmented Generation
(RAG). However, RAG continues to face challenges in handling complex reasoning
and logical connections between multiple sources of information. A novel
approach for enhancing Large Language Models (LLMs) in knowledge-intensive QA
tasks is presented through the automated generation of context-based QA pairs.
This methodology leverages LLMs to create fine-tuning data, reducing reliance
on human labelling and improving model comprehension and reasoning
capabilities. The proposed system includes an automated QA generator and a
model fine-tuner, evaluated using perplexity, ROUGE, BLEU, and BERTScore.
Comprehensive experiments demonstrate improvements in logical coherence and
factual accuracy, with implications for developing adaptable Artificial
Intelligence (AI) systems. Mistral-7b-v0.3 outperforms Llama-3-8b with BERT F1,
BLEU, and ROUGE scores 0.858, 0.172, and 0.260 of for the LLM generated QA
pairs compared to scores of 0.836, 0.083, and 0.139 for the human annotated QA
pairs.

---


### [Tokenization Constraints in LLMs: A Study of Symbolic and Arithmetic Reasoning Limits](http://arxiv.org/abs/2505.14178v1)

Tokenization is the first - and often underappreciated - layer of computation
in language models. While Chain-of-Thought (CoT) prompting enables transformer
models to approximate recurrent computation by externalizing intermediate
steps, we show that the success of such reasoning is fundamentally bounded by
the structure of tokenized inputs. This work presents a theoretical and
empirical investigation into how tokenization schemes, particularly
subword-based methods like byte-pair encoding (BPE), impede symbolic
computation by merging or obscuring atomic reasoning units. We introduce the
notion of Token Awareness to formalize how poor token granularity disrupts
logical alignment and prevents models from generalizing symbolic procedures.
Through systematic evaluation on arithmetic and symbolic tasks, we demonstrate
that token structure dramatically affect reasoning performance, causing failure
even with CoT, while atomically-aligned formats unlock strong generalization,
allowing small models (e.g., GPT-4o-mini) to outperform larger systems (e.g.,
o1) in structured reasoning. Our findings reveal that symbolic reasoning
ability in LLMs is not purely architectural, but deeply conditioned on
token-level representations.

---


### [DSMentor: Enhancing Data Science Agents with Curriculum Learning and Online Knowledge Accumulation](http://arxiv.org/abs/2505.14163v1)

Large language model (LLM) agents have shown promising performance in
generating code for solving complex data science problems. Recent studies
primarily focus on enhancing in-context learning through improved search,
sampling, and planning techniques, while overlooking the importance of the
order in which problems are tackled during inference. In this work, we develop
a novel inference-time optimization framework, referred to as DSMentor, which
leverages curriculum learning -- a strategy that introduces simpler task first
and progressively moves to more complex ones as the learner improves -- to
enhance LLM agent performance in challenging data science tasks. Our
mentor-guided framework organizes data science tasks in order of increasing
difficulty and incorporates a growing long-term memory to retain prior
experiences, guiding the agent's learning progression and enabling more
effective utilization of accumulated knowledge. We evaluate DSMentor through
extensive experiments on DSEval and QRData benchmarks. Experiments show that
DSMentor using Claude-3.5-Sonnet improves the pass rate by up to 5.2% on DSEval
and QRData compared to baseline agents. Furthermore, DSMentor demonstrates
stronger causal reasoning ability, improving the pass rate by 8.8% on the
causality problems compared to GPT-4 using Program-of-Thoughts prompts. Our
work underscores the importance of developing effective strategies for
accumulating and utilizing knowledge during inference, mirroring the human
learning process and opening new avenues for improving LLM performance through
curriculum-based inference optimization.

---


### [MM-Agent: LLM as Agents for Real-world Mathematical Modeling Problem](http://arxiv.org/abs/2505.14148v1)

Mathematical modeling is a cornerstone of scientific discovery and
engineering practice, enabling the translation of real-world problems into
formal systems across domains such as physics, biology, and economics. Unlike
mathematical reasoning, which assumes a predefined formulation, modeling
requires open-ended problem analysis, abstraction, and principled
formalization. While Large Language Models (LLMs) have shown strong reasoning
capabilities, they fall short in rigorous model construction, limiting their
utility in real-world problem-solving. To this end, we formalize the task of
LLM-powered real-world mathematical modeling, where agents must analyze
problems, construct domain-appropriate formulations, and generate complete
end-to-end solutions. We introduce MM-Bench, a curated benchmark of 111
problems from the Mathematical Contest in Modeling (MCM/ICM), spanning the
years 2000 to 2025 and across ten diverse domains such as physics, biology, and
economics. To tackle this task, we propose MM-Agent, an expert-inspired
framework that decomposes mathematical modeling into four stages: open-ended
problem analysis, structured model formulation, computational problem solving,
and report generation. Experiments on MM-Bench show that MM-Agent significantly
outperforms baseline agents, achieving an 11.88\% improvement over human expert
solutions while requiring only 15 minutes and \$0.88 per task using GPT-4o.
Furthermore, under official MCM/ICM protocols, MM-Agent assisted two
undergraduate teams in winning the Finalist Award (\textbf{top 2.0\% among
27,456 teams}) in MCM/ICM 2025, demonstrating its practical effectiveness as a
modeling copilot. Our code is available at
https://github.com/usail-hkust/LLM-MM-Agent

---


### [SHARP: Synthesizing High-quality Aligned Reasoning Problems for Large Reasoning Models Reinforcement Learning](http://arxiv.org/abs/2505.14147v1)

Training large reasoning models (LRMs) with reinforcement learning in STEM
domains is hindered by the scarcity of high-quality, diverse, and verifiable
problem sets. Existing synthesis methods, such as Chain-of-Thought prompting,
often generate oversimplified or uncheckable data, limiting model advancement
on complex tasks. To address these challenges, we introduce SHARP, a unified
approach to Synthesizing High-quality Aligned Reasoning Problems for LRMs
reinforcement learning with verifiable rewards (RLVR). SHARP encompasses a
strategic set of self-alignment principles -- targeting graduate and
Olympiad-level difficulty, rigorous logical consistency, and unambiguous,
verifiable answers -- and a structured three-phase framework (Alignment,
Instantiation, Inference) that ensures thematic diversity and fine-grained
control over problem generation. We implement SHARP by leveraging a
state-of-the-art LRM to infer and verify challenging STEM questions, then
employ a reinforcement learning loop to refine the model's reasoning through
verifiable reward signals. Experiments on benchmarks such as GPQA demonstrate
that SHARP-augmented training substantially outperforms existing methods,
markedly improving complex reasoning accuracy and pushing LRM performance
closer to expert-level proficiency. Our contributions include the SHARP
strategy, framework design, end-to-end implementation, and experimental
evaluation of its effectiveness in elevating LRM reasoning capabilities.

---


### [s3: You Don't Need That Much Data to Train a Search Agent via RL](http://arxiv.org/abs/2505.14146v1)

Retrieval-augmented generation (RAG) systems empower large language models
(LLMs) to access external knowledge during inference. Recent advances have
enabled LLMs to act as search agents via reinforcement learning (RL), improving
information acquisition through multi-turn interactions with retrieval engines.
However, existing approaches either optimize retrieval using search-only
metrics (e.g., NDCG) that ignore downstream utility or fine-tune the entire LLM
to jointly reason and retrieve-entangling retrieval with generation and
limiting the real search utility and compatibility with frozen or proprietary
models. In this work, we propose s3, a lightweight, model-agnostic framework
that decouples the searcher from the generator and trains the searcher using a
Gain Beyond RAG reward: the improvement in generation accuracy over naive RAG.
s3 requires only 2.4k training samples to outperform baselines trained on over
70x more data, consistently delivering stronger downstream performance across
six general QA and five medical QA benchmarks.

---


### [Building a Stable Planner: An Extended Finite State Machine Based Planning Module for Mobile GUI Agent](http://arxiv.org/abs/2505.14141v1)

Mobile GUI agents execute user commands by directly interacting with the
graphical user interface (GUI) of mobile devices, demonstrating significant
potential to enhance user convenience. However, these agents face considerable
challenges in task planning, as they must continuously analyze the GUI and
generate operation instructions step by step. This process often leads to
difficulties in making accurate task plans, as GUI agents lack a deep
understanding of how to effectively use the target applications, which can
cause them to become "lost" during task execution. To address the task planning
issue, we propose SPlanner, a plug-and-play planning module to generate
execution plans that guide vision language model(VLMs) in executing tasks. The
proposed planning module utilizes extended finite state machines (EFSMs) to
model the control logits and configurations of mobile applications. It then
decomposes a user instruction into a sequence of primary function modeled in
EFSMs, and generate the execution path by traversing the EFSMs. We further
refine the execution path into a natural language plan using an LLM. The final
plan is concise and actionable, and effectively guides VLMs to generate
interactive GUI actions to accomplish user tasks. SPlanner demonstrates strong
performance on dynamic benchmarks reflecting real-world mobile usage. On the
AndroidWorld benchmark, SPlanner achieves a 63.8% task success rate when paired
with Qwen2.5-VL-72B as the VLM executor, yielding a 28.8 percentage point
improvement compared to using Qwen2.5-VL-72B without planning assistance.

---


### [RL of Thoughts: Navigating LLM Reasoning with Inference-time Reinforcement Learning](http://arxiv.org/abs/2505.14140v1)

Despite rapid advancements in large language models (LLMs), the token-level
autoregressive nature constrains their complex reasoning capabilities. To
enhance LLM reasoning, inference-time techniques, including
Chain/Tree/Graph-of-Thought(s), successfully improve the performance, as they
are fairly cost-effective by guiding reasoning through sophisticated logical
structures without modifying LLMs' parameters. However, these manually
predefined, task-agnostic frameworks are applied uniformly across diverse
tasks, lacking adaptability. To improve this, we propose RL-of-Thoughts (RLoT),
where we train a lightweight navigator model with reinforcement learning (RL)
to adaptively enhance LLM reasoning at inference time. Specifically, we design
five basic logic blocks from the perspective of human cognition. During the
reasoning process, the trained RL navigator dynamically selects the suitable
logic blocks and combines them into task-specific logical structures according
to problem characteristics. Experiments across multiple reasoning benchmarks
(AIME, MATH, GPQA, etc.) with multiple LLMs (GPT, Llama, Qwen, and DeepSeek)
illustrate that RLoT outperforms established inference-time techniques by up to
13.4%. Remarkably, with less than 3K parameters, our RL navigator is able to
make sub-10B LLMs comparable to 100B-scale counterparts. Moreover, the RL
navigator demonstrates strong transferability: a model trained on one specific
LLM-task pair can effectively generalize to unseen LLMs and tasks. Our code is
open-source at https://anonymous.4open.science/r/RL-LLM-Reasoning-1A30 for
reproducibility.

---


### [NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI](http://arxiv.org/abs/2505.14064v1)

In many real-world applications, deployed models encounter inputs that differ
from the data seen during training. Out-of-distribution detection identifies
whether an input stems from an unseen distribution, while open-world
recognition flags such inputs to ensure the system remains robust as
ever-emerging, previously $unknown$ categories appear and must be addressed
without retraining. Foundation and vision-language models are pre-trained on
large and diverse datasets with the expectation of broad generalization across
domains, including medical imaging. However, benchmarking these models on test
sets with only a few common outlier types silently collapses the evaluation
back to a closed-set problem, masking failures on rare or truly novel
conditions encountered in clinical use.
  We therefore present $NOVA$, a challenging, real-life $evaluation-only$
benchmark of $\sim$900 brain MRI scans that span 281 rare pathologies and
heterogeneous acquisition protocols. Each case includes rich clinical
narratives and double-blinded expert bounding-box annotations. Together, these
enable joint assessment of anomaly localisation, visual captioning, and
diagnostic reasoning. Because NOVA is never used for training, it serves as an
$extreme$ stress-test of out-of-distribution generalisation: models must bridge
a distribution gap both in sample appearance and in semantic space. Baseline
results with leading vision-language models (GPT-4o, Gemini 2.0 Flash, and
Qwen2.5-VL-72B) reveal substantial performance drops across all tasks,
establishing NOVA as a rigorous testbed for advancing models that can detect,
localize, and reason about truly unknown anomalies.

---


### [ProMind-LLM: Proactive Mental Health Care via Causal Reasoning with Sensor Data](http://arxiv.org/abs/2505.14038v1)

Mental health risk is a critical global public health challenge,
necessitating innovative and reliable assessment methods. With the development
of large language models (LLMs), they stand out to be a promising tool for
explainable mental health care applications. Nevertheless, existing approaches
predominantly rely on subjective textual mental records, which can be distorted
by inherent mental uncertainties, leading to inconsistent and unreliable
predictions. To address these limitations, this paper introduces ProMind-LLM.
We investigate an innovative approach integrating objective behavior data as
complementary information alongside subjective mental records for robust mental
health risk assessment. Specifically, ProMind-LLM incorporates a comprehensive
pipeline that includes domain-specific pretraining to tailor the LLM for mental
health contexts, a self-refine mechanism to optimize the processing of
numerical behavioral data, and causal chain-of-thought reasoning to enhance the
reliability and interpretability of its predictions. Evaluations of two
real-world datasets, PMData and Globem, demonstrate the effectiveness of our
proposed methods, achieving substantial improvements over general LLMs. We
anticipate that ProMind-LLM will pave the way for more dependable,
interpretable, and scalable mental health case solutions.

---


### [Adaptive Cyclic Diffusion for Inference Scaling](http://arxiv.org/abs/2505.14036v1)

Diffusion models have demonstrated strong generative capabilities across
domains ranging from image synthesis to complex reasoning tasks. However, most
inference-time scaling methods rely on fixed denoising schedules, limiting
their ability to allocate computation based on instance difficulty or
task-specific demands adaptively. We introduce the challenge of adaptive
inference-time scaling-dynamically adjusting computational effort during
inference-and propose Adaptive Bi-directional Cyclic Diffusion (ABCD), a
flexible, search-based inference framework. ABCD refines outputs through
bi-directional diffusion cycles while adaptively controlling exploration depth
and termination. It comprises three components: Cyclic Diffusion Search,
Automatic Exploration-Exploitation Balancing, and Adaptive Thinking Time.
Experiments show that ABCD improves performance across diverse tasks while
maintaining computational efficiency.

---


### [FedGraM: Defending Against Untargeted Attacks in Federated Learning via Embedding Gram Matrix](http://arxiv.org/abs/2505.14024v1)

Federated Learning (FL) enables geographically distributed clients to
collaboratively train machine learning models by sharing only their local
models, ensuring data privacy. However, FL is vulnerable to untargeted attacks
that aim to degrade the global model's performance on the underlying data
distribution. Existing defense mechanisms attempt to improve FL's resilience
against such attacks, but their effectiveness is limited in practical FL
environments due to data heterogeneity. On the contrary, we aim to detect and
remove the attacks to mitigate their impact. Generalization contribution plays
a crucial role in distinguishing untargeted attacks. Our observations indicate
that, with limited data, the divergence between embeddings representing
different classes provides a better measure of generalization than direct
accuracy. In light of this, we propose a novel robust aggregation method,
FedGraM, designed to defend against untargeted attacks in FL. The server
maintains an auxiliary dataset containing one sample per class to support
aggregation. This dataset is fed to the local models to extract embeddings.
Then, the server calculates the norm of the Gram Matrix of the embeddings for
each local model. The norm serves as an indicator of each model's inter-class
separation capability in the embedding space. FedGraM identifies and removes
potentially malicious models by filtering out those with the largest norms,
then averages the remaining local models to form the global model. We conduct
extensive experiments to evaluate the performance of FedGraM. Our empirical
results show that with limited data samples used to construct the auxiliary
dataset, FedGraM achieves exceptional performance, outperforming
state-of-the-art defense methods.

---


### [Divide by Question, Conquer by Agent: SPLIT-RAG with Question-Driven Graph Partitioning](http://arxiv.org/abs/2505.13994v1)

Retrieval-Augmented Generation (RAG) systems empower large language models
(LLMs) with external knowledge, yet struggle with efficiency-accuracy
trade-offs when scaling to large knowledge graphs. Existing approaches often
rely on monolithic graph retrieval, incurring unnecessary latency for simple
queries and fragmented reasoning for complex multi-hop questions. To address
these challenges, this paper propose SPLIT-RAG, a multi-agent RAG framework
that addresses these limitations with question-driven semantic graph
partitioning and collaborative subgraph retrieval. The innovative framework
first create Semantic Partitioning of Linked Information, then use the
Type-Specialized knowledge base to achieve Multi-Agent RAG. The attribute-aware
graph segmentation manages to divide knowledge graphs into semantically
coherent subgraphs, ensuring subgraphs align with different query types, while
lightweight LLM agents are assigned to partitioned subgraphs, and only relevant
partitions are activated during retrieval, thus reduce search space while
enhancing efficiency. Finally, a hierarchical merging module resolves
inconsistencies across subgraph-derived answers through logical verifications.
Extensive experimental validation demonstrates considerable improvements
compared to existing approaches.

---


### [Solving Normalized Cut Problem with Constrained Action Space](http://arxiv.org/abs/2505.13986v1)

Reinforcement Learning (RL) has emerged as an important paradigm to solve
combinatorial optimization problems primarily due to its ability to learn
heuristics that can generalize across problem instances. However, integrating
external knowledge that will steer combinatorial optimization problem solutions
towards domain appropriate outcomes remains an extremely challenging task. In
this paper, we propose the first RL solution that uses constrained action
spaces to guide the normalized cut problem towards pre-defined template
instances. Using transportation networks as an example domain, we create a
Wedge and Ring Transformer that results in graph partitions that are shaped in
form of Wedges and Rings and which are likely to be closer to natural optimal
partitions. However, our approach is general as it is based on principles that
can be generalized to other domains.

---


### [Toward Effective Reinforcement Learning Fine-Tuning for Medical VQA in Vision-Language Models](http://arxiv.org/abs/2505.13973v1)

Recently, reinforcement learning (RL)-based tuning has shifted the trajectory
of Multimodal Large Language Models (MLLMs), particularly following the
introduction of Group Relative Policy Optimization (GRPO). However, directly
applying it to medical tasks remains challenging for achieving clinically
grounded model behavior. Motivated by the need to align model response with
clinical expectations, we investigate four critical dimensions that affect the
effectiveness of RL-based tuning in medical visual question answering (VQA):
base model initialization strategy, the role of medical semantic alignment, the
impact of length-based rewards on long-chain reasoning, and the influence of
bias. We conduct extensive experiments to analyze these factors for medical
MLLMs, providing new insights into how models are domain-specifically
fine-tuned. Additionally, our results also demonstrate that GRPO-based RL
tuning consistently outperforms standard supervised fine-tuning (SFT) in both
accuracy and reasoning quality.

---


### [EEG-to-Text Translation: A Model for Deciphering Human Brain Activity](http://arxiv.org/abs/2505.13936v1)

With the rapid advancement of large language models like Gemini, GPT, and
others, bridging the gap between the human brain and language processing has
become an important area of focus. To address this challenge, researchers have
developed various models to decode EEG signals into text. However, these models
still face significant performance limitations. To overcome these shortcomings,
we propose a new model, R1 Translator, which aims to improve the performance of
EEG-to-text decoding. The R1 Translator model combines a bidirectional LSTM
encoder with a pretrained transformer-based decoder, utilizing EEG features to
produce high-quality text outputs. The model processes EEG embeddings through
the LSTM to capture sequential dependencies, which are then fed into the
transformer decoder for effective text generation. The R1 Translator excels in
ROUGE metrics, outperforming both T5 (previous research) and Brain Translator.
Specifically, R1 achieves a ROUGE-1 score of 38.00% (P), which is up to 9%
higher than T5 (34.89%) and 3% better than Brain (35.69%). It also leads in
ROUGE-L, with a F1 score of 32.51%, outperforming T5 by 3% (29.67%) and Brain
by 2% (30.38%). In terms of CER, R1 achieves a CER of 0.5795, which is 2% lower
than T5 (0.5917) and 4% lower than Brain (0.6001). Additionally, R1 performs
better in WER with a score of 0.7280, outperforming T5 by 4.3% (0.7610) and
Brain by 3.6% (0.7553). Code is available at
https://github.com/Mmurrad/EEG-To-text.

---


### [APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight](http://arxiv.org/abs/2505.13921v1)

Large Language Models (LLMs) demonstrate strong reasoning and task planning
capabilities but remain fundamentally limited in physical interaction modeling.
Existing approaches integrate perception via Vision-Language Models (VLMs) or
adaptive decision-making through Reinforcement Learning (RL), but they fail to
capture dynamic object interactions or require task-specific training, limiting
their real-world applicability. We introduce APEX (Anticipatory
Physics-Enhanced Execution), a framework that equips LLMs with physics-driven
foresight for real-time task planning. APEX constructs structured graphs to
identify and model the most relevant dynamic interactions in the environment,
providing LLMs with explicit physical state updates. Simultaneously, APEX
provides low-latency forward simulations of physically feasible actions,
allowing LLMs to select optimal strategies based on predictive outcomes rather
than static observations. We evaluate APEX on three benchmarks designed to
assess perception, prediction, and decision-making: (1) Physics Reasoning
Benchmark, testing causal inference and object motion prediction; (2) Tetris,
evaluating whether physics-informed prediction enhances decision-making
performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance,
assessing the immediate integration of perception and action feasibility
analysis. APEX significantly outperforms standard LLMs and VLM-based models,
demonstrating the necessity of explicit physics reasoning for bridging the gap
between language-based intelligence and real-world task execution. The source
code and experiment setup are publicly available at
https://github.com/hwj20/APEX_EXP .

---


### [Do Language Models Use Their Depth Efficiently?](http://arxiv.org/abs/2505.13898v1)

Modern LLMs are increasingly deep, and depth correlates with performance,
albeit with diminishing returns. However, do these models use their depth
efficiently? Do they compose more features to create higher-order computations
that are impossible in shallow models, or do they merely spread the same kinds
of computation out over more layers? To address these questions, we analyze the
residual stream of the Llama 3.1 and Qwen 3 family of models. We find: First,
comparing the output of the sublayers to the residual stream reveals that
layers in the second half contribute much less than those in the first half,
with a clear phase transition between the two halves. Second, skipping layers
in the second half has a much smaller effect on future computations and output
predictions. Third, for multihop tasks, we are unable to find evidence that
models are using increased depth to compose subresults in examples involving
many hops. Fourth, we seek to directly address whether deeper models are using
their additional layers to perform new kinds of computation. To do this, we
train linear maps from the residual stream of a shallow model to a deeper one.
We find that layers with the same relative depth map best to each other,
suggesting that the larger model simply spreads the same computations out over
its many layers. All this evidence suggests that deeper models are not using
their depth to learn new kinds of computation, but only using the greater depth
to perform more fine-grained adjustments to the residual. This may help explain
why increasing scale leads to diminishing returns for stacked Transformer
architectures.

---


### [Domain Adaptation of VLM for Soccer Video Understanding](http://arxiv.org/abs/2505.13860v1)

Vision Language Models (VLMs) have demonstrated strong performance in
multi-modal tasks by effectively aligning visual and textual representations.
However, most video understanding VLM research has been domain-agnostic,
leaving the understanding of their transfer learning capability to specialized
domains under-explored. In this work, we address this by exploring the
adaptability of open-source VLMs to specific domains, and focusing on soccer as
an initial case study. Our approach uses large-scale soccer datasets and LLM to
create instruction-following data, and use them to iteratively fine-tune the
general-domain VLM in a curriculum learning fashion (first teaching the model
key soccer concepts to then question answering tasks). The final adapted model,
trained using a curated dataset of 20k video clips, exhibits significant
improvement in soccer-specific tasks compared to the base model, with a 37.5%
relative improvement for the visual question-answering task and an accuracy
improvement from 11.8% to 63.5% for the downstream soccer action classification
task.

---


### [EfficientLLM: Efficiency in Large Language Models](http://arxiv.org/abs/2505.13840v1)

Large Language Models (LLMs) have driven significant progress, yet their
growing parameter counts and context windows incur prohibitive compute, energy,
and monetary costs. We introduce EfficientLLM, a novel benchmark and the first
comprehensive empirical study evaluating efficiency techniques for LLMs at
scale. Conducted on a production-class cluster (48xGH200, 8xH200 GPUs), our
study systematically explores three key axes: (1) architecture pretraining
(efficient attention variants: MQA, GQA, MLA, NSA; sparse Mixture-of-Experts
(MoE)), (2) fine-tuning (parameter-efficient methods: LoRA, RSLoRA, DoRA), and
(3) inference (quantization methods: int4, float16). We define six fine-grained
metrics (Memory Utilization, Compute Utilization, Latency, Throughput, Energy
Consumption, Compression Rate) to capture hardware saturation,
latency-throughput balance, and carbon cost. Evaluating over 100
model-technique pairs (0.5B-72B parameters), we derive three core insights: (i)
Efficiency involves quantifiable trade-offs: no single method is universally
optimal; e.g., MoE reduces FLOPs and improves accuracy but increases VRAM by
40%, while int4 quantization cuts memory/energy by up to 3.9x at a 3-5%
accuracy drop. (ii) Optima are task- and scale-dependent: MQA offers optimal
memory-latency trade-offs for constrained devices, MLA achieves lowest
perplexity for quality-critical tasks, and RSLoRA surpasses LoRA efficiency
only beyond 14B parameters. (iii) Techniques generalize across modalities: we
extend evaluations to Large Vision Models (Stable Diffusion 3.5, Wan 2.1) and
Vision-Language Models (Qwen2.5-VL), confirming effective transferability. By
open-sourcing datasets, evaluation pipelines, and leaderboards, EfficientLLM
provides essential guidance for researchers and engineers navigating the
efficiency-performance landscape of next-generation foundation models.

---


### [Toward Real-World Cooperative and Competitive Soccer with Quadrupedal Robot Teams](http://arxiv.org/abs/2505.13834v1)

Achieving coordinated teamwork among legged robots requires both fine-grained
locomotion control and long-horizon strategic decision-making. Robot soccer
offers a compelling testbed for this challenge, combining dynamic, competitive,
and multi-agent interactions. In this work, we present a hierarchical
multi-agent reinforcement learning (MARL) framework that enables fully
autonomous and decentralized quadruped robot soccer. First, a set of highly
dynamic low-level skills is trained for legged locomotion and ball
manipulation, such as walking, dribbling, and kicking. On top of these, a
high-level strategic planning policy is trained with Multi-Agent Proximal
Policy Optimization (MAPPO) via Fictitious Self-Play (FSP). This learning
framework allows agents to adapt to diverse opponent strategies and gives rise
to sophisticated team behaviors, including coordinated passing, interception,
and dynamic role allocation. With an extensive ablation study, the proposed
learning method shows significant advantages in the cooperative and competitive
multi-agent soccer game. We deploy the learned policies to real quadruped
robots relying solely on onboard proprioception and decentralized localization,
with the resulting system supporting autonomous robot-robot and robot-human
soccer matches on indoor and outdoor soccer courts.

---


### [Multimodal RAG-driven Anomaly Detection and Classification in Laser Powder Bed Fusion using Large Language Models](http://arxiv.org/abs/2505.13828v1)

Additive manufacturing enables the fabrication of complex designs while
minimizing waste, but faces challenges related to defects and process
anomalies. This study presents a novel multimodal Retrieval-Augmented
Generation-based framework that automates anomaly detection across various
Additive Manufacturing processes leveraging retrieved information from
literature, including images and descriptive text, rather than training
datasets. This framework integrates text and image retrieval from scientific
literature and multimodal generation models to perform zero-shot anomaly
identification, classification, and explanation generation in a Laser Powder
Bed Fusion setting. The proposed framework is evaluated on four L-PBF
manufacturing datasets from Oak Ridge National Laboratory, featuring various
printer makes, models, and materials. This evaluation demonstrates the
framework's adaptability and generalizability across diverse images without
requiring additional training. Comparative analysis using Qwen2-VL-2B and
GPT-4o-mini as MLLM within the proposed framework highlights that GPT-4o-mini
outperforms Qwen2-VL-2B and proportional random baseline in manufacturing
anomalies classification. Additionally, the evaluation of the RAG system
confirms that incorporating retrieval mechanisms improves average accuracy by
12% by reducing the risk of hallucination and providing additional information.
The proposed framework can be continuously updated by integrating emerging
research, allowing seamless adaptation to the evolving landscape of AM
technologies. This scalable, automated, and zero-shot-capable framework
streamlines AM anomaly analysis, enhancing efficiency and accuracy.

---


### [Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](http://arxiv.org/abs/2505.13792v1)

Question Answering (QA) poses a challenging and critical problem,
particularly in today's age of interactive dialogue systems such as ChatGPT,
Perplexity, Microsoft Copilot, etc. where users demand both accuracy and
transparency in the model's outputs. Since smaller language models (SLMs) are
computationally more efficient but often under-perform compared to larger
models, Knowledge Distillation (KD) methods allow for finetuning these smaller
models to improve their final performance. Lately, the intermediate tokens or
the so called `reasoning' traces produced by Chain-of-Thought (CoT) or by
reasoning models such as DeepSeek R1 are used as a training signal for KD.
However, these reasoning traces are often verbose and difficult to interpret or
evaluate. In this work, we aim to address the challenge of evaluating the
faithfulness of these reasoning traces and their correlation with the final
performance. To this end, we employ a KD method leveraging rule-based problem
decomposition. This approach allows us to break down complex queries into
structured sub-problems, generating interpretable traces whose correctness can
be readily evaluated, even at inference time. Specifically, we demonstrate this
approach on Open Book QA, decomposing the problem into a Classification step
and an Information Retrieval step, thereby simplifying trace evaluation. Our
SFT experiments with correct and incorrect traces on the CoTemp QA, Microsoft
Machine Reading Comprehension QA, and Facebook bAbI QA datasets reveal the
striking finding that correct traces do not necessarily imply that the model
outputs the correct final solution. Similarly, we find a low correlation
between correct final solutions and intermediate trace correctness. These
results challenge the implicit assumption behind utilizing reasoning traces for
improving SLMs' final performance via KD.

---


### [Preference Learning with Lie Detectors can Induce Honesty or Evasion](http://arxiv.org/abs/2505.13787v1)

As AI systems become more capable, deceptive behaviors can undermine
evaluation and mislead users at deployment. Recent work has shown that lie
detectors can accurately classify deceptive behavior, but they are not
typically used in the training pipeline due to concerns around contamination
and objective hacking. We examine these concerns by incorporating a lie
detector into the labelling step of LLM post-training and evaluating whether
the learned policy is genuinely more honest, or instead learns to fool the lie
detector while remaining deceptive. Using DolusChat, a novel 65k-example
dataset with paired truthful/deceptive responses, we identify three key factors
that determine the honesty of learned policies: amount of exploration during
preference learning, lie detector accuracy, and KL regularization strength. We
find that preference learning with lie detectors and GRPO can lead to policies
which evade lie detectors, with deception rates of over 85\%. However, if the
lie detector true positive rate (TPR) or KL regularization is sufficiently
high, GRPO learns honest policies. In contrast, off-policy algorithms (DPO)
consistently lead to deception rates under 25\% for realistic TPRs. Our results
illustrate a more complex picture than previously assumed: depending on the
context, lie-detector-enhanced training can be a powerful tool for scalable
oversight, or a counterproductive method encouraging undetectable misalignment.

---


### [UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models](http://arxiv.org/abs/2505.14679v1)

Lifelong learning enables large language models (LLMs) to adapt to evolving
information by continually updating their internal knowledge. An ideal system
should support efficient, wide-ranging updates while preserving existing
capabilities and ensuring reliable deployment. Model editing stands out as a
promising solution for this goal, offering a focused and efficient way to
revise a model's internal knowledge. Although recent paradigms have made
notable progress, they often struggle to meet the demands of practical lifelong
adaptation at scale. To bridge this gap, we propose ULTRAEDIT-a fundamentally
new editing solution that is training-, subject- and memory-free, making it
particularly well-suited for ultra-scalable, real-world lifelong model editing.
ULTRAEDIT performs editing through a self-contained process that relies solely
on lightweight linear algebra operations to compute parameter shifts, enabling
fast and consistent parameter modifications with minimal overhead. To improve
scalability in lifelong settings, ULTRAEDIT employs a lifelong normalization
strategy that continuously updates feature statistics across turns, allowing it
to adapt to distributional shifts and maintain consistency over time. ULTRAEDIT
achieves editing speeds over 7x faster than the previous state-of-the-art
method-which was also the fastest known approach-while consuming less than 1/3
the VRAM, making it the only method currently capable of editing a 7B LLM on a
24GB consumer-grade GPU. Furthermore, we construct ULTRAEDITBENCH-the largest
dataset in the field to date, with over 2M editing pairs-and demonstrate that
our method supports up to 1M edits while maintaining high accuracy.
Comprehensive experiments on four datasets and six models show that ULTRAEDIT
consistently achieves superior performance across diverse model editing
scenarios. Our code is available at: https://github.com/XiaojieGu/UltraEdit.

---


### [General-Reasoner: Advancing LLM Reasoning Across All Domains](http://arxiv.org/abs/2505.14652v1)

Reinforcement learning (RL) has recently demonstrated strong potential in
enhancing the reasoning capabilities of large language models (LLMs).
Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero,
enables direct RL training of base LLMs without relying on an intermediate
supervised fine-tuning stage. Despite these advancements, current works for LLM
reasoning mainly focus on mathematical and coding domains, largely due to data
abundance and the ease of answer verification. This limits the applicability
and generalization of such models to broader domains, where questions often
have diverse answer representations, and data is more scarce. In this paper, we
propose General-Reasoner, a novel training paradigm designed to enhance LLM
reasoning capabilities across diverse domains. Our key contributions include:
(1) constructing a large-scale, high-quality dataset of questions with
verifiable answers curated by web crawling, covering a wide range of
disciplines; and (2) developing a generative model-based answer verifier, which
replaces traditional rule-based verification with the capability of
chain-of-thought and context-awareness. We train a series of models and
evaluate them on a wide range of datasets covering wide domains like physics,
chemistry, finance, electronics etc. Our comprehensive evaluation across these
12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC)
demonstrates that General-Reasoner outperforms existing baseline methods,
achieving robust and generalizable reasoning performance while maintaining
superior effectiveness in mathematical reasoning tasks.

---


### [Think Only When You Need with Large Hybrid-Reasoning Models](http://arxiv.org/abs/2505.14631v1)

Recent Large Reasoning Models (LRMs) have shown substantially improved
reasoning capabilities over traditional Large Language Models (LLMs) by
incorporating extended thinking processes prior to producing final responses.
However, excessively lengthy thinking introduces substantial overhead in terms
of token consumption and latency, which is particularly unnecessary for simple
queries. In this work, we introduce Large Hybrid-Reasoning Models (LHRMs), the
first kind of model capable of adaptively determining whether to perform
thinking based on the contextual information of user queries. To achieve this,
we propose a two-stage training pipeline comprising Hybrid Fine-Tuning (HFT) as
a cold start, followed by online reinforcement learning with the proposed
Hybrid Group Policy Optimization (HGPO) to implicitly learn to select the
appropriate thinking mode. Furthermore, we introduce a metric called Hybrid
Accuracy to quantitatively assess the model's capability for hybrid thinking.
Extensive experimental results show that LHRMs can adaptively perform hybrid
thinking on queries of varying difficulty and type. It outperforms existing
LRMs and LLMs in reasoning and general capabilities while significantly
improving efficiency. Together, our work advocates for a reconsideration of the
appropriate use of extended thinking processes and provides a solid starting
point for building hybrid thinking systems.

---


### [Enhancing Learned Knowledge in LoRA Adapters Through Efficient Contrastive Decoding on Ascend NPUs](http://arxiv.org/abs/2505.14620v1)

Huawei Cloud users leverage LoRA (Low-Rank Adaptation) as an efficient and
scalable method to fine-tune and customize large language models (LLMs) for
application-specific needs. However, tasks that require complex reasoning or
deep contextual understanding are often hindered by biases or interference from
the base model when using typical decoding methods like greedy or beam search.
These biases can lead to generic or task-agnostic responses from the base model
instead of leveraging the LoRA-specific adaptations. In this paper, we
introduce Contrastive LoRA Decoding (CoLD), a novel decoding framework designed
to maximize the use of task-specific knowledge in LoRA-adapted models,
resulting in better downstream performance. CoLD uses contrastive decoding by
scoring candidate tokens based on the divergence between the probability
distributions of a LoRA-adapted expert model and the corresponding base model.
This approach prioritizes tokens that better align with the LoRA's learned
representations, enhancing performance for specialized tasks. While effective,
a naive implementation of CoLD is computationally expensive because each
decoding step requires evaluating multiple token candidates across both models.
To address this, we developed an optimized kernel for Huawei's Ascend NPU. CoLD
achieves up to a 5.54% increase in task accuracy while reducing end-to-end
latency by 28% compared to greedy decoding. This work provides practical and
efficient decoding strategies for fine-tuned LLMs in resource-constrained
environments and has broad implications for applied data science in both cloud
and on-premises settings.

---


### [Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models](http://arxiv.org/abs/2505.14617v1)

Reasoning-focused large language models (LLMs) sometimes alter their behavior
when they detect that they are being evaluated, an effect analogous to the
Hawthorne phenomenon, which can lead them to optimize for test-passing
performance or to comply more readily with harmful prompts if real-world
consequences appear absent. We present the first quantitative study of how such
"test awareness" impacts model behavior, particularly its safety alignment. We
introduce a white-box probing framework that (i) linearly identifies
awareness-related activations and (ii) steers models toward or away from test
awareness while monitoring downstream performance. We apply our method to
different state-of-the-art open-source reasoning LLMs across both realistic and
hypothetical tasks. Our results demonstrate that test awareness significantly
impact safety alignment, and is different for different models. By providing
fine-grained control over this latent effect, our work aims to increase trust
in how we perform safety evaluation.

---


### [Context Reasoner: Incentivizing Reasoning Capability for Contextualized Privacy and Safety Compliance via Reinforcement Learning](http://arxiv.org/abs/2505.14585v1)

While Large Language Models (LLMs) exhibit remarkable capabilities, they also
introduce significant safety and privacy risks. Current mitigation strategies
often fail to preserve contextual reasoning capabilities in risky scenarios.
Instead, they rely heavily on sensitive pattern matching to protect LLMs, which
limits the scope. Furthermore, they overlook established safety and privacy
standards, leading to systemic risks for legal compliance. To address these
gaps, we formulate safety and privacy issues into contextualized compliance
problems following the Contextual Integrity (CI) theory. Under the CI
framework, we align our model with three critical regulatory standards: GDPR,
EU AI Act, and HIPAA. Specifically, we employ reinforcement learning (RL) with
a rule-based reward to incentivize contextual reasoning capabilities while
enhancing compliance with safety and privacy norms. Through extensive
experiments, we demonstrate that our method not only significantly enhances
legal compliance (achieving a +17.64% accuracy improvement in safety/privacy
benchmarks) but also further improves general reasoning capability. For
OpenThinker-7B, a strong reasoning model that significantly outperforms its
base model Qwen2.5-7B-Instruct across diverse subjects, our method enhances its
general reasoning capabilities, with +2.05% and +8.98% accuracy improvement on
the MMLU and LegalBench benchmark, respectively.

---


### [Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders](http://arxiv.org/abs/2505.14536v1)

Large language models (LLMs) are now ubiquitous in user-facing applications,
yet they still generate undesirable toxic outputs, including profanity,
vulgarity, and derogatory remarks. Although numerous detoxification methods
exist, most apply broad, surface-level fixes and can therefore easily be
circumvented by jailbreak attacks. In this paper we leverage sparse
autoencoders (SAEs) to identify toxicity-related directions in the residual
stream of models and perform targeted activation steering using the
corresponding decoder vectors. We introduce three tiers of steering
aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing
trade-offs between toxicity reduction and language fluency. At stronger
steering strengths, these causal interventions surpass competitive baselines in
reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2
Small depending on the aggressiveness. Crucially, standard NLP benchmark scores
upon steering remain stable, indicating that the model's knowledge and general
abilities are preserved. We further show that feature-splitting in wider SAEs
hampers safety interventions, underscoring the importance of disentangled
feature learning. Our findings highlight both the promise and the current
limitations of SAE-based causal interventions for LLM detoxification, further
suggesting practical guidelines for safer language-model deployment.

---


### [PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models](http://arxiv.org/abs/2505.14481v1)

In the field of urban planning, existing Vision-Language Models (VLMs)
frequently fail to effectively analyze and evaluate planning maps, despite the
critical importance of these visual elements for urban planners and related
educational contexts. Planning maps, which visualize land use, infrastructure
layouts, and functional zoning, require specialized understanding of spatial
configurations, regulatory requirements, and multi-scale analysis. To address
this challenge, we introduce PlanGPT-VL, the first domain-specific
Vision-Language Model tailored specifically for urban planning maps. PlanGPT-VL
employs three innovative approaches: (1) PlanAnno-V framework for high-quality
VQA data synthesis, (2) Critical Point Thinking to reduce hallucinations
through structured verification, and (3) comprehensive training methodology
combining Supervised Fine-Tuning with frozen vision encoder parameters. Through
systematic evaluation on our proposed PlanBench-V benchmark, we demonstrate
that PlanGPT-VL significantly outperforms general-purpose state-of-the-art VLMs
in specialized planning map interpretation tasks, offering urban planning
professionals a reliable tool for map analysis, assessment, and educational
applications while maintaining high factual accuracy. Our lightweight 7B
parameter model achieves comparable performance to models exceeding 72B
parameters, demonstrating efficient domain specialization without sacrificing
performance.

---


### [RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](http://arxiv.org/abs/2505.14462v1)

As vision-language models (VLMs) become increasingly integrated into daily
life, the need for accurate visual culture understanding is becoming critical.
Yet, these models frequently fall short in interpreting cultural nuances
effectively. Prior work has demonstrated the effectiveness of
retrieval-augmented generation (RAG) in enhancing cultural understanding in
text-only settings, while its application in multimodal scenarios remains
underexplored. To bridge this gap, we introduce RAVENEA (Retrieval-Augmented
Visual culturE uNdErstAnding), a new benchmark designed to advance visual
culture understanding through retrieval, focusing on two tasks: culture-focused
visual question answering (cVQA) and culture-informed image captioning (cIC).
RAVENEA extends existing datasets by integrating over 10,000 Wikipedia
documents curated and ranked by human annotators. With RAVENEA, we train and
evaluate seven multimodal retrievers for each image query, and measure the
downstream impact of retrieval-augmented inputs across fourteen
state-of-the-art VLMs. Our results show that lightweight VLMs, when augmented
with culture-aware retrieval, outperform their non-augmented counterparts (by
at least 3.2% absolute on cVQA and 6.2% absolute on cIC). This highlights the
value of retrieval-augmented methods and culturally inclusive benchmarks for
multimodal understanding.

---


### [Dual Decomposition of Weights and Singular Value Low Rank Adaptation](http://arxiv.org/abs/2505.14367v1)

Parameter-Efficient Fine-Tuning (PEFT) has emerged as a critical paradigm for
adapting Large Language Models (LLMs) to downstream tasks, among which Low-rank
Adaptation (LoRA) represents one of the most widely adopted methodologies.
However, existing LoRA-based approaches exhibit two fundamental limitations:
unstable training dynamics and inefficient knowledge transfer from pre-trained
models, both stemming from random initialization of adapter parameters. To
overcome these challenges, we propose DuDe, a novel approach that decomposes
weight matrices into magnitude and direction components, employing Singular
Value Decomposition (SVD) for principled initialization. Our comprehensive
evaluation demonstrates DuDe's superior performance and robustness, achieving
up to 48.35\% accuracy on MMLU and 62.53\% ($\pm$ 1.59) accuracy on GSM8K. Our
theoretical analysis and empirical validation collectively demonstrate that
DuDe's decomposition strategy enhances optimization stability and better
preserves pre-trained representations, particularly for domain-specific tasks
requiring specialized knowledge. The combination of robust empirical
performance and rigorous theoretical foundations establishes DuDe as a
significant contribution to PEFT methodologies for LLMs.

---


### [OSoRA: Output-Dimension and Singular-Value Initialized Low-Rank Adaptation](http://arxiv.org/abs/2505.14350v1)

Fine-tuning Large Language Models (LLMs) has become increasingly challenging
due to their massive scale and associated computational costs.
Parameter-Efficient Fine-Tuning (PEFT) methodologies have been proposed as
computational alternatives; however, their implementations still require
significant resources. In this paper, we present OSoRA (Output-Dimension and
Singular-Value Initialized Low-Rank Adaptation), a novel PEFT method for LLMs.
OSoRA extends Low-Rank Adaptation (LoRA) by integrating Singular Value
Decomposition (SVD) with learnable scaling vectors in a unified framework. It
first performs an SVD of pre-trained weight matrices, then optimizes an
output-dimension vector during training, while keeping the corresponding
singular vector matrices frozen. OSoRA substantially reduces computational
resource requirements by minimizing the number of trainable parameters during
fine-tuning. Comprehensive evaluations across mathematical reasoning, common
sense reasoning, and other benchmarks demonstrate that OSoRA achieves
comparable or superior performance to state-of-the-art methods like LoRA and
VeRA, while maintaining a linear parameter scaling even as the rank increases
to higher dimensions. Our ablation studies further confirm that jointly
training both the singular values and the output-dimension vector is critical
for optimal performance.

---


### [A MIND for Reasoning: Meta-learning for In-context Deduction](http://arxiv.org/abs/2505.14313v1)

Large language models (LLMs) are increasingly evaluated on formal tasks,
where strong reasoning abilities define the state of the art. However, their
ability to generalize to out-of-distribution problems remains limited. In this
paper, we investigate how LLMs can achieve a systematic understanding of
deductive rules. Our focus is on the task of identifying the appropriate subset
of premises within a knowledge base needed to derive a given hypothesis. To
tackle this challenge, we propose Meta-learning for In-context Deduction
(MIND), a novel few-shot meta-learning fine-tuning approach. The goal of MIND
is to enable models to generalize more effectively to unseen knowledge bases
and to systematically apply inference rules. Our results show that MIND
significantly improves generalization in small LMs ranging from 1.5B to 7B
parameters. The benefits are especially pronounced in smaller models and
low-data settings. Remarkably, small models fine-tuned with MIND outperform
state-of-the-art LLMs, such as GPT-4o and o3-mini, on this task.

---


### [Scaling Law for Quantization-Aware Training](http://arxiv.org/abs/2505.14302v1)

Large language models (LLMs) demand substantial computational and memory
resources, creating deployment challenges. Quantization-aware training (QAT)
addresses these challenges by reducing model precision while maintaining
performance. However, the scaling behavior of QAT, especially at 4-bit
precision (W4A4), is not well understood. Existing QAT scaling laws often
ignore key factors such as the number of training tokens and quantization
granularity, which limits their applicability. This paper proposes a unified
scaling law for QAT that models quantization error as a function of model size,
training data volume, and quantization group size. Through 268 QAT experiments,
we show that quantization error decreases as model size increases, but rises
with more training tokens and coarser quantization granularity. To identify the
sources of W4A4 quantization error, we decompose it into weight and activation
components. Both components follow the overall trend of W4A4 quantization
error, but with different sensitivities. Specifically, weight quantization
error increases more rapidly with more training tokens. Further analysis shows
that the activation quantization error in the FC2 layer, caused by outliers, is
the primary bottleneck of W4A4 QAT quantization error. By applying
mixed-precision quantization to address this bottleneck, we demonstrate that
weight and activation quantization errors can converge to similar levels.
Additionally, with more training data, weight quantization error eventually
exceeds activation quantization error, suggesting that reducing weight
quantization error is also important in such scenarios. These findings offer
key insights for improving QAT research and development.

---


### [Cross-Lingual Optimization for Language Transfer in Large Language Models](http://arxiv.org/abs/2505.14297v1)

Adapting large language models to other languages typically employs
supervised fine-tuning (SFT) as a standard approach. However, it often suffers
from an overemphasis on English performance, a phenomenon that is especially
pronounced in data-constrained environments. To overcome these challenges, we
propose \textbf{Cross-Lingual Optimization (CLO)} that efficiently transfers an
English-centric LLM to a target language while preserving its English
capabilities. CLO utilizes publicly available English SFT data and a
translation model to enable cross-lingual transfer. We conduct experiments
using five models on six languages, each possessing varying levels of resource.
Our results show that CLO consistently outperforms SFT in both acquiring target
language proficiency and maintaining English performance. Remarkably, in
low-resource languages, CLO with only 3,200 samples surpasses SFT with 6,400
samples, demonstrating that CLO can achieve better performance with less data.
Furthermore, we find that SFT is particularly sensitive to data quantity in
medium and low-resource languages, whereas CLO remains robust. Our
comprehensive analysis emphasizes the limitations of SFT and incorporates
additional training strategies in CLO to enhance efficiency.

---


### [AAPO: Enhance the Reasoning Capabilities of LLMs with Advantage Momentum](http://arxiv.org/abs/2505.14264v1)

Reinforcement learning (RL) has emerged as an effective approach for
enhancing the reasoning capabilities of large language models (LLMs),
especially in scenarios where supervised fine-tuning (SFT) falls short due to
limited chain-of-thought (CoT) data. Among RL-based post-training methods,
group relative advantage estimation, as exemplified by Group Relative Policy
Optimization (GRPO), has attracted considerable attention for eliminating the
dependency on the value model, thereby simplifying training compared to
traditional approaches like Proximal Policy Optimization (PPO). However, we
observe that exsiting group relative advantage estimation method still suffers
from training inefficiencies, particularly when the estimated advantage
approaches zero. To address this limitation, we propose Advantage-Augmented
Policy Optimization (AAPO), a novel RL algorithm that optimizes the
cross-entropy (CE) loss using advantages enhanced through a momentum-based
estimation scheme. This approach effectively mitigates the inefficiencies
associated with group relative advantage estimation. Experimental results on
multiple mathematical reasoning benchmarks demonstrate the superior performance
of AAPO.

---


### [Technical Report on classification of literature related to children speech disorder](http://arxiv.org/abs/2505.14242v1)

This technical report presents a natural language processing (NLP)-based
approach for systematically classifying scientific literature on childhood
speech disorders. We retrieved and filtered 4,804 relevant articles published
after 2015 from the PubMed database using domain-specific keywords. After
cleaning and pre-processing the abstracts, we applied two topic modeling
techniques - Latent Dirichlet Allocation (LDA) and BERTopic - to identify
latent thematic structures in the corpus. Our models uncovered 14 clinically
meaningful clusters, such as infantile hyperactivity and abnormal epileptic
behavior. To improve relevance and precision, we incorporated a custom stop
word list tailored to speech pathology. Evaluation results showed that the LDA
model achieved a coherence score of 0.42 and a perplexity of -7.5, indicating
strong topic coherence and predictive performance. The BERTopic model exhibited
a low proportion of outlier topics (less than 20%), demonstrating its capacity
to classify heterogeneous literature effectively. These results provide a
foundation for automating literature reviews in speech-language pathology.

---


### [Cheaper, Better, Faster, Stronger: Robust Text-to-SQL without Chain-of-Thought or Fine-Tuning](http://arxiv.org/abs/2505.14174v1)

LLMs are effective at code generation tasks like text-to-SQL, but is it worth
the cost? Many state-of-the-art approaches use non-task-specific LLM techniques
including Chain-of-Thought (CoT), self-consistency, and fine-tuning. These
methods can be costly at inference time, sometimes requiring over a hundred LLM
calls with reasoning, incurring average costs of up to \$0.46 per query, while
fine-tuning models can cost thousands of dollars. We introduce "N-rep"
consistency, a more cost-efficient text-to-SQL approach that achieves similar
BIRD benchmark scores as other more expensive methods, at only \$0.039 per
query. N-rep leverages multiple representations of the same schema input to
mitigate weaknesses in any single representation, making the solution more
robust and allowing the use of smaller and cheaper models without any reasoning
or fine-tuning. To our knowledge, N-rep is the best-performing text-to-SQL
approach in its cost range.

---


### [Temporal Alignment of Time Sensitive Facts with Activation Engineering](http://arxiv.org/abs/2505.14158v1)

Large Language Models (LLMs) are trained on diverse and often conflicting
knowledge spanning multiple domains and time periods. Some of this knowledge is
only valid within specific temporal contexts, such as answering the question,
"Who is the President of the United States in 2022?" Ensuring LLMs generate
time appropriate responses is crucial for maintaining relevance and accuracy.
In this work we explore activation engineering as a method for temporally
aligning LLMs to improve factual recall without any training or dataset
creation. In this research we explore an activation engineering technique to
ground three versions of LLaMA 2 to specific points in time and examine the
effects of varying injection layers and prompting strategies. Our experiments
demonstrate up to a 44% and 16% improvement in relative and explicit prompting
respectively, achieving comparable performance to the fine-tuning method
proposed by Zhao et al. (2024) . Notably, our approach achieves similar results
to the fine-tuning baseline while being significantly more computationally
efficient and requiring no pre-aligned datasets.

---


### [Self-Reasoning Language Models: Unfold Hidden Reasoning Chains with Few Reasoning Catalyst](http://arxiv.org/abs/2505.14116v1)

Inference-time scaling has attracted much attention which significantly
enhance the performance of Large Language Models (LLMs) in complex reasoning
tasks by increasing the length of Chain-of-Thought. These longer intermediate
reasoning rationales embody various meta-reasoning skills in human cognition,
such as reflection and decomposition, being difficult to create and acquire. In
this work, we introduce \textit{Self-Reasoning Language Model} (SRLM), where
the model itself can synthesize longer CoT data and iteratively improve
performance through self-training. By incorporating a few demonstration
examples (i.e., 1,000 samples) on how to unfold hidden reasoning chains from
existing responses, which act as a reasoning catalyst, we demonstrate that SRLM
not only enhances the model's initial performance but also ensures more stable
and consistent improvements in subsequent iterations. Our proposed SRLM
achieves an average absolute improvement of more than $+2.5$ points across five
reasoning tasks: MMLU, GSM8K, ARC-C, HellaSwag, and BBH on two backbone models.
Moreover, it brings more improvements with more times of sampling during
inference, such as absolute $+7.89$ average improvement with $64$ sampling
times, revealing the in-depth, diverse and creative reasoning paths in SRLM
against the strong baseline.

---


### [MultiHal: Multilingual Dataset for Knowledge-Graph Grounded Evaluation of LLM Hallucinations](http://arxiv.org/abs/2505.14101v1)

Large Language Models (LLMs) have inherent limitations of faithfulness and
factuality, commonly referred to as hallucinations. Several benchmarks have
been developed that provide a test bed for factuality evaluation within the
context of English-centric datasets, while relying on supplementary informative
context like web links or text passages but ignoring the available structured
factual resources. To this end, Knowledge Graphs (KGs) have been identified as
a useful aid for hallucination mitigation, as they provide a structured way to
represent the facts about entities and their relations with minimal linguistic
overhead. We bridge the lack of KG paths and multilinguality for factual
language modeling within the existing hallucination evaluation benchmarks and
propose a KG-based multilingual, multihop benchmark called \textbf{MultiHal}
framed for generative text evaluation. As part of our data collection pipeline,
we mined 140k KG-paths from open-domain KGs, from which we pruned noisy
KG-paths, curating a high-quality subset of 25.9k. Our baseline evaluation
shows an absolute scale increase by approximately 0.12 to 0.36 points for the
semantic similarity score in KG-RAG over vanilla QA across multiple languages
and multiple models, demonstrating the potential of KG integration. We
anticipate MultiHal will foster future research towards several graph-based
hallucination mitigation and fact-checking tasks.

---


### [Beyond Chains: Bridging Large Language Models and Knowledge Bases in Complex Question Answering](http://arxiv.org/abs/2505.14099v1)

Knowledge Base Question Answering (KBQA) aims to answer natural language
questions using structured knowledge from KBs. While LLM-only approaches offer
generalization, they suffer from outdated knowledge, hallucinations, and lack
of transparency. Chain-based KG-RAG methods address these issues by
incorporating external KBs, but are limited to simple chain-structured
questions due to the absence of planning and logical structuring. Inspired by
semantic parsing methods, we propose PDRR: a four-stage framework consisting of
Predict, Decompose, Retrieve, and Reason. Our method first predicts the
question type and decomposes the question into structured triples. Then
retrieves relevant information from KBs and guides the LLM as an agent to
reason over and complete the decomposed triples. Experimental results
demonstrate that PDRR consistently outperforms existing methods across various
LLM backbones and achieves superior performance on both chain-structured and
non-chain complex questions.

---


### [BAR: A Backward Reasoning based Agent for Complex Minecraft Tasks](http://arxiv.org/abs/2505.14079v1)

Large language model (LLM) based agents have shown great potential in
following human instructions and automatically completing various tasks. To
complete a task, the agent needs to decompose it into easily executed steps by
planning. Existing studies mainly conduct the planning by inferring what steps
should be executed next starting from the agent's initial state. However, this
forward reasoning paradigm doesn't work well for complex tasks. We propose to
study this issue in Minecraft, a virtual environment that simulates complex
tasks based on real-world scenarios. We believe that the failure of forward
reasoning is caused by the big perception gap between the agent's initial state
and task goal. To this end, we leverage backward reasoning and make the
planning starting from the terminal state, which can directly achieve the task
goal in one step. Specifically, we design a BAckward Reasoning based agent
(BAR). It is equipped with a recursive goal decomposition module, a state
consistency maintaining module and a stage memory module to make robust,
consistent, and efficient planning starting from the terminal state.
Experimental results demonstrate the superiority of BAR over existing methods
and the effectiveness of proposed modules.

---


### [AUTOLAW: Enhancing Legal Compliance in Large Language Models via Case Law Generation and Jury-Inspired Deliberation](http://arxiv.org/abs/2505.14015v1)

The rapid advancement of domain-specific large language models (LLMs) in
fields like law necessitates frameworks that account for nuanced regional legal
distinctions, which are critical for ensuring compliance and trustworthiness.
Existing legal evaluation benchmarks often lack adaptability and fail to
address diverse local contexts, limiting their utility in dynamically evolving
regulatory landscapes. To address these gaps, we propose AutoLaw, a novel
violation detection framework that combines adversarial data generation with a
jury-inspired deliberation process to enhance legal compliance of LLMs. Unlike
static approaches, AutoLaw dynamically synthesizes case law to reflect local
regulations and employs a pool of LLM-based "jurors" to simulate judicial
decision-making. Jurors are ranked and selected based on synthesized legal
expertise, enabling a deliberation process that minimizes bias and improves
detection accuracy. Evaluations across three benchmarks: Law-SG, Case-SG
(legality), and Unfair-TOS (policy), demonstrate AutoLaw's effectiveness:
adversarial data generation improves LLM discrimination, while the jury-based
voting strategy significantly boosts violation detection rates. Our results
highlight the framework's ability to adaptively probe legal misalignments and
deliver reliable, context-aware judgments, offering a scalable solution for
evaluating and enhancing LLMs in legally sensitive applications.

---


### [DRP: Distilled Reasoning Pruning with Skill-aware Step Decomposition for Efficient Large Reasoning Models](http://arxiv.org/abs/2505.13975v1)

While Large Reasoning Models (LRMs) have demonstrated success in complex
reasoning tasks through long chain-of-thought (CoT) reasoning, their inference
often involves excessively verbose reasoning traces, resulting in substantial
inefficiency. To address this, we propose Distilled Reasoning Pruning (DRP), a
hybrid framework that combines inference-time pruning with tuning-based
distillation, two widely used strategies for efficient reasoning. DRP uses a
teacher model to perform skill-aware step decomposition and content pruning,
and then distills the pruned reasoning paths into a student model, enabling it
to reason both efficiently and accurately. Across several challenging
mathematical reasoning datasets, we find that models trained with DRP achieve
substantial improvements in token efficiency without sacrificing accuracy.
Specifically, DRP reduces average token usage on GSM8K from 917 to 328 while
improving accuracy from 91.7% to 94.1%, and achieves a 43% token reduction on
AIME with no performance drop. Further analysis shows that aligning the
reasoning structure of training CoTs with the student's reasoning capacity is
critical for effective knowledge transfer and performance gains.

---


### [Through a Compressed Lens: Investigating the Impact of Quantization on LLM Explainability and Interpretability](http://arxiv.org/abs/2505.13963v1)

Quantization methods are widely used to accelerate inference and streamline
the deployment of large language models (LLMs). While prior research has
extensively investigated the degradation of various LLM capabilities due to
quantization, its effects on model explainability and interpretability, which
are crucial for understanding decision-making processes, remain unexplored. To
address this gap, we conduct comprehensive experiments using three common
quantization techniques at distinct bit widths, in conjunction with two
explainability methods, counterfactual examples and natural language
explanations, as well as two interpretability approaches, knowledge
memorization analysis and latent multi-hop reasoning analysis. We complement
our analysis with a thorough user study, evaluating selected explainability
methods. Our findings reveal that, depending on the configuration, quantization
can significantly impact model explainability and interpretability. Notably,
the direction of this effect is not consistent, as it strongly depends on (1)
the quantization method, (2) the explainability or interpretability approach,
and (3) the evaluation protocol. In some settings, human evaluation shows that
quantization degrades explainability, while in others, it even leads to
improvements. Our work serves as a cautionary tale, demonstrating that
quantization can unpredictably affect model transparency. This insight has
important implications for deploying LLMs in applications where transparency is
a critical requirement.

---


### [Beyond Text: Unveiling Privacy Vulnerabilities in Multi-modal Retrieval-Augmented Generation](http://arxiv.org/abs/2505.13957v1)

Multimodal Retrieval-Augmented Generation (MRAG) systems enhance LMMs by
integrating external multimodal databases, but introduce unexplored privacy
vulnerabilities. While text-based RAG privacy risks have been studied,
multimodal data presents unique challenges. We provide the first systematic
analysis of MRAG privacy vulnerabilities across vision-language and
speech-language modalities. Using a novel compositional structured prompt
attack in a black-box setting, we demonstrate how attackers can extract private
information by manipulating queries. Our experiments reveal that LMMs can both
directly generate outputs resembling retrieved content and produce descriptions
that indirectly expose sensitive information, highlighting the urgent need for
robust privacy-preserving MRAG techniques.

---


### [Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM](http://arxiv.org/abs/2505.13890v1)

Recent advances in test-time scaling have enabled Large Language Models
(LLMs) to display sophisticated reasoning abilities via extended
Chain-of-Thought (CoT) generation. Despite their potential, these Reasoning
LLMs (RLMs) often demonstrate counterintuitive and unstable behaviors, such as
performance degradation under few-shot prompting, that challenge our current
understanding of RLMs. In this work, we introduce a unified graph-based
analytical framework for better modeling the reasoning processes of RLMs. Our
method first clusters long, verbose CoT outputs into semantically coherent
reasoning steps, then constructs directed reasoning graphs to capture
contextual and logical dependencies among these steps. Through comprehensive
analysis across models and prompting regimes, we reveal that structural
properties, such as exploration density, branching, and convergence ratios,
strongly correlate with reasoning accuracy. Our findings demonstrate how
prompting strategies substantially reshape the internal reasoning structure of
RLMs, directly affecting task outcomes. The proposed framework not only enables
quantitative evaluation of reasoning quality beyond conventional metrics but
also provides practical insights for prompt engineering and the cognitive
analysis of LLMs. Code and resources will be released to facilitate future
research in this direction.

---


### [Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning](http://arxiv.org/abs/2505.13886v1)

Visual-language Chain-of-Thought (CoT) data resources are relatively scarce
compared to text-only counterparts, limiting the improvement of reasoning
capabilities in Vision Language Models (VLMs). However, high-quality
vision-language reasoning data is expensive and labor-intensive to annotate. To
address this issue, we leverage a promising resource: game code, which
naturally contains logical structures and state transition processes.
Therefore, we propose Code2Logic, a novel game-code-driven approach for
multimodal reasoning data synthesis. Our approach leverages Large Language
Models (LLMs) to adapt game code, enabling automatic acquisition of reasoning
processes and results through code execution. Using the Code2Logic approach, we
developed the GameQA dataset to train and evaluate VLMs. GameQA is
cost-effective and scalable to produce, challenging for state-of-the-art
models, and diverse with 30 games and 158 tasks. Surprisingly, despite training
solely on game data, VLMs demonstrated out of domain generalization,
specifically Qwen2.5-VL-7B improving performance by 2.33\% across 7 diverse
vision-language benchmarks. Our code and dataset are available at
https://github.com/tongjingqi/Code2Logic.

---


### [InfiFPO: Implicit Model Fusion via Preference Optimization in Large Language Models](http://arxiv.org/abs/2505.13878v1)

Model fusion combines multiple Large Language Models (LLMs) with different
strengths into a more powerful, integrated model through lightweight training
methods. Existing works on model fusion focus primarily on supervised
fine-tuning (SFT), leaving preference alignment (PA) --a critical phase for
enhancing LLM performance--largely unexplored. The current few fusion methods
on PA phase, like WRPO, simplify the process by utilizing only response outputs
from source models while discarding their probability information. To address
this limitation, we propose InfiFPO, a preference optimization method for
implicit model fusion. InfiFPO replaces the reference model in Direct
Preference Optimization (DPO) with a fused source model that synthesizes
multi-source probabilities at the sequence level, circumventing complex
vocabulary alignment challenges in previous works and meanwhile maintaining the
probability information. By introducing probability clipping and max-margin
fusion strategies, InfiFPO enables the pivot model to align with human
preferences while effectively distilling knowledge from source models.
Comprehensive experiments on 11 widely-used benchmarks demonstrate that InfiFPO
consistently outperforms existing model fusion and preference optimization
methods. When using Phi-4 as the pivot model, InfiFPO improve its average
performance from 79.95 to 83.33 on 11 benchmarks, significantly improving its
capabilities in mathematics, coding, and reasoning tasks.

---


### [Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning](http://arxiv.org/abs/2505.13866v1)

Recent reasoning-focused language models achieve high accuracy by generating
lengthy intermediate reasoning paths before producing final answers. While this
approach is effective in solving problems that require logical thinking, long
reasoning paths significantly increase memory usage and throughput of token
generation, limiting the practical deployment of such models. We propose
Reasoning Path Compression (RPC), a training-free method that accelerates
inference by leveraging the semantic sparsity of reasoning paths. RPC
periodically compresses the KV cache by retaining KV cache that receive high
importance score, which are computed using a selector window composed of
recently generated queries. Experiments show that RPC improves generation
throughput of QwQ-32B by up to 1.60$\times$ compared to the inference with full
KV cache, with an accuracy drop of 1.2% on the AIME 2024 benchmark. Our
findings demonstrate that semantic sparsity in reasoning traces can be
effectively exploited for compression, offering a practical path toward
efficient deployment of reasoning LLMs. Our code is available at
https://github.com/jiwonsong-dev/ReasoningPathCompression.

---


### [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](http://arxiv.org/abs/2505.14677v1)

Learning general-purpose reasoning capabilities has long been a challenging
problem in AI. Recent research in large language models (LLMs), such as
DeepSeek-R1, has shown that reinforcement learning techniques like GRPO can
enable pre-trained LLMs to develop reasoning capabilities using simple
question-answer pairs. In this paper, we aim to train visual language models
(VLMs) to perform reasoning on image data through reinforcement learning and
visual question-answer pairs, without any explicit chain-of-thought (CoT)
supervision. Our findings indicate that simply applying reinforcement learning
to a VLM -- by prompting the model to produce a reasoning chain before
providing an answer -- can lead the model to develop shortcuts from easy
questions, thereby reducing its ability to generalize across unseen data
distributions. We argue that the key to mitigating shortcut learning is to
encourage the model to interpret images prior to reasoning. Therefore, we train
the model to adhere to a caption-reason-answer output format: initially
generating a detailed caption for an image, followed by constructing an
extensive reasoning chain. When trained on 273K CoT-free visual question-answer
pairs and using only reinforcement learning, our model, named Visionary-R1,
outperforms strong multimodal models, such as GPT-4o, Claude3.5-Sonnet, and
Gemini-1.5-Pro, on multiple visual reasoning benchmarks.

---


### [VideoEval-Pro: Robust and Realistic Long Video Understanding Evaluation](http://arxiv.org/abs/2505.14640v1)

Large multimodal models (LMMs) have recently emerged as a powerful tool for
long video understanding (LVU), prompting the development of standardized LVU
benchmarks to evaluate their performance. However, our investigation reveals a
rather sober lesson for existing LVU benchmarks. First, most existing
benchmarks rely heavily on multiple-choice questions (MCQs), whose evaluation
results are inflated due to the possibility of guessing the correct answer;
Second, a significant portion of questions in these benchmarks have strong
priors to allow models to answer directly without even reading the input video.
For example, Gemini-1.5-Pro can achieve over 50\% accuracy given a random frame
from a long video on Video-MME. We also observe that increasing the number of
frames does not necessarily lead to improvement on existing benchmarks, which
is counterintuitive. As a result, the validity and robustness of current LVU
benchmarks are undermined, impeding a faithful assessment of LMMs' long-video
understanding capability. To tackle this problem, we propose VideoEval-Pro, a
realistic LVU benchmark containing questions with open-ended short-answer,
which truly require understanding the entire video. VideoEval-Pro assesses both
segment-level and full-video understanding through perception and reasoning
tasks. By evaluating 21 proprietary and open-source video LMMs, we conclude the
following findings: (1) video LMMs show drastic performance ($>$25\%) drops on
open-ended questions compared with MCQs; (2) surprisingly, higher MCQ scores do
not lead to higher open-ended scores on VideoEval-Pro; (3) compared to other
MCQ benchmarks, VideoEval-Pro benefits more from increasing the number of input
frames. Our results show that VideoEval-Pro offers a more realistic and
reliable measure of long video understanding, providing a clearer view of
progress in this domain.

---


### [Personalize Your Gaussian: Consistent 3D Scene Personalization from a Single Image](http://arxiv.org/abs/2505.14537v1)

Personalizing 3D scenes from a single reference image enables intuitive
user-guided editing, which requires achieving both multi-view consistency
across perspectives and referential consistency with the input image. However,
these goals are particularly challenging due to the viewpoint bias caused by
the limited perspective provided in a single image. Lacking the mechanisms to
effectively expand reference information beyond the original view, existing
methods of image-conditioned 3DGS personalization often suffer from this
viewpoint bias and struggle to produce consistent results. Therefore, in this
paper, we present Consistent Personalization for 3D Gaussian Splatting (CP-GS),
a framework that progressively propagates the single-view reference appearance
to novel perspectives. In particular, CP-GS integrates pre-trained image-to-3D
generation and iterative LoRA fine-tuning to extract and extend the reference
appearance, and finally produces faithful multi-view guidance images and the
personalized 3DGS outputs through a view-consistent generation process guided
by geometric cues. Extensive experiments on real-world scenes show that our
CP-GS effectively mitigates the viewpoint bias, achieving high-quality
personalization that significantly outperforms existing methods. The code will
be released at https://github.com/Yuxuan-W/CP-GS.

---


### [Investigating and Enhancing the Robustness of Large Multimodal Models Against Temporal Inconsistency](http://arxiv.org/abs/2505.14405v1)

Large Multimodal Models (LMMs) have recently demonstrated impressive
performance on general video comprehension benchmarks. Nevertheless, for
broader applications, the robustness of their temporal analysis capability
needs to be thoroughly investigated yet predominantly ignored. Motivated by
this, we propose a novel temporal robustness benchmark (TemRobBench), which
introduces temporal inconsistency perturbations separately at the visual and
textual modalities to assess the robustness of models. We evaluate 16
mainstream LMMs and find that they exhibit over-reliance on prior knowledge and
textual context in adversarial environments, while ignoring the actual temporal
dynamics in the video. To mitigate this issue, we design panoramic direct
preference optimization (PanoDPO), which encourages LMMs to incorporate both
visual and linguistic feature preferences simultaneously. Experimental results
show that PanoDPO can effectively enhance the model's robustness and
reliability in temporal analysis.

---


### [Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach](http://arxiv.org/abs/2505.14336v1)

Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy
environments by integrating visual cues. While recent advances integrate Large
Language Models (LLMs) into AVSR, their high computational cost hinders
deployment in resource-constrained settings. To address this, we propose
Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of
Projectors (SMoP) module to scale model capacity without increasing inference
costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors,
Llama-SMoP enables the use of smaller LLMs while maintaining strong
performance. We explore three SMoP configurations and show that Llama-SMoP DEDR
(Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and
experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation
studies confirm its effectiveness in expert activation, scalability, and noise
robustness.

---


### [UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning](http://arxiv.org/abs/2505.14231v1)

Traditional visual grounding methods primarily focus on single-image
scenarios with simple textual references. However, extending these methods to
real-world scenarios that involve implicit and complex instructions,
particularly in conjunction with multiple images, poses significant challenges,
which is mainly due to the lack of advanced reasoning ability across diverse
multi-modal contexts. In this work, we aim to address the more practical
universal grounding task, and propose UniVG-R1, a reasoning guided multimodal
large language model (MLLM) for universal visual grounding, which enhances
reasoning capabilities through reinforcement learning (RL) combined with
cold-start data. Specifically, we first construct a high-quality
Chain-of-Thought (CoT) grounding dataset, annotated with detailed reasoning
chains, to guide the model towards correct reasoning paths via supervised
fine-tuning. Subsequently, we perform rule-based reinforcement learning to
encourage the model to identify correct reasoning chains, thereby incentivizing
its reasoning capabilities. In addition, we identify a difficulty bias arising
from the prevalence of easy samples as RL training progresses, and we propose a
difficulty-aware weight adjustment strategy to further strengthen the
performance. Experimental results demonstrate the effectiveness of UniVG-R1,
which achieves state-of-the-art performance on MIG-Bench with a 9.1%
improvement over the previous method. Furthermore, our model exhibits strong
generalizability, achieving an average improvement of 23.4% in zero-shot
performance across four image and video reasoning grounding benchmarks. The
project page can be accessed at https://amap-ml.github.io/UniVG-R1-page/.

---


### [Towards Omnidirectional Reasoning with 360-R1: A Dataset, Benchmark, and GRPO-based Method](http://arxiv.org/abs/2505.14197v1)

Omnidirectional images (ODIs), with their 360{\deg} field of view, provide
unparalleled spatial awareness for immersive applications like augmented
reality and embodied AI. However, the capability of existing multi-modal large
language models (MLLMs) to comprehend and reason about such panoramic scenes
remains underexplored. This paper addresses this gap by introducing OmniVQA,
the first dataset and conducting the first benchmark for omnidirectional visual
question answering. Our evaluation of state-of-the-art MLLMs reveals
significant limitations in handling omnidirectional visual question answering,
highlighting persistent challenges in object localization, feature extraction,
and hallucination suppression within panoramic contexts. These results
underscore the disconnect between current MLLM capabilities and the demands of
omnidirectional visual understanding, which calls for dedicated architectural
or training innovations tailored to 360{\deg} imagery. Building on the OmniVQA
dataset and benchmark, we further introduce a rule-based reinforcement learning
method, 360-R1, based on Qwen2.5-VL-Instruct. Concretely, we modify the group
relative policy optimization (GRPO) by proposing three novel reward functions:
(1) reasoning process similarity reward, (2) answer semantic accuracy reward,
and (3) structured format compliance reward. Extensive experiments on our
OmniVQA demonstrate the superiority of our proposed method in omnidirectional
space (+6% improvement).

---


### [UHD Image Dehazing via anDehazeFormer with Atmospheric-aware KV Cache](http://arxiv.org/abs/2505.14010v1)

In this paper, we propose an efficient visual transformer framework for
ultra-high-definition (UHD) image dehazing that addresses the key challenges of
slow training speed and high memory consumption for existing methods. Our
approach introduces two key innovations: 1) an \textbf{a}daptive
\textbf{n}ormalization mechanism inspired by the nGPT architecture that enables
ultra-fast and stable training with a network with a restricted range of
parameter expressions; and 2) we devise an atmospheric scattering-aware KV
caching mechanism that dynamically optimizes feature preservation based on the
physical haze formation model. The proposed architecture improves the training
convergence speed by \textbf{5 $\times$} while reducing memory overhead,
enabling real-time processing of 50 high-resolution images per second on an
RTX4090 GPU. Experimental results show that our approach maintains
state-of-the-art dehazing quality while significantly improving computational
efficiency for 4K/8K image restoration tasks. Furthermore, we provide a new
dehazing image interpretable method with the help of an integrated gradient
attribution map. Our code can be found here:
https://anonymous.4open.science/r/anDehazeFormer-632E/README.md.

---


### [Every Pixel Tells a Story: End-to-End Urdu Newspaper OCR](http://arxiv.org/abs/2505.13943v1)

This paper introduces a comprehensive end-to-end pipeline for Optical
Character Recognition (OCR) on Urdu newspapers. In our approach, we address the
unique challenges of complex multi-column layouts, low-resolution archival
scans, and diverse font styles. Our process decomposes the OCR task into four
key modules: (1) article segmentation, (2) image super-resolution, (3) column
segmentation, and (4) text recognition. For article segmentation, we fine-tune
and evaluate YOLOv11x to identify and separate individual articles from
cluttered layouts. Our model achieves a precision of 0.963 and mAP@50 of 0.975.
For super-resolution, we fine-tune and benchmark the SwinIR model (reaching
32.71 dB PSNR) to enhance the quality of degraded newspaper scans. To do our
column segmentation, we use YOLOv11x to separate columns in text to further
enhance performance - this model reaches a precision of 0.970 and mAP@50 of
0.975. In the text recognition stage, we benchmark a range of LLMs from
different families, including Gemini, GPT, Llama, and Claude. The lowest WER of
0.133 is achieved by Gemini-2.5-Pro.

---


### [FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](http://arxiv.org/abs/2505.13813v1)

The Kolmogorov-Arnold Network (KAN) has been gaining popularity as an
alternative to the multi-layer perceptron (MLP) with its increased
expressiveness and interpretability. However, the KAN can be orders of
magnitude slower due to its increased computational cost and training
instability, limiting its applicability to larger-scale tasks. Recently, the
Kolmogorov-Arnold Transformer (KAT) has been proposed, which can achieve FLOPs
similar to the traditional Transformer with MLPs by leveraging Group-Rational
KAN (GR-KAN). Unfortunately, despite the comparable FLOPs, our
characterizations reveal that the KAT is still 123x slower in training speeds,
indicating that there are other performance bottlenecks beyond FLOPs. In this
paper, we conduct a series of experiments to understand the root cause of the
slowdown in KAT. We uncover that the slowdown can be isolated to memory stalls
and, more specifically, in the backward pass of GR-KAN caused by inefficient
gradient accumulation. To address this memory bottleneck, we propose FlashKAT,
which builds on our restructured kernel that minimizes gradient accumulation
with atomic adds and accesses to slow memory. Evaluations demonstrate that
FlashKAT can achieve a training speedup of 86.5x compared with the
state-of-the-art KAT, while reducing rounding errors in the coefficient
gradients. Our code is available at https://github.com/OSU-STARLAB/FlashKAT.

---


### [Quartet: Native FP4 Training Can Be Optimal for Large Language Models](http://arxiv.org/abs/2505.14669v1)

The rapid advancement of large language models (LLMs) has been paralleled by
unprecedented increases in computational demands, with training costs for
state-of-the-art models doubling every few months. Training models directly in
low-precision arithmetic offers a solution, by improving both computational
throughput and energy efficiency. Specifically, NVIDIA's recent Blackwell
architecture facilitates extremely low-precision operations, specifically FP4
variants, promising substantial efficiency gains. Yet, current algorithms for
training LLMs in FP4 precision face significant accuracy degradation and often
rely on mixed-precision fallbacks. In this paper, we systematically investigate
hardware-supported FP4 training and introduce Quartet, a new approach enabling
accurate, end-to-end FP4 training with all the major computations (in e.g.
linear layers) being performed in low precision. Through extensive evaluations
on Llama-type models, we reveal a new low-precision scaling law that quantifies
performance trade-offs across varying bit-widths and allows us to identify a
"near-optimal" low-precision training technique in terms of
accuracy-vs-computation, called Quartet. We implement Quartet using optimized
CUDA kernels tailored for NVIDIA Blackwell GPUs, and show that it can achieve
state-of-the-art accuracy for FP4 precision, successfully training
billion-scale models. Our method demonstrates that fully FP4-based training is
a competitive alternative to standard-precision and FP8 training. Our code is
available at https://github.com/IST-DASLab/Quartet.

---


### [Interpretable Dual-Stream Learning for Local Wind Hazard Prediction in Vulnerable Communities](http://arxiv.org/abs/2505.14522v1)

Wind hazards such as tornadoes and straight-line winds frequently affect
vulnerable communities in the Great Plains of the United States, where limited
infrastructure and sparse data coverage hinder effective emergency response.
Existing forecasting systems focus primarily on meteorological elements and
often fail to capture community-specific vulnerabilities, limiting their
utility for localized risk assessment and resilience planning. To address this
gap, we propose an interpretable dual-stream learning framework that integrates
structured numerical weather data with unstructured textual event narratives.
Our architecture combines a Random Forest and RoBERTa-based transformer through
a late fusion mechanism, enabling robust and context-aware wind hazard
prediction. The system is tailored for underserved tribal communities and
supports block-level risk assessment. Experimental results show significant
performance gains over traditional baselines. Furthermore, gradient-based
sensitivity and ablation studies provide insight into the model's
decision-making process, enhancing transparency and operational trust. The
findings demonstrate both predictive effectiveness and practical value in
supporting emergency preparedness and advancing community resilience.

---


### [ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs](http://arxiv.org/abs/2505.14468v1)

Serverless computing has grown rapidly for serving Large Language Model (LLM)
inference due to its pay-as-you-go pricing, fine-grained GPU usage, and rapid
scaling. However, our analysis reveals that current serverless can effectively
serve general LLM but fail with Low-Rank Adaptation (LoRA) inference due to
three key limitations: 1) massive parameter redundancy among functions where
99% of weights are unnecessarily duplicated, 2) costly artifact loading latency
beyond LLM loading, and 3) magnified resource contention when serving multiple
LoRA LLMs. These inefficiencies lead to massive GPU wastage, increased
Time-To-First-Token (TTFT), and high monetary costs.
  We propose ServerlessLoRA, a novel serverless inference system designed for
faster and cheaper LoRA LLM serving. ServerlessLoRA enables secure backbone LLM
sharing across isolated LoRA functions to reduce redundancy. We design a
pre-loading method that pre-loads comprehensive LoRA artifacts to minimize
cold-start latency. Furthermore, ServerlessLoRA employs contention aware
batching and offloading to mitigate GPU resource conflicts during bursty
workloads. Experiment on industrial workloads demonstrates that ServerlessLoRA
reduces TTFT by up to 86% and cuts monetary costs by up to 89% compared to
state-of-the-art LLM inference solutions.

---


### [Interpretable Reinforcement Learning for Load Balancing using Kolmogorov-Arnold Networks](http://arxiv.org/abs/2505.14459v1)

Reinforcement learning (RL) has been increasingly applied to network control
problems, such as load balancing. However, existing RL approaches often suffer
from lack of interpretability and difficulty in extracting controller
equations. In this paper, we propose the use of Kolmogorov-Arnold Networks
(KAN) for interpretable RL in network control. We employ a PPO agent with a
1-layer actor KAN model and an MLP Critic network to learn load balancing
policies that maximise throughput utility, minimize loss as well as delay. Our
approach allows us to extract controller equations from the learned neural
networks, providing insights into the decision-making process. We evaluate our
approach using different reward functions demonstrating its effectiveness in
improving network performance while providing interpretable policies.

---


### [Low-Cost FlashAttention with Fused Exponential and Multiplication Hardware Operators](http://arxiv.org/abs/2505.14314v1)

Attention mechanisms, particularly within Transformer architectures and large
language models (LLMs), have revolutionized sequence modeling in machine
learning and artificial intelligence applications. To compute attention for
increasingly long sequences, specialized accelerators have been proposed to
execute key attention steps directly in hardware. Among the various recently
proposed architectures, those based on variants of the FlashAttention
algorithm, originally designed for GPUs, stand out due to their optimized
computation, tiling capabilities, and reduced memory traffic. In this work, we
focus on optimizing the kernel of floating-point-based FlashAttention using new
hardware operators that fuse the computation of exponentials and vector
multiplications, e.g., e^x, V. The proposed ExpMul hardware operators
significantly reduce the area and power costs of FlashAttention-based hardware
accelerators. When implemented in a 28nm ASIC technology, they achieve
improvements of 28.8% in area and 17.6% in power, on average, compared to
state-of-the-art hardware architectures with separate exponentials and vector
multiplications hardware operators.

---


### [VAMO: Efficient Large-Scale Nonconvex Optimization via Adaptive Zeroth Order Variance Reduction](http://arxiv.org/abs/2505.13954v1)

Optimizing large-scale nonconvex problems, common in machine learning,
demands balancing rapid convergence with computational efficiency. First-order
(FO) stochastic methods like SVRG provide fast convergence and good
generalization but incur high costs due to full-batch gradients in large
models. Conversely, zeroth-order (ZO) algorithms reduce this burden using
estimated gradients, yet their slow convergence in high-dimensional settings
limits practicality. We introduce VAMO (VAriance-reduced Mixed-gradient
Optimizer), a stochastic variance-reduced method combining FO mini-batch
gradients with lightweight ZO finite-difference probes under an SVRG-style
framework. VAMO's hybrid design uses a two-point ZO estimator to achieve a
dimension-agnostic convergence rate of $\mathcal{O}(1/T + 1/b)$, where $T$ is
the number of iterations and $b$ is the batch-size, surpassing the
dimension-dependent slowdown of purely ZO methods and significantly improving
over SGD's $\mathcal{O}(1/\sqrt{T})$ rate. Additionally, we propose a
multi-point ZO variant that mitigates the $O(1/b)$ error by adjusting number of
estimation points to balance convergence and cost, making it ideal for a whole
range of computationally constrained scenarios. Experiments including
traditional neural network training and LLM finetuning show VAMO outperforms
established FO and ZO methods, offering a faster, more flexible option for
improved efficiency.

---


### [Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](http://arxiv.org/abs/2505.13925v1)

Symmetry is pervasive in robotics and has been widely exploited to improve
sample efficiency in deep reinforcement learning (DRL). However, existing
approaches primarily focus on spatial symmetries, such as reflection, rotation,
and translation, while largely neglecting temporal symmetries. To address this
gap, we explore time reversal symmetry, a form of temporal symmetry commonly
found in robotics tasks such as door opening and closing. We propose Time
Reversal symmetry enhanced Deep Reinforcement Learning (TR-DRL), a framework
that combines trajectory reversal augmentation and time reversal guided reward
shaping to efficiently solve temporally symmetric tasks. Our method generates
reversed transitions from fully reversible transitions, identified by a
proposed dynamics-consistent filter, to augment the training data. For
partially reversible transitions, we apply reward shaping to guide learning,
according to successful trajectories from the reversed task. Extensive
experiments on the Robosuite and MetaWorld benchmarks demonstrate that TR-DRL
is effective in both single-task and multi-task settings, achieving higher
sample efficiency and stronger final performance compared to baseline methods.

---


### [InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning](http://arxiv.org/abs/2505.13888v1)

Leveraging pretrained Vision-Language Models (VLMs) to map language
instruction and visual observations to raw low-level actions,
Vision-Language-Action models (VLAs) hold great promise for achieving
general-purpose robotic systems. Despite their advancements, existing VLAs tend
to spuriously correlate task-irrelevant visual features with actions, limiting
their generalization capacity beyond the training data. To tackle this
challenge, we propose Intrinsic Spatial Reasoning (InSpire), a simple yet
effective approach that mitigates the adverse effects of spurious correlations
by boosting the spatial reasoning ability of VLAs. Specifically, InSpire
redirects the VLA's attention to task-relevant factors by prepending the
question "In which direction is the [object] relative to the robot?" to the
language instruction and aligning the answer
"right/left/up/down/front/back/grasped" and predicted actions with the
ground-truth. Notably, InSpire can be used as a plugin to enhance existing
autoregressive VLAs, requiring no extra training data or interaction with other
large models. Extensive experimental results in both simulation and real-world
environments demonstrate the effectiveness and flexibility of our approach. Our
code, pretrained models and demos are publicly available at:
https://Koorye.github.io/proj/Inspire.

---


### [SkyMemory: A LEO Edge Cache for Transformer Inference Optimization and Scale Out](http://arxiv.org/abs/2505.14427v1)

We expand the scope of cache memory to include LEO constellations, which are
highly distributed systems with thousands of satellites connected with
free-space optics inter-satellite links (ISL) always only one hop from any
point on earth. We show how to increase the number of cache hits and improve
the speed of inference for the important use case of LLMs. These benefits apply
not only to LLMs, both terrestrially hosted and on satellites, but also
generalize to any cache distributed over multiple locations that needs to be
accessed in a timely manner. We show the benefit of our key value cache (KVC)
protocol in simulations and present a proof-of-concept implementation of the
protocol for KVCs on a testbed comprising 5 Intel NUC Linux mini PCs hosting a
19x5 constellation, with an NVIDIA Jetson Nano 8GB GPU hosting the LLM.

---


