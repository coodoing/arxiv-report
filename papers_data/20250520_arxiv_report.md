### [Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards](http://arxiv.org/abs/2505.13445v1)

Large Language Models (LLMs) show great promise in complex reasoning, with
Reinforcement Learning with Verifiable Rewards (RLVR) being a key enhancement
strategy. However, a prevalent issue is ``superficial self-reflection'', where
models fail to robustly verify their own outputs. We introduce RISE
(Reinforcing Reasoning with Self-Verification), a novel online RL framework
designed to tackle this. RISE explicitly and simultaneously trains an LLM to
improve both its problem-solving and self-verification abilities within a
single, integrated RL process. The core mechanism involves leveraging
verifiable rewards from an outcome verifier to provide on-the-fly feedback for
both solution generation and self-verification tasks. In each iteration, the
model generates solutions, then critiques its own on-policy generated
solutions, with both trajectories contributing to the policy update. Extensive
experiments on diverse mathematical reasoning benchmarks show that RISE
consistently improves model's problem-solving accuracy while concurrently
fostering strong self-verification skills. Our analyses highlight the
advantages of online verification and the benefits of increased verification
compute. Additionally, RISE models exhibit more frequent and accurate
self-verification behaviors during reasoning. These advantages reinforce RISE
as a flexible and effective path towards developing more robust and self-aware
reasoners.

---


### [VTBench: Evaluating Visual Tokenizers for Autoregressive Image Generation](http://arxiv.org/abs/2505.13439v1)

Autoregressive (AR) models have recently shown strong performance in image
generation, where a critical component is the visual tokenizer (VT) that maps
continuous pixel inputs to discrete token sequences. The quality of the VT
largely defines the upper bound of AR model performance. However, current
discrete VTs fall significantly behind continuous variational autoencoders
(VAEs), leading to degraded image reconstructions and poor preservation of
details and text. Existing benchmarks focus on end-to-end generation quality,
without isolating VT performance. To address this gap, we introduce VTBench, a
comprehensive benchmark that systematically evaluates VTs across three core
tasks: Image Reconstruction, Detail Preservation, and Text Preservation, and
covers a diverse range of evaluation scenarios. We systematically assess
state-of-the-art VTs using a set of metrics to evaluate the quality of
reconstructed images. Our findings reveal that continuous VAEs produce superior
visual representations compared to discrete VTs, particularly in retaining
spatial structure and semantic detail. In contrast, the degraded
representations produced by discrete VTs often lead to distorted
reconstructions, loss of fine-grained textures, and failures in preserving text
and object integrity. Furthermore, we conduct experiments on GPT-4o image
generation and discuss its potential AR nature, offering new insights into the
role of visual tokenization. We release our benchmark and codebase publicly to
support further research and call on the community to develop strong,
general-purpose open-source VTs.

---


### [Optimizing Anytime Reasoning via Budget Relative Policy Optimization](http://arxiv.org/abs/2505.13438v1)

Scaling test-time compute is crucial for enhancing the reasoning capabilities
of large language models (LLMs). Existing approaches typically employ
reinforcement learning (RL) to maximize a verifiable reward obtained at the end
of reasoning traces. However, such methods optimize only the final performance
under a large and fixed token budget, which hinders efficiency in both training
and deployment. In this work, we present a novel framework, AnytimeReasoner, to
optimize anytime reasoning performance, which aims to improve token efficiency
and the flexibility of reasoning under varying token budget constraints. To
achieve this, we truncate the complete thinking process to fit within sampled
token budgets from a prior distribution, compelling the model to summarize the
optimal answer for each truncated thinking for verification. This introduces
verifiable dense rewards into the reasoning process, facilitating more
effective credit assignment in RL optimization. We then optimize the thinking
and summary policies in a decoupled manner to maximize the cumulative reward.
Additionally, we introduce a novel variance reduction technique, Budget
Relative Policy Optimization (BRPO), to enhance the robustness and efficiency
of the learning process when reinforcing the thinking policy. Empirical results
in mathematical reasoning tasks demonstrate that our method consistently
outperforms GRPO across all thinking budgets under various prior distributions,
enhancing both training and token efficiency.

---


### [Learnware of Language Models: Specialized Small Language Models Can Do Big](http://arxiv.org/abs/2505.13425v1)

The learnware paradigm offers a novel approach to machine learning by
enabling users to reuse a set of well-trained models for tasks beyond the
models' original purposes. It eliminates the need to build models from scratch,
instead relying on specifications (representations of a model's capabilities)
to identify and leverage the most suitable models for new tasks. While
learnware has proven effective in many scenarios, its application to language
models has remained largely unexplored. At the same time, large language models
(LLMs) have demonstrated remarkable universal question-answering abilities, yet
they face challenges in specialized scenarios due to data scarcity, privacy
concerns, and high computational costs, thus more and more specialized small
language models (SLMs) are being trained for specific domains. To address these
limitations systematically, the learnware paradigm provides a promising
solution by enabling maximum utilization of specialized SLMs, and allowing
users to identify and reuse them in a collaborative and privacy-preserving
manner.
  This paper presents a preliminary attempt to apply the learnware paradigm to
language models. We simulated a learnware system comprising approximately 100
learnwares of specialized SLMs with 8B parameters, fine-tuned across finance,
healthcare, and mathematics domains. Each learnware contains an SLM and a
specification, which enables users to identify the most relevant models without
exposing their own data. Experimental results demonstrate promising
performance: by selecting one suitable learnware for each task-specific
inference, the system outperforms the base SLMs on all benchmarks. Compared to
LLMs, the system outperforms Qwen1.5-110B, Qwen2.5-72B, and
Llama3.1-70B-Instruct by at least 14% in finance domain tasks, and surpasses
Flan-PaLM-540B (ranked 7th on the Open Medical LLM Leaderboard) in medical
domain tasks.

---


### [AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database](http://arxiv.org/abs/2505.13406v1)

A mathematical knowledge graph (KG) presents knowledge within the field of
mathematics in a structured manner. Constructing a math KG using natural
language is an essential but challenging task. There are two major limitations
of existing works: first, they are constrained by corpus completeness, often
discarding or manually supplementing incomplete knowledge; second, they
typically fail to fully automate the integration of diverse knowledge sources.
This paper proposes AutoMathKG, a high-quality, wide-coverage, and
multi-dimensional math KG capable of automatic updates. AutoMathKG regards
mathematics as a vast directed graph composed of Definition, Theorem, and
Problem entities, with their reference relationships as edges. It integrates
knowledge from ProofWiki, textbooks, arXiv papers, and TheoremQA, enhancing
entities and relationships with large language models (LLMs) via in-context
learning for data augmentation. To search for similar entities, MathVD, a
vector database, is built through two designed embedding strategies using
SBERT. To automatically update, two mechanisms are proposed. For knowledge
completion mechanism, Math LLM is developed to interact with AutoMathKG,
providing missing proofs or solutions. For knowledge fusion mechanism, MathVD
is used to retrieve similar entities, and LLM is used to determine whether to
merge with a candidate or add as a new entity. A wide range of experiments
demonstrate the advanced performance and broad applicability of the AutoMathKG
system, including superior reachability query results in MathVD compared to
five baselines and robust mathematical reasoning capability in Math LLM.

---


### [How Adding Metacognitive Requirements in Support of AI Feedback in Practice Exams Transforms Student Learning Behaviors](http://arxiv.org/abs/2505.13381v1)

Providing personalized, detailed feedback at scale in large undergraduate
STEM courses remains a persistent challenge. We present an empirically
evaluated practice exam system that integrates AI generated feedback with
targeted textbook references, deployed in a large introductory biology course.
Our system encourages metacognitive behavior by asking students to explain
their answers and declare their confidence. It uses OpenAI's GPT-4o to generate
personalized feedback based on this information, while directing them to
relevant textbook sections. Through interaction logs from consenting
participants across three midterms (541, 342, and 413 students respectively),
totaling 28,313 question-student interactions across 146 learning objectives,
along with 279 surveys and 23 interviews, we examined the system's impact on
learning outcomes and engagement. Across all midterms, feedback types showed no
statistically significant performance differences, though some trends suggested
potential benefits. The most substantial impact came from the required
confidence ratings and explanations, which students reported transferring to
their actual exam strategies. About 40 percent of students engaged with
textbook references when prompted by feedback -- far higher than traditional
reading rates. Survey data revealed high satisfaction (mean rating 4.1 of 5),
with 82.1 percent reporting increased confidence on practiced midterm topics,
and 73.4 percent indicating they could recall and apply specific concepts. Our
findings suggest that embedding structured reflection requirements may be more
impactful than sophisticated feedback mechanisms.

---


### [Thinkless: LLM Learns When to Think](http://arxiv.org/abs/2505.13379v1)

Reasoning Language Models, capable of extended chain-of-thought reasoning,
have demonstrated remarkable performance on tasks requiring complex logical
inference. However, applying elaborate reasoning for all queries often results
in substantial computational inefficiencies, particularly when many problems
admit straightforward solutions. This motivates an open question: Can LLMs
learn when to think? To answer this, we propose Thinkless, a learnable
framework that empowers an LLM to adaptively select between short-form and
long-form reasoning, based on both task complexity and the model's ability.
Thinkless is trained under a reinforcement learning paradigm and employs two
control tokens, <short> for concise responses and <think> for detailed
reasoning. At the core of our method is a Decoupled Group Relative Policy
Optimization (DeGRPO) algorithm, which decomposes the learning objective of
hybrid reasoning into two components: (1) a control token loss that governs the
selection of the reasoning mode, and (2) a response loss that improves the
accuracy of the generated answers. This decoupled formulation enables
fine-grained control over the contributions of each objective, stabilizing
training and effectively preventing collapse observed in vanilla GRPO.
Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and
GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% -
90%, significantly improving the efficiency of Reasoning Language Models. The
code is available at https://github.com/VainF/Thinkless

---


### [Exploiting Symbolic Heuristics for the Synthesis of Domain-Specific Temporal Planning Guidance using Reinforcement Learning](http://arxiv.org/abs/2505.13372v1)

Recent work investigated the use of Reinforcement Learning (RL) for the
synthesis of heuristic guidance to improve the performance of temporal planners
when a domain is fixed and a set of training problems (not plans) is given. The
idea is to extract a heuristic from the value function of a particular
(possibly infinite-state) MDP constructed over the training problems.
  In this paper, we propose an evolution of this learning and planning
framework that focuses on exploiting the information provided by symbolic
heuristics during both the RL and planning phases. First, we formalize
different reward schemata for the synthesis and use symbolic heuristics to
mitigate the problems caused by the truncation of episodes needed to deal with
the potentially infinite MDP. Second, we propose learning a residual of an
existing symbolic heuristic, which is a "correction" of the heuristic value,
instead of eagerly learning the whole heuristic from scratch. Finally, we use
the learned heuristic in combination with a symbolic heuristic using a
multiple-queue planning approach to balance systematic search with imperfect
learned information. We experimentally compare all the approaches, highlighting
their strengths and weaknesses and significantly advancing the state of the art
for this planning and learning schema.

---


### [J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization](http://arxiv.org/abs/2505.13346v1)

To keep pace with the increasing pace of large language models (LLM)
development, model output evaluation has transitioned away from time-consuming
human evaluation to automatic evaluation, where LLMs themselves are tasked with
assessing and critiquing other model outputs. LLM-as-judge models are a class
of generative evaluators that excel in evaluating relatively simple domains,
like chat quality, but struggle in reasoning intensive domains where model
responses contain more substantive and challenging content. To remedy existing
judge shortcomings, we explore training judges with reinforcement learning
(RL). We make three key contributions: (1) We propose the Equivalent Initial
State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us
to train our judge to be robust to positional biases that arise in more complex
evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that
evaluates judges in diverse reasoning settings not covered by prior work. (3)
We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that
outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or
exceeding the performance of larger GRPO-trained judges on both JudgeBench and
ReasoningJudgeBench.

---


### [Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space](http://arxiv.org/abs/2505.13308v1)

Reasoning ability, a core component of human intelligence, continues to pose
a significant challenge for Large Language Models (LLMs) in the pursuit of AGI.
Although model performance has improved under the training scaling law,
significant challenges remain, particularly with respect to training
algorithms, such as catastrophic forgetting, and the limited availability of
novel training data. As an alternative, test-time scaling enhances reasoning
performance by increasing test-time computation without parameter updating.
Unlike prior methods in this paradigm focused on token space, we propose
leveraging latent space for more effective reasoning and better adherence to
the test-time scaling law. We introduce LatentSeek, a novel framework that
enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA)
within the model's latent space. Specifically, LatentSeek leverages policy
gradient to iteratively update latent representations, guided by self-generated
reward signals. LatentSeek is evaluated on a range of reasoning benchmarks,
including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures.
Results show that LatentSeek consistently outperforms strong baselines, such as
Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our
analysis demonstrates that LatentSeek is highly efficient, typically converging
within a few iterations for problems of average complexity, while also
benefiting from additional iterations, thereby highlighting the potential of
test-time scaling in the latent space. These findings position LatentSeek as a
lightweight, scalable, and effective solution for enhancing the reasoning
capabilities of LLMs.

---


### [RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning](http://arxiv.org/abs/2505.13307v1)

Chain-of-Thought (CoT) reasoning has proven effective in enhancing large
language models (LLMs) on complex tasks, spurring research into its underlying
mechanisms. However, two primary challenges remain for real-world applications:
(1) the lack of quantitative metrics and actionable guidelines for evaluating
and optimizing measurable boundaries of CoT capability, and (2) the absence of
methods to assess boundaries of unmeasurable CoT capability, such as multimodal
perception. To address these gaps, we introduce the Reasoning Boundary
Framework++ (RBF++). To tackle the first challenge, we define the reasoning
boundary (RB) as the maximum limit of CoT performance. We also propose a
combination law for RBs, enabling quantitative analysis and offering actionable
guidance across various CoT tasks. For the second challenge, particularly in
multimodal scenarios, we introduce a constant assumption, which replaces
unmeasurable RBs with scenario-specific constants. Additionally, we propose the
reasoning boundary division mechanism, which divides unmeasurable RBs into two
sub-boundaries, facilitating the quantification and optimization of both
unmeasurable domain knowledge and multimodal perception capabilities. Extensive
experiments involving 38 models across 13 tasks validate the feasibility of our
framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies,
offer insights into optimization and decay from two complementary perspectives,
and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope
this work advances the understanding of RBs and optimization strategies in
LLMs. Code and data are available at
https://github.com/LightChen233/reasoning-boundary.

---


### [StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment](http://arxiv.org/abs/2505.13232v1)

Learning robust representations from data often requires scale, which has led
to the success of recent zero-shot models such as CLIP. However, the obtained
robustness can easily be deteriorated when these models are fine-tuned on other
downstream tasks (e.g., of smaller scales). Previous works often interpret this
phenomenon in the context of domain shift, developing fine-tuning methods that
aim to preserve the original domain as much as possible. However, in a
different context, fine-tuned models with limited data are also prone to
learning features that are spurious to humans, such as background or texture.
In this paper, we propose StarFT (Spurious Textual Alignment Regularization), a
novel framework for fine-tuning zero-shot models to enhance robustness by
preventing them from learning spuriosity. We introduce a regularization that
aligns the output distribution for spuriosity-injected labels with the original
zero-shot model, ensuring that the model is not induced to extract irrelevant
features further from these descriptions.We leverage recent language models to
get such spuriosity-injected labels by generating alternative textual
descriptions that highlight potentially confounding features.Extensive
experiments validate the robust generalization of StarFT and its emerging
properties: zero-shot group robustness and improved zero-shot classification.
Notably, StarFT boosts both worst-group and average accuracy by 14.30% and
3.02%, respectively, in the Waterbirds group shift scenario, where other robust
fine-tuning baselines show even degraded performance.

---


### [Adversarial Testing in LLMs: Insights into Decision-Making Vulnerabilities](http://arxiv.org/abs/2505.13195v1)

As Large Language Models (LLMs) become increasingly integrated into
real-world decision-making systems, understanding their behavioural
vulnerabilities remains a critical challenge for AI safety and alignment. While
existing evaluation metrics focus primarily on reasoning accuracy or factual
correctness, they often overlook whether LLMs are robust to adversarial
manipulation or capable of using adaptive strategy in dynamic environments.
This paper introduces an adversarial evaluation framework designed to
systematically stress-test the decision-making processes of LLMs under
interactive and adversarial conditions. Drawing on methodologies from cognitive
psychology and game theory, our framework probes how models respond in two
canonical tasks: the two-armed bandit task and the Multi-Round Trust Task.
These tasks capture key aspects of exploration-exploitation trade-offs, social
cooperation, and strategic flexibility. We apply this framework to several
state-of-the-art LLMs, including GPT-3.5, GPT-4, Gemini-1.5, and DeepSeek-V3,
revealing model-specific susceptibilities to manipulation and rigidity in
strategy adaptation. Our findings highlight distinct behavioral patterns across
models and emphasize the importance of adaptability and fairness recognition
for trustworthy AI deployment. Rather than offering a performance benchmark,
this work proposes a methodology for diagnosing decision-making weaknesses in
LLM-based agents, providing actionable insights for alignment and safety
research.

---


### [ViPlan: A Benchmark for Visual Planning with Symbolic Predicates and Vision-Language Models](http://arxiv.org/abs/2505.13180v1)

Integrating Large Language Models with symbolic planners is a promising
direction for obtaining verifiable and grounded plans compared to planning in
natural language, with recent works extending this idea to visual domains using
Vision-Language Models (VLMs). However, rigorous comparison between
VLM-grounded symbolic approaches and methods that plan directly with a VLM has
been hindered by a lack of common environments, evaluation protocols and model
coverage. We introduce ViPlan, the first open-source benchmark for Visual
Planning with symbolic predicates and VLMs. ViPlan features a series of
increasingly challenging tasks in two domains: a visual variant of the classic
Blocksworld planning problem and a simulated household robotics environment. We
benchmark nine open-source VLM families across multiple sizes, along with
selected closed models, evaluating both VLM-grounded symbolic planning and
using the models directly to propose actions. We find symbolic planning to
outperform direct VLM planning in Blocksworld, where accurate image grounding
is crucial, whereas the opposite is true in the household robotics tasks, where
commonsense knowledge and the ability to recover from errors are beneficial.
Finally, we show that across most models and methods, there is no significant
benefit to using Chain-of-Thought prompting, suggesting that current VLMs still
struggle with visual reasoning.

---


### [Enhancing LLMs for Time Series Forecasting via Structure-Guided Cross-Modal Alignment](http://arxiv.org/abs/2505.13175v1)

The emerging paradigm of leveraging pretrained large language models (LLMs)
for time series forecasting has predominantly employed linguistic-temporal
modality alignment strategies through token-level or layer-wise feature
mapping. However, these approaches fundamentally neglect a critical insight:
the core competency of LLMs resides not merely in processing localized token
features but in their inherent capacity to model holistic sequence structures.
This paper posits that effective cross-modal alignment necessitates structural
consistency at the sequence level. We propose the Structure-Guided Cross-Modal
Alignment (SGCMA), a framework that fully exploits and aligns the
state-transition graph structures shared by time-series and linguistic data as
sequential modalities, thereby endowing time series with language-like
properties and delivering stronger generalization after modality alignment.
SGCMA consists of two key components, namely Structure Alignment and Semantic
Alignment. In Structure Alignment, a state transition matrix is learned from
text data through Hidden Markov Models (HMMs), and a shallow transformer-based
Maximum Entropy Markov Model (MEMM) receives the hot-start transition matrix
and annotates each temporal patch into state probability, ensuring that the
temporal representation sequence inherits language-like sequential dynamics. In
Semantic Alignment, cross-attention is applied between temporal patches and the
top-k tokens within each state, and the ultimate temporal embeddings are
derived by the expected value of these embeddings using a weighted average
based on state probabilities. Experiments on multiple benchmarks demonstrate
that SGCMA achieves state-of-the-art performance, offering a novel approach to
cross-modal alignment in time series forecasting.

---


### [ModernGBERT: German-only 1B Encoder Model Trained from Scratch](http://arxiv.org/abs/2505.13136v1)

Despite the prominence of decoder-only language models, encoders remain
crucial for resource-constrained applications. We introduce ModernGBERT (134M,
1B), a fully transparent family of German encoder models trained from scratch,
incorporating architectural innovations from ModernBERT. To evaluate the
practical trade-offs of training encoders from scratch, we also present
LL\"aMmlein2Vec (120M, 1B, 7B), a family of encoders derived from German
decoder-only models via LLM2Vec. We benchmark all models on natural language
understanding, text embedding, and long-context reasoning tasks, enabling a
controlled comparison between dedicated encoders and converted decoders. Our
results show that ModernGBERT 1B outperforms prior state-of-the-art German
encoders as well as encoders adapted via LLM2Vec, with regard to performance
and parameter-efficiency. All models, training data, checkpoints and code are
publicly available, advancing the German NLP ecosystem with transparent,
high-performance encoder models.

---


### [Zero-Shot Iterative Formalization and Planning in Partially Observable Environments](http://arxiv.org/abs/2505.13126v1)

In planning, using LLMs not to predict plans but to formalize an environment
into the Planning Domain Definition Language (PDDL) has been shown to greatly
improve performance and control. While most work focused on fully observable
environments, we tackle the more realistic and challenging partially observable
environments where existing methods are incapacitated by the lack of complete
information. We propose PDDLego+, a framework to iteratively formalize, plan,
grow, and refine PDDL representations in a zero-shot manner, without needing
access to any existing trajectories. On two textual simulated environments, we
show that PDDLego+ not only achieves superior performance, but also shows
robustness against problem complexity. We also show that the domain knowledge
captured after a successful trial is interpretable and benefits future tasks.

---


### [FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference](http://arxiv.org/abs/2505.13109v1)

Large language models (LLMs) have been widely deployed with rapidly expanding
context windows to support increasingly demanding applications. However, long
contexts pose significant deployment challenges, primarily due to the KV cache
whose size grows proportionally with context length. While KV cache compression
methods are proposed to address this issue, KV dropping methods incur
considerable accuracy loss, and KV retrieval methods suffer from significant
efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization
framework to enhance KV retrieval efficiency while preserving accuracy. On the
algorithm side, FreeKV introduces speculative retrieval to shift the KV
selection and recall processes out of the critical path, combined with
fine-grained correction to ensure accuracy. On the system side, FreeKV employs
hybrid KV layouts across CPU and GPU memory to eliminate fragmented data
transfers, and leverages double-buffered streamed recall to further improve
efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy
across various scenarios and models, delivering up to 13$\times$ speedup
compared to SOTA KV retrieval methods.

---


### [CAIM: Development and Evaluation of a Cognitive AI Memory Framework for Long-Term Interaction with Intelligent Agents](http://arxiv.org/abs/2505.13044v1)

Large language models (LLMs) have advanced the field of artificial
intelligence (AI) and are a powerful enabler for interactive systems. However,
they still face challenges in long-term interactions that require adaptation
towards the user as well as contextual knowledge and understanding of the
ever-changing environment. To overcome these challenges, holistic memory
modeling is required to efficiently retrieve and store relevant information
across interaction sessions for suitable responses. Cognitive AI, which aims to
simulate the human thought process in a computerized model, highlights
interesting aspects, such as thoughts, memory mechanisms, and decision-making,
that can contribute towards improved memory modeling for LLMs. Inspired by
these cognitive AI principles, we propose our memory framework CAIM. CAIM
consists of three modules: 1.) The Memory Controller as the central decision
unit; 2.) the Memory Retrieval, which filters relevant data for interaction
upon request; and 3.) the Post-Thinking, which maintains the memory storage. We
compare CAIM against existing approaches, focusing on metrics such as retrieval
accuracy, response correctness, contextual coherence, and memory storage. The
results demonstrate that CAIM outperforms baseline frameworks across different
metrics, highlighting its context-awareness and potential to improve long-term
human-AI interactions.

---


### [KIT's Offline Speech Translation and Instruction Following Submission for IWSLT 2025](http://arxiv.org/abs/2505.13036v1)

The scope of the International Workshop on Spoken Language Translation
(IWSLT) has recently broadened beyond traditional Speech Translation (ST) to
encompass a wider array of tasks, including Speech Question Answering and
Summarization. This shift is partly driven by the growing capabilities of
modern systems, particularly with the success of Large Language Models (LLMs).
In this paper, we present the Karlsruhe Institute of Technology's submissions
for the Offline ST and Instruction Following (IF) tracks, where we leverage
LLMs to enhance performance across all tasks. For the Offline ST track, we
propose a pipeline that employs multiple automatic speech recognition systems,
whose outputs are fused using an LLM with document-level context. This is
followed by a two-step translation process, incorporating additional refinement
step to improve translation quality. For the IF track, we develop an end-to-end
model that integrates a speech encoder with an LLM to perform a wide range of
instruction-following tasks. We complement it with a final document-level
refinement stage to further enhance output quality by using contextual
information.

---


### [MindOmni: Unleashing Reasoning Generation in Vision Language Models with RGPO](http://arxiv.org/abs/2505.13031v1)

Recent text-to-image systems face limitations in handling multimodal inputs
and complex reasoning tasks. We introduce MindOmni, a unified multimodal large
language model that addresses these challenges by incorporating reasoning
generation through reinforcement learning. MindOmni leverages a three-phase
training strategy: i) design of a unified vision language model with a
decoder-only diffusion module, ii) supervised fine-tuning with Chain-of-Thought
(CoT) instruction data, and iii) our proposed Reasoning Generation Policy
Optimization (RGPO) algorithm, utilizing multimodal feedback to effectively
guide policy updates. Experimental results demonstrate that MindOmni
outperforms existing models, achieving impressive performance on both
understanding and generation benchmarks, meanwhile showcasing advanced
fine-grained reasoning generation capabilities, especially with mathematical
reasoning instruction. All codes will be made public at
\href{https://github.com/EasonXiao-888/MindOmni}{https://github.com/EasonXiao-888/MindOmni}.

---


### [Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset](http://arxiv.org/abs/2505.13028v1)

Large Language Models (LLMs) are increasingly integrated into critical
systems in industries like healthcare and finance. Users can often submit
queries to LLM-enabled chatbots, some of which can enrich responses with
information retrieved from internal databases storing sensitive data. This
gives rise to a range of attacks in which a user submits a malicious query and
the LLM-system outputs a response that creates harm to the owner, such as
leaking internal data or creating legal liability by harming a third-party.
While security tools are being developed to counter these threats, there is
little formal evaluation of their effectiveness and usability. This study
addresses this gap by conducting a thorough comparative analysis of LLM
security tools. We identified 13 solutions (9 closed-source, 4 open-source),
but only 7 were evaluated due to a lack of participation by proprietary model
owners.To evaluate, we built a benchmark dataset of malicious prompts, and
evaluate these tools performance against a baseline LLM model
(ChatGPT-3.5-Turbo). Our results show that the baseline model has too many
false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard
emerged as the best overall tools showcasing the tradeoff between usability and
performance. The study concluded with recommendations for greater transparency
among closed source providers, improved context-aware detections, enhanced
open-source engagement, increased user awareness, and the adoption of more
representative performance metrics.

---


### [Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs](http://arxiv.org/abs/2505.13026v1)

Large language models (LLMs) excel at mathematical reasoning and logical
problem-solving. The current popular training paradigms primarily use
supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the
models' reasoning abilities. However, when using SFT or RL alone, there are
respective challenges: SFT may suffer from overfitting, while RL is prone to
mode collapse. The state-of-the-art methods have proposed hybrid training
schemes. However, static switching faces challenges such as poor generalization
across different tasks and high dependence on data quality. In response to
these challenges, inspired by the curriculum learning-quiz mechanism in human
reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training
framework that theoretically unifies SFT and RL and dynamically balances the
two throughout optimization. SASR uses SFT for initial warm-up to establish
basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm
based on gradient norm and divergence relative to the original distribution to
seamlessly integrate SFT with the online RL method GRPO. By monitoring the
training status of LLMs and adjusting the training process in sequence, SASR
ensures a smooth transition between training schemes, maintaining core
reasoning abilities while exploring different paths. Experimental results
demonstrate that SASR outperforms SFT, RL, and static hybrid training methods.

---


### [To Bias or Not to Bias: Detecting bias in News with bias-detector](http://arxiv.org/abs/2505.13010v1)

Media bias detection is a critical task in ensuring fair and balanced
information dissemination, yet it remains challenging due to the subjectivity
of bias and the scarcity of high-quality annotated data. In this work, we
perform sentence-level bias classification by fine-tuning a RoBERTa-based model
on the expert-annotated BABE dataset. Using McNemar's test and the 5x2
cross-validation paired t-test, we show statistically significant improvements
in performance when comparing our model to a domain-adaptively pre-trained
DA-RoBERTa baseline. Furthermore, attention-based analysis shows that our model
avoids common pitfalls like oversensitivity to politically charged terms and
instead attends more meaningfully to contextually relevant tokens. For a
comprehensive examination of media bias, we present a pipeline that combines
our model with an already-existing bias-type classifier. Our method exhibits
good generalization and interpretability, despite being constrained by
sentence-level analysis and dataset size because of a lack of larger and more
advanced bias corpora. We talk about context-aware modeling, bias
neutralization, and advanced bias type classification as potential future
directions. Our findings contribute to building more robust, explainable, and
socially responsible NLP systems for media bias detection.

---


### [Fractured Chain-of-Thought Reasoning](http://arxiv.org/abs/2505.12992v1)

Inference-time scaling techniques have significantly bolstered the reasoning
capabilities of large language models (LLMs) by harnessing additional
computational effort at inference without retraining. Similarly,
Chain-of-Thought (CoT) prompting and its extension, Long CoT, improve accuracy
by generating rich intermediate reasoning trajectories, but these approaches
incur substantial token costs that impede their deployment in latency-sensitive
settings. In this work, we first show that truncated CoT, which stops reasoning
before completion and directly generates the final answer, often matches full
CoT sampling while using dramatically fewer tokens. Building on this insight,
we introduce Fractured Sampling, a unified inference-time strategy that
interpolates between full CoT and solution-only sampling along three orthogonal
axes: (1) the number of reasoning trajectories, (2) the number of final
solutions per trajectory, and (3) the depth at which reasoning traces are
truncated. Through extensive experiments on five diverse reasoning benchmarks
and several model scales, we demonstrate that Fractured Sampling consistently
achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling
gains in Pass@k versus token budget. Our analysis reveals how to allocate
computation across these dimensions to maximize performance, paving the way for
more efficient and scalable LLM reasoning.

---


### [An Empirical Study of Many-to-Many Summarization with Large Language Models](http://arxiv.org/abs/2505.12983v1)

Many-to-many summarization (M2MS) aims to process documents in any language
and generate the corresponding summaries also in any language. Recently, large
language models (LLMs) have shown strong multi-lingual abilities, giving them
the potential to perform M2MS in real applications. This work presents a
systematic empirical study on LLMs' M2MS ability. Specifically, we first
reorganize M2MS data based on eight previous domain-specific datasets. The
reorganized data contains 47.8K samples spanning five domains and six
languages, which could be used to train and evaluate LLMs. Then, we benchmark
18 LLMs in a zero-shot manner and an instruction-tuning manner. Fine-tuned
traditional models (e.g., mBART) are also conducted for comparisons. Our
experiments reveal that, zero-shot LLMs achieve competitive results with
fine-tuned traditional models. After instruct-tuning, open-source LLMs can
significantly improve their M2MS ability, and outperform zero-shot LLMs
(including GPT-4) in terms of automatic evaluations. In addition, we
demonstrate that this task-specific improvement does not sacrifice the LLMs'
general task-solving abilities. However, as revealed by our human evaluation,
LLMs still face the factuality issue, and the instruction tuning might
intensify the issue. Thus, how to control factual errors becomes the key when
building LLM summarizers in real applications, and is worth noting in future
research.

---


### [DGRO: Enhancing LLM Reasoning via Exploration-Exploitation Control and Reward Variance Management](http://arxiv.org/abs/2505.12951v1)

Inference scaling further accelerates Large Language Models (LLMs) toward
Artificial General Intelligence (AGI), with large-scale Reinforcement Learning
(RL) to unleash long Chain-of-Thought reasoning. Most contemporary reasoning
approaches usually rely on handcrafted rule-based reward functions. However,
the tarde-offs of exploration and exploitation in RL algorithms involves
multiple complex considerations, and the theoretical and empirical impacts of
manually designed reward functions remain insufficiently explored. In this
paper, we propose Decoupled Group Reward Optimization (DGRO), a general RL
algorithm for LLM reasoning. On the one hand, DGRO decouples the traditional
regularization coefficient into two independent hyperparameters: one scales the
policy gradient term, and the other regulates the distance from the sampling
policy. This decoupling not only enables precise control over balancing
exploration and exploitation, but also can be seamlessly extended to Online
Policy Mirror Descent (OPMD) algorithms in Kimi k1.5 and Direct Reward
Optimization. On the other hand, we observe that reward variance significantly
affects both convergence speed and final model performance. We conduct both
theoretical analysis and extensive empirical validation to assess DGRO,
including a detailed ablation study that investigates its performance and
optimization dynamics. Experimental results show that DGRO achieves
state-of-the-art performance on the Logic dataset with an average accuracy of
96.9\%, and demonstrates strong generalization across mathematical benchmarks.

---


### [ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models](http://arxiv.org/abs/2505.13444v1)

Chart understanding presents a unique challenge for large vision-language
models (LVLMs), as it requires the integration of sophisticated textual and
visual reasoning capabilities. However, current LVLMs exhibit a notable
imbalance between these skills, falling short on visual reasoning that is
difficult to perform in text. We conduct a case study using a synthetic dataset
solvable only through visual reasoning and show that model performance degrades
significantly with increasing visual complexity, while human performance
remains robust. We then introduce ChartMuseum, a new Chart Question Answering
(QA) benchmark containing 1,162 expert-annotated questions spanning multiple
reasoning types, curated from real-world charts across 184 sources,
specifically built to evaluate complex visual and textual reasoning. Unlike
prior chart understanding benchmarks -- where frontier models perform similarly
and near saturation -- our benchmark exposes a substantial gap between model
and human performance, while effectively differentiating model capabilities:
although humans achieve 93% accuracy, the best-performing model Gemini-2.5-Pro
attains only 63.0%, and the leading open-source LVLM Qwen2.5-VL-72B-Instruct
achieves only 38.5%. Moreover, on questions requiring primarily visual
reasoning, all models experience a 35%-55% performance drop from
text-reasoning-heavy question performance. Lastly, our qualitative error
analysis reveals specific categories of visual reasoning that are challenging
for current LVLMs.

---


### [SMOTExT: SMOTE meets Large Language Models](http://arxiv.org/abs/2505.13434v1)

Data scarcity and class imbalance are persistent challenges in training
robust NLP models, especially in specialized domains or low-resource settings.
We propose a novel technique, SMOTExT, that adapts the idea of Synthetic
Minority Over-sampling (SMOTE) to textual data. Our method generates new
synthetic examples by interpolating between BERT-based embeddings of two
existing examples and then decoding the resulting latent point into text with
xRAG architecture. By leveraging xRAG's cross-modal retrieval-generation
framework, we can effectively turn interpolated vectors into coherent text.
While this is preliminary work supported by qualitative outputs only, the
method shows strong potential for knowledge distillation and data augmentation
in few-shot settings. Notably, our approach also shows promise for
privacy-preserving machine learning: in early experiments, training models
solely on generated data achieved comparable performance to models trained on
the original dataset. This suggests a viable path toward safe and effective
learning under data protection constraints.

---


### [Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](http://arxiv.org/abs/2505.13430v1)

As the size of large language models grows exponentially, GPU memory has
become a bottleneck for adapting these models to downstream tasks. In this
paper, we aim to push the limits of memory-efficient training by minimizing
memory usage on model weights, gradients, and optimizer states, within a
unified framework. Our idea is to eliminate both gradients and optimizer states
using zeroth-order optimization, which approximates gradients by perturbing
weights during forward passes to identify gradient directions. To minimize
memory usage on weights, we employ model quantization, e.g., converting from
bfloat16 to int4. However, directly applying zeroth-order optimization to
quantized weights is infeasible due to the precision gap between discrete
weights and continuous gradients, which would otherwise require de-quantization
and re-quantization. To overcome this challenge, we propose Quantized
Zeroth-order Optimization (QZO), a novel approach that perturbs the continuous
quantization scale for gradient estimation and uses a directional derivative
clipping method to stabilize training. QZO is orthogonal to both scalar-based
and codebook-based post-training quantization methods. Compared to
full-parameter fine-tuning in bfloat16, QZO can reduce the total memory cost by
more than 18$\times$ for 4-bit LLMs, and enables fine-tuning Llama-2-13B and
Stable Diffusion 3.5 Large within a single 24GB GPU.

---


### [MR. Judge: Multimodal Reasoner as a Judge](http://arxiv.org/abs/2505.13403v1)

The paradigm of using Large Language Models (LLMs) and Multimodal Large
Language Models (MLLMs) as evaluative judges has emerged as an effective
approach in RLHF and inference-time scaling. In this work, we propose
Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering
general-purpose MLLMs judges with strong reasoning capabilities. Instead of
directly assigning scores for each response, we formulate the judgement process
as a reasoning-inspired multiple-choice problem. Specifically, the judge model
first conducts deliberate reasoning covering different aspects of the responses
and eventually selects the best response from them. This reasoning process not
only improves the interpretibility of the judgement, but also greatly enhances
the performance of MLLM judges. To cope with the lack of questions with scored
responses, we propose the following strategy to achieve automatic annotation:
1) Reverse Response Candidates Synthesis: starting from a supervised
fine-tuning (SFT) dataset, we treat the original response as the best candidate
and prompt the MLLM to generate plausible but flawed negative candidates. 2)
Text-based reasoning extraction: we carefully design a data synthesis pipeline
for distilling the reasoning capability from a text-based reasoning model,
which is adopted to enable the MLLM judges to regain complex reasoning ability
via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge
is effective across a wide range of tasks. Specifically, our MR. Judge-7B
surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet
during inference-time scaling by up to 7.7%.

---


### [Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks](http://arxiv.org/abs/2505.13348v1)

Large Language Models (LLMs) are increasingly employed as evaluators
(LLM-as-a-Judge) for assessing the quality of machine-generated text. This
paradigm offers scalability and cost-effectiveness compared to human
annotation. However, the reliability and security of such systems, particularly
their robustness against adversarial manipulations, remain critical concerns.
This paper investigates the vulnerability of LLM-as-a-Judge architectures to
prompt-injection attacks, where malicious inputs are designed to compromise the
judge's decision-making process. We formalize two primary attack strategies:
Comparative Undermining Attack (CUA), which directly targets the final decision
output, and Justification Manipulation Attack (JMA), which aims to alter the
model's generated reasoning. Using the Greedy Coordinate Gradient (GCG)
optimization method, we craft adversarial suffixes appended to one of the
responses being compared. Experiments conducted on the MT-Bench Human Judgments
dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and
Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves
an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable
effectiveness. These findings highlight substantial vulnerabilities in current
LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and
further research into adversarial evaluation and trustworthiness in LLM-based
assessment frameworks.

---


### [I'll believe it when I see it: Images increase misinformation sharing in Vision-Language Models](http://arxiv.org/abs/2505.13302v1)

Large language models are increasingly integrated into news recommendation
systems, raising concerns about their role in spreading misinformation. In
humans, visual content is known to boost credibility and shareability of
information, yet its effect on vision-language models (VLMs) remains unclear.
We present the first study examining how images influence VLMs' propensity to
reshare news content, whether this effect varies across model families, and how
persona conditioning and content attributes modulate this behavior. To support
this analysis, we introduce two methodological contributions: a
jailbreaking-inspired prompting strategy that elicits resharing decisions from
VLMs while simulating users with antisocial traits and political alignments;
and a multimodal dataset of fact-checked political news from PolitiFact, paired
with corresponding images and ground-truth veracity labels. Experiments across
model families reveal that image presence increases resharing rates by 4.8% for
true news and 15.0% for false news. Persona conditioning further modulates this
effect: Dark Triad traits amplify resharing of false news, whereas
Republican-aligned profiles exhibit reduced veracity sensitivity. Of all the
tested models, only Claude-3-Haiku demonstrates robustness to visual
misinformation. These findings highlight emerging risks in multimodal model
behavior and motivate the development of tailored evaluation frameworks and
mitigation strategies for personalized AI systems. Code and dataset are
available at: https://github.com/3lis/misinfo_vlm

---


### [CSC-SQL: Corrective Self-Consistency in Text-to-SQL via Reinforcement Learning](http://arxiv.org/abs/2505.13271v1)

Large language models (LLMs) have demonstrated strong capabilities in
translating natural language questions about relational databases into SQL
queries. In particular, test-time scaling techniques such as Self-Consistency
and Self-Correction can enhance SQL generation accuracy by increasing
computational effort during inference. However, these methods have notable
limitations: Self-Consistency may select suboptimal outputs despite majority
votes, while Self-Correction typically addresses only syntactic errors. To
leverage the strengths of both approaches, we propose CSC-SQL, a novel method
that integrates Self-Consistency and Self-Correction. CSC-SQL selects the two
most frequently occurring outputs from parallel sampling and feeds them into a
merge revision model for correction. Additionally, we employ the Group Relative
Policy Optimization (GRPO) algorithm to fine-tune both the SQL generation and
revision models via reinforcement learning, significantly enhancing output
quality. Experimental results confirm the effectiveness and generalizability of
CSC-SQL. On the BIRD development set, our 3B model achieves 65.28% execution
accuracy, while the 7B model achieves 69.19%. The code will be open sourced at
https://github.com/CycloneBoy/csc_sql.

---


### [Effective and Transparent RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability](http://arxiv.org/abs/2505.13258v1)

Retrieval-Augmented Generation (RAG) has significantly improved the
performance of large language models (LLMs) on knowledge-intensive domains.
However, although RAG achieved successes across distinct domains, there are
still some unsolved challenges: 1) Effectiveness. Existing research mainly
focuses on developing more powerful RAG retrievers, but how to enhance the
generator's (LLM's) ability to utilize the retrieved information for reasoning
and generation? 2) Transparency. Most RAG methods ignore which retrieved
content actually contributes to the reasoning process, resulting in a lack of
interpretability and visibility. To address this, we propose ARENA
(Adaptive-Rewarded Evidence Navigation Agent), a transparent RAG generator
framework trained via reinforcement learning (RL) with our proposed rewards.
Based on the structured generation and adaptive reward calculation, our
RL-based training enables the model to identify key evidence, perform
structured reasoning, and generate answers with interpretable decision traces.
Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments
with various RAG baselines demonstrate that our model achieves 10-30%
improvements on all multi-hop QA datasets, which is comparable with the SOTA
Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses
show that ARENA has strong flexibility to be adopted on new datasets without
extra training. Our models and codes are publicly released.

---


### [JNLP at SemEval-2025 Task 11: Cross-Lingual Multi-Label Emotion Detection Using Generative Models](http://arxiv.org/abs/2505.13244v1)

With the rapid advancement of global digitalization, users from different
countries increasingly rely on social media for information exchange. In this
context, multilingual multi-label emotion detection has emerged as a critical
research area. This study addresses SemEval-2025 Task 11: Bridging the Gap in
Text-Based Emotion Detection. Our paper focuses on two sub-tracks of this task:
(1) Track A: Multi-label emotion detection, and (2) Track B: Emotion intensity.
To tackle multilingual challenges, we leverage pre-trained multilingual models
and focus on two architectures: (1) a fine-tuned BERT-based classification
model and (2) an instruction-tuned generative LLM. Additionally, we propose two
methods for handling multi-label classification: the base method, which maps an
input directly to all its corresponding emotion labels, and the pairwise
method, which models the relationship between the input text and each emotion
category individually. Experimental results demonstrate the strong
generalization ability of our approach in multilingual emotion recognition. In
Track A, our method achieved Top 4 performance across 10 languages, ranking 1st
in Hindi. In Track B, our approach also secured Top 5 performance in 7
languages, highlighting its simplicity and effectiveness\footnote{Our code is
available at https://github.com/yingjie7/mlingual_multilabel_emo_detection.

---


### [A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](http://arxiv.org/abs/2505.13173v1)

Large Language Models (LLMs) have demonstrated remarkable generalization
capabilities across diverse tasks and languages. In this study, we focus on
natural language understanding in three classical languages -- Sanskrit,
Ancient Greek and Latin -- to investigate the factors affecting cross-lingual
zero-shot generalization. First, we explore named entity recognition and
machine translation into English. While LLMs perform equal to or better than
fine-tuned baselines on out-of-domain data, smaller models often struggle,
especially with niche or abstract entity types. In addition, we concentrate on
Sanskrit by presenting a factoid question-answering (QA) dataset and show that
incorporating context via retrieval-augmented generation approach significantly
boosts performance. In contrast, we observe pronounced performance drops for
smaller LLMs across these QA tasks. These results suggest model scale as an
important factor influencing cross-lingual generalization. Assuming that models
used such as GPT-4o and Llama-3.1 are not instruction fine-tuned on classical
languages, our findings provide insights into how LLMs may generalize on these
languages and their consequent utility in classical studies.

---


### [Suicide Risk Assessment Using Multimodal Speech Features: A Study on the SW1 Challenge Dataset](http://arxiv.org/abs/2505.13069v1)

The 1st SpeechWellness Challenge conveys the need for speech-based suicide
risk assessment in adolescents. This study investigates a multimodal approach
for this challenge, integrating automatic transcription with WhisperX,
linguistic embeddings from Chinese RoBERTa, and audio embeddings from WavLM.
Additionally, handcrafted acoustic features -- including MFCCs, spectral
contrast, and pitch-related statistics -- were incorporated. We explored three
fusion strategies: early concatenation, modality-specific processing, and
weighted attention with mixup regularization. Results show that weighted
attention provided the best generalization, achieving 69% accuracy on the
development set, though a performance gap between development and test sets
highlights generalization challenges. Our findings, strictly tied to the
MINI-KID framework, emphasize the importance of refining embedding
representations and fusion mechanisms to enhance classification reliability.

---


### [MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix](http://arxiv.org/abs/2505.13032v1)

We introduce MMAR, a new benchmark designed to evaluate the deep reasoning
capabilities of Audio-Language Models (ALMs) across massive multi-disciplinary
tasks. MMAR comprises 1,000 meticulously curated audio-question-answer
triplets, collected from real-world internet videos and refined through
iterative error corrections and quality checks to ensure high quality. Unlike
existing benchmarks that are limited to specific domains of sound, music, or
speech, MMAR extends them to a broad spectrum of real-world audio scenarios,
including mixed-modality combinations of sound, music, and speech. Each
question in MMAR is hierarchically categorized across four reasoning layers:
Signal, Perception, Semantic, and Cultural, with additional sub-categories
within each layer to reflect task diversity and complexity. To further foster
research in this area, we annotate every question with a Chain-of-Thought (CoT)
rationale to promote future advancements in audio reasoning. Each item in the
benchmark demands multi-step deep reasoning beyond surface-level understanding.
Moreover, a part of the questions requires graduate-level perceptual and
domain-specific knowledge, elevating the benchmark's difficulty and depth. We
evaluate MMAR using a broad set of models, including Large Audio-Language
Models (LALMs), Large Audio Reasoning Models (LARMs), Omni Language Models
(OLMs), Large Language Models (LLMs), and Large Reasoning Models (LRMs), with
audio caption inputs. The performance of these models on MMAR highlights the
benchmark's challenging nature, and our analysis further reveals critical
limitations of understanding and reasoning capabilities among current models.
We hope MMAR will serve as a catalyst for future advances in this important but
little-explored area.

---


### [Evaluating the Performance of RAG Methods for Conversational AI in the Airport Domain](http://arxiv.org/abs/2505.13006v1)

Airports from the top 20 in terms of annual passengers are highly dynamic
environments with thousands of flights daily, and they aim to increase the
degree of automation. To contribute to this, we implemented a Conversational AI
system that enables staff in an airport to communicate with flight information
systems. This system not only answers standard airport queries but also
resolves airport terminology, jargon, abbreviations, and dynamic questions
involving reasoning. In this paper, we built three different
Retrieval-Augmented Generation (RAG) methods, including traditional RAG, SQL
RAG, and Knowledge Graph-based RAG (Graph RAG). Experiments showed that
traditional RAG achieved 84.84% accuracy using BM25 + GPT-4 but occasionally
produced hallucinations, which is risky to airport safety. In contrast, SQL RAG
and Graph RAG achieved 80.85% and 91.49% accuracy respectively, with
significantly fewer hallucinations. Moreover, Graph RAG was especially
effective for questions that involved reasoning. Based on our observations, we
thus recommend SQL RAG and Graph RAG are better for airport environments, due
to fewer hallucinations and the ability to handle dynamic questions.

---


### [Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs](http://arxiv.org/abs/2505.12929v1)

Reinforcement learning (RL) has become a cornerstone for enhancing the
reasoning capabilities of large language models (LLMs), with recent innovations
such as Group Relative Policy Optimization (GRPO) demonstrating exceptional
effectiveness. In this study, we identify a critical yet underexplored issue in
RL training: low-probability tokens disproportionately influence model updates
due to their large gradient magnitudes. This dominance hinders the effective
learning of high-probability tokens, whose gradients are essential for LLMs'
performance but are substantially suppressed. To mitigate this interference, we
propose two novel methods: Advantage Reweighting and Low-Probability Token
Isolation (Lopti), both of which effectively attenuate gradients from
low-probability tokens while emphasizing parameter updates driven by
high-probability tokens. Our approaches promote balanced updates across tokens
with varying probabilities, thereby enhancing the efficiency of RL training.
Experimental results demonstrate that they substantially improve the
performance of GRPO-trained LLMs, achieving up to a 46.2% improvement in K&K
Logic Puzzle reasoning tasks. Our implementation is available at
https://github.com/zhyang2226/AR-Lopti.

---


### [Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective](http://arxiv.org/abs/2505.12886v1)

Large Reasoning Models (LRMs) have shown impressive capabilities in
multi-step reasoning tasks. However, alongside these successes, a more
deceptive form of model error has emerged--Reasoning Hallucination--where
logically coherent but factually incorrect reasoning traces lead to persuasive
yet faulty conclusions. Unlike traditional hallucinations, these errors are
embedded within structured reasoning, making them more difficult to detect and
potentially more harmful. In this work, we investigate reasoning hallucinations
from a mechanistic perspective. We propose the Reasoning Score, which
quantifies the depth of reasoning by measuring the divergence between logits
obtained from projecting late layers of LRMs to the vocabulary space,
effectively distinguishing shallow pattern-matching from genuine deep
reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA
dataset and identify two key reasoning hallucination patterns: early-stage
fluctuation in reasoning depth and incorrect backtracking to flawed prior
steps. These insights motivate our Reasoning Hallucination Detection (RHD)
framework, which achieves state-of-the-art performance across multiple domains.
To mitigate reasoning hallucinations, we further introduce GRPO-R, an enhanced
reinforcement learning algorithm that incorporates step-level deep reasoning
rewards via potential-based shaping. Our theoretical analysis establishes
stronger generalization guarantees, and experiments demonstrate improved
reasoning quality and reduced hallucination rates.

---


### [Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?](http://arxiv.org/abs/2505.12871v1)

Low rank adaptation (LoRA) has emerged as a prominent technique for
fine-tuning large language models (LLMs) thanks to its superb efficiency gains
over previous methods. While extensive studies have examined the performance
and structural properties of LoRA, its behavior upon training-time attacks
remain underexplored, posing significant security risks. In this paper, we
theoretically investigate the security implications of LoRA's low-rank
structure during fine-tuning, in the context of its robustness against data
poisoning and backdoor attacks. We propose an analytical framework that models
LoRA's training dynamics, employs the neural tangent kernel to simplify the
analysis of the training process, and applies information theory to establish
connections between LoRA's low rank structure and its vulnerability against
training-time attacks. Our analysis indicates that LoRA exhibits better
robustness to backdoor attacks than full fine-tuning, while becomes more
vulnerable to untargeted data poisoning due to its over-simplified information
geometry. Extensive experimental evaluations have corroborated our theoretical
findings.

---


### [The Hidden Structure -- Improving Legal Document Understanding Through Explicit Text Formatting](http://arxiv.org/abs/2505.12837v1)

Legal contracts possess an inherent, semantically vital structure (e.g.,
sections, clauses) that is crucial for human comprehension but whose impact on
LLM processing remains under-explored. This paper investigates the effects of
explicit input text structure and prompt engineering on the performance of
GPT-4o and GPT-4.1 on a legal question-answering task using an excerpt of the
CUAD. We compare model exact-match accuracy across various input formats:
well-structured plain-text (human-generated from CUAD), plain-text cleaned of
line breaks, extracted plain-text from Azure OCR, plain-text extracted by
GPT-4o Vision, and extracted (and interpreted) Markdown (MD) from GPT-4o
Vision. To give an indication of the impact of possible prompt engineering, we
assess the impact of shifting task instructions to the system prompt and
explicitly informing the model about the structured nature of the input. Our
findings reveal that GPT-4o demonstrates considerable robustness to variations
in input structure, but lacks in overall performance. Conversely, GPT-4.1's
performance is markedly sensitive; poorly structured inputs yield suboptimal
results (but identical with GPT-4o), while well-structured formats (original
CUAD text, GPT-4o Vision text and GPT-4o MD) improve exact-match accuracy by
~20 percentage points. Optimizing the system prompt to include task details and
an advisory about structured input further elevates GPT-4.1's accuracy by an
additional ~10-13 percentage points, with Markdown ultimately achieving the
highest performance under these conditions (79 percentage points overall
exact-match accuracy). This research empirically demonstrates that while newer
models exhibit greater resilience, careful input structuring and strategic
prompt design remain critical for optimizing the performance of LLMs, and can
significantly affect outcomes in high-stakes legal applications.

---


### [FlightGPT: Towards Generalizable and Interpretable UAV Vision-and-Language Navigation with Vision-Language Models](http://arxiv.org/abs/2505.12835v1)

Unmanned Aerial Vehicle (UAV) Vision-and-Language Navigation (VLN) is vital
for applications such as disaster response, logistics delivery, and urban
inspection. However, existing methods often struggle with insufficient
multimodal fusion, weak generalization, and poor interpretability. To address
these challenges, we propose FlightGPT, a novel UAV VLN framework built upon
Vision-Language Models (VLMs) with powerful multimodal perception capabilities.
We design a two-stage training pipeline: first, Supervised Fine-Tuning (SFT)
using high-quality demonstrations to improve initialization and structured
reasoning; then, Group Relative Policy Optimization (GRPO) algorithm, guided by
a composite reward that considers goal accuracy, reasoning quality, and format
compliance, to enhance generalization and adaptability. Furthermore, FlightGPT
introduces a Chain-of-Thought (CoT)-based reasoning mechanism to improve
decision interpretability. Extensive experiments on the city-scale dataset
CityNav demonstrate that FlightGPT achieves state-of-the-art performance across
all scenarios, with a 9.22\% higher success rate than the strongest baseline in
unseen environments. Our implementation is publicly available.

---


### [EAVIT: Efficient and Accurate Human Value Identification from Text data via LLMs](http://arxiv.org/abs/2505.12792v1)

The rapid evolution of large language models (LLMs) has revolutionized
various fields, including the identification and discovery of human values
within text data. While traditional NLP models, such as BERT, have been
employed for this task, their ability to represent textual data is
significantly outperformed by emerging LLMs like GPTs. However, the performance
of online LLMs often degrades when handling long contexts required for value
identification, which also incurs substantial computational costs. To address
these challenges, we propose EAVIT, an efficient and accurate framework for
human value identification that combines the strengths of both locally
fine-tunable and online black-box LLMs. Our framework employs a value detector
- a small, local language model - to generate initial value estimations. These
estimations are then used to construct concise input prompts for online LLMs,
enabling accurate final value identification. To train the value detector, we
introduce explanation-based training and data generation techniques
specifically tailored for value identification, alongside sampling strategies
to optimize the brevity of LLM input prompts. Our approach effectively reduces
the number of input tokens by up to 1/6 compared to directly querying online
LLMs, while consistently outperforming traditional NLP methods and other
LLM-based strategies.

---


### [ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL](http://arxiv.org/abs/2505.12768v1)

In Text-to-SQL, execution feedback is essential for guiding large language
models (LLMs) to reason accurately and generate reliable SQL queries. However,
existing methods treat execution feedback solely as a post-hoc signal for
correction or selection, failing to integrate it into the generation process.
This limitation hinders their ability to address reasoning errors as they
occur, ultimately reducing query accuracy and robustness. To address this
issue, we propose ReEx-SQL (Reasoning with Execution-Aware Reinforcement
Learning), a framework for Text-to-SQL that enables models to interact with the
database during decoding and dynamically adjust their reasoning based on
execution feedback. ReEx-SQL introduces an execution-aware reasoning paradigm
that interleaves intermediate SQL execution into reasoning paths, facilitating
context-sensitive revisions. It achieves this through structured prompts with
markup tags and a stepwise rollout strategy that integrates execution feedback
into each stage of generation. To supervise policy learning, we develop a
composite reward function that includes an exploration reward, explicitly
encouraging effective database interaction. Additionally, ReEx-SQL adopts a
tree-based decoding strategy to support exploratory reasoning, enabling dynamic
expansion of alternative reasoning paths. Notably, ReEx-SQL achieves 88.8% on
Spider and 64.9% on BIRD at the 7B scale, surpassing the standard reasoning
baseline by 2.7% and 2.6%, respectively. It also shows robustness, achieving
85.2% on Spider-Realistic with leading performance. In addition, its
tree-structured decoding improves efficiency and performance over linear
decoding, reducing inference time by 51.9% on the BIRD development set.

---


### [G1: Bootstrapping Perception and Reasoning Abilities of Vision-Language Model via Reinforcement Learning](http://arxiv.org/abs/2505.13426v1)

Vision-Language Models (VLMs) excel in many direct multimodal tasks but
struggle to translate this prowess into effective decision-making within
interactive, visually rich environments like games. This ``knowing-doing'' gap
significantly limits their potential as autonomous agents, as leading VLMs
often performing badly in simple games. To address this, we introduce VLM-Gym,
a curated reinforcement learning (RL) environment featuring diverse visual
games with unified interfaces and adjustable, compositional difficulty,
specifically designed for scalable multi-game parallel training. Leveraging
VLM-Gym, we train G0 models using pure RL-driven self-evolution, which
demonstrate emergent perception and reasoning patterns. To further mitigate
challenges arising from game diversity, we develop G1 models. G1 incorporates a
perception-enhanced cold start prior to RL fine-tuning. Our resulting G1 models
consistently surpass their teacher across all games and outperform leading
proprietary models like Claude-3.7-Sonnet-Thinking. Systematic analysis reveals
an intriguing finding: perception and reasoning abilities mutually bootstrap
each other throughout the RL training process. Source code including VLM-Gym
and RL training are released at https://github.com/chenllliang/G1 to foster
future research in advancing VLMs as capable interactive agents.

---


### [From Local Details to Global Context: Advancing Vision-Language Models with Attention-Based Selection](http://arxiv.org/abs/2505.13233v1)

Pretrained vision-language models (VLMs), e.g., CLIP, demonstrate impressive
zero-shot capabilities on downstream tasks. Prior research highlights the
crucial role of visual augmentation techniques, like random cropping, in
alignment with fine-grained class descriptions generated by large language
models (LLMs), significantly enhancing zero-shot performance by incorporating
multi-view information. However, the inherent randomness of these augmentations
can inevitably introduce background artifacts and cause models to overly focus
on local details, compromising global semantic understanding. To address these
issues, we propose an \textbf{A}ttention-\textbf{B}ased \textbf{S}election
(\textbf{ABS}) method from local details to global context, which applies
attention-guided cropping in both raw images and feature space, supplement
global semantic information through strategic feature selection. Additionally,
we introduce a soft matching technique to effectively filter LLM descriptions
for better alignment. \textbf{ABS} achieves state-of-the-art performance on
out-of-distribution generalization and zero-shot classification tasks. Notably,
\textbf{ABS} is training-free and even rivals few-shot and test-time adaptation
methods. Our code is available at
\href{https://github.com/BIT-DA/ABS}{\textcolor{darkgreen}{https://github.com/BIT-DA/ABS}}.

---


### [Walking the Tightrope: Disentangling Beneficial and Detrimental Drifts in Non-Stationary Custom-Tuning](http://arxiv.org/abs/2505.13081v1)

This paper uncovers a critical yet overlooked phenomenon in multi-modal large
language models (MLLMs): detrimental concept drift within chain-of-thought
(CoT) reasoning during non-stationary reinforcement fine-tuning (RFT), where
reasoning token distributions evolve unpredictably, thereby introducing
significant biases in final predictions. To address this, we are pioneers in
establishing the theoretical bridge between concept drift theory and RFT
processes by formalizing CoT's autoregressive token streams as non-stationary
distributions undergoing arbitrary temporal shifts. Leveraging this framework,
we propose a novel counterfact-aware RFT that systematically decouples
beneficial distribution adaptation from harmful concept drift through concept
graph-empowered LLM experts generating counterfactual reasoning trajectories.
Our solution, Counterfactual Preference Optimization (CPO), enables stable RFT
in non-stationary environments, particularly within the medical domain, through
custom-tuning of counterfactual-aware preference alignment. Extensive
experiments demonstrate our superior performance of robustness, generalization
and coordination within RFT. Besides, we also contributed a large-scale dataset
CXR-CounterFact (CCF), comprising 320,416 meticulously curated counterfactual
reasoning trajectories derived from MIMIC-CXR. Our code and data are public.

---


### [AdaToken-3D: Dynamic Spatial Gating for Efficient 3D Large Multimodal-Models Reasoning](http://arxiv.org/abs/2505.12782v1)

Large Multimodal Models (LMMs) have become a pivotal research focus in deep
learning, demonstrating remarkable capabilities in 3D scene understanding.
However, current 3D LMMs employing thousands of spatial tokens for multimodal
reasoning suffer from critical inefficiencies: excessive computational overhead
and redundant information flows. Unlike 2D VLMs processing single images, 3D
LMMs exhibit inherent architectural redundancy due to the heterogeneous
mechanisms between spatial tokens and visual tokens. To address this challenge,
we propose AdaToken-3D, an adaptive spatial token optimization framework that
dynamically prunes redundant tokens through spatial contribution analysis. Our
method automatically tailors pruning strategies to different 3D LMM
architectures by quantifying token-level information flows via attention
pattern mining. Extensive experiments on LLaVA-3D (a 7B parameter 3D-LMM)
demonstrate that AdaToken-3D achieves 21\% faster inference speed and 63\%
FLOPs reduction while maintaining original task accuracy. Beyond efficiency
gains, this work systematically investigates redundancy patterns in multimodal
spatial information flows through quantitative token interaction analysis. Our
findings reveal that over 60\% of spatial tokens contribute minimally ($<$5\%)
to the final predictions, establishing theoretical foundations for efficient 3D
multimodal learning.

---


### [Reasoning-OCR: Can Large Multimodal Models Solve Complex Logical Reasoning Problems from OCR Cues?](http://arxiv.org/abs/2505.12766v1)

Large Multimodal Models (LMMs) have become increasingly versatile,
accompanied by impressive Optical Character Recognition (OCR) related
capabilities. Existing OCR-related benchmarks emphasize evaluating LMMs'
abilities of relatively simple visual question answering, visual-text parsing,
etc. However, the extent to which LMMs can deal with complex logical reasoning
problems based on OCR cues is relatively unexplored. To this end, we introduce
the Reasoning-OCR benchmark, which challenges LMMs to solve complex reasoning
problems based on the cues that can be extracted from rich visual-text.
Reasoning-OCR covers six visual scenarios and encompasses 150 meticulously
designed questions categorized into six reasoning challenges. Additionally,
Reasoning-OCR minimizes the impact of field-specialized knowledge. Our
evaluation offers some insights for proprietary and open-source LMMs in
different reasoning challenges, underscoring the urgent to improve the
reasoning performance. We hope Reasoning-OCR can inspire and facilitate future
research on enhancing complex reasoning ability based on OCR cues.
Reasoning-OCR is publicly available at
https://github.com/Hxyz-123/ReasoningOCR.

---


### [FLASH: Latent-Aware Semi-Autoregressive Speculative Decoding for Multimodal Tasks](http://arxiv.org/abs/2505.12728v1)

Large language and multimodal models (LLMs and LMMs) exhibit strong inference
capabilities but are often limited by slow decoding speeds. This challenge is
especially acute in LMMs, where visual inputs typically comprise more tokens
with lower information density than text -- an issue exacerbated by recent
trends toward finer-grained visual tokenizations to boost performance.
Speculative decoding has been effective in accelerating LLM inference by using
a smaller draft model to generate candidate tokens, which are then selectively
verified by the target model, improving speed without sacrificing output
quality. While this strategy has been extended to LMMs, existing methods
largely overlook the unique properties of visual inputs and depend solely on
text-based draft models. In this work, we propose \textbf{FLASH} (Fast
Latent-Aware Semi-Autoregressive Heuristics), a speculative decoding framework
designed specifically for LMMs, which leverages two key properties of
multimodal data to design the draft model. First, to address redundancy in
visual tokens, we propose a lightweight latent-aware token compression
mechanism. Second, recognizing that visual objects often co-occur within a
scene, we employ a semi-autoregressive decoding strategy to generate multiple
tokens per forward pass. These innovations accelerate draft decoding while
maintaining high acceptance rates, resulting in faster overall inference.
Experiments show that FLASH significantly outperforms prior speculative
decoding approaches in both unimodal and multimodal settings, achieving up to
\textbf{2.68$\times$} speed-up on video captioning and \textbf{2.55$\times$} on
visual instruction tuning tasks compared to the original LMM.

---


### [Gluon: Making Muon & Scion Great Again! (Bridging Theory and Practice of LMO-based Optimizers for LLMs)](http://arxiv.org/abs/2505.13416v1)

Recent developments in deep learning optimization have brought about
radically new algorithms based on the Linear Minimization Oracle (LMO)
framework, such as $\sf Muon$ and $\sf Scion$. After over a decade of $\sf
Adam$'s dominance, these LMO-based methods are emerging as viable replacements,
offering several practical advantages such as improved memory efficiency,
better hyperparameter transferability, and most importantly, superior empirical
performance on large-scale tasks, including LLM training. However, a
significant gap remains between their practical use and our current theoretical
understanding: prior analyses (1) overlook the layer-wise LMO application of
these optimizers in practice, and (2) rely on an unrealistic smoothness
assumption, leading to impractically small stepsizes. To address both, we
propose a new LMO-based method called $\sf Gluon$, capturing prior
theoretically analyzed methods as special cases, and introduce a new refined
generalized smoothness model that captures the layer-wise geometry of neural
networks, matches the layer-wise practical implementation of $\sf Muon$ and
$\sf Scion$, and leads to convergence guarantees with strong practical
predictive power. Unlike prior results, our theoretical stepsizes closely match
the fine-tuned values reported by Pethick et al. (2025). Our experiments with
NanoGPT and CNN confirm that our assumption holds along the optimization
trajectory, ultimately closing the gap between theory and practice.

---


### [A Dataless Reinforcement Learning Approach to Rounding Hyperplane Optimization for Max-Cut](http://arxiv.org/abs/2505.13405v1)

The Maximum Cut (MaxCut) problem is NP-Complete, and obtaining its optimal
solution is NP-hard in the worst case. As a result, heuristic-based algorithms
are commonly used, though their design often requires significant domain
expertise. More recently, learning-based methods trained on large (un)labeled
datasets have been proposed; however, these approaches often struggle with
generalizability and scalability. A well-known approximation algorithm for
MaxCut is the Goemans-Williamson (GW) algorithm, which relaxes the Quadratic
Unconstrained Binary Optimization (QUBO) formulation into a semidefinite
program (SDP). The GW algorithm then applies hyperplane rounding by uniformly
sampling a random hyperplane to convert the SDP solution into binary node
assignments. In this paper, we propose a training-data-free approach based on a
non-episodic reinforcement learning formulation, in which an agent learns to
select improved rounding hyperplanes that yield better cuts than those produced
by the GW algorithm. By optimizing over a Markov Decision Process (MDP), our
method consistently achieves better cuts across large-scale graphs with varying
densities and degree distributions.

---


### [Unlabeled Data or Pre-trained Model: Rethinking Semi-Supervised Learning and Pretrain-Finetuning](http://arxiv.org/abs/2505.13317v1)

Semi-supervised learning (SSL) alleviates the cost of data labeling process
by exploiting unlabeled data, and has achieved promising results on various
tasks such as image classification. Meanwhile, the Pretrain-Finetuning paradigm
has garnered significant attention in recent years, and exploiting pre-trained
models could also reduce the requirement of labeled data in downstream tasks.
Therefore, a question naturally occurs: \emph{When the labeled data is scarce
in the target tasks, should we exploit unlabeled data or pre-trained models?}
To answer this question, we select pre-trained Vision-Language Models (VLMs) as
representative pretrain-finetuning instances and propose \textit{Few-shot SSL}
-- a framework that enables fair comparison between these two paradigms by
controlling the amount of labeled data used. Extensive experiments across
various settings demonstrate that pre-trained VLMs generally outperform SSL
methods in nearly all cases, except when the data has low resolution or lacks
clear semantic structure. Therefore, we encourage future SSL research to
compare with pre-trained models and explore deeper integration, such as using
pre-trained knowledge to enhance pseudo-labeling. To support future research,
we release our unified reproduction and evaluation framework. Codes are
available at
https://anonymous.4open.science/r/Rethinking-SSL-and-Pretrain-Finetuning-5566

---


### [Automatic mixed precision for optimizing gained time with constrained loss mean-squared-error based on model partition to sequential sub-graphs](http://arxiv.org/abs/2505.13060v1)

Quantization is essential for Neural Network (NN) compression, reducing model
size and computational demands by using lower bit-width data types, though
aggressive reduction often hampers accuracy. Mixed Precision (MP) mitigates
this tradeoff by varying the numerical precision across network layers. This
study focuses on automatically selecting an optimal MP configuration within
Post-Training Quantization (PTQ) for inference. The first key contribution is a
novel sensitivity metric derived from a first-order Taylor series expansion of
the loss function as a function of quantization errors in weights and
activations. This metric, based on the Mean Square Error (MSE) of the loss, is
efficiently calculated per layer using high-precision forward and backward
passes over a small calibration dataset. The metric is additive across layers,
with low calibration memory overhead as weight optimization is unnecessary. The
second contribution is an accurate hardware-aware method for predicting MP time
gain by modeling it as additive for sequential sub-graphs. An algorithm
partitions the model graph into sequential subgraphs, measuring time gain for
each configuration using a few samples. After calibrating per-layer sensitivity
and time gain, an Integer Programming (IP) problem is formulated to maximize
time gain while keeping loss MSE below a set threshold. Memory gain and
theoretical time gain based on Multiply and Accumulate (MAC) operations are
also considered. Rigorous experiments on the Intel Gaudi 2 accelerator validate
the approach on several Large Language Models (LLMs).

---


### [Seeing, Saying, Solving: An LLM-to-TL Framework for Cooperative Robots](http://arxiv.org/abs/2505.13376v1)

Increased robot deployment, such as in warehousing, has revealed a need for
seamless collaboration among heterogeneous robot teams to resolve unforeseen
conflicts. To address this challenge, we propose a novel, decentralized
framework for robots to request and provide help. The framework begins with
robots detecting conflicts using a Vision Language Model (VLM), then reasoning
over whether help is needed. If so, it crafts and broadcasts a natural language
(NL) help request using a Large Language Model (LLM). Potential helper robots
reason over the request and offer help (if able), along with information about
impact to their current tasks. Helper reasoning is implemented via an LLM
grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar
to guarantee syntactically valid NL-to-STL translations, which are then solved
as a Mixed Integer Linear Program (MILP). Finally, the requester robot chooses
a helper by reasoning over impact on the overall system. We evaluate our system
via experiments considering different strategies for choosing a helper, and
find that a requester robot can minimize overall time impact on the system by
considering multiple help offers versus simple heuristics (e.g., selecting the
nearest robot to help).

---


### [HydraInfer: Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving](http://arxiv.org/abs/2505.12658v1)

Multimodal Large Language Models (MLLMs) have been rapidly advancing,
enabling cross-modal understanding and generation, and propelling artificial
intelligence towards artificial general intelligence. However, existing MLLM
inference systems are typically designed based on the architecture of language
models, integrating image processing and language processing as a single
scheduling unit. This design struggles to accommodate the heterogeneous demands
of different stages in terms of computational resources, memory access
patterns, and service-level objectives (SLOs), leading to low resource
utilization and high request latency, ultimately failing to meet the service
requirements of diverse inference scenarios.
  To address these challenges, we propose HydraInfer, an efficient MLLM
inference system that adopts a Hybrid Encode-Prefill-Decode (EPD)
Disaggregation architecture. By scheduling the three stages - encode, prefill,
and decode - onto separate heterogeneous inference instances, the system
flexibly reallocates resources across stages, significantly reducing idle
computation, alleviating resource bottlenecks, and improving overall system
throughput and scalability. In addition, HydraInfer supports a stage-level
batching strategy that enhances load balancing, enables parallel execution of
visual and language models, and further optimizes inference performance.
Experiments under real multimodal inference workloads demonstrate that
HydraInfer can achieve up to 4x higher inference throughput compared to
state-of-the-art systems (e.g., vLLM) on a single-node 8xH800 GPU cluster,
while meeting the 90th percentile request SLO.

---


