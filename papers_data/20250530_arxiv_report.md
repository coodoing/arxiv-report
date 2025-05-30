### [Differential Information: An Information-Theoretic Perspective on Preference Optimization](http://arxiv.org/abs/2505.23761v1)

Direct Preference Optimization (DPO) has become a standard technique for
aligning language models with human preferences in a supervised manner. Despite
its empirical success, the theoretical justification behind its log-ratio
reward parameterization remains incomplete. In this work, we address this gap
by utilizing the Differential Information Distribution (DID): a distribution
over token sequences that captures the information gained during policy
updates. First, we show that when preference labels encode the differential
information required to transform a reference policy into a target policy, the
log-ratio reward in DPO emerges as the uniquely optimal form for learning the
target policy via preference optimization. This result naturally yields a
closed-form expression for the optimal sampling distribution over rejected
responses. Second, we find that the condition for preferences to encode
differential information is fundamentally linked to an implicit assumption
regarding log-margin ordered policies-an inductive bias widely used in
preference optimization yet previously unrecognized. Finally, by analyzing the
entropy of the DID, we characterize how learning low-entropy differential
information reinforces the policy distribution, while high-entropy differential
information induces a smoothing effect, which explains the log-likelihood
displacement phenomenon. We validate our theoretical findings in synthetic
experiments and extend them to real-world instruction-following datasets. Our
results suggest that learning high-entropy differential information is crucial
for general instruction-following, while learning low-entropy differential
information benefits knowledge-intensive question answering. Overall, our work
presents a unifying perspective on the DPO objective, the structure of
preference data, and resulting policy behaviors through the lens of
differential information.

---


### [DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning](http://arxiv.org/abs/2505.23754v1)

Theorem proving serves as a major testbed for evaluating complex reasoning
abilities in large language models (LLMs). However, traditional automated
theorem proving (ATP) approaches rely heavily on formal proof systems that
poorly align with LLMs' strength derived from informal, natural language
knowledge acquired during pre-training. In this work, we propose DeepTheorem, a
comprehensive informal theorem-proving framework exploiting natural language to
enhance LLM mathematical reasoning. DeepTheorem includes a large-scale
benchmark dataset consisting of 121K high-quality IMO-level informal theorems
and proofs spanning diverse mathematical domains, rigorously annotated for
correctness, difficulty, and topic categories, accompanied by systematically
constructed verifiable theorem variants. We devise a novel reinforcement
learning strategy (RL-Zero) explicitly tailored to informal theorem proving,
leveraging the verified theorem variants to incentivize robust mathematical
inference. Additionally, we propose comprehensive outcome and process
evaluation metrics examining proof correctness and the quality of reasoning
steps. Extensive experimental analyses demonstrate DeepTheorem significantly
improves LLM theorem-proving performance compared to existing datasets and
supervised fine-tuning protocols, achieving state-of-the-art accuracy and
reasoning quality. Our findings highlight DeepTheorem's potential to
fundamentally advance automated informal theorem proving and mathematical
exploration.

---


### [Boosting Domain Incremental Learning: Selecting the Optimal Parameters is All You Need](http://arxiv.org/abs/2505.23744v1)

Deep neural networks (DNNs) often underperform in real-world, dynamic
settings where data distributions change over time. Domain Incremental Learning
(DIL) offers a solution by enabling continual model adaptation, with
Parameter-Isolation DIL (PIDIL) emerging as a promising paradigm to reduce
knowledge conflicts. However, existing PIDIL methods struggle with parameter
selection accuracy, especially as the number of domains and corresponding
classes grows. To address this, we propose SOYO, a lightweight framework that
improves domain selection in PIDIL. SOYO introduces a Gaussian Mixture
Compressor (GMC) and Domain Feature Resampler (DFR) to store and balance prior
domain data efficiently, while a Multi-level Domain Feature Fusion Network
(MDFN) enhances domain feature extraction. Our framework supports multiple
Parameter-Efficient Fine-Tuning (PEFT) methods and is validated across tasks
such as image classification, object detection, and speech enhancement.
Experimental results on six benchmarks demonstrate SOYO's consistent
superiority over existing baselines, showcasing its robustness and adaptability
in complex, evolving environments. The codes will be released in
https://github.com/qwangcv/SOYO.

---


### [Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time](http://arxiv.org/abs/2505.23729v1)

Aligning large language models with humans is challenging due to the
inherently multifaceted nature of preference feedback. While existing
approaches typically frame this as a multi-objective optimization problem, they
often overlook how humans actually make decisions. Research on bounded
rationality suggests that human decision making follows satisficing
strategies-optimizing primary objectives while ensuring others meet acceptable
thresholds. To bridge this gap and operationalize the notion of satisficing
alignment, we propose SITAlign: an inference time framework that addresses the
multifaceted nature of alignment by maximizing a primary objective while
satisfying threshold-based constraints on secondary criteria. We provide
theoretical insights by deriving sub-optimality bounds of our satisficing based
inference alignment approach. We empirically validate SITAlign's performance
through extensive experimentation on multiple benchmarks. For instance, on the
PKU-SafeRLHF dataset with the primary objective of maximizing helpfulness while
ensuring a threshold on harmlessness, SITAlign outperforms the state-of-the-art
multi objective decoding strategy by a margin of 22.3% in terms of GPT-4
win-tie rate for helpfulness reward while adhering to the threshold on
harmlessness.

---


### [SC-LoRA: Balancing Efficient Fine-tuning and Knowledge Preservation via Subspace-Constrained LoRA](http://arxiv.org/abs/2505.23724v1)

Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank
Adaptation (LoRA), are indispensable for efficiently customizing Large Language
Models (LLMs). However, vanilla LoRA suffers from slow convergence speed and
knowledge forgetting problems. Recent studies have leveraged the power of
designed LoRA initialization, to enhance the fine-tuning efficiency, or to
preserve knowledge in the pre-trained LLM. However, none of these works can
address the two cases at the same time. To this end, we introduce
Subspace-Constrained LoRA (SC-LoRA), a novel LoRA initialization framework
engineered to navigate the trade-off between efficient fine-tuning and
knowledge preservation. We achieve this by constraining the output of trainable
LoRA adapters in a low-rank subspace, where the context information of
fine-tuning data is most preserved while the context information of preserved
knowledge is least retained, in a balanced way. Such constraint enables the
trainable weights to primarily focus on the main features of fine-tuning data
while avoiding damaging the preserved knowledge features. We provide
theoretical analysis on our method, and conduct extensive experiments including
safety preservation and world knowledge preservation, on various downstream
tasks. In our experiments, SC-LoRA succeeds in delivering superior fine-tuning
performance while markedly diminishing knowledge forgetting, surpassing
contemporary LoRA initialization methods.

---


### [ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering](http://arxiv.org/abs/2505.23723v1)

The emergence of large language model (LLM)-based agents has significantly
advanced the development of autonomous machine learning (ML) engineering.
However, most existing approaches rely heavily on manual prompt engineering,
failing to adapt and optimize based on diverse experimental experiences.
Focusing on this, for the first time, we explore the paradigm of learning-based
agentic ML, where an LLM agent learns through interactive experimentation on ML
tasks using online reinforcement learning (RL). To realize this, we propose a
novel agentic ML training framework with three key components: (1)
exploration-enriched fine-tuning, which enables LLM agents to generate diverse
actions for enhanced RL exploration; (2) step-wise RL, which enables training
on a single action step, accelerating experience collection and improving
training efficiency; (3) an agentic ML-specific reward module, which unifies
varied ML feedback signals into consistent rewards for RL optimization.
Leveraging this framework, we train ML-Agent, driven by a 7B-sized Qwen-2.5 LLM
for autonomous ML. Remarkably, despite being trained on merely 9 ML tasks, our
7B-sized ML-Agent outperforms the 671B-sized DeepSeek-R1 agent. Furthermore, it
achieves continuous performance improvements and demonstrates exceptional
cross-task generalization capabilities.

---


### [Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability](http://arxiv.org/abs/2505.23703v1)

Enhancing the mathematical reasoning capabilities of LLMs has garnered
significant attention in both the mathematical and computer science
communities. Recent works have made substantial progress in both Natural
Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the
potential of pure Reinforcement Learning (RL) methods on base models. However,
RL approaches struggle to impart new capabilities not presented in the base
model, highlighting the need to integrate more knowledge like FL into NL math
reasoning effectively. Yet, this integration is challenging due to inherent
disparities in problem structure and reasoning format between NL and FL. To
address these challenges, we introduce **NL-FL HybridReasoning**, an end-to-end
framework designed to incorporate the FL expert into NL math problem-solving.
To bridge the NL and FL input format gap, we propose the *NL-FL Problem
Alignment* method, which reformulates the Question-Answering (QA) problems in
NL as existence theorems in FL. Subsequently, the *Mixed Problem Input*
technique we provide enables the FL reasoner to handle both QA and existence
problems concurrently. Lastly, we mitigate the NL and FL output format gap in
reasoning through an LLM-based *Answer Extraction* mechanism. Comprehensive
experiments demonstrate that the **HybridReasoning** framework achieves
**89.80%** and **84.34%** accuracy rates on the MATH-500 and the AMC
benchmarks, surpassing the NL baseline by 4.60% and 4.82%, respectively.
Notably, some problems resolved by our framework remain unsolved by the NL
baseline model even under a larger number of trials.

---


### [CLDTracker: A Comprehensive Language Description for Visual Tracking](http://arxiv.org/abs/2505.23704v1)

VOT remains a fundamental yet challenging task in computer vision due to
dynamic appearance changes, occlusions, and background clutter. Traditional
trackers, relying primarily on visual cues, often struggle in such complex
scenarios. Recent advancements in VLMs have shown promise in semantic
understanding for tasks like open-vocabulary detection and image captioning,
suggesting their potential for VOT. However, the direct application of VLMs to
VOT is hindered by critical limitations: the absence of a rich and
comprehensive textual representation that semantically captures the target
object's nuances, limiting the effective use of language information;
inefficient fusion mechanisms that fail to optimally integrate visual and
textual features, preventing a holistic understanding of the target; and a lack
of temporal modeling of the target's evolving appearance in the language
domain, leading to a disconnect between the initial description and the
object's subsequent visual changes. To bridge these gaps and unlock the full
potential of VLMs for VOT, we propose CLDTracker, a novel Comprehensive
Language Description framework for robust visual Tracking. Our tracker
introduces a dual-branch architecture consisting of a textual and a visual
branch. In the textual branch, we construct a rich bag of textual descriptions
derived by harnessing the powerful VLMs such as CLIP and GPT-4V, enriched with
semantic and contextual cues to address the lack of rich textual
representation. Experiments on six standard VOT benchmarks demonstrate that
CLDTracker achieves SOTA performance, validating the effectiveness of
leveraging robust and temporally-adaptive vision-language representations for
tracking. Code and models are publicly available at:
https://github.com/HamadYA/CLDTracker

---


### [Data-to-Dashboard: Multi-Agent LLM Framework for Insightful Visualization in Enterprise Analytics](http://arxiv.org/abs/2505.23695v1)

The rapid advancement of LLMs has led to the creation of diverse agentic
systems in data analysis, utilizing LLMs' capabilities to improve insight
generation and visualization. In this paper, we present an agentic system that
automates the data-to-dashboard pipeline through modular LLM agents capable of
domain detection, concept extraction, multi-perspective analysis generation,
and iterative self-reflection. Unlike existing chart QA systems, our framework
simulates the analytical reasoning process of business analysts by retrieving
domain-relevant knowledge and adapting to diverse datasets without relying on
closed ontologies or question templates.
  We evaluate our system on three datasets across different domains.
Benchmarked against GPT-4o with a single-prompt baseline, our approach shows
improved insightfulness, domain relevance, and analytical depth, as measured by
tailored evaluation metrics and qualitative human assessment.
  This work contributes a novel modular pipeline to bridge the path from raw
data to visualization, and opens new opportunities for human-in-the-loop
validation by domain experts in business analytics. All code can be found here:
https://github.com/77luvC/D2D_Data2Dashboard

---


### [VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos](http://arxiv.org/abs/2505.23693v1)

MLLMs have been widely studied for video question answering recently.
However, most existing assessments focus on natural videos, overlooking
synthetic videos, such as AI-generated content (AIGC). Meanwhile, some works in
video generation rely on MLLMs to evaluate the quality of generated videos, but
the capabilities of MLLMs on interpreting AIGC videos remain largely
underexplored. To address this, we propose a new benchmark, VF-Eval, which
introduces four tasks-coherence validation, error awareness, error type
detection, and reasoning evaluation-to comprehensively evaluate the abilities
of MLLMs on AIGC videos. We evaluate 13 frontier MLLMs on VF-Eval and find that
even the best-performing model, GPT-4.1, struggles to achieve consistently good
performance across all tasks. This highlights the challenging nature of our
benchmark. Additionally, to investigate the practical applications of VF-Eval
in improving video generation, we conduct an experiment, RePrompt,
demonstrating that aligning MLLMs more closely with human feedback can benefit
video generation.

---


### [Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles](http://arxiv.org/abs/2505.23590v1)

The application of rule-based reinforcement learning (RL) to multimodal large
language models (MLLMs) introduces unique challenges and potential deviations
from findings in text-only domains, particularly for perception-heavy tasks.
This paper provides a comprehensive study of rule-based visual RL using jigsaw
puzzles as a structured experimental framework, revealing several key findings.
\textit{Firstly,} we find that MLLMs, initially performing near to random
guessing on simple puzzles, achieve near-perfect accuracy and generalize to
complex, unseen configurations through fine-tuning. \textit{Secondly,} training
on jigsaw puzzles can induce generalization to other visual tasks, with
effectiveness tied to specific task configurations. \textit{Thirdly,} MLLMs can
learn and generalize with or without explicit reasoning, though open-source
models often favor direct answering. Consequently, even when trained for
step-by-step reasoning, they can ignore the thinking process in deriving the
final answer. \textit{Fourthly,} we observe that complex reasoning patterns
appear to be pre-existing rather than emergent, with their frequency increasing
alongside training and task difficulty. \textit{Finally,} our results
demonstrate that RL exhibits more effective generalization than Supervised
Fine-Tuning (SFT), and an initial SFT cold start phase can hinder subsequent RL
optimization. Although these observations are based on jigsaw puzzles and may
vary across other visual tasks, this research contributes a valuable piece of
jigsaw to the larger puzzle of collective understanding rule-based visual RL
and its potential in multimodal learning. The code is available at:
\href{https://github.com/zifuwanggg/Jigsaw-R1}{https://github.com/zifuwanggg/Jigsaw-R1}.

---


### [Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](http://arxiv.org/abs/2505.23564v1)

Enhancing the reasoning capabilities of large language models effectively
using reinforcement learning (RL) remains a crucial challenge. Existing
approaches primarily adopt two contrasting advantage estimation granularities:
Token-level methods (e.g., PPO) aim to provide the fine-grained advantage
signals but suffer from inaccurate estimation due to difficulties in training
an accurate critic model. On the other extreme, trajectory-level methods (e.g.,
GRPO) solely rely on a coarse-grained advantage signal from the final reward,
leading to imprecise credit assignment. To address these limitations, we
propose Segment Policy Optimization (SPO), a novel RL framework that leverages
segment-level advantage estimation at an intermediate granularity, achieving a
better balance by offering more precise credit assignment than trajectory-level
methods and requiring fewer estimation points than token-level methods,
enabling accurate advantage estimation based on Monte Carlo (MC) without a
critic model. SPO features three components with novel strategies: (1) flexible
segment partition; (2) accurate segment advantage estimation; and (3) policy
optimization using segment advantages, including a novel probability-mask
strategy. We further instantiate SPO for two specific scenarios: (1) SPO-chain
for short chain-of-thought (CoT), featuring novel cutpoint-based partition and
chain-based advantage estimation, achieving $6$-$12$ percentage point
improvements in accuracy over PPO and GRPO on GSM8K. (2) SPO-tree for long CoT,
featuring novel tree-based advantage estimation, which significantly reduces
the cost of MC estimation, achieving $7$-$11$ percentage point improvements
over GRPO on MATH500 under 2K and 4K context evaluation. We make our code
publicly available at https://github.com/AIFrameResearch/SPO.

---


### [SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents](http://arxiv.org/abs/2505.23559v1)

Recent advancements in large language model (LLM) agents have significantly
accelerated scientific discovery automation, yet concurrently raised critical
ethical and safety concerns. To systematically address these challenges, we
introduce \textbf{SafeScientist}, an innovative AI scientist framework
explicitly designed to enhance safety and ethical responsibility in AI-driven
scientific exploration. SafeScientist proactively refuses ethically
inappropriate or high-risk tasks and rigorously emphasizes safety throughout
the research process. To achieve comprehensive safety oversight, we integrate
multiple defensive mechanisms, including prompt monitoring, agent-collaboration
monitoring, tool-use monitoring, and an ethical reviewer component.
Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel
benchmark specifically designed to evaluate AI safety in scientific contexts,
comprising 240 high-risk scientific tasks across 6 domains, alongside 30
specially designed scientific tools and 120 tool-related risk tasks. Extensive
experiments demonstrate that SafeScientist significantly improves safety
performance by 35\% compared to traditional AI scientist frameworks, without
compromising scientific output quality. Additionally, we rigorously validate
the robustness of our safety pipeline against diverse adversarial attack
methods, further confirming the effectiveness of our integrated approach. The
code and data will be available at https://github.com/ulab-uiuc/SafeScientist.
\textcolor{red}{Warning: this paper contains example data that may be offensive
or harmful.}

---


### [Sustainable Carbon-Aware and Water-Efficient LLM Scheduling in Geo-Distributed Cloud Datacenters](http://arxiv.org/abs/2505.23554v1)

In recent years, Large Language Models (LLM) such as ChatGPT, CoPilot, and
Gemini have been widely adopted in different areas. As the use of LLMs
continues to grow, many efforts have focused on reducing the massive training
overheads of these models. But it is the environmental impact of handling user
requests to LLMs that is increasingly becoming a concern. Recent studies
estimate that the costs of operating LLMs in their inference phase can exceed
training costs by 25x per year. As LLMs are queried incessantly, the cumulative
carbon footprint for the operational phase has been shown to far exceed the
footprint during the training phase. Further, estimates indicate that 500 ml of
fresh water is expended for every 20-50 requests to LLMs during inference. To
address these important sustainability issues with LLMs, we propose a novel
framework called SLIT to co-optimize LLM quality of service (time-to-first
token), carbon emissions, water usage, and energy costs. The framework utilizes
a machine learning (ML) based metaheuristic to enhance the sustainability of
LLM hosting across geo-distributed cloud datacenters. Such a framework will
become increasingly vital as LLMs proliferate.

---


### [Subgraph Gaussian Embedding Contrast for Self-Supervised Graph Representation Learning](http://arxiv.org/abs/2505.23529v1)

Graph Representation Learning (GRL) is a fundamental task in machine
learning, aiming to encode high-dimensional graph-structured data into
low-dimensional vectors. Self-Supervised Learning (SSL) methods are widely used
in GRL because they can avoid expensive human annotation. In this work, we
propose a novel Subgraph Gaussian Embedding Contrast (SubGEC) method. Our
approach introduces a subgraph Gaussian embedding module, which adaptively maps
subgraphs to a structured Gaussian space, ensuring the preservation of input
subgraph characteristics while generating subgraphs with a controlled
distribution. We then employ optimal transport distances, more precisely the
Wasserstein and Gromov-Wasserstein distances, to effectively measure the
similarity between subgraphs, enhancing the robustness of the contrastive
learning process. Extensive experiments across multiple benchmarks demonstrate
that \method~outperforms or presents competitive performance against
state-of-the-art approaches. Our findings provide insights into the design of
SSL methods for GRL, emphasizing the importance of the distribution of the
generated contrastive pairs.

---


### [Enhanced DACER Algorithm with High Diffusion Efficiency](http://arxiv.org/abs/2505.23426v1)

Due to their expressive capacity, diffusion models have shown great promise
in offline RL and imitation learning. Diffusion Actor-Critic with Entropy
Regulator (DACER) extended this capability to online RL by using the reverse
diffusion process as a policy approximator, trained end-to-end with policy
gradient methods, achieving strong performance. However, this comes at the cost
of requiring many diffusion steps, which significantly hampers training
efficiency, while directly reducing the steps leads to noticeable performance
degradation. Critically, the lack of inference efficiency becomes a significant
bottleneck for applying diffusion policies in real-time online RL settings. To
improve training and inference efficiency while maintaining or even enhancing
performance, we propose a Q-gradient field objective as an auxiliary
optimization target to guide the denoising process at each diffusion step.
Nonetheless, we observe that the independence of the Q-gradient field from the
diffusion time step negatively impacts the performance of the diffusion policy.
To address this, we introduce a temporal weighting mechanism that enables the
model to efficiently eliminate large-scale noise in the early stages and refine
actions in the later stages. Experimental results on MuJoCo benchmarks and
several multimodal tasks demonstrate that the DACER2 algorithm achieves
state-of-the-art performance in most MuJoCo control tasks with only five
diffusion steps, while also exhibiting stronger multimodality compared to
DACER.

---


### [GAM-Agent: Game-Theoretic and Uncertainty-Aware Collaboration for Complex Visual Reasoning](http://arxiv.org/abs/2505.23399v1)

We propose GAM-Agent, a game-theoretic multi-agent framework for enhancing
vision-language reasoning. Unlike prior single-agent or monolithic models,
GAM-Agent formulates the reasoning process as a non-zero-sum game between base
agents--each specializing in visual perception subtasks--and a critical agent
that verifies logic consistency and factual correctness. Agents communicate via
structured claims, evidence, and uncertainty estimates. The framework
introduces an uncertainty-aware controller to dynamically adjust agent
collaboration, triggering multi-round debates when disagreement or ambiguity is
detected. This process yields more robust and interpretable predictions.
Experiments on four challenging benchmarks--MMMU, MMBench, MVBench, and
V*Bench--demonstrate that GAM-Agent significantly improves performance across
various VLM backbones. Notably, GAM-Agent boosts the accuracy of small-to-mid
scale models (e.g., Qwen2.5-VL-7B, InternVL3-14B) by 5--6\%, and still enhances
strong models like GPT-4o by up to 2--3\%. Our approach is modular, scalable,
and generalizable, offering a path toward reliable and explainable multi-agent
multimodal reasoning.

---


### [Afterburner: Reinforcement Learning Facilitates Self-Improving Code Efficiency Optimization](http://arxiv.org/abs/2505.23387v1)

Large Language Models (LLMs) generate functionally correct solutions but
often fall short in code efficiency, a critical bottleneck for real-world
deployment. In this paper, we introduce a novel test-time iterative
optimization framework to address this, employing a closed-loop system where
LLMs iteratively refine code based on empirical performance feedback from an
execution sandbox. We explore three training strategies: Supervised Fine-Tuning
(SFT), Direct Preference Optimization (DPO), and Group Relative Policy
Optimization~(GRPO). Experiments on our Venus dataset and the APPS benchmark
show that SFT and DPO rapidly saturate in efficiency gains. In contrast, GRPO,
using reinforcement learning (RL) with execution feedback, continuously
optimizes code performance, significantly boosting both pass@1 (from 47% to
62%) and the likelihood of outperforming human submissions in efficiency (from
31% to 45%). Our work demonstrates effective test-time code efficiency
improvement and critically reveals the power of RL in teaching LLMs to truly
self-improve code efficiency.

---


### [Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](http://arxiv.org/abs/2505.23349v1)

Rewards serve as proxies for human preferences and play a crucial role in
Reinforcement Learning from Human Feedback (RLHF). However, if these rewards
are inherently imperfect, exhibiting various biases, they can adversely affect
the alignment of large language models (LLMs). In this paper, we collectively
define the various biases present in rewards as the problem of reward
unfairness. We propose a bias-agnostic method to address the issue of reward
fairness from a resource allocation perspective, without specifically designing
for each type of bias, yet effectively mitigating them. Specifically, we model
preference learning as a resource allocation problem, treating rewards as
resources to be allocated while considering the trade-off between utility and
fairness in their distribution. We propose two methods, Fairness Regularization
and Fairness Coefficient, to achieve fairness in rewards. We apply our methods
in both verification and reinforcement learning scenarios to obtain a fairness
reward model and a policy model, respectively. Experiments conducted in these
scenarios demonstrate that our approach aligns LLMs with human preferences in a
more fair manner.

---


### [Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization](http://arxiv.org/abs/2505.23331v1)

Fine-tuning pre-trained generative models with Reinforcement Learning (RL)
has emerged as an effective approach for aligning outputs more closely with
nuanced human preferences. In this paper, we investigate the application of
Group Relative Policy Optimization (GRPO) to fine-tune next-scale visual
autoregressive (VAR) models. Our empirical results demonstrate that this
approach enables alignment to intricate reward signals derived from aesthetic
predictors and CLIP embeddings, significantly enhancing image quality and
enabling precise control over the generation style. Interestingly, by
leveraging CLIP, our method can help VAR models generalize beyond their initial
ImageNet distribution: through RL-driven exploration, these models can generate
images aligned with prompts referencing image styles that were absent during
pre-training. In summary, we show that RL-based fine-tuning is both efficient
and effective for VAR models, benefiting particularly from their fast inference
speeds, which are advantageous for online sampling, an aspect that poses
significant challenges for diffusion-based alternatives.

---


### [Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective](http://arxiv.org/abs/2505.23277v1)

Retrieval-augmented generation (RAG) enhances large language models (LLMs)
with external context, but retrieved passages are often lengthy, noisy, or
exceed input limits. Existing compression methods typically require supervised
training of dedicated compression models, increasing cost and reducing
portability. We propose Sentinel, a lightweight sentence-level compression
framework that reframes context filtering as an attention-based understanding
task. Rather than training a compression model, Sentinel probes decoder
attention from an off-the-shelf 0.5B proxy LLM using a lightweight classifier
to identify sentence relevance. Empirically, we find that query-context
relevance estimation is consistent across model scales, with 0.5B proxies
closely matching the behaviors of larger models. On the LongBench benchmark,
Sentinel achieves up to 5$\times$ compression while matching the QA performance
of 7B-scale compression systems. Our results suggest that probing native
attention signals enables fast, effective, and question-aware context
compression. Code available at: https://github.com/yzhangchuck/Sentinel.

---


### [The Arabic AI Fingerprint: Stylometric Analysis and Detection of Large Language Models Text](http://arxiv.org/abs/2505.23276v1)

Large Language Models (LLMs) have achieved unprecedented capabilities in
generating human-like text, posing subtle yet significant challenges for
information integrity across critical domains, including education, social
media, and academia, enabling sophisticated misinformation campaigns,
compromising healthcare guidance, and facilitating targeted propaganda. This
challenge becomes severe, particularly in under-explored and low-resource
languages like Arabic. This paper presents a comprehensive investigation of
Arabic machine-generated text, examining multiple generation strategies
(generation from the title only, content-aware generation, and text refinement)
across diverse model architectures (ALLaM, Jais, Llama, and GPT-4) in academic,
and social media domains. Our stylometric analysis reveals distinctive
linguistic patterns differentiating human-written from machine-generated Arabic
text across these varied contexts. Despite their human-like qualities, we
demonstrate that LLMs produce detectable signatures in their Arabic outputs,
with domain-specific characteristics that vary significantly between different
contexts. Based on these insights, we developed BERT-based detection models
that achieved exceptional performance in formal contexts (up to 99.9\%
F1-score) with strong precision across model architectures. Our cross-domain
analysis confirms generalization challenges previously reported in the
literature. To the best of our knowledge, this work represents the most
comprehensive investigation of Arabic machine-generated text to date, uniquely
combining multiple prompt generation methods, diverse model architectures, and
in-depth stylometric analysis across varied textual domains, establishing a
foundation for developing robust, linguistically-informed detection systems
essential for preserving information integrity in Arabic-language contexts.

---


### [Unsupervised Transcript-assisted Video Summarization and Highlight Detection](http://arxiv.org/abs/2505.23268v1)

Video consumption is a key part of daily life, but watching entire videos can
be tedious. To address this, researchers have explored video summarization and
highlight detection to identify key video segments. While some works combine
video frames and transcripts, and others tackle video summarization and
highlight detection using Reinforcement Learning (RL), no existing work, to the
best of our knowledge, integrates both modalities within an RL framework. In
this paper, we propose a multimodal pipeline that leverages video frames and
their corresponding transcripts to generate a more condensed version of the
video and detect highlights using a modality fusion mechanism. The pipeline is
trained within an RL framework, which rewards the model for generating diverse
and representative summaries while ensuring the inclusion of video segments
with meaningful transcript content. The unsupervised nature of the training
allows for learning from large-scale unannotated datasets, overcoming the
challenge posed by the limited size of existing annotated datasets. Our
experiments show that using the transcript in video summarization and highlight
detection achieves superior results compared to relying solely on the visual
content of the video.

---


### [Disrupting Vision-Language Model-Driven Navigation Services via Adversarial Object Fusion](http://arxiv.org/abs/2505.23266v1)

We present Adversarial Object Fusion (AdvOF), a novel attack framework
targeting vision-and-language navigation (VLN) agents in service-oriented
environments by generating adversarial 3D objects. While foundational models
like Large Language Models (LLMs) and Vision Language Models (VLMs) have
enhanced service-oriented navigation systems through improved perception and
decision-making, their integration introduces vulnerabilities in
mission-critical service workflows. Existing adversarial attacks fail to
address service computing contexts, where reliability and quality-of-service
(QoS) are paramount. We utilize AdvOF to investigate and explore the impact of
adversarial environments on the VLM-based perception module of VLN agents. In
particular, AdvOF first precisely aggregates and aligns the victim object
positions in both 2D and 3D space, defining and rendering adversarial objects.
Then, we collaboratively optimize the adversarial object with regularization
between the adversarial and victim object across physical properties and VLM
perceptions. Through assigning importance weights to varying views, the
optimization is processed stably and multi-viewedly by iterative fusions from
local updates and justifications. Our extensive evaluations demonstrate AdvOF
can effectively degrade agent performance under adversarial conditions while
maintaining minimal interference with normal navigation tasks. This work
advances the understanding of service security in VLM-powered navigation
systems, providing computational foundations for robust service composition in
physical-world deployments.

---


### [Accelerating RLHF Training with Reward Variance Increase](http://arxiv.org/abs/2505.23247v1)

Reinforcement learning from human feedback (RLHF) is an essential technique
for ensuring that large language models (LLMs) are aligned with human values
and preferences during the post-training phase. As an effective RLHF approach,
group relative policy optimization (GRPO) has demonstrated success in many
LLM-based applications. However, efficient GRPO-based RLHF training remains a
challenge. Recent studies reveal that a higher reward variance of the initial
policy model leads to faster RLHF training. Inspired by this finding, we
propose a practical reward adjustment model to accelerate RLHF training by
provably increasing the reward variance and preserving the relative preferences
and reward expectation. Our reward adjustment method inherently poses a
nonconvex optimization problem, which is NP-hard to solve in general. To
overcome the computational challenges, we design a novel $O(n \log n)$
algorithm to find a global solution of the nonconvex reward adjustment model by
explicitly characterizing the extreme points of the feasible set. As an
important application, we naturally integrate this reward adjustment model into
the GRPO algorithm, leading to a more efficient GRPO with reward variance
increase (GRPOVI) algorithm for RLHF training. As an interesting byproduct, we
provide an indirect explanation for the empirical effectiveness of GRPO with
rule-based reward for RLHF training, as demonstrated in DeepSeek-R1. Experiment
results demonstrate that the GRPOVI algorithm can significantly improve the
RLHF training efficiency compared to the original GRPO algorithm.

---


### [VERINA: Benchmarking Verifiable Code Generation](http://arxiv.org/abs/2505.23135v1)

Large language models (LLMs) are increasingly integrated in software
development, but ensuring correctness in LLM-generated code remains challenging
and often requires costly manual review. Verifiable code generation -- jointly
generating code, specifications, and proofs of code-specification alignment --
offers a promising path to address this limitation and further unleash LLMs'
benefits in coding. Yet, there exists a significant gap in evaluation: current
benchmarks often lack support for end-to-end verifiable code generation. In
this paper, we introduce Verina (Verifiable Code Generation Arena), a
high-quality benchmark enabling a comprehensive and modular evaluation of code,
specification, and proof generation as well as their compositions. Verina
consists of 189 manually curated coding tasks in Lean, with detailed problem
descriptions, reference implementations, formal specifications, and extensive
test suites. Our extensive evaluation of state-of-the-art LLMs reveals
significant challenges in verifiable code generation, especially in proof
generation, underscoring the need for improving LLM-based theorem provers in
verification domains. The best model, OpenAI o4-mini, generates only 61.4%
correct code, 51.0% sound and complete specifications, and 3.6% successful
proofs, with one trial per task. We hope Verina will catalyze progress in
verifiable code generation by providing a rigorous and comprehensive benchmark.
We release our dataset on https://huggingface.co/datasets/sunblaze-ucb/verina
and our evaluation code on https://github.com/sunblaze-ucb/verina.

---


### [Decom-Renorm-Merge: Model Merging on the Right Space Improves Multitasking](http://arxiv.org/abs/2505.23117v1)

In the era of large-scale training, model merging has evolved into a tool for
creating multitasking models efficiently. It enables the knowledge of models to
be fused, without the need for heavy computation as required in traditional
multitask learning. Existing merging methods often assume that entries at
identical positions in weight matrices serve the same function, enabling
straightforward entry-wise comparison and merging. However, this assumption
overlooks the complexity of finetuned neural networks, where neurons may
develop distinct feature compositions, making direct entry-wise merging
problematic. We present Decom-Renorm-Merge (DRM), a simple yet effective
approach that leverages Singular Value Decomposition to decompose and
coordinate weight matrices into an aligned joint space, where entry-wise
merging becomes possible. We showcase the effectiveness of DRM across various
settings ranging from smaller encoder-based such as ViT and DeBERTa,
encoder-decoder-based such as T5, and larger decoder-based such as Llama3.1-8B.
Our experimental results show that DRM outperforms several state-of-the-art
merging techniques across full finetuning and low-rank adaptation settings.
Moreover, our analysis reveals renormalization as the crucial component for
creating a robust and even joint space for merging, significantly contributing
to the method's performance.

---


### [Infi-MMR: Curriculum-based Unlocking Multimodal Reasoning via Phased Reinforcement Learning in Multimodal Small Language Models](http://arxiv.org/abs/2505.23091v1)

Recent advancements in large language models (LLMs) have demonstrated
substantial progress in reasoning capabilities, such as DeepSeek-R1, which
leverages rule-based reinforcement learning to enhance logical reasoning
significantly. However, extending these achievements to multimodal large
language models (MLLMs) presents critical challenges, which are frequently more
pronounced for Multimodal Small Language Models (MSLMs) given their typically
weaker foundational reasoning abilities: (1) the scarcity of high-quality
multimodal reasoning datasets, (2) the degradation of reasoning capabilities
due to the integration of visual processing, and (3) the risk that direct
application of reinforcement learning may produce complex yet incorrect
reasoning processes. To address these challenges, we design a novel framework
Infi-MMR to systematically unlock the reasoning potential of MSLMs through a
curriculum of three carefully structured phases and propose our multimodal
reasoning model Infi-MMR-3B. The first phase, Foundational Reasoning
Activation, leverages high-quality textual reasoning datasets to activate and
strengthen the model's logical reasoning capabilities. The second phase,
Cross-Modal Reasoning Adaptation, utilizes caption-augmented multimodal data to
facilitate the progressive transfer of reasoning skills to multimodal contexts.
The third phase, Multimodal Reasoning Enhancement, employs curated,
caption-free multimodal data to mitigate linguistic biases and promote robust
cross-modal reasoning. Infi-MMR-3B achieves both state-of-the-art multimodal
math reasoning ability (43.68% on MathVerse testmini, 27.04% on MathVision
test, and 21.33% on OlympiadBench) and general reasoning ability (67.2% on
MathVista testmini).

---


### [Second Opinion Matters: Towards Adaptive Clinical AI via the Consensus of Expert Model Ensemble](http://arxiv.org/abs/2505.23075v1)

Despite the growing clinical adoption of large language models (LLMs),
current approaches heavily rely on single model architectures. To overcome
risks of obsolescence and rigid dependence on single model systems, we present
a novel framework, termed the Consensus Mechanism. Mimicking clinical triage
and multidisciplinary clinical decision-making, the Consensus Mechanism
implements an ensemble of specialized medical expert agents enabling improved
clinical decision making while maintaining robust adaptability. This
architecture enables the Consensus Mechanism to be optimized for cost, latency,
or performance, purely based on its interior model configuration.
  To rigorously evaluate the Consensus Mechanism, we employed three medical
evaluation benchmarks: MedMCQA, MedQA, and MedXpertQA Text, and the
differential diagnosis dataset, DDX+. On MedXpertQA, the Consensus Mechanism
achieved an accuracy of 61.0% compared to 53.5% and 45.9% for OpenAI's O3 and
Google's Gemini 2.5 Pro. Improvement was consistent across benchmarks with an
increase in accuracy on MedQA
($\Delta\mathrm{Accuracy}_{\mathrm{consensus\text{-}O3}} = 3.4\%$) and MedMCQA
($\Delta\mathrm{Accuracy}_{\mathrm{consensus\text{-}O3}} = 9.1\%$). These
accuracy gains extended to differential diagnosis generation, where our system
demonstrated improved recall and precision (F1$_\mathrm{consensus}$ = 0.326 vs.
F1$_{\mathrm{O3\text{-}high}}$ = 0.2886) and a higher top-1 accuracy for DDX
(Top1$_\mathrm{consensus}$ = 52.0% vs. Top1$_{\mathrm{O3\text{-}high}}$ =
45.2%).

---


### [From Token to Action: State Machine Reasoning to Mitigate Overthinking in Information Retrieval](http://arxiv.org/abs/2505.23059v1)

Chain-of-Thought (CoT) prompting enables complex reasoning in large language
models (LLMs), including applications in information retrieval (IR). However,
it often leads to overthinking, where models produce excessively long and
semantically redundant traces with little or no benefit. We identify two key
challenges in IR: redundant trajectories that revisit similar states and
misguided reasoning that diverges from user intent. To address these, we
propose State Machine Reasoning (SMR), a transition-based reasoning framework
composed of discrete actions (Refine, Rerank, Stop) that support early stopping
and fine-grained control. Experiments on the BEIR and BRIGHT benchmarks show
that SMR improves retrieval performance (nDCG@10) by 3.4% while reducing token
usage by 74.4%. It generalizes across LLMs and retrievers without requiring
task-specific tuning, offering a practical alternative to conventional CoT
reasoning. The code and details are available at https://github.com/ldilab/SMR.

---


### [AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic Large Language Models](http://arxiv.org/abs/2505.23020v1)

The acquisition of agentic capabilities has transformed LLMs from "knowledge
providers" to "action executors", a trend that while expanding LLMs' capability
boundaries, significantly increases their susceptibility to malicious use.
Previous work has shown that current LLM-based agents execute numerous
malicious tasks even without being attacked, indicating a deficiency in agentic
use safety alignment during the post-training phase. To address this gap, we
propose AgentAlign, a novel framework that leverages abstract behavior chains
as a medium for safety alignment data synthesis. By instantiating these
behavior chains in simulated environments with diverse tool instances, our
framework enables the generation of highly authentic and executable
instructions while capturing complex multi-step dynamics. The framework further
ensures model utility by proportionally synthesizing benign instructions
through non-malicious interpretations of behavior chains, precisely calibrating
the boundary between helpfulness and harmlessness. Evaluation results on
AgentHarm demonstrate that fine-tuning three families of open-source models
using our method substantially improves their safety (35.8% to 79.5%
improvement) while minimally impacting or even positively enhancing their
helpfulness, outperforming various prompting methods. The dataset and code have
both been open-sourced.

---


### [Synthetic Document Question Answering in Hungarian](http://arxiv.org/abs/2505.23008v1)

Modern VLMs have achieved near-saturation accuracy in English document visual
question-answering (VQA). However, this task remains challenging in lower
resource languages due to a dearth of suitable training and evaluation data. In
this paper we present scalable methods for curating such datasets by focusing
on Hungarian, approximately the 17th highest resource language on the internet.
Specifically, we present HuDocVQA and HuDocVQA-manual, document VQA datasets
that modern VLMs significantly underperform on compared to English DocVQA.
HuDocVQA-manual is a small manually curated dataset based on Hungarian
documents from Common Crawl, while HuDocVQA is a larger synthetically generated
VQA data set from the same source. We apply multiple rounds of quality
filtering and deduplication to HuDocVQA in order to match human-level quality
in this dataset. We also present HuCCPDF, a dataset of 117k pages from
Hungarian Common Crawl PDFs along with their transcriptions, which can be used
for training a model for Hungarian OCR. To validate the quality of our
datasets, we show how finetuning on a mixture of these datasets can improve
accuracy on HuDocVQA for Llama 3.2 11B Instruct by +7.2%. Our datasets and code
will be released to the public to foster further research in multilingual
DocVQA.

---


### [Model-Preserving Adaptive Rounding](http://arxiv.org/abs/2505.22988v1)

The main goal of post-training quantization (PTQ) is to produced a compressed
model whose output distribution is as close to the original model's as
possible. To do this tractably, almost all LLM PTQ algorithms quantize linear
layers by independently minimizing the immediate activation error. However,
this localized objective ignores the effect of subsequent layers, so reducing
it does not necessarily give a closer model. In this work, we introduce Yet
Another Quantization Algorithm (YAQA), an adaptive rounding algorithm that uses
Kronecker-factored approximations of each linear layer's Hessian with respect
to the \textit{full model} KL divergence. YAQA consists of two components:
Kronecker-factored sketches of the full layerwise Hessian that can be tractably
computed for hundred-billion parameter LLMs, and a quantizer-independent
rounding algorithm that uses these sketches and comes with theoretical
guarantees. Across a wide range of models and quantizers, YAQA empirically
reduces the KL divergence to the original model by $\approx 30\%$ while
achieving state of the art performance on downstream tasks.

---


### [MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence](http://arxiv.org/abs/2505.23764v1)

Spatial intelligence is essential for multimodal large language models
(MLLMs) operating in the complex physical world. Existing benchmarks, however,
probe only single-image relations and thus fail to assess the multi-image
spatial reasoning that real-world deployments demand. We introduce MMSI-Bench,
a VQA benchmark dedicated to multi-image spatial intelligence. Six 3D-vision
researchers spent more than 300 hours meticulously crafting 1,000 challenging,
unambiguous multiple-choice questions from over 120,000 images, each paired
with carefully designed distractors and a step-by-step reasoning process. We
conduct extensive experiments and thoroughly evaluate 34 open-source and
proprietary MLLMs, observing a wide gap: the strongest open-source model
attains roughly 30% accuracy and OpenAI's o3 reasoning model reaches 40%, while
humans score 97%. These results underscore the challenging nature of MMSI-Bench
and the substantial headroom for future research. Leveraging the annotated
reasoning processes, we also provide an automated error analysis pipeline that
diagnoses four dominant failure modes, including (1) grounding errors, (2)
overlap-matching and scene-reconstruction errors, (3) situation-transformation
reasoning errors, and (4) spatial-logic errors, offering valuable insights for
advancing multi-image spatial intelligence. Project page:
https://runsenxu.com/projects/MMSI_Bench .

---


### [Label-Guided In-Context Learning for Named Entity Recognition](http://arxiv.org/abs/2505.23722v1)

In-context learning (ICL) enables large language models (LLMs) to perform new
tasks using only a few demonstrations. In Named Entity Recognition (NER),
demonstrations are typically selected based on semantic similarity to the test
instance, ignoring training labels and resulting in suboptimal performance. We
introduce DEER, a new method that leverages training labels through token-level
statistics to improve ICL performance. DEER first enhances example selection
with a label-guided, token-based retriever that prioritizes tokens most
informative for entity recognition. It then prompts the LLM to revisit
error-prone tokens, which are also identified using label statistics, and make
targeted corrections. Evaluated on five NER datasets using four different LLMs,
DEER consistently outperforms existing ICL methods and approaches the
performance of supervised fine-tuning. Further analysis shows its effectiveness
on both seen and unseen entities and its robustness in low-resource settings.

---


### [Don't Take the Premise for Granted: Evaluating the Premise Critique Ability of Large Language Models](http://arxiv.org/abs/2505.23715v1)

Large language models (LLMs) have witnessed rapid advancements, demonstrating
remarkable capabilities. However, a notable vulnerability persists: LLMs often
uncritically accept flawed or contradictory premises, leading to inefficient
reasoning and unreliable outputs. This emphasizes the significance of
possessing the \textbf{Premise Critique Ability} for LLMs, defined as the
capacity to proactively identify and articulate errors in input premises. Most
existing studies assess LLMs' reasoning ability in ideal settings, largely
ignoring their vulnerabilities when faced with flawed premises. Thus, we
introduce the \textbf{Premise Critique Bench (PCBench)}, designed by
incorporating four error types across three difficulty levels, paired with
multi-faceted evaluation metrics. We conducted systematic evaluations of 15
representative LLMs. Our findings reveal: (1) Most models rely heavily on
explicit prompts to detect errors, with limited autonomous critique; (2)
Premise critique ability depends on question difficulty and error type, with
direct contradictions being easier to detect than complex or procedural errors;
(3) Reasoning ability does not consistently correlate with the premise critique
ability; (4) Flawed premises trigger overthinking in reasoning models, markedly
lengthening responses due to repeated attempts at resolving conflicts. These
insights underscore the urgent need to enhance LLMs' proactive evaluation of
input validity, positioning premise critique as a foundational capability for
developing reliable, human-centric systems. The code is available at
https://github.com/MLGroupJLU/Premise_Critique.

---


### [Table-R1: Inference-Time Scaling for Table Reasoning](http://arxiv.org/abs/2505.23621v1)

In this work, we present the first study to explore inference-time scaling on
table reasoning tasks. We develop and evaluate two post-training strategies to
enable inference-time scaling: distillation from frontier model reasoning
traces and reinforcement learning with verifiable rewards (RLVR). For
distillation, we introduce a large-scale dataset of reasoning traces generated
by DeepSeek-R1, which we use to fine-tune LLMs into the Table-R1-SFT model. For
RLVR, we propose task-specific verifiable reward functions and apply the GRPO
algorithm to obtain the Table-R1-Zero model. We evaluate our Table-R1-series
models across diverse table reasoning tasks, including short-form QA, fact
verification, and free-form QA. Notably, the Table-R1-Zero model matches or
exceeds the performance of GPT-4.1 and DeepSeek-R1, while using only a
7B-parameter LLM. It also demonstrates strong generalization to out-of-domain
datasets. Extensive ablation and qualitative analyses reveal the benefits of
instruction tuning, model architecture choices, and cross-task generalization,
as well as emergence of essential table reasoning skills during RL training.

---


### [Evaluating AI capabilities in detecting conspiracy theories on YouTube](http://arxiv.org/abs/2505.23570v1)

As a leading online platform with a vast global audience, YouTube's extensive
reach also makes it susceptible to hosting harmful content, including
disinformation and conspiracy theories. This study explores the use of
open-weight Large Language Models (LLMs), both text-only and multimodal, for
identifying conspiracy theory videos shared on YouTube. Leveraging a labeled
dataset of thousands of videos, we evaluate a variety of LLMs in a zero-shot
setting and compare their performance to a fine-tuned RoBERTa baseline. Results
show that text-based LLMs achieve high recall but lower precision, leading to
increased false positives. Multimodal models lag behind their text-only
counterparts, indicating limited benefits from visual data integration. To
assess real-world applicability, we evaluate the most accurate models on an
unlabeled dataset, finding that RoBERTa achieves performance close to LLMs with
a larger number of parameters. Our work highlights the strengths and
limitations of current LLM-based approaches for online harmful content
detection, emphasizing the need for more precise and robust systems.

---


### [Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](http://arxiv.org/abs/2505.23540v1)

Recent advances in preference optimization have demonstrated significant
potential for improving mathematical reasoning capabilities in large language
models (LLMs). While current approaches leverage high-quality pairwise
preference data through outcome-based criteria like answer correctness or
consistency, they fundamentally neglect the internal logical coherence of
responses. To overcome this, we propose Probability-Consistent Preference
Optimization (PCPO), a novel framework that establishes dual quantitative
metrics for preference selection: (1) surface-level answer correctness and (2)
intrinsic token-level probability consistency across responses. Extensive
experiments show that our PCPO consistently outperforms existing outcome-only
criterion approaches across a diverse range of LLMs and benchmarks. Our code is
publicly available at https://github.com/YunqiaoYang/PCPO.

---


### [Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](http://arxiv.org/abs/2505.23495v1)

Knowledge Graph Question Answering (KGQA) systems rely on high-quality
benchmarks to evaluate complex multi-hop reasoning. However, despite their
widespread use, popular datasets such as WebQSP and CWQ suffer from critical
quality issues, including inaccurate or incomplete ground-truth annotations,
poorly constructed questions that are ambiguous, trivial, or unanswerable, and
outdated or inconsistent knowledge. Through a manual audit of 16 popular KGQA
datasets, including WebQSP and CWQ, we find that the average factual
correctness rate is only 57 %. To address these issues, we introduce KGQAGen,
an LLM-in-the-loop framework that systematically resolves these pitfalls.
KGQAGen combines structured knowledge grounding, LLM-guided generation, and
symbolic verification to produce challenging and verifiable QA instances. Using
KGQAGen, we construct KGQAGen-10k, a ten-thousand scale benchmark grounded in
Wikidata, and evaluate a diverse set of KG-RAG models. Experimental results
demonstrate that even state-of-the-art systems struggle on this benchmark,
highlighting its ability to expose limitations of existing models. Our findings
advocate for more rigorous benchmark construction and position KGQAGen as a
scalable framework for advancing KGQA evaluation.

---


### [Revisiting Overthinking in Long Chain-of-Thought from the Perspective of Self-Doubt](http://arxiv.org/abs/2505.23480v1)

Reasoning Large Language Models (RLLMs) have demonstrated impressive
performance on complex tasks, largely due to the adoption of Long
Chain-of-Thought (Long CoT) reasoning. However, they often exhibit overthinking
-- performing unnecessary reasoning steps even after arriving at the correct
answer. Prior work has largely focused on qualitative analyses of overthinking
through sample-based observations of long CoTs. In contrast, we present a
quantitative analysis of overthinking from the perspective of self-doubt,
characterized by excessive token usage devoted to re-verifying already-correct
answer. We find that self-doubt significantly contributes to overthinking. In
response, we introduce a simple and effective prompting method to reduce the
model's over-reliance on input questions, thereby avoiding self-doubt.
Specifically, we first prompt the model to question the validity of the input
question, and then respond concisely based on the outcome of that evaluation.
Experiments on three mathematical reasoning tasks and four datasets with
missing premises demonstrate that our method substantially reduces answer
length and yields significant improvements across nearly all datasets upon 4
widely-used RLLMs. Further analysis demonstrates that our method effectively
minimizes the number of reasoning steps and reduces self-doubt.

---


### [From Parameters to Prompts: Understanding and Mitigating the Factuality Gap between Fine-Tuned LLMs](http://arxiv.org/abs/2505.23410v1)

Factual knowledge extraction aims to explicitly extract knowledge
parameterized in pre-trained language models for application in downstream
tasks. While prior work has been investigating the impact of supervised
fine-tuning data on the factuality of large language models (LLMs), its
mechanism remains poorly understood. We revisit this impact through systematic
experiments, with a particular focus on the factuality gap that arises when
fine-tuning on known versus unknown knowledge. Our findings show that this gap
can be mitigated at the inference stage, either under out-of-distribution (OOD)
settings or by using appropriate in-context learning (ICL) prompts (i.e.,
few-shot learning and Chain of Thought (CoT)). We prove this phenomenon
theoretically from the perspective of knowledge graphs, showing that the
test-time prompt may diminish or even overshadow the impact of fine-tuning data
and play a dominant role in knowledge extraction. Ultimately, our results shed
light on the interaction between finetuning data and test-time prompt,
demonstrating that ICL can effectively compensate for shortcomings in
fine-tuning data, and highlighting the need to reconsider the use of ICL
prompting as a means to evaluate the effectiveness of fine-tuning data
selection methods.

---


### [Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models](http://arxiv.org/abs/2505.23404v1)

Adversarial attacks on Large Language Models (LLMs) via jailbreaking
techniques-methods that circumvent their built-in safety and ethical
constraints-have emerged as a critical challenge in AI security. These attacks
compromise the reliability of LLMs by exploiting inherent weaknesses in their
comprehension capabilities. This paper investigates the efficacy of
jailbreaking strategies that are specifically adapted to the diverse levels of
understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking
Strategies Based on the Semantic Understanding Capabilities of Large Language
Models, a novel framework that classifies LLMs into Type I and Type II
categories according to their semantic comprehension abilities. For each
category, we design tailored jailbreaking strategies aimed at leveraging their
vulnerabilities to facilitate successful attacks. Extensive experiments
conducted on multiple LLMs demonstrate that our adaptive strategy markedly
improves the success rate of jailbreaking. Notably, our approach achieves an
exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)

---


### [Discriminative Policy Optimization for Token-Level Reward Models](http://arxiv.org/abs/2505.23363v1)

Process reward models (PRMs) provide more nuanced supervision compared to
outcome reward models (ORMs) for optimizing policy models, positioning them as
a promising approach to enhancing the capabilities of LLMs in complex reasoning
tasks. Recent efforts have advanced PRMs from step-level to token-level
granularity by integrating reward modeling into the training of generative
models, with reward scores derived from token generation probabilities.
However, the conflict between generative language modeling and reward modeling
may introduce instability and lead to inaccurate credit assignments. To address
this challenge, we revisit token-level reward assignment by decoupling reward
modeling from language generation and derive a token-level reward model through
the optimization of a discriminative policy, termed the Q-function Reward Model
(Q-RM). We theoretically demonstrate that Q-RM explicitly learns token-level
Q-functions from preference data without relying on fine-grained annotations.
In our experiments, Q-RM consistently outperforms all baseline methods across
various benchmarks. For example, when integrated into PPO/REINFORCE algorithms,
Q-RM enhances the average Pass@1 score by 5.85/4.70 points on mathematical
reasoning tasks compared to the ORM baseline, and by 4.56/5.73 points compared
to the token-level PRM counterpart. Moreover, reinforcement learning with Q-RM
significantly enhances training efficiency, achieving convergence 12 times
faster than ORM on GSM8K and 11 times faster than step-level PRM on MATH. Code
and data are available at https://github.com/homzer/Q-RM.

---


### [Proximalized Preference Optimization for Diverse Feedback Types: A Decomposed Perspective on DPO](http://arxiv.org/abs/2505.23316v1)

Direct alignment methods typically optimize large language models (LLMs) by
contrasting the likelihoods of preferred versus dispreferred responses. While
effective in steering LLMs to match relative preference, these methods are
frequently noted for decreasing the absolute likelihoods of example responses.
As a result, aligned models tend to generate outputs that deviate from the
expected patterns, exhibiting reward-hacking effect even without a reward
model. This undesired consequence exposes a fundamental limitation in
contrastive alignment, which we characterize as likelihood underdetermination.
In this work, we revisit direct preference optimization (DPO) -- the seminal
direct alignment method -- and demonstrate that its loss theoretically admits a
decomposed reformulation. The reformulated loss not only broadens applicability
to a wider range of feedback types, but also provides novel insights into the
underlying cause of likelihood underdetermination. Specifically, the standard
DPO implementation implicitly oversimplifies a regularizer in the reformulated
loss, and reinstating its complete version effectively resolves the
underdetermination issue. Leveraging these findings, we introduce PRoximalized
PReference Optimization (PRO), a unified method to align with diverse feeback
types, eliminating likelihood underdetermination through an efficient
approximation of the complete regularizer. Comprehensive experiments show the
superiority of PRO over existing methods in scenarios involving pairwise,
binary and scalar feedback.

---


### [Generalized Category Discovery in Event-Centric Contexts: Latent Pattern Mining with LLMs](http://arxiv.org/abs/2505.23304v1)

Generalized Category Discovery (GCD) aims to classify both known and novel
categories using partially labeled data that contains only known classes.
Despite achieving strong performance on existing benchmarks, current textual
GCD methods lack sufficient validation in realistic settings. We introduce
Event-Centric GCD (EC-GCD), characterized by long, complex narratives and
highly imbalanced class distributions, posing two main challenges: (1)
divergent clustering versus classification groupings caused by subjective
criteria, and (2) Unfair alignment for minority classes. To tackle these, we
propose PaMA, a framework leveraging LLMs to extract and refine event patterns
for improved cluster-class alignment. Additionally, a ranking-filtering-mining
pipeline ensures balanced representation of prototypes across imbalanced
categories. Evaluations on two EC-GCD benchmarks, including a newly constructed
Scam Report dataset, demonstrate that PaMA outperforms prior methods with up to
12.58% H-score gains, while maintaining strong generalization on base GCD
datasets.

---


### [Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs](http://arxiv.org/abs/2505.23299v1)

Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems
are increasingly deployed in industry applications, yet their reliability
remains hampered by challenges in detecting hallucinations. While supervised
state-of-the-art (SOTA) methods that leverage LLM hidden states -- such as
activation tracing and representation analysis -- show promise, their
dependence on extensively annotated datasets limits scalability in real-world
applications. This paper addresses the critical bottleneck of data annotation
by investigating the feasibility of reducing training data requirements for two
SOTA hallucination detection frameworks: Lookback Lens, which analyzes
attention head dynamics, and probing-based approaches, which decode internal
model representations. We propose a methodology combining efficient
classification algorithms with dimensionality reduction techniques to minimize
sample size demands while maintaining competitive performance. Evaluations on
standardized question-answering RAG benchmarks show that our approach achieves
performance comparable to strong proprietary LLM-based baselines with only 250
training samples. These results highlight the potential of lightweight,
data-efficient paradigms for industrial deployment, particularly in
annotation-constrained scenarios.

---


### [PBEBench: A Multi-Step Programming by Examples Reasoning Benchmark inspired by Historical Linguistics](http://arxiv.org/abs/2505.23126v1)

Recently, long chain of thought (LCoT), Large Language Models (LLMs), have
taken the machine learning world by storm with their breathtaking reasoning
capabilities. However, are the abstract reasoning abilities of these models
general enough for problems of practical importance? Unlike past work, which
has focused mainly on math, coding, and data wrangling, we focus on a
historical linguistics-inspired inductive reasoning problem, formulated as
Programming by Examples. We develop a fully automated pipeline for dynamically
generating a benchmark for this task with controllable difficulty in order to
tackle scalability and contamination issues to which many reasoning benchmarks
are subject. Using our pipeline, we generate a test set with nearly 1k
instances that is challenging for all state-of-the-art reasoning LLMs, with the
best model (Claude-3.7-Sonnet) achieving a mere 54% pass rate, demonstrating
that LCoT LLMs still struggle with a class or reasoning that is ubiquitous in
historical linguistics as well as many other domains.

---


### [Dataset Cartography for Large Language Model Alignment: Mapping and Diagnosing Preference Data](http://arxiv.org/abs/2505.23114v1)

Human preference data plays a critical role in aligning large language models
(LLMs) with human values. However, collecting such data is often expensive and
inefficient, posing a significant scalability challenge. To address this, we
introduce Alignment Data Map, a GPT-4o-assisted tool for analyzing and
diagnosing preference data. Using GPT-4o as a proxy for LLM alignment, we
compute alignment scores for LLM-generated responses to instructions from
existing preference datasets. These scores are then used to construct an
Alignment Data Map based on their mean and variance. Our experiments show that
using only 33 percent of the data, specifically samples in the high-mean,
low-variance region, achieves performance comparable to or better than using
the entire dataset. This finding suggests that the Alignment Data Map can
significantly improve data collection efficiency by identifying high-quality
samples for LLM alignment without requiring explicit annotations. Moreover, the
Alignment Data Map can diagnose existing preference datasets. Our analysis
shows that it effectively detects low-impact or potentially misannotated
samples. Source code is available online.

---


### [Generating Diverse Training Samples for Relation Extraction with Large Language Models](http://arxiv.org/abs/2505.23108v1)

Using Large Language Models (LLMs) to generate training data can potentially
be a preferable way to improve zero or few-shot NLP tasks. However, many
problems remain to be investigated for this direction. For the task of Relation
Extraction (RE), we find that samples generated by directly prompting LLMs may
easily have high structural similarities with each other. They tend to use a
limited variety of phrasing while expressing the relation between a pair of
entities. Therefore, in this paper, we study how to effectively improve the
diversity of the training samples generated with LLMs for RE, while also
maintaining their correctness. We first try to make the LLMs produce dissimilar
samples by directly giving instructions in In-Context Learning (ICL) prompts.
Then, we propose an approach to fine-tune LLMs for diversity training sample
generation through Direct Preference Optimization (DPO). Our experiments on
commonly used RE datasets show that both attempts can improve the quality of
the generated training data. We also find that comparing with directly
performing RE with an LLM, training a non-LLM RE model with its generated
samples may lead to better performance.

---


### [MAP: Revisiting Weight Decomposition for Low-Rank Adaptation](http://arxiv.org/abs/2505.23094v1)

The rapid development of large language models has revolutionized natural
language processing, but their fine-tuning remains computationally expensive,
hindering broad deployment. Parameter-efficient fine-tuning (PEFT) methods,
such as LoRA, have emerged as solutions. Recent work like DoRA attempts to
further decompose weight adaptation into direction and magnitude components.
However, existing formulations often define direction heuristically at the
column level, lacking a principled geometric foundation. In this paper, we
propose MAP, a novel framework that reformulates weight matrices as
high-dimensional vectors and decouples their adaptation into direction and
magnitude in a rigorous manner. MAP normalizes the pre-trained weights, learns
a directional update, and introduces two scalar coefficients to independently
scale the magnitude of the base and update vectors. This design enables more
interpretable and flexible adaptation, and can be seamlessly integrated into
existing PEFT methods. Extensive experiments show that MAP significantly
improves performance when coupling with existing methods, offering a simple yet
powerful enhancement to existing PEFT methods. Given the universality and
simplicity of MAP, we hope it can serve as a default setting for designing
future PEFT methods.

---


### [SNS-Bench-VL: Benchmarking Multimodal Large Language Models in Social Networking Services](http://arxiv.org/abs/2505.23065v1)

With the increasing integration of visual and textual content in Social
Networking Services (SNS), evaluating the multimodal capabilities of Large
Language Models (LLMs) is crucial for enhancing user experience, content
understanding, and platform intelligence. Existing benchmarks primarily focus
on text-centric tasks, lacking coverage of the multimodal contexts prevalent in
modern SNS ecosystems. In this paper, we introduce SNS-Bench-VL, a
comprehensive multimodal benchmark designed to assess the performance of
Vision-Language LLMs in real-world social media scenarios. SNS-Bench-VL
incorporates images and text across 8 multimodal tasks, including note
comprehension, user engagement analysis, information retrieval, and
personalized recommendation. It comprises 4,001 carefully curated multimodal
question-answer pairs, covering single-choice, multiple-choice, and open-ended
tasks. We evaluate over 25 state-of-the-art multimodal LLMs, analyzing their
performance across tasks. Our findings highlight persistent challenges in
multimodal social context comprehension. We hope SNS-Bench-VL will inspire
future research towards robust, context-aware, and human-aligned multimodal
intelligence for next-generation social networking services.

---


### [Query Routing for Retrieval-Augmented Language Models](http://arxiv.org/abs/2505.23052v1)

Retrieval-Augmented Generation (RAG) significantly improves the performance
of Large Language Models (LLMs) on knowledge-intensive tasks. However, varying
response quality across LLMs under RAG necessitates intelligent routing
mechanisms, which select the most suitable model for each query from multiple
retrieval-augmented LLMs via a dedicated router model. We observe that external
documents dynamically affect LLMs' ability to answer queries, while existing
routing methods, which rely on static parametric knowledge representations,
exhibit suboptimal performance in RAG scenarios. To address this, we formally
define the new retrieval-augmented LLM routing problem, incorporating the
influence of retrieved documents into the routing framework. We propose
RAGRouter, a RAG-aware routing design, which leverages document embeddings and
RAG capability embeddings with contrastive learning to capture knowledge
representation shifts and enable informed routing decisions. Extensive
experiments on diverse knowledge-intensive tasks and retrieval settings show
that RAGRouter outperforms the best individual LLM by 3.61% on average and
existing routing methods by 3.29%-9.33%. With an extended score-threshold-based
mechanism, it also achieves strong performance-efficiency trade-offs under
low-latency constraints.

---


### [DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration](http://arxiv.org/abs/2505.23049v1)

Pruning is a widely used technique to compress large language models (LLMs)
by removing unimportant weights, but it often suffers from significant
performance degradation - especially under semi-structured sparsity
constraints. Existing pruning methods primarily focus on estimating the
importance of individual weights, which limits their ability to preserve
critical capabilities of the model. In this work, we propose a new perspective:
rather than merely selecting which weights to prune, we first redistribute
parameter importance to make the model inherently more amenable to pruning. By
minimizing the information entropy of normalized importance scores, our
approach concentrates importance onto a smaller subset of weights, thereby
enhancing pruning robustness. We instantiate this idea through DenoiseRotator,
which applies learnable orthogonal transformations to the model's weight
matrices. Our method is model-agnostic and can be seamlessly integrated with
existing pruning techniques such as Magnitude, SparseGPT, and Wanda. Evaluated
on LLaMA3, Qwen2.5, and Mistral models under 50% unstructured and 2:4
semi-structured sparsity, DenoiseRotator consistently improves perplexity and
zero-shot accuracy. For instance, on LLaMA3-70B pruned with SparseGPT at 2:4
semi-structured sparsity, DenoiseRotator reduces the perplexity gap to the
dense model by 58%, narrowing the degradation from 8.1 to 3.4 points. Codes are
available at https://github.com/Axel-gu/DenoiseRotator.

---


### [EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models](http://arxiv.org/abs/2505.23038v1)

In-Context Learning (ICL) technique based on Large Language Models (LLMs) has
gained prominence in Named Entity Recognition (NER) tasks for its lower
computing resource consumption, less manual labeling overhead, and stronger
generalizability. Nevertheless, most ICL-based NER methods depend on
large-parameter LLMs: the open-source models demand substantial computational
resources for deployment and inference, while the closed-source ones incur high
API costs, raise data-privacy concerns, and hinder community collaboration. To
address this question, we propose an Ensemble Learning Method for Named Entity
Recognition (EL4NER), which aims at aggregating the ICL outputs of multiple
open-source, small-parameter LLMs to enhance overall performance in NER tasks
at less deployment and inference cost. Specifically, our method comprises three
key components. First, we design a task decomposition-based pipeline that
facilitates deep, multi-stage ensemble learning. Second, we introduce a novel
span-level sentence similarity algorithm to establish an ICL demonstration
retrieval mechanism better suited for NER tasks. Third, we incorporate a
self-validation mechanism to mitigate the noise introduced during the ensemble
process. We evaluated EL4NER on multiple widely adopted NER datasets from
diverse domains. Our experimental results indicate that EL4NER surpasses most
closed-source, large-parameter LLM-based methods at a lower parameter cost and
even attains state-of-the-art (SOTA) performance among ICL-based methods on
certain datasets. These results show the parameter efficiency of EL4NER and
underscore the feasibility of employing open-source, small-parameter LLMs
within the ICL paradigm for NER tasks.

---


### [Improving Multilingual Social Media Insights: Aspect-based Comment Analysis](http://arxiv.org/abs/2505.23037v1)

The inherent nature of social media posts, characterized by the freedom of
language use with a disjointed array of diverse opinions and topics, poses
significant challenges to downstream NLP tasks such as comment clustering,
comment summarization, and social media opinion analysis. To address this, we
propose a granular level of identifying and generating aspect terms from
individual comments to guide model attention. Specifically, we leverage
multilingual large language models with supervised fine-tuning for comment
aspect term generation (CAT-G), further aligning the model's predictions with
human expectations through DPO. We demonstrate the effectiveness of our method
in enhancing the comprehension of social media discourse on two NLP tasks.
Moreover, this paper contributes the first multilingual CAT-G test set on
English, Chinese, Malay, and Bahasa Indonesian. As LLM capabilities vary among
languages, this test set allows for a comparative analysis of performance
across languages with varying levels of LLM proficiency.

---


### [ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind](http://arxiv.org/abs/2505.22961v1)

Large language models (LLMs) have shown promising potential in persuasion,
but existing works on training LLM persuaders are still preliminary. Notably,
while humans are skilled in modeling their opponent's thoughts and opinions
proactively and dynamically, current LLMs struggle with such Theory of Mind
(ToM) reasoning, resulting in limited diversity and opponent awareness. To
address this limitation, we introduce Theory of Mind Augmented Persuader
(ToMAP), a novel approach for building more flexible persuader agents by
incorporating two theory of mind modules that enhance the persuader's awareness
and analysis of the opponent's mental state. Specifically, we begin by
prompting the persuader to consider possible objections to the target central
claim, and then use a text encoder paired with a trained MLP classifier to
predict the opponent's current stance on these counterclaims. Our carefully
designed reinforcement learning schema enables the persuader learns how to
analyze opponent-related information and utilize it to generate more effective
arguments. Experiments show that the ToMAP persuader, while containing only 3B
parameters, outperforms much larger baselines, like GPT-4o, with a relative
gain of 39.4% across multiple persuadee models and diverse corpora. Notably,
ToMAP exhibits complex reasoning chains and reduced repetition during training,
which leads to more diverse and effective arguments. The opponent-aware feature
of ToMAP also makes it suitable for long conversations and enables it to employ
more logical and opponent-aware strategies. These results underscore our
method's effectiveness and highlight its potential for developing more
persuasive language agents. Code is available at:
https://github.com/ulab-uiuc/ToMAP.

---


### [LLM-based HSE Compliance Assessment: Benchmark, Performance, and Advancements](http://arxiv.org/abs/2505.22959v1)

Health, Safety, and Environment (HSE) compliance assessment demands dynamic
real-time decision-making under complicated regulations and complex
human-machine-environment interactions. While large language models (LLMs) hold
significant potential for decision intelligence and contextual dialogue, their
capacity for domain-specific knowledge in HSE and structured legal reasoning
remains underexplored. We introduce HSE-Bench, the first benchmark dataset
designed to evaluate the HSE compliance assessment capabilities of LLM.
HSE-Bench comprises over 1,000 manually curated questions drawn from
regulations, court cases, safety exams, and fieldwork videos, and integrates a
reasoning flow based on Issue spotting, rule Recall, rule Application, and rule
Conclusion (IRAC) to assess the holistic reasoning pipeline. We conduct
extensive evaluations on different prompting strategies and more than 10 LLMs,
including foundation models, reasoning models and multimodal vision models. The
results show that, although current LLMs achieve good performance, their
capabilities largely rely on semantic matching rather than principled reasoning
grounded in the underlying HSE compliance context. Moreover, their native
reasoning trace lacks the systematic legal reasoning required for rigorous HSE
compliance assessment. To alleviate these, we propose a new prompting
technique, Reasoning of Expert (RoE), which guides LLMs to simulate the
reasoning process of different experts for compliance assessment and reach a
more accurate unified decision. We hope our study highlights reasoning gaps in
LLMs for HSE compliance and inspires further research on related tasks.

---


### [Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](http://arxiv.org/abs/2505.23766v1)

Recent advances in multimodal large language models (MLLMs) have demonstrated
remarkable capabilities in vision-language tasks, yet they often struggle with
vision-centric scenarios where precise visual focus is needed for accurate
reasoning. In this paper, we introduce Argus to address these limitations with
a new visual attention grounding mechanism. Our approach employs object-centric
grounding as visual chain-of-thought signals, enabling more effective
goal-conditioned visual attention during multimodal reasoning tasks.
Evaluations on diverse benchmarks demonstrate that Argus excels in both
multimodal reasoning tasks and referring object grounding tasks. Extensive
analysis further validates various design choices of Argus, and reveals the
effectiveness of explicit language-guided visual region-of-interest engagement
in MLLMs, highlighting the importance of advancing multimodal intelligence from
a visual-centric perspective. Project page: https://yunzeman.github.io/argus/

---


### [ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks](http://arxiv.org/abs/2505.23752v1)

Recent progress in large language models (LLMs) has enabled tool-augmented
agents capable of solving complex real-world tasks through step-by-step
reasoning. However, existing evaluations often focus on general-purpose or
multimodal scenarios, leaving a gap in domain-specific benchmarks that assess
tool-use capabilities in complex remote sensing use cases. We present ThinkGeo,
an agentic benchmark designed to evaluate LLM-driven agents on remote sensing
tasks via structured tool use and multi-step planning. Inspired by
tool-interaction paradigms, ThinkGeo includes human-curated queries spanning a
wide range of real-world applications such as urban planning, disaster
assessment and change analysis, environmental monitoring, transportation
analysis, aviation monitoring, recreational infrastructure, and industrial site
analysis. Each query is grounded in satellite or aerial imagery and requires
agents to reason through a diverse toolset. We implement a ReAct-style
interaction loop and evaluate both open and closed-source LLMs (e.g., GPT-4o,
Qwen2.5) on 436 structured agentic tasks. The benchmark reports both step-wise
execution metrics and final answer correctness. Our analysis reveals notable
disparities in tool accuracy and planning consistency across models. ThinkGeo
provides the first extensive testbed for evaluating how tool-enabled LLMs
handle spatial reasoning in remote sensing. Our code and dataset are publicly
available

---


### [How Animals Dance (When You're Not Looking)](http://arxiv.org/abs/2505.23738v1)

We present a keyframe-based framework for generating music-synchronized,
choreography aware animal dance videos. Starting from a few keyframes
representing distinct animal poses -- generated via text-to-image prompting or
GPT-4o -- we formulate dance synthesis as a graph optimization problem: find
the optimal keyframe structure that satisfies a specified choreography pattern
of beats, which can be automatically estimated from a reference dance video. We
also introduce an approach for mirrored pose image generation, essential for
capturing symmetry in dance. In-between frames are synthesized using an video
diffusion model. With as few as six input keyframes, our method can produce up
to 30 second dance videos across a wide range of animals and music tracks.

---


### [ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](http://arxiv.org/abs/2505.23734v1)

Feed-forward 3D Gaussian Splatting (3DGS) models have recently emerged as a
promising solution for novel view synthesis, enabling one-pass inference
without the need for per-scene 3DGS optimization. However, their scalability is
fundamentally constrained by the limited capacity of their encoders, leading to
degraded performance or excessive memory consumption as the number of input
views increases. In this work, we analyze feed-forward 3DGS frameworks through
the lens of the Information Bottleneck principle and introduce ZPressor, a
lightweight architecture-agnostic module that enables efficient compression of
multi-view inputs into a compact latent state $Z$ that retains essential scene
information while discarding redundancy. Concretely, ZPressor enables existing
feed-forward 3DGS models to scale to over 100 input views at 480P resolution on
an 80GB GPU, by partitioning the views into anchor and support sets and using
cross attention to compress the information from the support views into anchor
views, forming the compressed latent state $Z$. We show that integrating
ZPressor into several state-of-the-art feed-forward 3DGS models consistently
improves performance under moderate input views and enhances robustness under
dense view settings on two large-scale benchmarks DL3DV-10K and RealEstate10K.
The video results, code and trained models are available on our project page:
https://lhmd.top/zpressor.

---


### [DA-VPT: Semantic-Guided Visual Prompt Tuning for Vision Transformers](http://arxiv.org/abs/2505.23694v1)

Visual Prompt Tuning (VPT) has become a promising solution for
Parameter-Efficient Fine-Tuning (PEFT) approach for Vision Transformer (ViT)
models by partially fine-tuning learnable tokens while keeping most model
parameters frozen. Recent research has explored modifying the connection
structures of the prompts. However, the fundamental correlation and
distribution between the prompts and image tokens remain unexplored. In this
paper, we leverage metric learning techniques to investigate how the
distribution of prompts affects fine-tuning performance. Specifically, we
propose a novel framework, Distribution Aware Visual Prompt Tuning (DA-VPT), to
guide the distributions of the prompts by learning the distance metric from
their class-related semantic data. Our method demonstrates that the prompts can
serve as an effective bridge to share semantic information between image
patches and the class token. We extensively evaluated our approach on popular
benchmarks in both recognition and segmentation tasks. The results demonstrate
that our approach enables more effective and efficient fine-tuning of ViT
models by leveraging semantic information to guide the learning of the prompts,
leading to improved performance on various downstream vision tasks.

---


### [A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](http://arxiv.org/abs/2505.23601v1)

Endoscopic procedures are essential for diagnosing and treating internal
diseases, and multi-modal large language models (MLLMs) are increasingly
applied to assist in endoscopy analysis. However, current benchmarks are
limited, as they typically cover specific endoscopic scenarios and a small set
of clinical tasks, failing to capture the real-world diversity of endoscopic
scenarios and the full range of skills needed in clinical workflows. To address
these issues, we introduce EndoBench, the first comprehensive benchmark
specifically designed to assess MLLMs across the full spectrum of endoscopic
practice with multi-dimensional capacities. EndoBench encompasses 4 distinct
endoscopic scenarios, 12 specialized clinical tasks with 12 secondary subtasks,
and 5 levels of visual prompting granularities, resulting in 6,832 rigorously
validated VQA pairs from 21 diverse datasets. Our multi-dimensional evaluation
framework mirrors the clinical workflow--spanning anatomical recognition,
lesion analysis, spatial localization, and surgical operations--to holistically
gauge the perceptual and diagnostic abilities of MLLMs in realistic scenarios.
We benchmark 23 state-of-the-art models, including general-purpose,
medical-specialized, and proprietary MLLMs, and establish human clinician
performance as a reference standard. Our extensive experiments reveal: (1)
proprietary MLLMs outperform open-source and medical-specialized models
overall, but still trail human experts; (2) medical-domain supervised
fine-tuning substantially boosts task-specific accuracy; and (3) model
performance remains sensitive to prompt format and clinical task complexity.
EndoBench establishes a new standard for evaluating and advancing MLLMs in
endoscopy, highlighting both progress and persistent gaps between current
models and expert clinical reasoning. We publicly release our benchmark and
code.

---


### [Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition](http://arxiv.org/abs/2505.23566v1)

Handwritten Mathematical Expression Recognition (HMER) remains a persistent
challenge in Optical Character Recognition (OCR) due to the inherent freedom of
symbol layout and variability in handwriting styles. Prior methods have faced
performance bottlenecks, proposing isolated architectural modifications that
are difficult to integrate coherently into a unified framework. Meanwhile,
recent advances in pretrained vision-language models (VLMs) have demonstrated
strong cross-task generalization, offering a promising foundation for
developing unified solutions. In this paper, we introduce Uni-MuMER, which
fully fine-tunes a VLM for the HMER task without modifying its architecture,
effectively injecting domain-specific knowledge into a generalist framework.
Our method integrates three data-driven tasks: Tree-Aware Chain-of-Thought
(Tree-CoT) for structured spatial reasoning, Error-Driven Learning (EDL) for
reducing confusion among visually similar characters, and Symbol Counting (SC)
for improving recognition consistency in long expressions. Experiments on the
CROHME and HME100K datasets show that Uni-MuMER achieves new state-of-the-art
performance, surpassing the best lightweight specialized model SSAN by 16.31%
and the top-performing VLM Gemini2.5-flash by 24.42% in the zero-shot setting.
Our datasets, models, and code are open-sourced at:
https://github.com/BFlameSwift/Uni-MuMER

---


### [Qwen Look Again: Guiding Vision-Language Reasoning Models to Re-attention Visual Information](http://arxiv.org/abs/2505.23558v1)

Inference time scaling drives extended reasoning to enhance the performance
of Vision-Language Models (VLMs), thus forming powerful Vision-Language
Reasoning Models (VLRMs). However, long reasoning dilutes visual tokens,
causing visual information to receive less attention and may trigger
hallucinations. Although introducing text-only reflection processes shows
promise in language models, we demonstrate that it is insufficient to suppress
hallucinations in VLMs. To address this issue, we introduce Qwen-LookAgain
(Qwen-LA), a novel VLRM designed to mitigate hallucinations by incorporating a
vision-text reflection process that guides the model to re-attention visual
information during reasoning. We first propose a reinforcement learning method
Balanced Reflective Policy Optimization (BRPO), which guides the model to
decide when to generate vision-text reflection on its own and balance the
number and length of reflections. Then, we formally prove that VLRMs lose
attention to visual tokens as reasoning progresses, and demonstrate that
supplementing visual information during reflection enhances visual attention.
Therefore, during training and inference, Visual Token COPY and Visual Token
ROUTE are introduced to force the model to re-attention visual information at
the visual level, addressing the limitations of text-only reflection.
Experiments on multiple visual QA datasets and hallucination metrics indicate
that Qwen-LA achieves leading accuracy performance while reducing
hallucinations. Our code is available at:
https://github.com/Liar406/Look_Again.

---


### [OmniEarth-Bench: Towards Holistic Evaluation of Earth's Six Spheres and Cross-Spheres Interactions with Multimodal Observational Earth Data](http://arxiv.org/abs/2505.23522v1)

Existing benchmarks for Earth science multimodal learning exhibit critical
limitations in systematic coverage of geosystem components and cross-sphere
interactions, often constrained to isolated subsystems (only in
Human-activities sphere or atmosphere) with limited evaluation dimensions (less
than 16 tasks). To address these gaps, we introduce OmniEarth-Bench, the first
comprehensive multimodal benchmark spanning all six Earth science spheres
(atmosphere, lithosphere, Oceansphere, cryosphere, biosphere and
Human-activities sphere) and cross-spheres with one hundred expert-curated
evaluation dimensions. Leveraging observational data from satellite sensors and
in-situ measurements, OmniEarth-Bench integrates 29,779 annotations across four
tiers: perception, general reasoning, scientific knowledge reasoning and
chain-of-thought (CoT) reasoning. This involves the efforts of 2-5 experts per
sphere to establish authoritative evaluation dimensions and curate relevant
observational datasets, 40 crowd-sourcing annotators to assist experts for
annotations, and finally, OmniEarth-Bench is validated via hybrid expert-crowd
workflows to reduce label ambiguity. Experiments on 9 state-of-the-art MLLMs
reveal that even the most advanced models struggle with our benchmarks, where
none of them reach 35\% accuracy. Especially, in some cross-spheres tasks, the
performance of leading models like GPT-4o drops to 0.0\%. OmniEarth-Bench sets
a new standard for geosystem-aware AI, advancing both scientific discovery and
practical applications in environmental monitoring and disaster prediction. The
dataset, source code, and trained models were released.

---


### [VCapsBench: A Large-scale Fine-grained Benchmark for Video Caption Quality Evaluation](http://arxiv.org/abs/2505.23484v1)

Video captions play a crucial role in text-to-video generation tasks, as
their quality directly influences the semantic coherence and visual fidelity of
the generated videos. Although large vision-language models (VLMs) have
demonstrated significant potential in caption generation, existing benchmarks
inadequately address fine-grained evaluation, particularly in capturing
spatial-temporal details critical for video generation. To address this gap, we
introduce the Fine-grained Video Caption Evaluation Benchmark (VCapsBench), the
first large-scale fine-grained benchmark comprising 5,677 (5K+) videos and
109,796 (100K+) question-answer pairs. These QA-pairs are systematically
annotated across 21 fine-grained dimensions (e.g., camera movement, and shot
type) that are empirically proven critical for text-to-video generation. We
further introduce three metrics (Accuracy (AR), Inconsistency Rate (IR),
Coverage Rate (CR)), and an automated evaluation pipeline leveraging large
language model (LLM) to verify caption quality via contrastive QA-pairs
analysis. By providing actionable insights for caption optimization, our
benchmark can advance the development of robust text-to-video models. The
dataset and codes are available at website: https://github.com/GXYM/VCapsBench.

---


### [LAFR: Efficient Diffusion-based Blind Face Restoration via Latent Codebook Alignment Adapter](http://arxiv.org/abs/2505.23462v1)

Blind face restoration from low-quality (LQ) images is a challenging task
that requires not only high-fidelity image reconstruction but also the
preservation of facial identity. While diffusion models like Stable Diffusion
have shown promise in generating high-quality (HQ) images, their VAE modules
are typically trained only on HQ data, resulting in semantic misalignment when
encoding LQ inputs. This mismatch significantly weakens the effectiveness of LQ
conditions during the denoising process. Existing approaches often tackle this
issue by retraining the VAE encoder, which is computationally expensive and
memory-intensive. To address this limitation efficiently, we propose LAFR
(Latent Alignment for Face Restoration), a novel codebook-based latent space
adapter that aligns the latent distribution of LQ images with that of HQ
counterparts, enabling semantically consistent diffusion sampling without
altering the original VAE. To further enhance identity preservation, we
introduce a multi-level restoration loss that combines constraints from
identity embeddings and facial structural priors. Additionally, by leveraging
the inherent structural regularity of facial images, we show that lightweight
finetuning of diffusion prior on just 0.9% of FFHQ dataset is sufficient to
achieve results comparable to state-of-the-art methods, reduce training time by
70%. Extensive experiments on both synthetic and real-world face restoration
benchmarks demonstrate the effectiveness and efficiency of LAFR, achieving
high-quality, identity-preserving face reconstruction from severely degraded
inputs.

---


### [UniRL: Self-Improving Unified Multimodal Models via Supervised and Reinforcement Learning](http://arxiv.org/abs/2505.23380v1)

Unified multimodal large language models such as Show-o and Janus have
achieved strong performance across both generation and understanding tasks.
However, these models typically rely on large-scale datasets and require
substantial computation during the pretraining stage. In addition, several
post-training methods have been proposed, but they often depend on external
data or are limited to task-specific customization. In this work, we introduce
UniRL, a self-improving post-training approach. Our approach enables the model
to generate images from prompts and use them as training data in each
iteration, without relying on any external image data. Moreover, it enables the
two tasks to enhance each other: the generated images are used for
understanding, and the understanding results are used to supervise generation.
We explore supervised fine-tuning (SFT) and Group Relative Policy Optimization
(GRPO) to optimize the models. UniRL offers three key advantages: (1) it
requires no external image data, as all training samples are generated by the
model itself during training; (2) it not only improves individual task
performance, but also reduces the imbalance between generation and
understanding; and (3) it requires only several additional training steps
during the post-training stage. We evaluate UniRL on top of Show-o and Janus,
achieving a GenEval score of 0.77 for Show-o and 0.65 for Janus. Code and
models will be released in https://github.com/showlab/UniRL.

---


### [VideoReasonBench: Can MLLMs Perform Vision-Centric Complex Video Reasoning?](http://arxiv.org/abs/2505.23359v1)

Recent studies have shown that long chain-of-thought (CoT) reasoning can
significantly enhance the performance of large language models (LLMs) on
complex tasks. However, this benefit is yet to be demonstrated in the domain of
video understanding, since most existing benchmarks lack the reasoning depth
required to demonstrate the advantages of extended CoT chains. While recent
efforts have proposed benchmarks aimed at video reasoning, the tasks are often
knowledge-driven and do not rely heavily on visual content. To bridge this gap,
we introduce VideoReasonBench, a benchmark designed to evaluate vision-centric,
complex video reasoning. To ensure visual richness and high reasoning
complexity, each video in VideoReasonBench depicts a sequence of fine-grained
operations on a latent state that is only visible in part of the video. The
questions evaluate three escalating levels of video reasoning skills: recalling
observed visual information, inferring the content of latent states, and
predicting information beyond the video. Under such task setting, models have
to precisely recall multiple operations in the video, and perform step-by-step
reasoning to get correct final answers for these questions. Using
VideoReasonBench, we comprehensively evaluate 18 state-of-the-art multimodal
LLMs (MLLMs), finding that most perform poorly on complex video reasoning,
e.g., GPT-4o achieves only 6.9% accuracy, while the thinking-enhanced
Gemini-2.5-Pro significantly outperforms others with 56.0% accuracy. Our
investigations on "test-time scaling" further reveal that extended thinking
budget, while offering none or minimal benefits on existing video benchmarks,
is essential for improving the performance on VideoReasonBench.

---


### [Holistic Large-Scale Scene Reconstruction via Mixed Gaussian Splatting](http://arxiv.org/abs/2505.23280v1)

Recent advances in 3D Gaussian Splatting have shown remarkable potential for
novel view synthesis. However, most existing large-scale scene reconstruction
methods rely on the divide-and-conquer paradigm, which often leads to the loss
of global scene information and requires complex parameter tuning due to scene
partitioning and local optimization. To address these limitations, we propose
MixGS, a novel holistic optimization framework for large-scale 3D scene
reconstruction. MixGS models the entire scene holistically by integrating
camera pose and Gaussian attributes into a view-aware representation, which is
decoded into fine-detailed Gaussians. Furthermore, a novel mixing operation
combines decoded and original Gaussians to jointly preserve global coherence
and local fidelity. Extensive experiments on large-scale scenes demonstrate
that MixGS achieves state-of-the-art rendering quality and competitive speed,
while significantly reducing computational requirements, enabling large-scale
scene reconstruction training on a single 24GB VRAM GPU. The code will be
released at https://github.com/azhuantou/MixGS.

---


### [Image Aesthetic Reasoning: A New Benchmark for Medical Image Screening with MLLMs](http://arxiv.org/abs/2505.23265v1)

Multimodal Large Language Models (MLLMs) are of great application across many
domains, such as multimodal understanding and generation. With the development
of diffusion models (DM) and unified MLLMs, the performance of image generation
has been significantly improved, however, the study of image screening is rare
and its performance with MLLMs is unsatisfactory due to the lack of data and
the week image aesthetic reasoning ability in MLLMs. In this work, we propose a
complete solution to address these problems in terms of data and methodology.
For data, we collect a comprehensive medical image screening dataset with 1500+
samples, each sample consists of a medical image, four generated images, and a
multiple-choice answer. The dataset evaluates the aesthetic reasoning ability
under four aspects: \textit{(1) Appearance Deformation, (2) Principles of
Physical Lighting and Shadow, (3) Placement Layout, (4) Extension Rationality}.
For methodology, we utilize long chains of thought (CoT) and Group Relative
Policy Optimization with Dynamic Proportional Accuracy reward, called DPA-GRPO,
to enhance the image aesthetic reasoning ability of MLLMs. Our experimental
results reveal that even state-of-the-art closed-source MLLMs, such as GPT-4o
and Qwen-VL-Max, exhibit performance akin to random guessing in image aesthetic
reasoning. In contrast, by leveraging the reinforcement learning approach, we
are able to surpass the score of both large-scale models and leading
closed-source models using a much smaller model. We hope our attempt on medical
image screening will serve as a regular configuration in image aesthetic
reasoning in the future.

---


### [TrackVLA: Embodied Visual Tracking in the Wild](http://arxiv.org/abs/2505.23189v1)

Embodied visual tracking is a fundamental skill in Embodied AI, enabling an
agent to follow a specific target in dynamic environments using only egocentric
vision. This task is inherently challenging as it requires both accurate target
recognition and effective trajectory planning under conditions of severe
occlusion and high scene dynamics. Existing approaches typically address this
challenge through a modular separation of recognition and planning. In this
work, we propose TrackVLA, a Vision-Language-Action (VLA) model that learns the
synergy between object recognition and trajectory planning. Leveraging a shared
LLM backbone, we employ a language modeling head for recognition and an
anchor-based diffusion model for trajectory planning. To train TrackVLA, we
construct an Embodied Visual Tracking Benchmark (EVT-Bench) and collect diverse
difficulty levels of recognition samples, resulting in a dataset of 1.7 million
samples. Through extensive experiments in both synthetic and real-world
environments, TrackVLA demonstrates SOTA performance and strong
generalizability. It significantly outperforms existing methods on public
benchmarks in a zero-shot manner while remaining robust to high dynamics and
occlusion in real-world scenarios at 10 FPS inference speed. Our project page
is: https://pku-epic.github.io/TrackVLA-web.

---


### [DIP-R1: Deep Inspection and Perception with RL Looking Through and Understanding Complex Scenes](http://arxiv.org/abs/2505.23179v1)

Multimodal Large Language Models (MLLMs) have demonstrated significant visual
understanding capabilities, yet their fine-grained visual perception in complex
real-world scenarios, such as densely crowded public areas, remains limited.
Inspired by the recent success of reinforcement learning (RL) in both LLMs and
MLLMs, in this paper, we explore how RL can enhance visual perception ability
of MLLMs. Then we develop a novel RL-based framework, Deep Inspection and
Perception with RL (DIP-R1) designed to enhance the visual perception
capabilities of MLLMs, by comprehending complex scenes and looking through
visual instances closely. DIP-R1 guides MLLMs through detailed inspection of
visual scene via three simply designed rule-based reward modelings. First, we
adopt a standard reasoning reward encouraging the model to include three
step-by-step processes: 1) reasoning for understanding visual scenes, 2)
observing for looking through interested but ambiguous regions, and 3)
decision-making for predicting answer. Second, a variance-guided looking reward
is designed to examine uncertain regions for the second observing process. It
explicitly enables the model to inspect ambiguous areas, improving its ability
to mitigate perceptual uncertainties. Third, we model a weighted
precision-recall accuracy reward enhancing accurate decision-making. We explore
its effectiveness across diverse fine-grained object detection data consisting
of challenging real-world environments, such as densely crowded scenes. Built
upon existing MLLMs, DIP-R1 achieves consistent and significant improvement
across various in-domain and out-of-domain scenarios. It also outperforms
various existing baseline models and supervised fine-tuning methods. Our
findings highlight the substantial potential of integrating RL into MLLMs for
enhancing capabilities in complex real-world perception tasks.

---


### [Interpreting Chest X-rays Like a Radiologist: A Benchmark with Clinical Reasoning](http://arxiv.org/abs/2505.23143v1)

Artificial intelligence (AI)-based chest X-ray (CXR) interpretation
assistants have demonstrated significant progress and are increasingly being
applied in clinical settings. However, contemporary medical AI models often
adhere to a simplistic input-to-output paradigm, directly processing an image
and an instruction to generate a result, where the instructions may be integral
to the model's architecture. This approach overlooks the modeling of the
inherent diagnostic reasoning in chest X-ray interpretation. Such reasoning is
typically sequential, where each interpretive stage considers the images, the
current task, and the contextual information from previous stages. This
oversight leads to several shortcomings, including misalignment with clinical
scenarios, contextless reasoning, and untraceable errors. To fill this gap, we
construct CXRTrek, a new multi-stage visual question answering (VQA) dataset
for CXR interpretation. The dataset is designed to explicitly simulate the
diagnostic reasoning process employed by radiologists in real-world clinical
settings for the first time. CXRTrek covers 8 sequential diagnostic stages,
comprising 428,966 samples and over 11 million question-answer (Q&A) pairs,
with an average of 26.29 Q&A pairs per sample. Building on the CXRTrek dataset,
we propose a new vision-language large model (VLLM), CXRTrekNet, specifically
designed to incorporate the clinical reasoning flow into the VLLM framework.
CXRTrekNet effectively models the dependencies between diagnostic stages and
captures reasoning patterns within the radiological context. Trained on our
dataset, the model consistently outperforms existing medical VLLMs on the
CXRTrek benchmarks and demonstrates superior generalization across multiple
tasks on five diverse external datasets. The dataset and model can be found in
our repository (https://github.com/guanjinquan/CXRTrek).

---


### [Distortion of AI Alignment: Does Preference Optimization Optimize for Preferences?](http://arxiv.org/abs/2505.23749v1)

After pre-training, large language models are aligned with human preferences
based on pairwise comparisons. State-of-the-art alignment methods (such as
PPO-based RLHF and DPO) are built on the assumption of aligning with a single
preference model, despite being deployed in settings where users have diverse
preferences. As a result, it is not even clear that these alignment methods
produce models that satisfy users on average -- a minimal requirement for
pluralistic alignment. Drawing on social choice theory and modeling users'
comparisons through individual Bradley-Terry (BT) models, we introduce an
alignment method's distortion: the worst-case ratio between the optimal
achievable average utility, and the average utility of the learned policy.
  The notion of distortion helps draw sharp distinctions between alignment
methods: Nash Learning from Human Feedback achieves the minimax optimal
distortion of $(\frac{1}{2} + o(1)) \cdot \beta$ (for the BT temperature
$\beta$), robustly across utility distributions, distributions of comparison
pairs, and permissible KL divergences from the reference policy. RLHF and DPO,
by contrast, suffer $\geq (1 - o(1)) \cdot \beta$ distortion already without a
KL constraint, and $e^{\Omega(\beta)}$ or even unbounded distortion in the full
setting, depending on how comparison pairs are sampled.

---


### [MuLoCo: Muon is a practical inner optimizer for DiLoCo](http://arxiv.org/abs/2505.23725v1)

DiLoCo is a powerful framework for training large language models (LLMs)
under networking constraints with advantages for increasing parallelism and
accelerator utilization in data center settings. Despite significantly reducing
communication frequency, however, DiLoCo's communication steps still involve
all-reducing a complete copy of the model's parameters. While existing works
have explored ways to reduce communication in DiLoCo, the role of error
feedback accumulators and the effect of the inner-optimizer on compressibility
remain under-explored. In this work, we investigate the effectiveness of
standard compression methods including Top-k sparsification and quantization
for reducing the communication overhead of DiLoCo when paired with two local
optimizers (AdamW and Muon). Our experiments pre-training decoder-only
transformer language models (LMs) reveal that leveraging Muon as the inner
optimizer for DiLoCo along with an error-feedback accumulator allows to
aggressively compress the communicated delta to 2-bits with next to no
performance degradation. Crucially, MuLoCo (Muon inner optimizer DiLoCo)
significantly outperforms DiLoCo while communicating 8X less and having
identical memory complexity.

---


### [MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment](http://arxiv.org/abs/2505.23634v1)

The model context protocol (MCP) has been widely adapted as an open standard
enabling the seamless integration of generative AI agents. However, recent work
has shown the MCP is susceptible to retrieval-based "falsely benign" attacks
(FBAs), allowing malicious system access and credential theft, but requiring
that users download compromised files directly to their systems. Herein, we
show that the threat model of MCP-based attacks is significantly broader than
previously thought, i.e., attackers need only post malicious content online to
deceive MCP agents into carrying out their attacks on unsuspecting victims'
systems.
  To improve alignment guardrails against such attacks, we introduce a new MCP
dataset of FBAs and (truly) benign samples to explore the effectiveness of
direct preference optimization (DPO) for the refusal training of large language
models (LLMs). While DPO improves model guardrails against such attacks, we
show that the efficacy of refusal learning varies drastically depending on the
model's original post-training alignment scheme--e.g., GRPO-based LLMs learn to
refuse extremely poorly. Thus, to further improve FBA refusals, we introduce
Retrieval Augmented Generation for Preference alignment (RAG-Pref), a novel
preference alignment strategy based on RAG. We show that RAG-Pref significantly
improves the ability of LLMs to refuse FBAs, particularly when combined with
DPO alignment, thus drastically improving guardrails against MCP-based attacks.

---


### [Adaptive Federated LoRA in Heterogeneous Wireless Networks with Independent Sampling](http://arxiv.org/abs/2505.23555v1)

Federated LoRA has emerged as a promising technique for efficiently
fine-tuning large language models (LLMs) on distributed devices by reducing the
number of trainable parameters. However, existing approaches often inadequately
overlook the theoretical and practical implications of system and data
heterogeneity, thereby failing to optimize the overall training efficiency,
particularly in terms of wall-clock time. In this paper, we propose an adaptive
federated LoRA strategy with independent client sampling to minimize the
convergence wall-clock time of federated fine-tuning under both computation and
communication heterogeneity. We first derive a new convergence bound for
federated LoRA with arbitrary and independent client sampling, notably without
requiring the stringent bounded gradient assumption. Then, we introduce an
adaptive bandwidth allocation scheme that accounts for heterogeneous client
resources and system bandwidth constraints. Based on the derived theory, we
formulate and solve a non-convex optimization problem to jointly determine the
LoRA sketching ratios and sampling probabilities, aiming to minimize wall-clock
convergence time. An efficient and low-complexity algorithm is developed to
approximate the solution. Finally, extensive experiments demonstrate that our
approach significantly reduces wall-clock training time compared to
state-of-the-art methods across various models and datasets.

---


### [AnchorAttention: Difference-Aware Sparse Attention with Stripe Granularity](http://arxiv.org/abs/2505.23520v1)

Large Language Models (LLMs) with extended context lengths face significant
computational challenges during the pre-filling phase, primarily due to the
quadratic complexity of self-attention. Existing methods typically employ
dynamic pattern matching and block-sparse low-level implementations. However,
their reliance on local information for pattern identification fails to capture
global contexts, and the coarse granularity of blocks leads to persistent
internal sparsity, resulting in suboptimal accuracy and efficiency. To address
these limitations, we propose \textbf{AnchorAttention}, a difference-aware,
dynamic sparse attention mechanism that efficiently identifies critical
attention regions at a finer stripe granularity while adapting to global
contextual information, achieving superior speed and accuracy. AnchorAttention
comprises three key components: (1) \textbf{Pattern-based Anchor Computation},
leveraging the commonalities present across all inputs to rapidly compute a set
of near-maximum scores as the anchor; (2) \textbf{Difference-aware Stripe
Sparsity Identification}, performing difference-aware comparisons with the
anchor to quickly obtain discrete coordinates of significant regions in a
stripe-like sparsity pattern; (3) \textbf{Fine-grained Sparse Computation},
replacing the traditional contiguous KV block loading approach with
simultaneous discrete KV position loading to maximize sparsity rates while
preserving full hardware computational potential. With its finer-grained
sparsity strategy, \textbf{AnchorAttention} achieves higher sparsity rates at
the same recall level, significantly reducing computation time. Compared to
previous state-of-the-art methods, at a text length of 128k, it achieves a
speedup of 1.44$\times$ while maintaining higher recall rates.

---


### [Diversity-Aware Policy Optimization for Large Language Model Reasoning](http://arxiv.org/abs/2505.23433v1)

The reasoning capabilities of large language models (LLMs) have advanced
rapidly, particularly following the release of DeepSeek R1, which has inspired
a surge of research into data quality and reinforcement learning (RL)
algorithms. Despite the pivotal role diversity plays in RL, its influence on
LLM reasoning remains largely underexplored. To bridge this gap, this work
presents a systematic investigation into the impact of diversity in RL-based
training for LLM reasoning, and proposes a novel diversity-aware policy
optimization method. Across evaluations on 12 LLMs, we observe a strong
positive correlation between the solution diversity and Potential at k (a novel
metric quantifying an LLM's reasoning potential) in high-performing models.
This finding motivates our method to explicitly promote diversity during RL
training. Specifically, we design a token-level diversity and reformulate it
into a practical objective, then we selectively apply it to positive samples.
Integrated into the R1-zero training framework, our method achieves a 3.5
percent average improvement across four mathematical reasoning benchmarks,
while generating more diverse and robust solutions.

---


### [KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](http://arxiv.org/abs/2505.23416v1)

Transformer-based large language models (LLMs) cache context as key-value
(KV) pairs during inference. As context length grows, KV cache sizes expand,
leading to substantial memory overhead and increased attention latency. This
paper introduces KVzip, a query-agnostic KV cache eviction method enabling
effective reuse of compressed KV caches across diverse queries. KVzip
quantifies the importance of a KV pair using the underlying LLM to reconstruct
original contexts from cached KV pairs, subsequently evicting pairs with lower
importance. Extensive empirical evaluations demonstrate that KVzip reduces KV
cache size by 3-4$\times$ and FlashAttention decoding latency by approximately
2$\times$, with negligible performance loss in question-answering, retrieval,
reasoning, and code comprehension tasks. Evaluations include various models
such as LLaMA3.1-8B, Qwen2.5-14B, and Gemma3-12B, with context lengths reaching
up to 170K tokens. KVzip significantly outperforms existing query-aware KV
eviction methods, which suffer from performance degradation even at a 90% cache
budget ratio under multi-query scenarios.

---


### [Daunce: Data Attribution through Uncertainty Estimation](http://arxiv.org/abs/2505.23223v1)

Training data attribution (TDA) methods aim to identify which training
examples influence a model's predictions on specific test data most. By
quantifying these influences, TDA supports critical applications such as data
debugging, curation, and valuation. Gradient-based TDA methods rely on
gradients and second-order information, limiting their applicability at scale.
While recent random projection-based methods improve scalability, they often
suffer from degraded attribution accuracy. Motivated by connections between
uncertainty and influence functions, we introduce Daunce - a simple yet
effective data attribution approach through uncertainty estimation. Our method
operates by fine-tuning a collection of perturbed models and computing the
covariance of per-example losses across these models as the attribution score.
Daunce is scalable to large language models (LLMs) and achieves more accurate
attribution compared to existing TDA methods. We validate Daunce on tasks
ranging from vision tasks to LLM fine-tuning, and further demonstrate its
compatibility with black-box model access. Applied to OpenAI's GPT models, our
method achieves, to our knowledge, the first instance of data attribution on
proprietary LLMs.

---


### [Two Is Better Than One: Rotations Scale LoRAs](http://arxiv.org/abs/2505.23184v1)

Scaling Low-Rank Adaptation (LoRA)-based Mixture-of-Experts (MoE) facilitates
large language models (LLMs) to efficiently adapt to diverse tasks. However,
traditional gating mechanisms that route inputs to the best experts may
fundamentally hinder LLMs' scalability, leading to poor generalization and
underfitting issues. We identify that the root cause lies in the restricted
expressiveness of existing weighted-sum mechanisms, both within and outside the
convex cone of LoRA representations. This motivates us to propose RadarGate, a
novel geometrically inspired gating method that introduces rotational
operations of LoRAs representations to boost the expressiveness and facilitate
richer feature interactions among multiple LoRAs for scalable LLMs.
Specifically, we first fuse each LoRA representation to other LoRAs using a
learnable component and then feed the output to a rotation matrix. This matrix
involves learnable parameters that define the relative angular relationship
between LoRA representations. Such a simple yet effective mechanism provides an
extra degree of freedom, facilitating the learning of cross-LoRA synergies and
properly tracking the challenging poor generalization and underfitting issues
as the number of LoRA grows. Extensive experiments on 6 public benchmarks
across 21 tasks show the effectiveness of our RadarGate for scaling LoRAs. We
also provide valuable insights, revealing that the rotations to each pair of
representations are contrastive, encouraging closer alignment of semantically
similar representations during geometrical transformation while pushing
distance ones further apart. We will release our code to the community.

---


### [LUMION: Fast Fault Recovery for ML Jobs Using Programmable Optical Fabrics](http://arxiv.org/abs/2505.23105v1)

When accelerators fail in modern ML datacenters, operators migrate the
affected ML training or inference jobs to entirely new racks. This approach,
while preserving network performance, is highly inefficient, requiring
datacenters to reserve full racks of idle accelerators for fault tolerance. In
this paper, we address this resource inefficiency by introducing LUMION, a
novel reconfigurable optical fabric for connecting accelerators within a
datacenter rack. Instead of migrating entire ML jobs, LUMION dynamically
integrates spare accelerators into ongoing workloads as failures occur, thereby
maintaining consistent performance without costly migrations. We show the
benefits of LUMION by building an end-to-end hardware prototype. Our
experiments fine-tune Llama 3.2 and show that LUMION swaps a failed GPU with a
healthy one and restarts the ML job within ~ 1 second of the failure. LUMION
achieves higher inter-GPU bandwidth compared to traditional electrical racks
after replacing failed accelerators with spare ones, leading to nearly 2X
improvement in fine-tuning throughput.

---


### [Weight Spectra Induced Efficient Model Adaptation](http://arxiv.org/abs/2505.23099v1)

Large-scale foundation models have demonstrated remarkable versatility across
a wide range of downstream tasks. However, fully fine-tuning these models
incurs prohibitive computational costs, motivating the development of
Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA, which introduces
low-rank updates to pre-trained weights. Despite their empirical success, the
underlying mechanisms by which PEFT modifies model parameters remain
underexplored. In this work, we present a systematic investigation into the
structural changes of weight matrices during fully fine-tuning. Through
singular value decomposition (SVD), we reveal that fine-tuning predominantly
amplifies the top singular values while leaving the remainder largely intact,
suggesting that task-specific knowledge is injected into a low-dimensional
subspace. Furthermore, we find that the dominant singular vectors are
reoriented in task-specific directions, whereas the non-dominant subspace
remains stable. Building on these insights, we propose a novel method that
leverages learnable rescaling of top singular directions, enabling precise
modulation of the most influential components without disrupting the global
structure. Our approach achieves consistent improvements over strong baselines
across multiple tasks, highlighting the efficacy of structurally informed
fine-tuning.

---


### [SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference](http://arxiv.org/abs/2505.23022v1)

Existing Large Language Model (LLM) serving systems prioritize maximum
throughput. They often neglect Service Level Objectives (SLOs) such as Time to
First Token (TTFT) and Time Per Output Token (TPOT), which leads to suboptimal
SLO attainment. This paper introduces SCORPIO, an SLO-oriented LLM serving
system designed to maximize system goodput and SLO attainment for workloads
with heterogeneous SLOs. Our core insight is to exploit SLO heterogeneity for
adaptive scheduling across admission control, queue management, and batch
selection. SCORPIO features a TTFT Guard, which employs least-deadline-first
reordering and rejects unattainable requests, and a TPOT Guard, which utilizes
a VBS-based admission control and a novel credit-based batching mechanism. Both
guards are supported by a predictive module. Evaluations demonstrate that
SCORPIO improves system goodput by up to 14.4X and SLO adherence by up to 46.5%
compared to state-of-the-art baselines.

---


### [EmergentTTS-Eval: Evaluating TTS Models on Complex Prosodic, Expressiveness, and Linguistic Challenges Using Model-as-a-Judge](http://arxiv.org/abs/2505.23009v1)

Text-to-Speech (TTS) benchmarks often fail to capture how well models handle
nuanced and semantically complex text. Building on $\textit{EmergentTTS}$, we
introduce $\textit{EmergentTTS-Eval}$, a comprehensive benchmark covering six
challenging TTS scenarios: emotions, paralinguistics, foreign words, syntactic
complexity, complex pronunciation (e.g. URLs, formulas), and questions.
Crucially, our framework automates both test-case generation and evaluation,
making the benchmark easily extensible. Starting from a small set of
human-written seed prompts, we iteratively extend them using LLMs to target
specific structural, phonetic and prosodic challenges, resulting in 1,645
diverse test cases. Moreover, we employ a model-as-a-judge approach, using a
Large Audio Language Model (LALM) to assess the speech across multiple
dimensions such as expressed emotion, prosodic, intonational, and pronunciation
accuracy. We evaluate state-of-the-art open-source and proprietary TTS systems,
such as 11Labs, Deepgram, and OpenAI's 4o-mini-TTS, on EmergentTTS-Eval,
demonstrating its ability to reveal fine-grained performance differences.
Results show that the model-as-a-judge approach offers robust TTS assessment
and a high correlation with human preferences. We open source the evaluation
$\href{https://github.com/boson-ai/EmergentTTS-Eval-public}{code}$ and the
$\href{https://huggingface.co/datasets/bosonai/EmergentTTS-Eval}{dataset}$.

---


### [LLM Agents for Bargaining with Utility-based Feedback](http://arxiv.org/abs/2505.22998v1)

Bargaining, a critical aspect of real-world interactions, presents challenges
for large language models (LLMs) due to limitations in strategic depth and
adaptation to complex human factors. Existing benchmarks often fail to capture
this real-world complexity. To address this and enhance LLM capabilities in
realistic bargaining, we introduce a comprehensive framework centered on
utility-based feedback. Our contributions are threefold: (1) BargainArena, a
novel benchmark dataset with six intricate scenarios (e.g., deceptive
practices, monopolies) to facilitate diverse strategy modeling; (2)
human-aligned, economically-grounded evaluation metrics inspired by utility
theory, incorporating agent utility and negotiation power, which implicitly
reflect and promote opponent-aware reasoning (OAR); and (3) a structured
feedback mechanism enabling LLMs to iteratively refine their bargaining
strategies. This mechanism can positively collaborate with in-context learning
(ICL) prompts, including those explicitly designed to foster OAR. Experimental
results show that LLMs often exhibit negotiation strategies misaligned with
human preferences, and that our structured feedback mechanism significantly
improves their performance, yielding deeper strategic and opponent-aware
reasoning.

---


### [Stairway to Success: Zero-Shot Floor-Aware Object-Goal Navigation via LLM-Driven Coarse-to-Fine Exploration](http://arxiv.org/abs/2505.23019v1)

Object-Goal Navigation (OGN) remains challenging in real-world, multi-floor
environments and under open-vocabulary object descriptions. We observe that
most episodes in widely used benchmarks such as HM3D and MP3D involve
multi-floor buildings, with many requiring explicit floor transitions. However,
existing methods are often limited to single-floor settings or predefined
object categories. To address these limitations, we tackle two key challenges:
(1) efficient cross-level planning and (2) zero-shot object-goal navigation
(ZS-OGN), where agents must interpret novel object descriptions without prior
exposure. We propose ASCENT, a framework that combines a Multi-Floor Spatial
Abstraction module for hierarchical semantic mapping and a Coarse-to-Fine
Frontier Reasoning module leveraging Large Language Models (LLMs) for
context-aware exploration, without requiring additional training on new object
semantics or locomotion data. Our method outperforms state-of-the-art ZS-OGN
approaches on HM3D and MP3D benchmarks while enabling efficient multi-floor
navigation. We further validate its practicality through real-world deployment
on a quadruped robot, achieving successful object exploration across unseen
floors.

---


### [MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning](http://arxiv.org/abs/2505.23254v1)

Owing to the huge success of generative artificial intelligence (AI), large
language models (LLMs) have emerged as a core subclass, underpinning
applications such as question answering, text generation, and code completion.
While fine-tuning these models on domain-specific data can yield significant
performance gains, it also poses daunting computational challenges, especially
for researchers and small organizations with limited hardware resources.
Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy
to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU
DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily
targets model-centric performance issues. As a result, key system-level issues,
including system memory fragmentation, inefficient pinned buffer allocation,
peak CPU usage spikes, and file system overhead, remain unaddressed, stifling
scalability and inflating costs. Such an observation motivates this paper to
introduce MemAscend, a framework that systematically tackles the underexplored
system memory bottlenecks in SSD-offloaded LLM training, with a focus on
resource-constrained environments. By streamlining pinned-memory allocation,
eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a
substantial system memory budget, enabling larger models, longer context
windows, and higher batch sizes without exceeding modest hardware limits.
Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption
by an average of 55.7% compared with standard SSD offloading techniques,
lowering the hardware barrier for fine-tuning and unlocking new possibilities
for cost-effective large-scale training on limited-resource machines.

---


