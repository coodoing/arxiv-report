### [Maximizing Confidence Alone Improves Reasoning](http://arxiv.org/abs/2505.22660v1)

Reinforcement learning (RL) has enabled machine learning models to achieve
significant advances in many fields. Most recently, RL has empowered frontier
language models to solve challenging math, science, and coding problems.
However, central to any RL algorithm is the reward function, and reward
engineering is a notoriously difficult problem in any domain. In this paper, we
propose RENT: Reinforcement Learning via Entropy Minimization -- a fully
unsupervised RL method that requires no external reward or ground-truth
answers, and instead uses the model's entropy of its underlying distribution as
an intrinsic reward. We find that by reinforcing the chains of thought that
yield high model confidence on its generated answers, the model improves its
reasoning ability. In our experiments, we showcase these improvements on an
extensive suite of commonly-used reasoning benchmarks, including GSM8K,
MATH500, AMC, AIME, and GPQA, and models of varying sizes from the Qwen and
Mistral families. The generality of our unsupervised learning method lends
itself to applicability in a wide range of domains where external supervision
is limited or unavailable.

---


### [FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control](http://arxiv.org/abs/2505.22642v1)

Reinforcement learning (RL) has driven significant progress in robotics, but
its complexity and long training times remain major bottlenecks. In this
report, we introduce FastTD3, a simple, fast, and capable RL algorithm that
significantly speeds up training for humanoid robots in popular suites such as
HumanoidBench, IsaacLab, and MuJoCo Playground. Our recipe is remarkably
simple: we train an off-policy TD3 agent with several modifications -- parallel
simulation, large-batch updates, a distributional critic, and carefully tuned
hyperparameters. FastTD3 solves a range of HumanoidBench tasks in under 3 hours
on a single A100 GPU, while remaining stable during training. We also provide a
lightweight and easy-to-use implementation of FastTD3 to accelerate RL research
in robotics.

---


### [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](http://arxiv.org/abs/2505.22617v1)

This paper aims to overcome a major obstacle in scaling RL for reasoning with
LLMs, namely the collapse of policy entropy. Such phenomenon is consistently
observed across vast RL runs without entropy intervention, where the policy
entropy dropped sharply at the early training stage, this diminished
exploratory ability is always accompanied with the saturation of policy
performance. In practice, we establish a transformation equation R=-a*e^H+b
between entropy H and downstream performance R. This empirical law strongly
indicates that, the policy performance is traded from policy entropy, thus
bottlenecked by its exhaustion, and the ceiling is fully predictable H=0,
R=-a+b. Our finding necessitates entropy management for continuous exploration
toward scaling compute for RL. To this end, we investigate entropy dynamics
both theoretically and empirically. Our derivation highlights that, the change
in policy entropy is driven by the covariance between action probability and
the change in logits, which is proportional to its advantage when using Policy
Gradient-like algorithms. Empirical study shows that, the values of covariance
term and entropy differences matched exactly, supporting the theoretical
conclusion. Moreover, the covariance term stays mostly positive throughout
training, further explaining why policy entropy would decrease monotonically.
Through understanding the mechanism behind entropy dynamics, we motivate to
control entropy by restricting the update of high-covariance tokens.
Specifically, we propose two simple yet effective techniques, namely Clip-Cov
and KL-Cov, which clip and apply KL penalty to tokens with high covariances
respectively. Experiments show that these methods encourage exploration, thus
helping policy escape entropy collapse and achieve better downstream
performance.

---


### [RICO: Improving Accuracy and Completeness in Image Recaptioning via Visual Reconstruction](http://arxiv.org/abs/2505.22613v1)

Image recaptioning is widely used to generate training datasets with enhanced
quality for various multimodal tasks. Existing recaptioning methods typically
rely on powerful multimodal large language models (MLLMs) to enhance textual
descriptions, but often suffer from inaccuracies due to hallucinations and
incompleteness caused by missing fine-grained details. To address these
limitations, we propose RICO, a novel framework that refines captions through
visual reconstruction. Specifically, we leverage a text-to-image model to
reconstruct a caption into a reference image, and prompt an MLLM to identify
discrepancies between the original and reconstructed images to refine the
caption. This process is performed iteratively, further progressively promoting
the generation of more faithful and comprehensive descriptions. To mitigate the
additional computational cost induced by the iterative process, we introduce
RICO-Flash, which learns to generate captions like RICO using DPO. Extensive
experiments demonstrate that our approach significantly improves caption
accuracy and completeness, outperforms most baselines by approximately 10% on
both CapsBench and CompreCap. Code released at
https://github.com/wangyuchi369/RICO.

---


### [HDDLGym: A Tool for Studying Multi-Agent Hierarchical Problems Defined in HDDL with OpenAI Gym](http://arxiv.org/abs/2505.22597v1)

In recent years, reinforcement learning (RL) methods have been widely tested
using tools like OpenAI Gym, though many tasks in these environments could also
benefit from hierarchical planning. However, there is a lack of a tool that
enables seamless integration of hierarchical planning with RL. Hierarchical
Domain Definition Language (HDDL), used in classical planning, introduces a
structured approach well-suited for model-based RL to address this gap. To
bridge this integration, we introduce HDDLGym, a Python-based tool that
automatically generates OpenAI Gym environments from HDDL domains and problems.
HDDLGym serves as a link between RL and hierarchical planning, supporting
multi-agent scenarios and enabling collaborative planning among agents. This
paper provides an overview of HDDLGym's design and implementation, highlighting
the challenges and design choices involved in integrating HDDL with the Gym
interface, and applying RL policies to support hierarchical planning. We also
provide detailed instructions and demonstrations for using the HDDLGym
framework, including how to work with existing HDDL domains and problems from
International Planning Competitions, exemplified by the Transport domain.
Additionally, we offer guidance on creating new HDDL domains for multi-agent
scenarios and demonstrate the practical use of HDDLGym in the Overcooked
domain. By leveraging the advantages of HDDL and Gym, HDDLGym aims to be a
valuable tool for studying RL in hierarchical planning, particularly in
multi-agent contexts.

---


### [Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](http://arxiv.org/abs/2505.22591v1)

Although large language models demonstrate strong performance across various
domains, they still struggle with numerous bad cases in mathematical reasoning.
Previous approaches to learning from errors synthesize training data by solely
extrapolating from isolated bad cases, thereby failing to generalize the
extensive patterns inherent within these cases. This paper presents
Self-Error-Instruct (SEI), a framework that addresses these model weaknesses
and synthesizes more generalized targeted training data. Specifically, we
explore a target model on two mathematical datasets, GSM8K and MATH, to
pinpoint bad cases. Then, we generate error keyphrases for these cases based on
the instructor model's (GPT-4o) analysis and identify error types by clustering
these keyphrases. Next, we sample a few bad cases during each generation for
each identified error type and input them into the instructor model, which
synthesizes additional training data using a self-instruct approach. This new
data is refined through a one-shot learning process to ensure that only the
most effective examples are kept. Finally, we use these curated data to
fine-tune the target model, iteratively repeating the process to enhance
performance. We apply our framework to various models and observe improvements
in their reasoning abilities across both in-domain and out-of-domain
mathematics datasets. These results demonstrate the effectiveness of self-error
instruction in improving LLMs' mathematical reasoning through error
generalization.

---


### [GitGoodBench: A Novel Benchmark For Evaluating Agentic Performance On Git](http://arxiv.org/abs/2505.22583v1)

Benchmarks for Software Engineering (SE) AI agents, most notably SWE-bench,
have catalyzed progress in programming capabilities of AI agents. However, they
overlook critical developer workflows such as Version Control System (VCS)
operations. To address this issue, we present GitGoodBench, a novel benchmark
for evaluating AI agent performance on VCS tasks. GitGoodBench covers three
core Git scenarios extracted from permissive open-source Python, Java, and
Kotlin repositories. Our benchmark provides three datasets: a comprehensive
evaluation suite (900 samples), a rapid prototyping version (120 samples), and
a training corpus (17,469 samples). We establish baseline performance on the
prototyping version of our benchmark using GPT-4o equipped with custom tools,
achieving a 21.11% solve rate overall. We expect GitGoodBench to serve as a
crucial stepping stone toward truly comprehensive SE agents that go beyond mere
programming.

---


### [Fusion Steering: Prompt-Specific Activation Control](http://arxiv.org/abs/2505.22572v1)

We present Fusion Steering, an activation steering methodology that improves
factual accuracy in large language models (LLMs) for question-answering (QA)
tasks. This approach introduces flexible steering configurations, including
full-layer steering and segmented steering. Unlike traditional methods
constrained to single-layer or fixed-layer operations, Fusion Steering employs
dynamic injection of prompt-specific activation deltas across all transformer
layers. These activation deltas are derived from reference completions that
combine the ground-truth answer with a model-generated explanation to
facilitate semantically enriched, example-specific steering. The injection
weights are optimized per prompt using Optuna, targeting a joint objective that
balances token overlap (factual alignment) and perplexity (fluency proxy).
Evaluation employs a composite score integrating token overlap and LLM-graded
quality, encompassing factual accuracy, coherence, and relevance. Empirical
results on 260 SimpleQA prompts (selected from 500 where the baseline failed)
showcase the efficacy of segmented steering. Using Gemma-2-2B-IT with 8-bit
quantization, segmented steering achieves an accuracy of 25.4% (outputs scoring
$\geq 0.6$), outperforming the baseline at 3.5% and full-layer steering at
16.2%. Under the stricter SimpleQA rubric, segmented steering boosts fully
correct responses from 0.0% to 13.1%. These findings highlight the strengths of
segmented, dynamic intervention strategies and the promise of per-prompt,
full-network activation control. Fusion Steering is also amenable to sparse
representations, such as Neuronpedia or sparse crosscoders, suggesting a
promising direction for interpretable and scalable activation-level control in
LLMs.

---


### [Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems](http://arxiv.org/abs/2505.22571v1)

This paper presents a novel approach for unified retrieval-augmented
generation (RAG) systems using the recent emerging large language model (LLM)
agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental
controllers, has become a promising approach to enable the interpretability of
RAG tasks, especially for complex reasoning question-answering systems (e.g.,
multi-hop queries). Nonetheless, previous works mainly focus on solving RAG
systems with either single-hop or multi-hop approaches separately, which limits
the application of those approaches to real-world applications. In this study,
we propose a trainable agent framework called Agent-UniRAG for unified
retrieval-augmented LLM systems, which enhances the effectiveness and
interpretability of RAG systems. The main idea is to design an LLM agent
framework to solve RAG tasks step-by-step based on the complexity of the
inputs, simultaneously including single-hop and multi-hop queries in an
end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset
to enable the proposed agent framework for small open-source LLMs (e.g.,
Llama-3-8B). The results show comparable performances with closed-source and
larger open-source LLMs across various RAG benchmarks. Our source code and
dataset are publicly available for further exploitation.

---


### [Scaling-up Perceptual Video Quality Assessment](http://arxiv.org/abs/2505.22543v1)

The data scaling law has been shown to significantly enhance the performance
of large multi-modal models (LMMs) across various downstream tasks. However, in
the domain of perceptual video quality assessment (VQA), the potential of
scaling law remains unprecedented due to the scarcity of labeled resources and
the insufficient scale of datasets. To address this, we propose
\textbf{OmniVQA}, an efficient framework designed to efficiently build
high-quality, human-in-the-loop VQA multi-modal instruction databases (MIDBs).
We then scale up to create \textbf{OmniVQA-Chat-400K}, the largest MIDB in the
VQA field concurrently. Our focus is on the technical and aesthetic quality
dimensions, with abundant in-context instruction data to provide fine-grained
VQA knowledge. Additionally, we have built the \textbf{OmniVQA-MOS-20K} dataset
to enhance the model's quantitative quality rating capabilities. We then
introduce a \textbf{complementary} training strategy that effectively leverages
the knowledge from datasets for quality understanding and quality rating tasks.
Furthermore, we propose the \textbf{OmniVQA-FG (fine-grain)-Benchmark} to
evaluate the fine-grained performance of the models. Our results demonstrate
that our models achieve state-of-the-art performance in both quality
understanding and rating tasks.

---


### [On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling](http://arxiv.org/abs/2505.22491v1)

The dominant paradigm for training large-scale vision and language models is
He initialization and a single global learning rate (\textit{standard
parameterization}, SP). Despite its practical success, standard parametrization
remains poorly understood from a theoretical perspective: Existing
infinite-width theory would predict instability under large learning rates and
vanishing feature learning under stable learning rates. However, empirically
optimal learning rates consistently decay much slower than theoretically
predicted. By carefully studying neural network training dynamics, we
demonstrate that this discrepancy is not fully explained by finite-width
phenomena such as catapult effects or a lack of alignment between weights and
incoming activations. We instead show that the apparent contradiction can be
fundamentally resolved by taking the loss function into account: In contrast to
Mean Squared Error (MSE) loss, we prove that under cross-entropy (CE) loss, an
intermediate \textit{controlled divergence} regime emerges, where logits
diverge but loss, gradients, and activations remain stable. Stable training
under large learning rates enables persistent feature evolution at scale in all
hidden layers, which is crucial for the practical success of SP. In experiments
across optimizers (SGD, Adam), architectures (MLPs, GPT) and data modalities
(vision, language), we validate that neural networks operate in this controlled
divergence regime under CE loss but not under MSE loss. Our empirical evidence
suggests that width-scaling considerations are surprisingly useful for
predicting empirically optimal learning rate exponents. Finally, our analysis
clarifies the effectiveness and limitations of recently proposed layerwise
learning rate scalings for standard initialization.

---


### [Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO](http://arxiv.org/abs/2505.22453v1)

Improving Multi-modal Large Language Models (MLLMs) in the post-training
stage typically relies on supervised fine-tuning (SFT) or reinforcement
learning (RL). However, these supervised methods require expensive and manually
annotated multi-modal data--an ultimately unsustainable resource. While recent
efforts have explored unsupervised post-training, their methods are complex and
difficult to iterate. In this work, we are the first to investigate the use of
GRPO, a stable and scalable online RL algorithm, for enabling continual
self-improvement without any external supervision. We propose MM-UPT, a simple
yet effective framework for unsupervised post-training of MLLMs. MM-UPT builds
upon GRPO, replacing traditional reward signals with a self-rewarding mechanism
based on majority voting over multiple sampled responses. Our experiments
demonstrate that MM-UPT significantly improves the reasoning ability of
Qwen2.5-VL-7B (e.g., 66.3 %$\rightarrow$72.9 % on MathVista, 62.9
%$\rightarrow$68.7 % on We-Math), using standard dataset without ground truth
labels. MM-UPT also outperforms prior unsupervised baselines and even
approaches the results of supervised GRPO. Furthermore, we show that
incorporating synthetic questions, generated solely by MLLM itself, can boost
performance as well, highlighting a promising approach for scalable
self-improvement. Overall, MM-UPT offers a new paradigm for continual,
autonomous enhancement of MLLMs in the absence of external supervision. Our
code is available at https://github.com/waltonfuture/MM-UPT.

---


### [Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start](http://arxiv.org/abs/2505.22334v1)

Recent advancements in large language models (LLMs) have demonstrated
impressive chain-of-thought reasoning capabilities, with reinforcement learning
(RL) playing a crucial role in this progress. While "aha moment"
patterns--where models exhibit self-correction through reflection--are often
attributed to emergent properties from RL, we first demonstrate that these
patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not
necessarily correlate with improved reasoning performance. Building on these
insights, we present a comprehensive study on enhancing multimodal reasoning
through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start
with structured chain-of-thought reasoning patterns, followed by (2)
reinforcement learning via GRPO to further refine these capabilities. Our
extensive experiments show that this combined approach consistently outperforms
both SFT-only and RL-only methods across challenging multimodal reasoning
benchmarks. The resulting models achieve state-of-the-art performance among
open-source MLLMs at both 3B and 7B scales, with our 7B model showing
substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on
MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving
performance competitive with several 7B models. Overall, this work provides
practical guidance for building advanced multimodal reasoning models. Our code
is available at https://github.com/waltonfuture/RL-with-Cold-Start.

---


### [Skywork Open Reasoner 1 Technical Report](http://arxiv.org/abs/2505.22312v1)

The success of DeepSeek-R1 underscores the significant role of reinforcement
learning (RL) in enhancing the reasoning capabilities of large language models
(LLMs). In this work, we present Skywork-OR1, an effective and scalable RL
implementation for long Chain-of-Thought (CoT) models. Building on the
DeepSeek-R1-Distill model series, our RL approach achieves notable performance
gains, increasing average accuracy across AIME24, AIME25, and LiveCodeBench
from 57.8% to 72.8% (+15.0%) for the 32B model and from 43.6% to 57.5% (+13.9%)
for the 7B model. Our Skywork-OR1-32B model surpasses both DeepSeek-R1 and
Qwen3-32B on the AIME24 and AIME25 benchmarks, while achieving comparable
results on LiveCodeBench. The Skywork-OR1-7B and Skywork-OR1-Math-7B models
demonstrate competitive reasoning capabilities among models of similar size. We
perform comprehensive ablation studies on the core components of our training
pipeline to validate their effectiveness. Additionally, we thoroughly
investigate the phenomenon of entropy collapse, identify key factors affecting
entropy dynamics, and demonstrate that mitigating premature entropy collapse is
critical for improved test performance. To support community research, we fully
open-source our model weights, training code, and training datasets.

---


### [From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications](http://arxiv.org/abs/2505.22311v1)

With the advent of 6G communications, intelligent communication systems face
multiple challenges, including constrained perception and response
capabilities, limited scalability, and low adaptability in dynamic
environments. This tutorial provides a systematic introduction to the
principles, design, and applications of Large Artificial Intelligence Models
(LAMs) and Agentic AI technologies in intelligent communication systems, aiming
to offer researchers a comprehensive overview of cutting-edge technologies and
practical guidance. First, we outline the background of 6G communications,
review the technological evolution from LAMs to Agentic AI, and clarify the
tutorial's motivation and main contributions. Subsequently, we present a
comprehensive review of the key components required for constructing LAMs. We
further categorize LAMs and analyze their applicability, covering Large
Language Models (LLMs), Large Vision Models (LVMs), Large Multimodal Models
(LMMs), Large Reasoning Models (LRMs), and lightweight LAMs. Next, we propose a
LAM-centric design paradigm tailored for communications, encompassing dataset
construction and both internal and external learning approaches. Building upon
this, we develop an LAM-based Agentic AI system for intelligent communications,
clarifying its core components such as planners, knowledge bases, tools, and
memory modules, as well as its interaction mechanisms. We also introduce a
multi-agent framework with data retrieval, collaborative planning, and
reflective evaluation for 6G. Subsequently, we provide a detailed overview of
the applications of LAMs and Agentic AI in communication scenarios. Finally, we
summarize the research challenges and future directions in current studies,
aiming to support the development of efficient, secure, and sustainable
next-generation intelligent communication systems.

---


### [Investigating Mechanisms for In-Context Vision Language Binding](http://arxiv.org/abs/2505.22200v1)

To understand a prompt, Vision-Language models (VLMs) must perceive the
image, comprehend the text, and build associations within and across both
modalities. For instance, given an 'image of a red toy car', the model should
associate this image to phrases like 'car', 'red toy', 'red object', etc. Feng
and Steinhardt propose the Binding ID mechanism in LLMs, suggesting that the
entity and its corresponding attribute tokens share a Binding ID in the model
activations. We investigate this for image-text binding in VLMs using a
synthetic dataset and task that requires models to associate 3D objects in an
image with their descriptions in the text. Our experiments demonstrate that
VLMs assign a distinct Binding ID to an object's image tokens and its textual
references, enabling in-context association.

---


### [Breaking the Cloak! Unveiling Chinese Cloaked Toxicity with Homophone Graph and Toxic Lexicon](http://arxiv.org/abs/2505.22184v1)

Social media platforms have experienced a significant rise in toxic content,
including abusive language and discriminatory remarks, presenting growing
challenges for content moderation. Some users evade censorship by deliberately
disguising toxic words through homophonic cloak, which necessitates the task of
unveiling cloaked toxicity. Existing methods are mostly designed for English
texts, while Chinese cloaked toxicity unveiling has not been solved yet. To
tackle the issue, we propose C$^2$TU, a novel training-free and prompt-free
method for Chinese cloaked toxic content unveiling. It first employs substring
matching to identify candidate toxic words based on Chinese homo-graph and
toxic lexicon. Then it filters those candidates that are non-toxic and corrects
cloaks to be their corresponding toxicities. Specifically, we develop two model
variants for filtering, which are based on BERT and LLMs, respectively. For
LLMs, we address the auto-regressive limitation in computing word occurrence
probability and utilize the full semantic contexts of a text sequence to reveal
cloaked toxic words. Extensive experiments demonstrate that C$^2$TU can achieve
superior performance on two Chinese toxic datasets. In particular, our method
outperforms the best competitor by up to 71% on the F1 score and 35% on
accuracy, respectively.

---


### [Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design](http://arxiv.org/abs/2505.22179v1)

Speculative decoding and quantization effectively accelerate memory-bound
inference of large language models. Speculative decoding mitigates the memory
bandwidth bottleneck by verifying multiple tokens within a single forward pass,
which increases computational effort. Quantization achieves this optimization
by compressing weights and activations into lower bit-widths and also reduces
computations via low-bit matrix multiplications. To further leverage their
strengths, we investigate the integration of these two techniques.
Surprisingly, experiments applying the advanced speculative decoding method
EAGLE-2 to various quantized models reveal that the memory benefits from 4-bit
weight quantization are diminished by the computational load from speculative
decoding. Specifically, verifying a tree-style draft incurs significantly more
time overhead than a single-token forward pass on 4-bit weight quantized
models. This finding led to our new speculative decoding design: a hierarchical
framework that employs a small model as an intermediate stage to turn
tree-style drafts into sequence drafts, leveraging the memory access benefits
of the target quantized model. Experimental results show that our hierarchical
approach achieves a 2.78$\times$ speedup across various tasks for the 4-bit
weight Llama-3-70B model on an A100 GPU, outperforming EAGLE-2 by 1.31$\times$.
Code available at https://github.com/AI9Stars/SpecMQuant.

---


### [Flexible Tool Selection through Low-dimensional Attribute Alignment of Vision and Language](http://arxiv.org/abs/2505.22146v1)

Flexible tool selection reflects a complex cognitive ability that
distinguishes humans from other species, yet computational models that capture
this ability remain underdeveloped. We developed a framework using
low-dimensional attribute representations to bridge visual tool perception and
linguistic task understanding. We constructed a comprehensive dataset (ToolNet)
containing 115 common tools labeled with 13 carefully designed attributes
spanning physical, functional, and psychological properties, paired with
natural language scenarios describing tool usage. Visual encoders (ResNet or
ViT) extract attributes from tool images while fine-tuned language models
(GPT-2, LLaMA, DeepSeek) derive required attributes from task descriptions. Our
approach achieves 74% accuracy in tool selection tasks-significantly
outperforming direct tool matching (20%) and smaller multimodal models
(21%-58%), while approaching performance of much larger models like GPT-4o
(73%) with substantially fewer parameters. Ablation studies revealed that
manipulation-related attributes (graspability, hand-relatedness, elongation)
consistently prove most critical across modalities. This work provides a
parameter-efficient, interpretable solution that mimics human-like tool
cognition, advancing both cognitive science understanding and practical
applications in tool selection tasks.

---


### [SridBench: Benchmark of Scientific Research Illustration Drawing of Image Generation Model](http://arxiv.org/abs/2505.22126v1)

Recent years have seen rapid advances in AI-driven image generation. Early
diffusion models emphasized perceptual quality, while newer multimodal models
like GPT-4o-image integrate high-level reasoning, improving semantic
understanding and structural composition. Scientific illustration generation
exemplifies this evolution: unlike general image synthesis, it demands accurate
interpretation of technical content and transformation of abstract ideas into
clear, standardized visuals. This task is significantly more
knowledge-intensive and laborious, often requiring hours of manual work and
specialized tools. Automating it in a controllable, intelligent manner would
provide substantial practical value. Yet, no benchmark currently exists to
evaluate AI on this front. To fill this gap, we introduce SridBench, the first
benchmark for scientific figure generation. It comprises 1,120 instances
curated from leading scientific papers across 13 natural and computer science
disciplines, collected via human experts and MLLMs. Each sample is evaluated
along six dimensions, including semantic fidelity and structural accuracy.
Experimental results reveal that even top-tier models like GPT-4o-image lag
behind human performance, with common issues in text/visual clarity and
scientific correctness. These findings highlight the need for more advanced
reasoning-driven visual generation capabilities.

---


### [Visual Large Language Models Exhibit Human-Level Cognitive Flexibility in the Wisconsin Card Sorting Test](http://arxiv.org/abs/2505.22112v1)

Cognitive flexibility has been extensively studied in human cognition but
remains relatively unexplored in the context of Visual Large Language Models
(VLLMs). This study assesses the cognitive flexibility of state-of-the-art
VLLMs (GPT-4o, Gemini-1.5 Pro, and Claude-3.5 Sonnet) using the Wisconsin Card
Sorting Test (WCST), a classic measure of set-shifting ability. Our results
reveal that VLLMs achieve or surpass human-level set-shifting capabilities
under chain-of-thought prompting with text-based inputs. However, their
abilities are highly influenced by both input modality and prompting strategy.
In addition, we find that through role-playing, VLLMs can simulate various
functional deficits aligned with patients having impairments in cognitive
flexibility, suggesting that VLLMs may possess a cognitive architecture, at
least regarding the ability of set-shifting, similar to the brain. This study
reveals the fact that VLLMs have already approached the human level on a key
component underlying our higher cognition, and highlights the potential to use
them to emulate complex brain processes.

---


### [Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO](http://arxiv.org/abs/2505.22068v1)

Previous study suggest that powerful Large Language Models (LLMs) trained
with Reinforcement Learning with Verifiable Rewards (RLVR) only refines
reasoning path without improving the reasoning capacity in math tasks while
supervised-finetuning(SFT) with distillation can. We study this from the view
of Scientific information extraction (SciIE) where LLMs and reasoning LLMs
underperforms small Bert-based models. SciIE require both the reasoning and
memorization. We argue that both SFT and RLVR can refine the reasoning path and
improve reasoning capacity in a simple way based on SciIE. We propose two-stage
training with 1. MimicSFT, using structured reasoning templates without needing
high-quality chain-of-thought data, 2. R$^2$GRPO with relevance and
rule-induced rewards. Experiments on scientific IE benchmarks show that both
methods can improve the reasoning capacity. R$^2$GRPO with mimicSFT surpasses
baseline LLMs and specialized supervised models in relation extraction. Our
code is available at https://github.com/ranlislz/R2GRPO.

---


### [Reinforced Reasoning for Embodied Planning](http://arxiv.org/abs/2505.22050v1)

Embodied planning requires agents to make coherent multi-step decisions based
on dynamic visual observations and natural language goals. While recent
vision-language models (VLMs) excel at static perception tasks, they struggle
with the temporal reasoning, spatial understanding, and commonsense grounding
needed for planning in interactive environments. In this work, we introduce a
reinforcement fine-tuning framework that brings R1-style reasoning enhancement
into embodied planning. We first distill a high-quality dataset from a powerful
closed-source model and perform supervised fine-tuning (SFT) to equip the model
with structured decision-making priors. We then design a rule-based reward
function tailored to multi-step action quality and optimize the policy via
Generalized Reinforced Preference Optimization (GRPO). Our approach is
evaluated on Embench, a recent benchmark for interactive embodied tasks,
covering both in-domain and out-of-domain scenarios. Experimental results show
that our method significantly outperforms models of similar or larger scale,
including GPT-4o-mini and 70B+ open-source baselines, and exhibits strong
generalization to unseen environments. This work highlights the potential of
reinforcement-driven reasoning to advance long-horizon planning in embodied AI.

---


### [Estimating the Effects of Sample Training Orders for Large Language Models without Retraining](http://arxiv.org/abs/2505.22042v1)

The order of training samples plays a crucial role in large language models
(LLMs), significantly impacting both their external performance and internal
learning dynamics. Traditional methods for investigating this effect generally
require retraining the model with various sample orders, which is
computationally infeasible for LLMs. In this work, we improve traditional
methods by designing a retraining-free framework. By approximating Adam
optimizer updates with first- and second-order Taylor expansions and utilizing
random projection methods to store intermediate checkpoints, our framework can
efficiently estimate model parameters for arbitrary training sample orders.
Next, we apply our framework to two downstream research problems: (1) Training
curriculum design for LLMs -- we base our retraining-free framework to propose
a novel curriculum learning strategy that augments curriculum proposals with
estimated model performances, enabling more informed sample scheduling. (2)
LLMs' memorization and generalization effect analysis -- we use our
retraining-free framework to estimate how the positions of training samples
influence LLMs' capacity for memorization and generalization. We conduct
extensive experiments to validate the effectiveness of our retraining-free
framework in reproducing the true model performances, and further demonstrate
its potential in optimizing LLM training curricula and analyzing the
memorization and generalization effects of LLMs.

---


### [Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](http://arxiv.org/abs/2505.22038v1)

Large Vision-Language Models (LVLMs) have shown impressive performance across
multi-modal tasks by encoding images into thousands of tokens. However, the
large number of image tokens results in significant computational overhead, and
the use of dynamic high-resolution inputs further increases this burden.
Previous approaches have attempted to reduce the number of image tokens through
token pruning, typically by selecting tokens based on attention scores or image
token diversity. Through empirical studies, we observe that existing methods
often overlook the joint impact of pruning on both the current layer's output
(local) and the outputs of subsequent layers (global), leading to suboptimal
pruning decisions. To address this challenge, we propose Balanced Token Pruning
(BTP), a plug-and-play method for pruning vision tokens. Specifically, our
method utilizes a small calibration set to divide the pruning process into
multiple stages. In the early stages, our method emphasizes the impact of
pruning on subsequent layers, whereas in the deeper stages, the focus shifts
toward preserving the consistency of local outputs. Extensive experiments
across various LVLMs demonstrate the broad effectiveness of our approach on
multiple benchmarks. Our method achieves a 78% compression rate while
preserving 96.7% of the original models' performance on average.

---


### [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](http://arxiv.org/abs/2505.22019v1)

Effectively retrieving, reasoning and understanding visually rich information
remains a challenge for RAG methods. Traditional text-based methods cannot
handle visual-related information. On the other hand, current vision-based RAG
approaches are often limited by fixed pipelines and frequently struggle to
reason effectively due to the insufficient activation of the fundamental
capabilities of models. As RL has been proven to be beneficial for model
reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex
reasoning across visually rich information. With this framework, VLMs interact
with search engines, autonomously sampling single-turn or multi-turn reasoning
trajectories with the help of visual perception tokens and undergoing continual
optimization based on these samples. Our approach highlights key limitations of
RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely
incorporate images into the context, leading to insufficient reasoning token
allocation and neglecting visual-specific perception; and (ii) When models
interact with search engines, their queries often fail to retrieve relevant
information due to the inability to articulate requirements, thereby leading to
suboptimal performance. To address these challenges, we define an action space
tailored for visually rich inputs, with actions including cropping and scaling,
allowing the model to gather information from a coarse-to-fine perspective.
Furthermore, to bridge the gap between users' original inquiries and the
retriever, we employ a simple yet effective reward that integrates query
rewriting and retrieval performance with a model-based reward. Our VRAG-RL
optimizes VLMs for RAG tasks using specially designed RL strategies, aligning
the model with real-world applications. The code is available at
\hyperlink{https://github.com/Alibaba-NLP/VRAG}{https://github.com/Alibaba-NLP/VRAG}.

---


### [Legal Assist AI: Leveraging Transformer-Based Model for Effective Legal Assistance](http://arxiv.org/abs/2505.22003v1)

Pursuit of accessible legal assistance in India faces a critical gap, as many
citizens struggle to leverage their legal rights due to limited awareness and
access to relevant legal information. This paper introduces Legal Assist AI, a
transformer-based model designed to bridge this gap by offering effective legal
assistance through large language models (LLMs). The system retrieves relevant
legal information from a curated database and generates accurate responses,
enabling effective assistance for diverse users, including legal professionals,
scholars, and the general public. The model was fine-tuned on extensive
datasets from the Indian legal domain, including Indian Constitution, Bharatiya
Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS) and so forth,
providing a robust understanding of the complexities of Indian law. By
incorporating domain-specific legal datasets, the proposed model demonstrated
remarkable efficiency and specialization in legal Question-Answering. The model
was evaluated against state-of-the-art models such as GPT-3.5 Turbo and Mistral
7B, achieving a 60.08% score on the AIBE, outperforming its competitors in
legal reasoning and accuracy. Unlike other models, Legal Assist AI avoided
common issues such as hallucinations, making it highly reliable for practical
legal applications. It showcases the model's applicability in real-world legal
scenarios, with future iterations aiming to enhance performance and expand its
dataset to cover a broader range of multilingual and case-specific queries as
well.

---


### [Reward-Independent Messaging for Decentralized Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2505.21985v1)

In multi-agent reinforcement learning (MARL), effective communication
improves agent performance, particularly under partial observability. We
propose MARL-CPC, a framework that enables communication among fully
decentralized, independent agents without parameter sharing. MARL-CPC
incorporates a message learning model based on collective predictive coding
(CPC) from emergent communication research. Unlike conventional methods that
treat messages as part of the action space and assume cooperation, MARL-CPC
links messages to state inference, supporting communication in non-cooperative,
reward-independent settings. We introduce two algorithms -Bandit-CPC and
IPPO-CPC- and evaluate them in non-cooperative MARL tasks. Benchmarks show that
both outperform standard message-as-action approaches, establishing effective
communication even when messages offer no direct benefit to the sender. These
results highlight MARL-CPC's potential for enabling coordination in complex,
decentralized environments.

---


### [DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation](http://arxiv.org/abs/2505.21969v1)

Adaptive navigation in unfamiliar environments is crucial for household
service robots but remains challenging due to the need for both low-level path
planning and high-level scene understanding. While recent vision-language model
(VLM) based zero-shot approaches reduce dependence on prior maps and
scene-specific training data, they face significant limitations: spatiotemporal
discontinuity from discrete observations, unstructured memory representations,
and insufficient task understanding leading to navigation failures. We propose
DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory
Oriented Navigation), a novel cognitive-inspired framework consisting of
Ventral and Dorsal Streams that mimics human navigation capabilities. The
Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology
Map to handle spatiotemporal discontinuities, while the Ventral Stream combines
RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops
Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON
on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art
performance on both success rate (SR) and success weighted by path length (SPL)
metrics, significantly outperforming existing methods. We also introduce a new
evaluation metric (AORI) to assess navigation intelligence better.
Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot
autonomous navigation without requiring prior map building or pre-training.

---


### [Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs](http://arxiv.org/abs/2505.21955v1)

Large vision-language models (LVLMs) are increasingly deployed in interactive
applications such as virtual and augmented reality, where first-person
(egocentric) view captured by head-mounted cameras serves as key input. While
this view offers fine-grained cues about user attention and hand-object
interactions, their narrow field of view and lack of global context often lead
to failures on spatially or contextually demanding queries. To address this, we
introduce a framework that augments egocentric inputs with third-person
(exocentric) views, providing complementary information such as global scene
layout and object visibility to LVLMs. We present E3VQA, the first benchmark
for multi-view question answering with 4K high-quality question-answer pairs
grounded in synchronized ego-exo image pairs. Additionally, we propose M3CoT, a
training-free prompting technique that constructs a unified scene
representation by integrating scene graphs from three complementary
perspectives. M3CoT enables LVLMs to reason more effectively across views,
yielding consistent performance gains (4.84% for GPT-4o and 5.94% for Gemini
2.0 Flash) over a recent CoT baseline. Our extensive evaluation reveals key
strengths and limitations of LVLMs in multi-view reasoning and highlights the
value of leveraging both egocentric and exocentric inputs.

---


### [Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference](http://arxiv.org/abs/2505.21919v1)

The increasing adoption of large language models (LLMs) with extended context
windows necessitates efficient Key-Value Cache (KVC) management to optimize
inference performance. Inference workloads like Retrieval-Augmented Generation
(RAG) and agents exhibit high cache reusability, making efficient caching
critical to reducing redundancy and improving speed. We analyze real-world KVC
access patterns using publicly available traces and evaluate commercial
key-value stores like Redis and state-of-the-art RDMA-based systems (CHIME [1]
and Sherman [2]) for KVC metadata management. Our work demonstrates the lack of
tailored storage solution for KVC prefilling, underscores the need for an
efficient distributed caching system with optimized metadata management for LLM
workloads, and provides insights into designing improved KVC management systems
for scalable, low-latency inference.

---


### [Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding](http://arxiv.org/abs/2505.21908v1)

Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement
and operations but require labor-intensive assignment. Large Language Models
(LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of
the task: pretraining corpora rarely contain private clinical or billing data.
We introduce DRG-Sapphire, which uses large-scale reinforcement learning (RL)
for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained
with Group Relative Policy Optimization (GRPO) using rule-based rewards,
DRG-Sapphire introduces a series of RL enhancements to address domain-specific
challenges not seen in previous mathematical tasks. Our model achieves
state-of-the-art accuracy on the MIMIC-IV benchmark and generates
physician-validated reasoning for DRG assignments, significantly enhancing
explainability. Our study further sheds light on broader challenges of applying
RL to knowledge-intensive, OOD tasks. We observe that RL performance scales
approximately linearly with the logarithm of the number of supervised
fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally
constrained by the domain knowledge encoded in the base model. For OOD tasks
like DRG coding, strong RL performance requires sufficient knowledge infusion
prior to RL. Consequently, scaling SFT may be more effective and
computationally efficient than scaling RL alone for such tasks.

---


### [Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge](http://arxiv.org/abs/2505.21906v1)

Vision-language-action (VLA) models have emerged as the next generation of
models in robotics. However, despite leveraging powerful pre-trained
Vision-Language Models (VLMs), existing end-to-end VLA systems often lose key
capabilities during fine-tuning as the model adapts to specific robotic tasks.
We argue that a generalizable VLA model should retain and expand upon the VLM's
core competencies: 1) Open-world embodied reasoning - the VLA should inherit
the knowledge from VLM, i.e., recognize anything that the VLM can recognize,
capable of solving math problems, possessing visual-spatial intelligence, 2)
Reasoning following - effectively translating the open-world reasoning into
actionable steps for the robot. In this work, we introduce ChatVLA-2, a novel
mixture-of-expert VLA model coupled with a specialized three-stage training
pipeline designed to preserve the VLM's original strengths while enabling
actionable reasoning. To validate our approach, we design a math-matching task
wherein a robot interprets math problems written on a whiteboard and picks
corresponding number cards from a table to solve equations. Remarkably, our
method exhibits exceptional mathematical reasoning and OCR capabilities,
despite these abilities not being explicitly trained within the VLA.
Furthermore, we demonstrate that the VLA possesses strong spatial reasoning
skills, enabling it to interpret novel directional instructions involving
previously unseen objects. Overall, our method showcases reasoning and
comprehension abilities that significantly surpass state-of-the-art imitation
learning methods such as OpenVLA, DexVLA, and pi-zero. This work represents a
substantial advancement toward developing truly generalizable robotic
foundation models endowed with robust reasoning capacities.

---


### [Compressing Sine-Activated Low-Rank Adapters through Post-Training Quantization](http://arxiv.org/abs/2505.21895v1)

Low-Rank Adaptation (LoRA) has become a standard approach for
parameter-efficient fine-tuning, offering substantial reductions in trainable
parameters by modeling updates as the product of two low-rank matrices. While
effective, the low-rank constraint inherently limits representational capacity,
often resulting in reduced performance compared to full-rank fine-tuning.
Recent work by Ji et al. (2025) has addressed this limitation by applying a
fixed-frequency sinusoidal transformation to low-rank adapters, increasing
their stable rank without introducing additional parameters. This raises a
crucial question: can the same sine-activated technique be successfully applied
within the context of Post-Training Quantization to retain benefits even after
model compression? In this paper, we investigate this question by extending the
sinusoidal transformation framework to quantized LoRA adapters. We develop a
theoretical analysis showing that the stable rank of a quantized adapter is
tightly linked to that of its full-precision counterpart, motivating the use of
such rank-enhancing functions even under quantization. Our results demonstrate
that the expressivity gains from a sinusoidal non-linearity persist after
quantization, yielding highly compressed adapters with negligible loss in
performance. We validate our approach across a range of fine-tuning tasks for
language, vision and text-to-image generation achieving significant memory
savings while maintaining competitive accuracy.

---


### [SDPO: Importance-Sampled Direct Preference Optimization for Stable Diffusion Training](http://arxiv.org/abs/2505.21893v1)

Preference learning has become a central technique for aligning generative
models with human expectations. Recently, it has been extended to diffusion
models through methods like Direct Preference Optimization (DPO). However,
existing approaches such as Diffusion-DPO suffer from two key challenges:
timestep-dependent instability, caused by a mismatch between the reverse and
forward diffusion processes and by high gradient variance in early noisy
timesteps, and off-policy bias arising from the mismatch between optimization
and data collection policies. We begin by analyzing the reverse diffusion
trajectory and observe that instability primarily occurs at early timesteps
with low importance weights. To address these issues, we first propose
DPO-C\&M, a practical strategy that improves stability by clipping and masking
uninformative timesteps while partially mitigating off-policy bias. Building on
this, we introduce SDPO (Importance-Sampled Direct Preference Optimization), a
principled framework that incorporates importance sampling into the objective
to fully correct for off-policy bias and emphasize informative updates during
the diffusion process. Experiments on CogVideoX-2B, CogVideoX-5B, and
Wan2.1-1.3B demonstrate that both methods outperform standard Diffusion-DPO,
with SDPO achieving superior VBench scores, human preference alignment, and
training robustness. These results highlight the importance of timestep-aware,
distribution-corrected optimization in diffusion-based preference learning.

---


### [SVRPBench: A Realistic Benchmark for Stochastic Vehicle Routing Problem](http://arxiv.org/abs/2505.21887v1)

Robust routing under uncertainty is central to real-world logistics, yet most
benchmarks assume static, idealized settings. We present SVRPBench, the first
open benchmark to capture high-fidelity stochastic dynamics in vehicle routing
at urban scale. Spanning more than 500 instances with up to 1000 customers, it
simulates realistic delivery conditions: time-dependent congestion, log-normal
delays, probabilistic accidents, and empirically grounded time windows for
residential and commercial clients. Our pipeline generates diverse,
constraint-rich scenarios, including multi-depot and multi-vehicle setups.
Benchmarking reveals that state-of-the-art RL solvers like POMO and AM degrade
by over 20% under distributional shift, while classical and metaheuristic
methods remain robust. To enable reproducible research, we release the dataset
and evaluation suite. SVRPBench challenges the community to design solvers that
generalize beyond synthetic assumptions and adapt to real-world uncertainty.

---


### [Evaluating the Retrieval Robustness of Large Language Models](http://arxiv.org/abs/2505.21870v1)

Retrieval-augmented generation (RAG) generally enhances large language
models' (LLMs) ability to solve knowledge-intensive tasks. But RAG may also
lead to performance degradation due to imperfect retrieval and the model's
limited ability to leverage retrieved content. In this work, we evaluate the
robustness of LLMs in practical RAG setups (henceforth retrieval robustness).
We focus on three research questions: (1) whether RAG is always better than
non-RAG; (2) whether more retrieved documents always lead to better
performance; (3) and whether document orders impact results. To facilitate this
study, we establish a benchmark of 1500 open-domain questions, each with
retrieved documents from Wikipedia. We introduce three robustness metrics, each
corresponds to one research question. Our comprehensive experiments, involving
11 LLMs and 3 prompting strategies, reveal that all of these LLMs exhibit
surprisingly high retrieval robustness; nonetheless, different degrees of
imperfect robustness hinders them from fully utilizing the benefits of RAG.

---


### [Beyond Perception: Evaluating Abstract Visual Reasoning through Multi-Stage Task](http://arxiv.org/abs/2505.21850v1)

Current Multimodal Large Language Models (MLLMs) excel in general visual
reasoning but remain underexplored in Abstract Visual Reasoning (AVR), which
demands higher-order reasoning to identify abstract rules beyond simple
perception. Existing AVR benchmarks focus on single-step reasoning, emphasizing
the end result but neglecting the multi-stage nature of reasoning process. Past
studies found MLLMs struggle with these benchmarks, but it doesn't explain how
they fail. To address this gap, we introduce MultiStAR, a Multi-Stage AVR
benchmark, based on RAVEN, designed to assess reasoning across varying levels
of complexity. Additionally, existing metrics like accuracy only focus on the
final outcomes while do not account for the correctness of intermediate steps.
Therefore, we propose a novel metric, MSEval, which considers the correctness
of intermediate steps in addition to the final outcomes. We conduct
comprehensive experiments on MultiStAR using 17 representative close-source and
open-source MLLMs. The results reveal that while existing MLLMs perform
adequately on basic perception tasks, they continue to face challenges in more
complex rule detection stages.

---


### [An Optimistic Algorithm for online CMDPS with Anytime Adversarial Constraints](http://arxiv.org/abs/2505.21841v1)

Online safe reinforcement learning (RL) plays a key role in dynamic
environments, with applications in autonomous driving, robotics, and
cybersecurity. The objective is to learn optimal policies that maximize rewards
while satisfying safety constraints modeled by constrained Markov decision
processes (CMDPs). Existing methods achieve sublinear regret under stochastic
constraints but often fail in adversarial settings, where constraints are
unknown, time-varying, and potentially adversarially designed. In this paper,
we propose the Optimistic Mirror Descent Primal-Dual (OMDPD) algorithm, the
first to address online CMDPs with anytime adversarial constraints. OMDPD
achieves optimal regret O(sqrt(K)) and strong constraint violation O(sqrt(K))
without relying on Slater's condition or the existence of a strictly known safe
policy. We further show that access to accurate estimates of rewards and
transitions can further improve these bounds. Our results offer practical
guarantees for safe decision-making in adversarial environments.

---


### [AutoL2S: Auto Long-Short Reasoning for Efficient Large Language Models](http://arxiv.org/abs/2505.22662v1)

The reasoning-capable large language models (LLMs) demonstrate strong
performance on complex reasoning tasks but often suffer from overthinking,
generating unnecessarily long chain-of-thought (CoT) reasoning paths for easy
reasoning questions, thereby increasing inference cost and latency. Recent
approaches attempt to address this challenge by manually deciding when to apply
long or short reasoning. However, they lack the flexibility to adapt CoT length
dynamically based on question complexity. In this paper, we propose Auto
Long-Short Reasoning (AutoL2S), a dynamic and model-agnostic framework that
enables LLMs to dynamically compress their generated reasoning path based on
the complexity of the reasoning question. AutoL2S enables a learned paradigm,
in which LLMs themselves can decide when longer reasoning is necessary and when
shorter reasoning suffices, by training on data annotated with our proposed
method, which includes both long and short CoT paths and a special <EASY>
token. We then use <EASY> token to indicate when the model can skip generating
lengthy CoT reasoning. This proposed annotation strategy can enhance the LLMs'
ability to generate shorter CoT reasoning paths with improved quality after
training. Extensive evaluation results show that AutoL2S reduces the length of
reasoning generation by up to 57% without compromising performance,
demonstrating the effectiveness of AutoL2S for scalable and efficient LLM
reasoning.

---


### [The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason](http://arxiv.org/abs/2505.22653v1)

Recent studies on post-training large language models (LLMs) for reasoning
through reinforcement learning (RL) typically focus on tasks that can be
accurately verified and rewarded, such as solving math problems. In contrast,
our research investigates the impact of reward noise, a more practical
consideration for real-world scenarios involving the post-training of LLMs
using reward models. We found that LLMs demonstrate strong robustness to
substantial reward noise. For example, manually flipping 40% of the reward
function's outputs in math tasks still allows a Qwen-2.5-7B model to achieve
rapid convergence, improving its performance on math tasks from 5% to 72%,
compared to the 75% accuracy achieved by a model trained with noiseless
rewards. Surprisingly, by only rewarding the appearance of key reasoning
phrases (namely reasoning pattern reward, RPR), such as ``first, I need
to''-without verifying the correctness of answers, the model achieved peak
downstream performance (over 70% accuracy for Qwen-2.5-7B) comparable to models
trained with strict correctness verification and accurate rewards. Recognizing
the importance of the reasoning process over the final results, we combined RPR
with noisy reward models. RPR helped calibrate the noisy reward models,
mitigating potential false negatives and enhancing the LLM's performance on
open-ended tasks. These findings suggest the importance of improving models'
foundational abilities during the pre-training phase while providing insights
for advancing post-training techniques. Our code and scripts are available at
https://github.com/trestad/Noisy-Rewards-in-Learning-to-Reason.

---


### [Sherlock: Self-Correcting Reasoning in Vision-Language Models](http://arxiv.org/abs/2505.22651v1)

Reasoning Vision-Language Models (VLMs) have shown promising performance on
complex multimodal tasks. However, they still face significant challenges: they
are highly sensitive to reasoning errors, require large volumes of annotated
data or accurate verifiers, and struggle to generalize beyond specific domains.
To address these limitations, we explore self-correction as a strategy to
enhance reasoning VLMs. We first conduct an in-depth analysis of reasoning
VLMs' self-correction abilities and identify key gaps. Based on our findings,
we introduce Sherlock, a self-correction and self-improvement training
framework. Sherlock introduces a trajectory-level self-correction objective, a
preference data construction method based on visual perturbation, and a dynamic
$\beta$ for preference tuning. Once the model acquires self-correction
capabilities using only 20k randomly sampled annotated data, it continues to
self-improve without external supervision. Built on the Llama3.2-Vision-11B
model, Sherlock achieves remarkable results across eight benchmarks, reaching
an average accuracy of 64.1 with direct generation and 65.4 after
self-correction. It outperforms LLaVA-CoT (63.2), Mulberry (63.9), and
LlamaV-o1 (63.4) while using less than 20% of the annotated data.

---


### [Precise In-Parameter Concept Erasure in Large Language Models](http://arxiv.org/abs/2505.22586v1)

Large language models (LLMs) often acquire knowledge during pretraining that
is undesirable in downstream deployments, e.g., sensitive information or
copyrighted content. Existing approaches for removing such knowledge rely on
fine-tuning, training low-rank adapters or fact-level editing, but these are
either too coarse, too shallow, or ineffective. In this work, we propose PISCES
(Precise In-parameter Suppression for Concept EraSure), a novel framework for
precisely erasing entire concepts from model parameters by directly editing
directions that encode them in parameter space. PISCES uses a disentangler
model to decompose MLP vectors into interpretable features, identifies those
associated with a target concept using automated interpretability techniques,
and removes them from model parameters. Experiments on Gemma 2 and Llama 3.1
over various concepts show that PISCES achieves modest gains in efficacy over
leading erasure methods, reducing accuracy on the target concept to as low as
7.7%, while dramatically improving erasure specificity (by up to 31%) and
robustness (by up to 38%). Overall, these results demonstrate that
feature-based in-parameter editing enables a more precise and reliable approach
for removing conceptual knowledge in language models.

---


### [Multi-MLLM Knowledge Distillation for Out-of-Context News Detection](http://arxiv.org/abs/2505.22517v1)

Multimodal out-of-context news is a type of misinformation in which the image
is used outside of its original context. Many existing works have leveraged
multimodal large language models (MLLMs) for detecting out-of-context news.
However, observing the limited zero-shot performance of smaller MLLMs, they
generally require label-rich fine-tuning and/or expensive API calls to GPT
models to improve the performance, which is impractical in low-resource
scenarios. In contrast, we aim to improve the performance of small MLLMs in a
more label-efficient and cost-effective manner. To this end, we first prompt
multiple teacher MLLMs to generate both label predictions and corresponding
rationales, which collectively serve as the teachers' knowledge. We then
introduce a two-stage knowledge distillation framework to transfer this
knowledge to a student MLLM. In Stage 1, we apply LoRA fine-tuning to the
student model using all training data. In Stage 2, we further fine-tune the
student model using both LoRA fine-tuning and DPO on the data points where
teachers' predictions conflict. This two-stage strategy reduces annotation
costs and helps the student model uncover subtle patterns in more challenging
cases. Experimental results demonstrate that our approach achieves
state-of-the-art performance using less than 10% labeled data.

---


### [EvolveSearch: An Iterative Self-Evolving Search Agent](http://arxiv.org/abs/2505.22501v1)

The rapid advancement of large language models (LLMs) has transformed the
landscape of agentic information seeking capabilities through the integration
of tools such as search engines and web browsers. However, current mainstream
approaches for enabling LLM web search proficiency face significant challenges:
supervised fine-tuning struggles with data production in open-search domains,
while RL converges quickly, limiting their data utilization efficiency. To
address these issues, we propose EvolveSearch, a novel iterative self-evolution
framework that combines SFT and RL to enhance agentic web search capabilities
without any external human-annotated reasoning data. Extensive experiments on
seven multi-hop question-answering (MHQA) benchmarks demonstrate that
EvolveSearch consistently improves performance across iterations, ultimately
achieving an average improvement of 4.7\% over the current state-of-the-art
across seven benchmarks, opening the door to self-evolution agentic
capabilities in open web search domains.

---


### [RAG-Zeval: Towards Robust and Interpretable Evaluation on RAG Responses through End-to-End Rule-Guided Reasoning](http://arxiv.org/abs/2505.22430v1)

Robust evaluation is critical for deploying trustworthy retrieval-augmented
generation (RAG) systems. However, current LLM-based evaluation frameworks
predominantly rely on directly prompting resource-intensive models with complex
multi-stage prompts, underutilizing models' reasoning capabilities and
introducing significant computational cost. In this paper, we present RAG-Zeval
(RAG-Zero Evaluator), a novel end-to-end framework that formulates faithfulness
and correctness evaluation as a rule-guided reasoning task. Our approach trains
evaluators with reinforcement learning, facilitating compact models to generate
comprehensive and sound assessments with detailed explanation in one-pass. We
introduce a ranking-based outcome reward mechanism, using preference judgments
rather than absolute scores, to address the challenge of obtaining precise
pointwise reward signals. To this end, we synthesize the ranking references by
generating quality-controlled responses with zero human annotation. Experiments
demonstrate RAG-Zeval's superior performance, achieving the strongest
correlation with human judgments and outperforming baselines that rely on LLMs
with 10-100 times more parameters. Our approach also exhibits superior
interpretability in response evaluation.

---


### [Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition](http://arxiv.org/abs/2505.22375v1)

This work presents Pangu Embedded, an efficient Large Language Model (LLM)
reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible
fast and slow thinking capabilities. Pangu Embedded addresses the significant
computational costs and inference latency challenges prevalent in existing
reasoning-optimized LLMs. We propose a two-stage training framework for its
construction. In Stage 1, the model is finetuned via an iterative distillation
process, incorporating inter-iteration model merging to effectively aggregate
complementary knowledge. This is followed by reinforcement learning on Ascend
clusters, optimized by a latency-tolerant scheduler that combines stale
synchronous parallelism with prioritized data queues. The RL process is guided
by a Multi-source Adaptive Reward System (MARS), which generates dynamic,
task-specific reward signals using deterministic metrics and lightweight LLM
evaluators for mathematics, coding, and general problem-solving tasks. Stage 2
introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode
for routine queries and a deeper "slow" mode for complex inference. This
framework offers both manual mode switching for user control and an automatic,
complexity-aware mode selection mechanism that dynamically allocates
computational resources to balance latency and reasoning depth. Experimental
results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate
that Pangu Embedded with 7B parameters, outperforms similar-size models like
Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art
reasoning quality within a single, unified model architecture, highlighting a
promising direction for developing powerful yet practically deployable LLM
reasoners.

---


### [LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High](http://arxiv.org/abs/2505.22354v1)

This paper examines how LLMs handle false presuppositions and whether certain
linguistic factors influence their responses to falsely presupposed content.
Presuppositions subtly introduce information as given, making them highly
effective at embedding disputable or false information. This raises concerns
about whether LLMs, like humans, may fail to detect and correct misleading
assumptions introduced as false presuppositions, even when the stakes of
misinformation are high. Using a systematic approach based on linguistic
presupposition analysis, we investigate the conditions under which LLMs are
more or less sensitive to adopt or reject false presuppositions. Focusing on
political contexts, we examine how factors like linguistic construction,
political party, and scenario probability impact the recognition of false
presuppositions. We conduct experiments with a newly created dataset and
examine three LLMs: OpenAI's GPT-4-o, Meta's LLama-3-8B, and MistralAI's
Mistral-7B-v03. Our results show that the models struggle to recognize false
presuppositions, with performance varying by condition. This study highlights
that linguistic presupposition analysis is a valuable tool for uncovering the
reinforcement of political misinformation in LLM responses.

---


### [Compensating for Data with Reasoning: Low-Resource Machine Translation with LLMs](http://arxiv.org/abs/2505.22293v1)

Large Language Models (LLMs) have demonstrated strong capabilities in
multilingual machine translation, sometimes even outperforming traditional
neural systems. However, previous research has highlighted the challenges of
using LLMs, particularly with prompt engineering, for low-resource languages.
In this work, we introduce Fragment-Shot Prompting, a novel in-context learning
method that segments input and retrieves translation examples based on
syntactic coverage, along with Pivoted Fragment-Shot, an extension that enables
translation without direct parallel data. We evaluate these methods using
GPT-3.5, GPT-4o, o1-mini, LLaMA-3.3, and DeepSeek-R1 for translation between
Italian and two Ladin variants, revealing three key findings: (1) Fragment-Shot
Prompting is effective for translating into and between the studied
low-resource languages, with syntactic coverage positively correlating with
translation quality; (2) Models with stronger reasoning abilities make more
effective use of retrieved knowledge, generally produce better translations,
and enable Pivoted Fragment-Shot to significantly improve translation quality
between the Ladin variants; and (3) prompt engineering offers limited, if any,
improvements when translating from a low-resource to a high-resource language,
where zero-shot prompting already yields satisfactory results. We publicly
release our code and the retrieval corpora.

---


### [BioHopR: A Benchmark for Multi-Hop, Multi-Answer Reasoning in Biomedical Domain](http://arxiv.org/abs/2505.22240v1)

Biomedical reasoning often requires traversing interconnected relationships
across entities such as drugs, diseases, and proteins. Despite the increasing
prominence of large language models (LLMs), existing benchmarks lack the
ability to evaluate multi-hop reasoning in the biomedical domain, particularly
for queries involving one-to-many and many-to-many relationships. This gap
leaves the critical challenges of biomedical multi-hop reasoning underexplored.
To address this, we introduce BioHopR, a novel benchmark designed to evaluate
multi-hop, multi-answer reasoning in structured biomedical knowledge graphs.
Built from the comprehensive PrimeKG, BioHopR includes 1-hop and 2-hop
reasoning tasks that reflect real-world biomedical complexities.
  Evaluations of state-of-the-art models reveal that O3-mini, a proprietary
reasoning-focused model, achieves 37.93% precision on 1-hop tasks and 14.57% on
2-hop tasks, outperforming proprietary models such as GPT4O and open-source
biomedical models including HuatuoGPT-o1-70B and Llama-3.3-70B. However, all
models exhibit significant declines in multi-hop performance, underscoring the
challenges of resolving implicit reasoning steps in the biomedical domain. By
addressing the lack of benchmarks for multi-hop reasoning in biomedical domain,
BioHopR sets a new standard for evaluating reasoning capabilities and
highlights critical gaps between proprietary and open-source models while
paving the way for future advancements in biomedical LLMs.

---


### [Reverse Preference Optimization for Complex Instruction Following](http://arxiv.org/abs/2505.22172v1)

Instruction following (IF) is a critical capability for large language models
(LLMs). However, handling complex instructions with multiple constraints
remains challenging. Previous methods typically select preference pairs based
on the number of constraints they satisfy, introducing noise where chosen
examples may fail to follow some constraints and rejected examples may excel in
certain respects over the chosen ones. To address the challenge of aligning
with multiple preferences, we propose a simple yet effective method called
Reverse Preference Optimization (RPO). It mitigates noise in preference pairs
by dynamically reversing the constraints within the instruction to ensure the
chosen response is perfect, alleviating the burden of extensive sampling and
filtering to collect perfect responses. Besides, reversal also enlarges the gap
between chosen and rejected responses, thereby clarifying the optimization
direction and making it more robust to noise. We evaluate RPO on two multi-turn
IF benchmarks, Sysbench and Multi-IF, demonstrating average improvements over
the DPO baseline of 4.6 and 2.5 points (on Llama-3.1 8B), respectively.
Moreover, RPO scales effectively across model sizes (8B to 70B parameters),
with the 70B RPO model surpassing GPT-4o.

---


### [ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments](http://arxiv.org/abs/2505.22169v1)

LLMs are highly sensitive to prompt phrasing, yet standard benchmarks
typically report performance using a single prompt, raising concerns about the
reliability of such evaluations. In this work, we argue for a stochastic method
of moments evaluation over the space of meaning-preserving prompt
perturbations. We introduce a formal definition of reliable evaluation that
accounts for prompt sensitivity, and suggest ReliableEval - a method for
estimating the number of prompt resamplings needed to obtain meaningful
results. Using our framework, we stochastically evaluate five frontier LLMs and
find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit
substantial prompt sensitivity. Our approach is model-, task-, and
metric-agnostic, offering a recipe for meaningful and robust LLM evaluation.

---


### [InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing](http://arxiv.org/abs/2505.22156v1)

Although existing model editing methods perform well in recalling exact edit
facts, they often struggle in complex scenarios that require deeper semantic
understanding rather than mere knowledge regurgitation. Leveraging the strong
contextual reasoning abilities of large language models (LLMs), in-context
learning (ICL) becomes a promising editing method by comprehending edit
information through context encoding. However, this method is constrained by
the limited context window of LLMs, leading to degraded performance and
efficiency as the number of edits increases. To overcome this limitation, we
propose InComeS, a flexible framework that enhances LLMs' ability to process
editing contexts through explicit compression and selection mechanisms.
Specifically, InComeS compresses each editing context into the key-value (KV)
cache of a special gist token, enabling efficient handling of multiple edits
without being restricted by the model's context window. Furthermore,
specialized cross-attention modules are added to dynamically select the most
relevant information from the gist pools, enabling adaptive and effective
utilization of edit information. We conduct experiments on diverse model
editing benchmarks with various editing formats, and the results demonstrate
the effectiveness and efficiency of our method.

---


### [LoKI: Low-damage Knowledge Implanting of Large Language Models](http://arxiv.org/abs/2505.22120v1)

Fine-tuning adapts pretrained models for specific tasks but poses the risk of
catastrophic forgetting (CF), where critical knowledge from pre-training is
overwritten. Current Parameter-Efficient Fine-Tuning (PEFT) methods for Large
Language Models (LLMs), while efficient, often sacrifice general capabilities.
To address the issue of CF in a general-purpose PEFT framework, we propose
\textbf{Lo}w-damage \textbf{K}nowledge \textbf{I}mplanting (\textbf{LoKI}), a
PEFT technique that is based on a mechanistic understanding of how knowledge is
stored in transformer architectures. In two real-world scenarios, LoKI
demonstrates task-specific performance that is comparable to or even surpasses
that of full fine-tuning and LoRA-based methods across various model types,
while significantly better preserving general capabilities. Our work connects
mechanistic insights into LLM knowledge storage with practical fine-tuning
objectives, achieving state-of-the-art trade-offs between task specialization
and the preservation of general capabilities. Our implementation is publicly
available as ready-to-use code\footnote{https://github.com/Nexround/LoKI}.

---


### [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](http://arxiv.org/abs/2505.22101v1)

Large Language Models (LLMs) have emerged as foundational infrastructure in
the pursuit of Artificial General Intelligence (AGI). Despite their remarkable
capabilities in language perception and generation, current LLMs fundamentally
lack a unified and structured architecture for handling memory. They primarily
rely on parametric memory (knowledge encoded in model weights) and ephemeral
activation memory (context-limited runtime states). While emerging methods like
Retrieval-Augmented Generation (RAG) incorporate plaintext memory, they lack
lifecycle management and multi-modal integration, limiting their capacity for
long-term knowledge evolution. To address this, we introduce MemOS, a memory
operating system designed for LLMs that, for the first time, elevates memory to
a first-class operational resource. It builds unified mechanisms for
representation, organization, and governance across three core memory types:
parametric, activation, and plaintext. At its core is the MemCube, a
standardized memory abstraction that enables tracking, fusion, and migration of
heterogeneous memory, while offering structured, traceable access across tasks
and contexts. MemOS establishes a memory-centric execution framework with
strong controllability, adaptability, and evolvability. It fills a critical gap
in current LLM infrastructure and lays the groundwork for continual adaptation,
personalized intelligence, and cross-platform coordination in next-generation
intelligent systems.

---


### [Learning to Route Queries Across Knowledge Bases for Step-wise Retrieval-Augmented Reasoning](http://arxiv.org/abs/2505.22095v1)

Multimodal Retrieval-Augmented Generation (MRAG) has shown promise in
mitigating hallucinations in Multimodal Large Language Models (MLLMs) by
incorporating external knowledge during generation. Existing MRAG methods
typically adopt a static retrieval pipeline that fetches relevant information
from multiple Knowledge Bases (KBs), followed by a refinement step. However,
these approaches overlook the reasoning and planning capabilities of MLLMs to
dynamically determine how to interact with different KBs during the reasoning
process. To address this limitation, we propose R1-Router, a novel MRAG
framework that learns to decide when and where to retrieve knowledge based on
the evolving reasoning state. Specifically, R1-Router can generate follow-up
queries according to the current reasoning step, routing these intermediate
queries to the most suitable KB, and integrating external knowledge into a
coherent reasoning trajectory to answer the original query. Furthermore, we
introduce Step-wise Group Relative Policy Optimization (Step-GRPO), a tailored
reinforcement learning algorithm that assigns step-specific rewards to optimize
the reasoning behavior of MLLMs. Experimental results on various open-domain QA
benchmarks across multiple modalities demonstrate that R1-Router outperforms
baseline models by over 7%. Further analysis shows that R1-Router can
adaptively and effectively leverage diverse KBs, reducing unnecessary
retrievals and improving both efficiency and accuracy.

---


### [Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?](http://arxiv.org/abs/2505.22061v1)

Retrieval-augmented generation (RAG) mitigates the hallucination problem in
large language models (LLMs) and has proven effective for specific,
personalized applications. However, passing private retrieved documents
directly to LLMs introduces vulnerability to membership inference attacks
(MIAs), which try to determine whether the target datum exists in the private
external database or not. Based on the insight that MIA queries typically
exhibit high similarity to only one target document, we introduce Mirabel, a
similarity-based MIA detection framework designed for the RAG system. With the
proposed Mirabel, we show that simple detect-and-hide strategies can
successfully obfuscate attackers, maintain data utility, and remain
system-agnostic. We experimentally prove its detection and defense against
various state-of-the-art MIA methods and its adaptability to existing private
RAG systems.

---


### [Leveraging Interview-Informed LLMs to Model Survey Responses: Comparative Insights from AI-Generated and Human Data](http://arxiv.org/abs/2505.21997v1)

Mixed methods research integrates quantitative and qualitative data but faces
challenges in aligning their distinct structures, particularly in examining
measurement characteristics and individual response patterns. Advances in large
language models (LLMs) offer promising solutions by generating synthetic survey
responses informed by qualitative data. This study investigates whether LLMs,
guided by personal interviews, can reliably predict human survey responses,
using the Behavioral Regulations in Exercise Questionnaire (BREQ) and
interviews from after-school program staff as a case study. Results indicate
that LLMs capture overall response patterns but exhibit lower variability than
humans. Incorporating interview data improves response diversity for some
models (e.g., Claude, GPT), while well-crafted prompts and low-temperature
settings enhance alignment between LLM and human responses. Demographic
information had less impact than interview content on alignment accuracy. These
findings underscore the potential of interview-informed LLMs to bridge
qualitative and quantitative methodologies while revealing limitations in
response variability, emotional interpretation, and psychometric fidelity.
Future research should refine prompt design, explore bias mitigation, and
optimize model settings to enhance the validity of LLM-generated survey data in
social science research.

---


### [RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](http://arxiv.org/abs/2505.21940v1)

Large Language Models (LLMs) excel in many areas but continue to face
challenges with complex reasoning tasks, such as Multi-Hop Question Answering
(MHQA). MHQA requires integrating evidence from diverse sources while managing
intricate logical dependencies, often leads to errors in reasoning.
Retrieval-Augmented Generation (RAG), widely employed in MHQA tasks, faces
challenges in effectively filtering noisy data and retrieving all necessary
evidence, thereby limiting its effectiveness in addressing MHQA challenges. To
address these challenges, we propose RISE:Reasoning Enhancement via Iterative
Self-Exploration, a novel framework designed to enhance models' reasoning
capability through iterative self-exploration. Specifically, RISE involves
three key steps in addressing MHQA tasks: question decomposition,
retrieve-then-read, and self-critique. By leveraging continuous
self-exploration, RISE identifies accurate reasoning paths, iteratively
self-improving the model's capability to integrate evidence, maintain logical
consistency, and enhance performance in MHQA tasks. Extensive experiments on
multiple MHQA benchmarks demonstrate that RISE significantly improves reasoning
accuracy and task performance.

---


### [RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments](http://arxiv.org/abs/2505.21936v1)

Computer-use agents (CUAs) promise to automate complex tasks across operating
systems (OS) and the web, but remain vulnerable to indirect prompt injection.
Current evaluations of this threat either lack support realistic but controlled
environments or ignore hybrid web-OS attack scenarios involving both
interfaces. To address this, we propose RedTeamCUA, an adversarial testing
framework featuring a novel hybrid sandbox that integrates a VM-based OS
environment with Docker-based web platforms. Our sandbox supports key features
tailored for red teaming, such as flexible adversarial scenario configuration,
and a setting that decouples adversarial evaluation from navigational
limitations of CUAs by initializing tests directly at the point of an
adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive
benchmark with 864 examples that investigate realistic, hybrid web-OS attack
scenarios and fundamental security vulnerabilities. Benchmarking current
frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA
demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated,
still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute
adversarial tasks with an Attempt Rate as high as 92.5%, although failing to
complete them due to capability limitations. Nevertheless, we observe
concerning ASRs of up to 50% in realistic end-to-end settings, with the
recently released frontier Claude 4 Opus | CUA showing an alarming ASR of 48%,
demonstrating that indirect prompt injection presents tangible risks for even
advanced CUAs despite their capabilities and safeguards. Overall, RedTeamCUA
provides an essential framework for advancing realistic, controlled, and
systematic analysis of CUA vulnerabilities, highlighting the urgent need for
robust defenses to indirect prompt injection prior to real-world deployment.

---


### [Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](http://arxiv.org/abs/2505.21930v1)

This paper develops an ensemble method for fine-tuning a language model to
multiple datasets. Existing methods, such as quantized LoRA (QLoRA), are
efficient when adapting to a single dataset. When training on multiple datasets
of different tasks, a common setup in practice, it remains unclear how to
design an efficient adaptation for fine-tuning language models. We propose to
use an ensemble of multiple smaller adapters instead of a single adapter per
task. We design an efficient algorithm that partitions $n$ datasets into $m$
groups, where $m$ is typically much smaller than $n$ in practice, and train one
adapter for each group before taking a weighted combination to form the
ensemble. The algorithm leverages a first-order approximation property of
low-rank adaptation to quickly obtain the fine-tuning performances of dataset
combinations since methods like LoRA stay close to the base model. Hence, we
use the gradients of the base model to estimate its behavior during
fine-tuning. Empirically, this approximation holds with less than $1\%$ error
on models with up to $34$ billion parameters, leading to an estimation of true
fine-tuning performances under $5\%$ error while speeding up computation
compared to base fine-tuning by $105$ times. When applied to fine-tune Llama
and GPT models on ten text classification tasks, our approach provides up to
$10\%$ higher average test accuracy over QLoRA, with only $9\%$ more FLOPs. On
a Llama model with $34$ billion parameters, an ensemble of QLoRA increases test
accuracy by $3\%$ compared to QLoRA, with only $8\%$ more FLOPs.

---


### [GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](http://arxiv.org/abs/2505.21863v1)

Publicly significant images from events hold valuable contextual information,
crucial for journalism and education. However, existing methods often struggle
to extract this relevance accurately. To address this, we introduce GETReason
(Geospatial Event Temporal Reasoning), a framework that moves beyond
surface-level image descriptions to infer deeper contextual meaning. We propose
that extracting global event, temporal, and geospatial information enhances
understanding of an image's significance. Additionally, we introduce GREAT
(Geospatial Reasoning and Event Accuracy with Temporal Alignment), a new metric
for evaluating reasoning-based image understanding. Our layered multi-agent
approach, assessed using a reasoning-weighted metric, demonstrates that
meaningful insights can be inferred, effectively linking images to their
broader event context.

---


### [Zero-Shot Vision Encoder Grafting via LLM Surrogates](http://arxiv.org/abs/2505.22664v1)

Vision language models (VLMs) typically pair a modestly sized vision encoder
with a large language model (LLM), e.g., Llama-70B, making the decoder the
primary computational burden during training. To reduce costs, a potential
promising strategy is to first train the vision encoder using a small language
model before transferring it to the large one. We construct small "surrogate
models" that share the same embedding space and representation language as the
large target LLM by directly inheriting its shallow layers. Vision encoders
trained on the surrogate can then be directly transferred to the larger model,
a process we call zero-shot grafting -- when plugged directly into the
full-size target LLM, the grafted pair surpasses the encoder-surrogate pair
and, on some benchmarks, even performs on par with full decoder training with
the target LLM. Furthermore, our surrogate training approach reduces overall
VLM training costs by ~45% when using Llama-70B as the decoder.

---


### [Training Free Stylized Abstraction](http://arxiv.org/abs/2505.22663v1)

Stylized abstraction synthesizes visually exaggerated yet semantically
faithful representations of subjects, balancing recognizability with perceptual
distortion. Unlike image-to-image translation, which prioritizes structural
fidelity, stylized abstraction demands selective retention of identity cues
while embracing stylistic divergence, especially challenging for
out-of-distribution individuals. We propose a training-free framework that
generates stylized abstractions from a single image using inference-time
scaling in vision-language models (VLLMs) to extract identity-relevant
features, and a novel cross-domain rectified flow inversion strategy that
reconstructs structure based on style-dependent priors. Our method adapts
structural restoration dynamically through style-aware temporal scheduling,
enabling high-fidelity reconstructions that honor both subject and style. It
supports multi-round abstraction-aware generation without fine-tuning. To
evaluate this task, we introduce StyleBench, a GPT-based human-aligned metric
suited for abstract styles where pixel-level similarity fails. Experiments
across diverse abstraction (e.g., LEGO, knitted dolls, South Park) show strong
generalization to unseen identities and styles in a fully open-source setup.

---


### [Universal Domain Adaptation for Semantic Segmentation](http://arxiv.org/abs/2505.22458v1)

Unsupervised domain adaptation for semantic segmentation (UDA-SS) aims to
transfer knowledge from labeled source data to unlabeled target data. However,
traditional UDA-SS methods assume that category settings between source and
target domains are known, which is unrealistic in real-world scenarios. This
leads to performance degradation if private classes exist. To address this
limitation, we propose Universal Domain Adaptation for Semantic Segmentation
(UniDA-SS), achieving robust adaptation even without prior knowledge of
category settings. We define the problem in the UniDA-SS scenario as low
confidence scores of common classes in the target domain, which leads to
confusion with private classes. To solve this problem, we propose UniMAP:
UniDA-SS with Image Matching and Prototype-based Distinction, a novel framework
composed of two key components. First, Domain-Specific Prototype-based
Distinction (DSPD) divides each class into two domain-specific prototypes,
enabling finer separation of domain-specific features and enhancing the
identification of common classes across domains. Second, Target-based Image
Matching (TIM) selects a source image containing the most common-class pixels
based on the target pseudo-label and pairs it in a batch to promote effective
learning of common classes. We also introduce a new UniDA-SS benchmark and
demonstrate through various experiments that UniMAP significantly outperforms
baselines. The code is available at
\href{https://github.com/KU-VGI/UniMAP}{this https URL}.

---


### [Zero-Shot 3D Visual Grounding from Vision-Language Models](http://arxiv.org/abs/2505.22429v1)

3D Visual Grounding (3DVG) seeks to locate target objects in 3D scenes using
natural language descriptions, enabling downstream applications such as
augmented reality and robotics. Existing approaches typically rely on labeled
3D data and predefined categories, limiting scalability to open-world settings.
We present SeeGround, a zero-shot 3DVG framework that leverages 2D
Vision-Language Models (VLMs) to bypass the need for 3D-specific training. To
bridge the modality gap, we introduce a hybrid input format that pairs
query-aligned rendered views with spatially enriched textual descriptions. Our
framework incorporates two core components: a Perspective Adaptation Module
that dynamically selects optimal viewpoints based on the query, and a Fusion
Alignment Module that integrates visual and spatial signals to enhance
localization precision. Extensive evaluations on ScanRefer and Nr3D confirm
that SeeGround achieves substantial improvements over existing zero-shot
baselines -- outperforming them by 7.7% and 7.1%, respectively -- and even
rivals fully supervised alternatives, demonstrating strong generalization under
challenging conditions.

---


### [Self-Reflective Reinforcement Learning for Diffusion-based Image Reasoning Generation](http://arxiv.org/abs/2505.22407v1)

Diffusion models have recently demonstrated exceptional performance in image
generation task. However, existing image generation methods still significantly
suffer from the dilemma of image reasoning, especially in logic-centered image
generation tasks. Inspired by the success of Chain of Thought (CoT) and
Reinforcement Learning (RL) in LLMs, we propose SRRL, a self-reflective RL
algorithm for diffusion models to achieve reasoning generation of logical
images by performing reflection and iteration across generation trajectories.
The intermediate samples in the denoising process carry noise, making accurate
reward evaluation difficult. To address this challenge, SRRL treats the entire
denoising trajectory as a CoT step with multi-round reflective denoising
process and introduces condition guided forward process, which allows for
reflective iteration between CoT steps. Through SRRL-based iterative diffusion
training, we introduce image reasoning through CoT into generation tasks
adhering to physical laws and unconventional physical phenomena for the first
time. Notably, experimental results of case study exhibit that the superior
performance of our SRRL algorithm even compared with GPT-4o. The project page
is https://jadenpan0.github.io/srrl.github.io/.

---


### [Zooming from Context to Cue: Hierarchical Preference Optimization for Multi-Image MLLMs](http://arxiv.org/abs/2505.22396v1)

Multi-modal Large Language Models (MLLMs) excel at single-image tasks but
struggle with multi-image understanding due to cross-modal misalignment,
leading to hallucinations (context omission, conflation, and
misinterpretation). Existing methods using Direct Preference Optimization (DPO)
constrain optimization to a solitary image reference within the input sequence,
neglecting holistic context modeling. We propose Context-to-Cue Direct
Preference Optimization (CcDPO), a multi-level preference optimization
framework that enhances per-image perception in multi-image settings by zooming
into visual clues -- from sequential context to local details. It features: (i)
Context-Level Optimization : Re-evaluates cognitive biases underlying MLLMs'
multi-image context comprehension and integrates a spectrum of low-cost global
sequence preferences for bias mitigation. (ii) Needle-Level Optimization :
Directs attention to fine-grained visual details through region-targeted visual
prompts and multimodal preference supervision. To support scalable
optimization, we also construct MultiScope-42k, an automatically generated
dataset with high-quality multi-level preference pairs. Experiments show that
CcDPO significantly reduces hallucinations and yields consistent performance
gains across general single- and multi-image tasks.

---


### [YH-MINER: Multimodal Intelligent System for Natural Ecological Reef Metric Extraction](http://arxiv.org/abs/2505.22250v1)

Coral reefs, crucial for sustaining marine biodiversity and ecological
processes (e.g., nutrient cycling, habitat provision), face escalating threats,
underscoring the need for efficient monitoring. Coral reef ecological
monitoring faces dual challenges of low efficiency in manual analysis and
insufficient segmentation accuracy in complex underwater scenarios. This study
develops the YH-OSI system, establishing an intelligent framework centered on
the Multimodal Large Model (MLLM) for "object detection-semantic
segmentation-prior input". The system uses the object detection module
(mAP@0.5=0.78) to generate spatial prior boxes for coral instances, driving the
segment module to complete pixel-level segmentation in low-light and densely
occluded scenarios. The segmentation masks and finetuned classification
instructions are fed into the Qwen2-VL-based multimodal model as prior inputs,
achieving a genus-level classification accuracy of 88% and simultaneously
extracting core ecological metrics. Meanwhile, the system retains the
scalability of the multimodal model through standardized interfaces, laying a
foundation for future integration into multimodal agent-based underwater robots
and supporting the full-process automation of "image acquisition-prior
generation-real-time analysis."

---


### [On the Transferability and Discriminability of Repersentation Learning in Unsupervised Domain Adaptation](http://arxiv.org/abs/2505.22099v1)

In this paper, we addressed the limitation of relying solely on distribution
alignment and source-domain empirical risk minimization in Unsupervised Domain
Adaptation (UDA). Our information-theoretic analysis showed that this standard
adversarial-based framework neglects the discriminability of target-domain
features, leading to suboptimal performance. To bridge this
theoretical-practical gap, we defined "good representation learning" as
guaranteeing both transferability and discriminability, and proved that an
additional loss term targeting target-domain discriminability is necessary.
Building on these insights, we proposed a novel adversarial-based UDA framework
that explicitly integrates a domain alignment objective with a
discriminability-enhancing constraint. Instantiated as Domain-Invariant
Representation Learning with Global and Local Consistency (RLGLC), our method
leverages Asymmetrically-Relaxed Wasserstein of Wasserstein Distance (AR-WWD)
to address class imbalance and semantic dimension weighting, and employs a
local consistency mechanism to preserve fine-grained target-domain
discriminative information. Extensive experiments across multiple benchmark
datasets demonstrate that RLGLC consistently surpasses state-of-the-art
methods, confirming the value of our theoretical perspective and underscoring
the necessity of enforcing both transferability and discriminability in
adversarial-based UDA.

---


### [Fast Feature Matching of UAV Images via Matrix Band Reduction-based GPU Data Schedule](http://arxiv.org/abs/2505.22089v1)

Feature matching dominats the time costs in structure from motion (SfM). The
primary contribution of this study is a GPU data schedule algorithm for
efficient feature matching of Unmanned aerial vehicle (UAV) images. The core
idea is to divide the whole dataset into blocks based on the matrix band
reduction (MBR) and achieve efficient feature matching via GPU-accelerated
cascade hashing. First, match pairs are selected by using an image retrieval
technique, which converts images into global descriptors and searches
high-dimension nearest neighbors with graph indexing. Second, compact image
blocks are iteratively generated from a MBR-based data schedule strategy, which
exploits image connections to avoid redundant data IO (input/output) burden and
increases the usage of GPU computing power. Third, guided by the generated
image blocks, feature matching is executed sequentially within the framework of
GPU-accelerated cascade hashing, and initial candidate matches are refined by
combining a local geometric constraint and RANSAC-based global verification.
For further performance improvement, these two seps are designed to execute
parallelly in GPU and CPU. Finally, the performance of the proposed solution is
evaluated by using large-scale UAV datasets. The results demonstrate that it
increases the efficiency of feature matching with speedup ratios ranging from
77.0 to 100.0 compared with KD-Tree based matching methods, and achieves
comparable accuracy in relative and absolute bundle adjustment (BA). The
proposed algorithm is an efficient solution for feature matching of UAV images.

---


### [OmniAD: Detect and Understand Industrial Anomaly via Multimodal Reasoning](http://arxiv.org/abs/2505.22039v1)

While anomaly detection has made significant progress, generating detailed
analyses that incorporate industrial knowledge remains a challenge. To address
this gap, we introduce OmniAD, a novel framework that unifies anomaly detection
and understanding for fine-grained analysis. OmniAD is a multimodal reasoner
that combines visual and textual reasoning processes. The visual reasoning
provides detailed inspection by leveraging Text-as-Mask Encoding to perform
anomaly detection through text generation without manually selected thresholds.
Following this, Visual Guided Textual Reasoning conducts comprehensive analysis
by integrating visual perception. To enhance few-shot generalization, we employ
an integrated training strategy that combines supervised fine-tuning (SFT) with
reinforcement learning (GRPO), incorporating three sophisticated reward
functions. Experimental results demonstrate that OmniAD achieves a performance
of 79.1 on the MMAD benchmark, surpassing models such as Qwen2.5-VL-7B and
GPT-4o. It also shows strong results across multiple anomaly detection
benchmarks. These results highlight the importance of enhancing visual
perception for effective reasoning in anomaly understanding. All codes and
models will be publicly available.

---


### [A2Seek: Towards Reasoning-Centric Benchmark for Aerial Anomaly Understanding](http://arxiv.org/abs/2505.21962v1)

While unmanned aerial vehicles (UAVs) offer wide-area, high-altitude coverage
for anomaly detection, they face challenges such as dynamic viewpoints, scale
variations, and complex scenes. Existing datasets and methods, mainly designed
for fixed ground-level views, struggle to adapt to these conditions, leading to
significant performance drops in drone-view scenarios. To bridge this gap, we
introduce A2Seek (Aerial Anomaly Seek), a large-scale, reasoning-centric
benchmark dataset for aerial anomaly understanding. This dataset covers various
scenarios and environmental conditions, providing high-resolution real-world
aerial videos with detailed annotations, including anomaly categories,
frame-level timestamps, region-level bounding boxes, and natural language
explanations for causal reasoning. Building on this dataset, we propose
A2Seek-R1, a novel reasoning framework that generalizes R1-style strategies to
aerial anomaly understanding, enabling a deeper understanding of "Where"
anomalies occur and "Why" they happen in aerial frames. To this end, A2Seek-R1
first employs a graph-of-thought (GoT)-guided supervised fine-tuning approach
to activate the model's latent reasoning capabilities on A2Seek. Then, we
introduce Aerial Group Relative Policy Optimization (A-GRPO) to design
rule-based reward functions tailored to aerial scenarios. Furthermore, we
propose a novel "seeking" mechanism that simulates UAV flight behavior by
directing the model's attention to informative regions. Extensive experiments
demonstrate that A2Seek-R1 achieves up to a 22.04% improvement in AP for
prediction accuracy and a 13.9% gain in mIoU for anomaly localization,
exhibiting strong generalization across complex environments and
out-of-distribution scenarios. Our dataset and code will be released at
https://hayneyday.github.io/A2Seek/.

---


### [Look Within or Look Beyond? A Theoretical Comparison Between Parameter-Efficient and Full Fine-Tuning](http://arxiv.org/abs/2505.22355v1)

Parameter-Efficient Fine-Tuning (PEFT) methods achieve performance comparable
to Full Fine-Tuning (FFT) while requiring significantly fewer computing
resources, making it the go-to choice for researchers. We find that although
PEFT can achieve competitive results on some benchmarks, its performance falls
short of FFT in complex tasks, such as reasoning and instruction-based
fine-tuning. In this paper, we compare the characteristics of PEFT and FFT in
terms of representational capacity and robustness based on optimization theory.
We theoretically demonstrate that PEFT is a strict subset of FFT. By providing
theoretical upper bounds for PEFT, we show that the limited parameter space
constrains the model's representational ability, making it more susceptible to
perturbations. Experiments on 15 datasets encompassing classification,
generation, reasoning, instruction fine-tuning tasks and 11 adversarial test
sets validate our theories. We hope that these results spark further research
beyond the realms of well established PEFT. The source code is in the anonymous
Github repository\footnote{https://github.com/misonsky/PEFTEval}.

---


### [Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training](http://arxiv.org/abs/2505.22257v1)

We revisit Group Relative Policy Optimization (GRPO) in both on-policy and
off-policy optimization regimes. Our motivation comes from recent work on
off-policy Proximal Policy Optimization (PPO), which improves training
stability, sampling efficiency, and memory usage. In addition, a recent
analysis of GRPO suggests that estimating the advantage function with
off-policy samples could be beneficial. Building on these observations, we
adapt GRPO to the off-policy setting. We show that both on-policy and
off-policy GRPO objectives yield an improvement in the reward. This result
motivates the use of clipped surrogate objectives in the off-policy version of
GRPO. We then compare the empirical performance of reinforcement learning with
verifiable rewards in post-training using both GRPO variants. Our results show
that off-policy GRPO either significantly outperforms or performs on par with
its on-policy counterpart.

---


### [Oryx: a Performant and Scalable Algorithm for Many-Agent Coordination in Offline MARL](http://arxiv.org/abs/2505.22151v1)

A key challenge in offline multi-agent reinforcement learning (MARL) is
achieving effective many-agent multi-step coordination in complex environments.
In this work, we propose Oryx, a novel algorithm for offline cooperative MARL
to directly address this challenge. Oryx adapts the recently proposed
retention-based architecture Sable and combines it with a sequential form of
implicit constraint Q-learning (ICQ), to develop a novel offline
auto-regressive policy update scheme. This allows Oryx to solve complex
coordination challenges while maintaining temporal coherence over lengthy
trajectories. We evaluate Oryx across a diverse set of benchmarks from prior
works (SMAC, RWARE, and Multi-Agent MuJoCo) covering tasks of both discrete and
continuous control, varying in scale and difficulty. Oryx achieves
state-of-the-art performance on more than 80% of the 65 tested datasets,
outperforming prior offline MARL methods and demonstrating robust
generalisation across domains with many agents and long horizons. Finally, we
introduce new datasets to push the limits of many-agent coordination in offline
MARL, and demonstrate Oryx's superior ability to scale effectively in such
settings. We will make all of our datasets, experimental data, and code
available upon publication.

---


### [ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning](http://arxiv.org/abs/2505.22094v1)

We propose ReinFlow, a simple yet effective online reinforcement learning
(RL) framework that fine-tunes a family of flow matching policies for
continuous robotic control. Derived from rigorous RL theory, ReinFlow injects
learnable noise into a flow policy's deterministic path, converting the flow
into a discrete-time Markov Process for exact and straightforward likelihood
computation. This conversion facilitates exploration and ensures training
stability, enabling ReinFlow to fine-tune diverse flow model variants,
including Rectified Flow [35] and Shortcut Models [19], particularly at very
few or even one denoising step. We benchmark ReinFlow in representative
locomotion and manipulation tasks, including long-horizon planning with visual
input and sparse reward. The episode reward of Rectified Flow policies obtained
an average net growth of 135.36% after fine-tuning in challenging legged
locomotion tasks while saving denoising steps and 82.63% of wall time compared
to state-of-the-art diffusion RL fine-tuning method DPPO [43]. The success rate
of the Shortcut Model policies in state and visual manipulation tasks achieved
an average net increase of 40.34% after fine-tuning with ReinFlow at four or
even one denoising step, whose performance is comparable to fine-tuned DDIM
policies while saving computation time for an average of 23.20%. Project
Webpage: https://reinflow.github.io/

---


### [PADAM: Parallel averaged Adam reduces the error for stochastic optimization in scientific machine learning](http://arxiv.org/abs/2505.22085v1)

Averaging techniques such as Ruppert--Polyak averaging and exponential
movering averaging (EMA) are powerful approaches to accelerate optimization
procedures of stochastic gradient descent (SGD) optimization methods such as
the popular ADAM optimizer. However, depending on the specific optimization
problem under consideration, the type and the parameters for the averaging need
to be adjusted to achieve the smallest optimization error. In this work we
propose an averaging approach, which we refer to as parallel averaged ADAM
(PADAM), in which we compute parallely different averaged variants of ADAM and
during the training process dynamically select the variant with the smallest
optimization error. A central feature of this approach is that this procedure
requires no more gradient evaluations than the usual ADAM optimizer as each of
the averaged trajectories relies on the same underlying ADAM trajectory and
thus on the same underlying gradients. We test the proposed PADAM optimizer in
13 stochastic optimization and deep neural network (DNN) learning problems and
compare its performance with known optimizers from the literature such as
standard SGD, momentum SGD, Adam with and without EMA, and ADAMW. In
particular, we apply the compared optimizers to physics-informed neural
network, deep Galerkin, deep backward stochastic differential equation and deep
Kolmogorov approximations for boundary value partial differential equation
problems from scientific machine learning, as well as to DNN approximations for
optimal control and optimal stopping problems. In nearly all of the considered
examples PADAM achieves, sometimes among others and sometimes exclusively,
essentially the smallest optimization error. This work thus strongly suggest to
consider PADAM for scientific machine learning problems and also motivates
further research for adaptive averaging procedures within the training of DNNs.

---


### [Detecting Undesired Process Behavior by Means of Retrieval Augmented Generation](http://arxiv.org/abs/2505.22041v1)

Conformance checking techniques detect undesired process behavior by
comparing process executions that are recorded in event logs to desired
behavior that is captured in a dedicated process model. If such models are not
available, conformance checking techniques are not applicable, but
organizations might still be interested in detecting undesired behavior in
their processes. To enable this, existing approaches use Large Language Models
(LLMs), assuming that they can learn to distinguish desired from undesired
behavior through fine-tuning. However, fine-tuning is highly resource-intensive
and the fine-tuned LLMs often do not generalize well. To address these
limitations, we propose an approach that requires neither a dedicated process
model nor resource-intensive fine-tuning to detect undesired process behavior.
Instead, we use Retrieval Augmented Generation (RAG) to provide an LLM with
direct access to a knowledge base that contains both desired and undesired
process behavior from other processes, assuming that the LLM can transfer this
knowledge to the process at hand. Our evaluation shows that our approach
outperforms fine-tuned LLMs in detecting undesired behavior, demonstrating that
RAG is a viable alternative to resource-intensive fine-tuning, particularly
when enriched with relevant context from the event log, such as frequent traces
and activities.

---


### [ACE: Exploring Activation Cosine Similarity and Variance for Accurate and Calibration-Efficient LLM Pruning](http://arxiv.org/abs/2505.21987v1)

With the rapid expansion of large language models (LLMs), the demand for
memory and computational resources has grown significantly. Recent advances in
LLM pruning aim to reduce the size and computational cost of these models.
However, existing methods often suffer from either suboptimal pruning
performance or low time efficiency during the pruning process. In this work, we
propose an efficient and effective pruning method that simultaneously achieves
high pruning performance and fast pruning speed with improved calibration
efficiency. Our approach introduces two key innovations: (1) An activation
cosine similarity loss-guided pruning metric, which considers the angular
deviation of the output activation between the dense and pruned models. (2) An
activation variance-guided pruning metric, which helps preserve semantic
distinctions in output activations after pruning, enabling effective pruning
with shorter input sequences. These two components can be readily combined to
enhance LLM pruning in both accuracy and efficiency. Experimental results show
that our method achieves up to an 18% reduction in perplexity and up to 63%
decrease in pruning time on prevalent LLMs such as LLaMA, LLaMA-2, and OPT.

---


### [Two-Stage Feature Generation with Transformer and Reinforcement Learning](http://arxiv.org/abs/2505.21978v1)

Feature generation is a critical step in machine learning, aiming to enhance
model performance by capturing complex relationships within the data and
generating meaningful new features. Traditional feature generation methods
heavily rely on domain expertise and manual intervention, making the process
labor-intensive and challenging to adapt to different scenarios. Although
automated feature generation techniques address these issues to some extent,
they often face challenges such as feature redundancy, inefficiency in feature
space exploration, and limited adaptability to diverse datasets and tasks. To
address these problems, we propose a Two-Stage Feature Generation (TSFG)
framework, which integrates a Transformer-based encoder-decoder architecture
with Proximal Policy Optimization (PPO). The encoder-decoder model in TSFG
leverages the Transformer's self-attention mechanism to efficiently represent
and transform features, capturing complex dependencies within the data. PPO
further enhances TSFG by dynamically adjusting the feature generation strategy
based on task-specific feedback, optimizing the process for improved
performance and adaptability. TSFG dynamically generates high-quality feature
sets, significantly improving the predictive performance of machine learning
models. Experimental results demonstrate that TSFG outperforms existing
state-of-the-art methods in terms of feature quality and adaptability.

---


