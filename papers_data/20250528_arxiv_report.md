### [How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](http://arxiv.org/abs/2505.21505v1)

Multilingual Alignment is an effective and representative paradigm to enhance
LLMs' multilingual capabilities, which transfers the capabilities from the
high-resource languages to the low-resource languages. Meanwhile, some
researches on language-specific neurons reveal that there are language-specific
neurons that are selectively activated in LLMs when processing different
languages. This provides a new perspective to analyze and understand LLMs'
mechanisms more specifically in multilingual scenarios. In this work, we
propose a new finer-grained neuron identification algorithm, which detects
language neurons~(including language-specific neurons and language-related
neurons) and language-agnostic neurons. Furthermore, based on the
distributional characteristics of different types of neurons, we divide the
LLMs' internal process for multilingual inference into four parts: (1)
multilingual understanding, (2) shared semantic space reasoning, (3)
multilingual output space transformation, and (4) vocabulary space outputting.
Additionally, we systematically analyze the models before and after alignment
with a focus on different types of neurons. We also analyze the phenomenon of
''Spontaneous Multilingual Alignment''. Overall, our work conducts a
comprehensive investigation based on different types of neurons, providing
empirical results and valuable insights for better understanding multilingual
alignment and multilingual capabilities of LLMs.

---


### [Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making](http://arxiv.org/abs/2505.21503v1)

Large language models (LLMs) have demonstrated strong potential in clinical
question answering, with recent multi-agent frameworks further improving
diagnostic accuracy via collaborative reasoning. However, we identify a
recurring issue of Silent Agreement, where agents prematurely converge on
diagnoses without sufficient critical analysis, particularly in complex or
ambiguous cases. We present a new concept called Catfish Agent, a
role-specialized LLM designed to inject structured dissent and counter silent
agreement. Inspired by the ``catfish effect'' in organizational psychology, the
Catfish Agent is designed to challenge emerging consensus to stimulate deeper
reasoning. We formulate two mechanisms to encourage effective and context-aware
interventions: (i) a complexity-aware intervention that modulates agent
engagement based on case difficulty, and (ii) a tone-calibrated intervention
articulated to balance critique and collaboration. Evaluations on nine medical
Q&A and three medical VQA benchmarks show that our approach consistently
outperforms both single- and multi-agent LLMs frameworks, including leading
commercial models such as GPT-4o and DeepSeek-R1.

---


### [Paper2Poster: Towards Multimodal Poster Automation from Scientific Papers](http://arxiv.org/abs/2505.21497v1)

Academic poster generation is a crucial yet challenging task in scientific
communication, requiring the compression of long-context interleaved documents
into a single, visually coherent page. To address this challenge, we introduce
the first benchmark and metric suite for poster generation, which pairs recent
conference papers with author-designed posters and evaluates outputs on
(i)Visual Quality-semantic alignment with human posters, (ii)Textual
Coherence-language fluency, (iii)Holistic Assessment-six fine-grained aesthetic
and informational criteria scored by a VLM-as-judge, and notably
(iv)PaperQuiz-the poster's ability to convey core paper content as measured by
VLMs answering generated quizzes. Building on this benchmark, we propose
PosterAgent, a top-down, visual-in-the-loop multi-agent pipeline: the (a)Parser
distills the paper into a structured asset library; the (b)Planner aligns
text-visual pairs into a binary-tree layout that preserves reading order and
spatial balance; and the (c)Painter-Commenter loop refines each panel by
executing rendering code and using VLM feedback to eliminate overflow and
ensure alignment. In our comprehensive evaluation, we find that GPT-4o
outputs-though visually appealing at first glance-often exhibit noisy text and
poor PaperQuiz scores, and we find that reader engagement is the primary
aesthetic bottleneck, as human-designed posters rely largely on visual
semantics to convey meaning. Our fully open-source variants (e.g. based on the
Qwen-2.5 series) outperform existing 4o-driven multi-agent systems across
nearly all metrics, while using 87% fewer tokens. It transforms a 22-page paper
into a finalized yet editable .pptx poster - all for just $0.005. These
findings chart clear directions for the next generation of fully automated
poster-generation models. The code and datasets are available at
https://github.com/Paper2Poster/Paper2Poster.

---


### [Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO](http://arxiv.org/abs/2505.21457v1)

Active vision, also known as active perception, refers to the process of
actively selecting where and how to look in order to gather task-relevant
information. It is a critical component of efficient perception and
decision-making in humans and advanced embodied agents. Recently, the use of
Multimodal Large Language Models (MLLMs) as central planning and
decision-making modules in robotic systems has gained extensive attention.
However, despite the importance of active perception in embodied intelligence,
there is little to no exploration of how MLLMs can be equipped with or learn
active perception capabilities. In this paper, we first provide a systematic
definition of MLLM-based active perception tasks. We point out that the
recently proposed GPT-o3 model's zoom-in search strategy can be regarded as a
special case of active perception; however, it still suffers from low search
efficiency and inaccurate region selection. To address these issues, we propose
ACTIVE-O3, a purely reinforcement learning based training framework built on
top of GRPO, designed to equip MLLMs with active perception capabilities. We
further establish a comprehensive benchmark suite to evaluate ACTIVE-O3 across
both general open-world tasks, such as small-object and dense object grounding,
and domain-specific scenarios, including small object detection in remote
sensing and autonomous driving, as well as fine-grained interactive
segmentation. In addition, ACTIVE-O3 also demonstrates strong zero-shot
reasoning abilities on the V* Benchmark, without relying on any explicit
reasoning data. We hope that our work can provide a simple codebase and
evaluation protocol to facilitate future research on active perception in
MLLMs.

---


### [Hume: Introducing System-2 Thinking in Visual-Language-Action Model](http://arxiv.org/abs/2505.21432v1)

Humans practice slow thinking before performing actual actions when handling
complex tasks in the physical world. This thinking paradigm, recently, has
achieved remarkable advancement in boosting Large Language Models (LLMs) to
solve complex tasks in digital domains. However, the potential of slow thinking
remains largely unexplored for robotic foundation models interacting with the
physical world. In this work, we propose Hume: a dual-system
Vision-Language-Action (VLA) model with value-guided System-2 thinking and
cascaded action denoising, exploring human-like thinking capabilities of
Vision-Language-Action models for dexterous robot control. System 2 of Hume
implements value-Guided thinking by extending a Vision-Language-Action Model
backbone with a novel value-query head to estimate the state-action value of
predicted actions. The value-guided thinking is conducted by repeat sampling
multiple action candidates and selecting one according to state-action value.
System 1 of Hume is a lightweight reactive visuomotor policy that takes System
2 selected action and performs cascaded action denoising for dexterous robot
control. At deployment time, System 2 performs value-guided thinking at a low
frequency while System 1 asynchronously receives the System 2 selected action
candidate and predicts fluid actions in real time. We show that Hume
outperforms the existing state-of-the-art Vision-Language-Action models across
multiple simulation benchmark and real-robot deployments.

---


### [Policy Induction: Predicting Startup Success via Explainable Memory-Augmented In-Context Learning](http://arxiv.org/abs/2505.21427v1)

Early-stage startup investment is a high-risk endeavor characterized by
scarce data and uncertain outcomes. Traditional machine learning approaches
often require large, labeled datasets and extensive fine-tuning, yet remain
opaque and difficult for domain experts to interpret or improve. In this paper,
we propose a transparent and data-efficient investment decision framework
powered by memory-augmented large language models (LLMs) using in-context
learning (ICL). Central to our method is a natural language policy embedded
directly into the LLM prompt, enabling the model to apply explicit reasoning
patterns and allowing human experts to easily interpret, audit, and iteratively
refine the logic. We introduce a lightweight training process that combines
few-shot learning with an in-context learning loop, enabling the LLM to update
its decision policy iteratively based on structured feedback. With only minimal
supervision and no gradient-based optimization, our system predicts startup
success far more accurately than existing benchmarks. It is over 20x more
precise than random chance, which succeeds 1.9% of the time. It is also 7.1x
more precise than the typical 5.6% success rate of top-tier venture capital
(VC) firms.

---


### [Diagnosing and Resolving Cloud Platform Instability with Multi-modal RAG LLMs](http://arxiv.org/abs/2505.21419v1)

Today's cloud-hosted applications and services are complex systems, and a
performance or functional instability can have dozens or hundreds of potential
root causes. Our hypothesis is that by combining the pattern matching
capabilities of modern AI tools with a natural multi-modal RAG LLM interface,
problem identification and resolution can be simplified. ARCA is a new
multi-modal RAG LLM system that targets this domain. Step-wise evaluations show
that ARCA outperforms state-of-the-art alternatives.

---


### [A Framework for Adversarial Analysis of Decision Support Systems Prior to Deployment](http://arxiv.org/abs/2505.21414v1)

This paper introduces a comprehensive framework designed to analyze and
secure decision-support systems trained with Deep Reinforcement Learning (DRL),
prior to deployment, by providing insights into learned behavior patterns and
vulnerabilities discovered through simulation. The introduced framework aids in
the development of precisely timed and targeted observation perturbations,
enabling researchers to assess adversarial attack outcomes within a strategic
decision-making context. We validate our framework, visualize agent behavior,
and evaluate adversarial outcomes within the context of a custom-built
strategic game, CyberStrike. Utilizing the proposed framework, we introduce a
method for systematically discovering and ranking the impact of attacks on
various observation indices and time-steps, and we conduct experiments to
evaluate the transferability of adversarial attacks across agent architectures
and DRL training algorithms. The findings underscore the critical need for
robust adversarial defense mechanisms to protect decision-making policies in
high-stakes environments.

---


### [MRSD: Multi-Resolution Skill Discovery for HRL Agents](http://arxiv.org/abs/2505.21410v1)

Hierarchical reinforcement learning (HRL) relies on abstract skills to solve
long-horizon tasks efficiently. While existing skill discovery methods learns
these skills automatically, they are limited to a single skill per task. In
contrast, humans learn and use both fine-grained and coarse motor skills
simultaneously. Inspired by human motor control, we propose Multi-Resolution
Skill Discovery (MRSD), an HRL framework that learns multiple skill encoders at
different temporal resolutions in parallel. A high-level manager dynamically
selects among these skills, enabling adaptive control strategies over time. We
evaluate MRSD on tasks from the DeepMind Control Suite and show that it
outperforms prior state-of-the-art skill discovery and HRL methods, achieving
faster convergence and higher final performance. Our findings highlight the
benefits of integrating multi-resolution skills in HRL, paving the way for more
versatile and efficient agents.

---


### [MME-Reasoning: A Comprehensive Benchmark for Logical Reasoning in MLLMs](http://arxiv.org/abs/2505.21327v1)

Logical reasoning is a fundamental aspect of human intelligence and an
essential capability for multimodal large language models (MLLMs). Despite the
significant advancement in multimodal reasoning, existing benchmarks fail to
comprehensively evaluate their reasoning abilities due to the lack of explicit
categorization for logical reasoning types and an unclear understanding of
reasoning. To address these issues, we introduce MME-Reasoning, a comprehensive
benchmark designed to evaluate the reasoning ability of MLLMs, which covers all
three types of reasoning (i.e., inductive, deductive, and abductive) in its
questions. We carefully curate the data to ensure that each question
effectively evaluates reasoning ability rather than perceptual skills or
knowledge breadth, and extend the evaluation protocols to cover the evaluation
of diverse questions. Our evaluation reveals substantial limitations of
state-of-the-art MLLMs when subjected to holistic assessments of logical
reasoning capabilities. Even the most advanced MLLMs show limited performance
in comprehensive logical reasoning, with notable performance imbalances across
reasoning types. In addition, we conducted an in-depth analysis of approaches
such as ``thinking mode'' and Rule-based RL, which are commonly believed to
enhance reasoning abilities. These findings highlight the critical limitations
and performance imbalances of current MLLMs in diverse logical reasoning
scenarios, providing comprehensive and systematic insights into the
understanding and evaluation of reasoning capabilities.

---


### [Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations](http://arxiv.org/abs/2505.21318v1)

While large language models (LLMs) with Chain-of-Thought (CoT) reasoning
excel in mathematics and coding, their potential for systematic reasoning in
chemistry, a domain demanding rigorous structural analysis for real-world tasks
like drug design and reaction engineering, remains untapped. Current benchmarks
focus on simple knowledge retrieval, neglecting step-by-step reasoning required
for complex tasks such as molecular optimization and reaction prediction. To
address this, we introduce ChemCoTBench, a reasoning framework that bridges
molecular structure understanding with arithmetic-inspired operations,
including addition, deletion, and substitution, to formalize chemical
problem-solving into transparent, step-by-step workflows. By treating molecular
transformations as modular "chemical operations", the framework enables
slow-thinking reasoning, mirroring the logic of mathematical proofs while
grounding solutions in real-world chemical constraints. We evaluate models on
two high-impact tasks: Molecular Property Optimization and Chemical Reaction
Prediction. These tasks mirror real-world challenges while providing structured
evaluability. By providing annotated datasets, a reasoning taxonomy, and
baseline evaluations, ChemCoTBench bridges the gap between abstract reasoning
methods and practical chemical discovery, establishing a foundation for
advancing LLMs as tools for AI-driven scientific innovation.

---


### [Complex System Diagnostics Using a Knowledge Graph-Informed and Large Language Model-Enhanced Framework](http://arxiv.org/abs/2505.21291v1)

In this paper, we present a novel diagnostic framework that integrates
Knowledge Graphs (KGs) and Large Language Models (LLMs) to support system
diagnostics in high-reliability systems such as nuclear power plants.
Traditional diagnostic modeling struggles when systems become too complex,
making functional modeling a more attractive approach. Our approach introduces
a diagnostic framework grounded in the functional modeling principles of the
Dynamic Master Logic (DML) model. It incorporates two coordinated LLM
components, including an LLM-based workflow for automated construction of DML
logic from system documentation and an LLM agent that facilitates interactive
diagnostics. The generated logic is encoded into a structured KG, referred to
as KG-DML, which supports hierarchical fault reasoning. Expert knowledge or
operational data can also be incorporated to refine the model's precision and
diagnostic depth. In the interaction phase, users submit natural language
queries, which are interpreted by the LLM agent. The agent selects appropriate
tools for structured reasoning, including upward and downward propagation
across the KG-DML. Rather than embedding KG content into every prompt, the LLM
agent distinguishes between diagnostic and interpretive tasks. For diagnostics,
the agent selects and executes external tools that perform structured KG
reasoning. For general queries, a Graph-based Retrieval-Augmented Generation
(Graph-RAG) approach is used, retrieving relevant KG segments and embedding
them into the prompt to generate natural explanations. A case study on an
auxiliary feedwater system demonstrated the framework's effectiveness, with
over 90% accuracy in key elements and consistent tool and argument extraction,
supporting its use in safety-critical diagnostics.

---


### [XBOUND: Exploring the Capability Boundaries of Device-Control Agents through Trajectory Tree Exploration](http://arxiv.org/abs/2505.21279v1)

Recent advancements in vision-language models (VLMs) have spurred increased
interest in Device-Control Agents (DC agents), such as utilizing in-the-wild
device control to manage graphical user interfaces. Conventional methods for
assessing the capabilities of DC agents, such as computing step-wise action
accuracy and overall task success rates, provide a macroscopic view of DC
agents' performance; however, they fail to offer microscopic insights into
potential errors that may occur in real-world applications. Conducting a
finer-grained performance evaluation of DC agents presents significant
challenges. This study introduces a new perspective on evaluation methods for
DC agents by proposing the XBOUND evaluation method, which employs the
calculation of a novel Explore Metric to delineate the capability boundaries of
DC agents. Compared to previous evaluation methods, XBOUND focuses on
individual states to assess the proficiency of DC agents in mastering these
states. Furthermore, we have developed a ``pseudo'' episode tree dataset
derived from Android Control test data. Utilizing this dataset and XBOUND, we
comprehensively evaluate the OS-Atlas and UI-TARS series, examining both the
overall and specific performance across five common tasks. Additionally, we
select representative cases to highlight the current deficiencies and
limitations inherent in both series. Code is available at
https://github.com/sqzhang-lazy/XBOUND.

---


### [Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space](http://arxiv.org/abs/2505.21277v1)

Large Language Models (LLMs), despite advanced general capabilities, still
suffer from numerous safety risks, especially jailbreak attacks that bypass
safety protocols. Understanding these vulnerabilities through black-box
jailbreak attacks, which better reflect real-world scenarios, offers critical
insights into model robustness. While existing methods have shown improvements
through various prompt engineering techniques, their success remains limited
against safety-aligned models, overlooking a more fundamental problem: the
effectiveness is inherently bounded by the predefined strategy spaces. However,
expanding this space presents significant challenges in both systematically
capturing essential attack patterns and efficiently navigating the increased
complexity. To better explore the potential of expanding the strategy space, we
address these challenges through a novel framework that decomposes jailbreak
strategies into essential components based on the Elaboration Likelihood Model
(ELM) theory and develops genetic-based optimization with intention evaluation
mechanisms. To be striking, our experiments reveal unprecedented jailbreak
capabilities by expanding the strategy space: we achieve over 90% success rate
on Claude-3.5 where prior methods completely fail, while demonstrating strong
cross-model transferability and surpassing specialized safeguard models in
evaluation accuracy. The code is open-sourced at:
https://github.com/Aries-iai/CL-GSO.

---


### [A Lightweight Multi-Expert Generative Language Model System for Engineering Information and Knowledge Extraction](http://arxiv.org/abs/2505.21109v1)

Despite recent advancements in domain adaptation techniques for large
language models, these methods remain computationally intensive, and the
resulting models can still exhibit hallucination issues. Most existing
adaptation methods do not prioritize reducing the computational resources
required for fine-tuning and inference of language models. Hallucination issues
have gradually decreased with each new model release. However, they remain
prevalent in engineering contexts, where generating well-structured text with
minimal errors and inconsistencies is critical. This work introduces a novel
approach called the Small Language Graph (SLG), which is a lightweight
adaptation solution designed to address the two key challenges outlined above.
The system is structured in the form of a graph, where each node represents a
lightweight expert - a small language model fine-tuned on specific and concise
texts. The results of this study have shown that SLG was able to surpass
conventional fine-tuning methods on the Exact Match metric by 3 times.
Additionally, the fine-tuning process was 1.7 times faster compared to that of
a larger stand-alone language model. These findings introduce a potential for
small to medium-sized engineering companies to confidently use generative AI
technologies, such as LLMs, without the necessity to invest in expensive
computational resources. Also, the graph architecture and the small size of
expert nodes offer a possible opportunity for distributed AI systems, thus
potentially diverting the global need for expensive centralized compute
clusters.

---


### [Thinker: Learning to Think Fast and Slow](http://arxiv.org/abs/2505.21097v1)

Recent studies show that the reasoning capabilities of Large Language Models
(LLMs) can be improved by applying Reinforcement Learning (RL) to
question-answering (QA) tasks in areas such as math and coding. With a long
context length, LLMs may learn to perform search, as indicated by the
self-correction behavior observed in DeepSeek R1. However, this search behavior
is often imprecise and lacks confidence, resulting in long, redundant responses
and highlighting deficiencies in intuition and verification. Inspired by the
Dual Process Theory in psychology, we introduce a simple modification to the QA
task that includes four stages: Fast Thinking, where the LLM must answer within
a strict token budget; Verification, where the model evaluates its initial
response; Slow Thinking, where it refines the initial response with more
deliberation; and Summarization, where it distills the refinement from the
previous stage into precise steps. Our proposed task improves average accuracy
from 24.9% to 27.9% for Qwen2.5-1.5B, and from 45.9% to 49.8% for
DeepSeek-R1-Qwen-1.5B. Notably, for Qwen2.5-1.5B, the Fast Thinking mode alone
achieves 26.8% accuracy using fewer than 1000 tokens, demonstrating substantial
inference efficiency gains. These findings suggest that intuition and
deliberative reasoning are distinct, complementary systems benefiting from
targeted training.

---


### [BLUCK: A Benchmark Dataset for Bengali Linguistic Understanding and Cultural Knowledge](http://arxiv.org/abs/2505.21092v1)

In this work, we introduce BLUCK, a new dataset designed to measure the
performance of Large Language Models (LLMs) in Bengali linguistic understanding
and cultural knowledge. Our dataset comprises 2366 multiple-choice questions
(MCQs) carefully curated from compiled collections of several college and job
level examinations and spans 23 categories covering knowledge on Bangladesh's
culture and history and Bengali linguistics. We benchmarked BLUCK using 6
proprietary and 3 open-source LLMs - including GPT-4o, Claude-3.5-Sonnet,
Gemini-1.5-Pro, Llama-3.3-70B-Instruct, and DeepSeekV3. Our results show that
while these models perform reasonably well overall, they, however, struggles in
some areas of Bengali phonetics. Although current LLMs' performance on Bengali
cultural and linguistic contexts is still not comparable to that of mainstream
languages like English, our results indicate Bengali's status as a mid-resource
language. Importantly, BLUCK is also the first MCQ-based evaluation benchmark
that is centered around native Bengali culture, history, and linguistics.

---


### [Why Distillation can Outperform Zero-RL: The Role of Flexible Reasoning](http://arxiv.org/abs/2505.21067v1)

Reinforcement learning (RL) has played an important role in improving the
reasoning ability of large language models (LLMs). Some studies apply RL
directly to \textit{smaller} base models (known as zero-RL) and also achieve
notable progress. However, in this paper, we show that using only 920 examples,
a simple distillation method based on the base model can clearly outperform
zero-RL, which typically requires much more data and computational cost. By
analyzing the token frequency in model outputs, we find that the distilled
model shows more flexible reasoning. It uses anthropomorphic tokens and logical
connectors much more often than the zero-RL model. Further analysis reveals
that distillation enhances the presence of two advanced cognitive behaviors:
Multi-Perspective Thinking or Attempting and Metacognitive Awareness. Frequent
occurrences of these two advanced cognitive behaviors give rise to flexible
reasoning, which is essential for solving complex reasoning problems, while
zero-RL fails to significantly boost the frequency of these behaviors.

---


### [LPOI: Listwise Preference Optimization for Vision Language Models](http://arxiv.org/abs/2505.21061v1)

Aligning large VLMs with human preferences is a challenging task, as methods
like RLHF and DPO often overfit to textual information or exacerbate
hallucinations. Although augmenting negative image samples partially addresses
these pitfalls, no prior work has employed listwise preference optimization for
VLMs, due to the complexity and cost of constructing listwise image samples. In
this work, we propose LPOI, the first object-aware listwise preference
optimization developed for reducing hallucinations in VLMs. LPOI identifies and
masks a critical object in the image, and then interpolates the masked region
between the positive and negative images to form a sequence of incrementally
more complete images. The model is trained to rank these images in ascending
order of object visibility, effectively reducing hallucinations while retaining
visual fidelity. LPOI requires no extra annotations beyond standard pairwise
preference data, as it automatically constructs the ranked lists through object
masking and interpolation. Comprehensive experiments on MMHalBench, AMBER, and
Object HalBench confirm that LPOI outperforms existing preference optimization
methods in reducing hallucinations and enhancing VLM performance. We make the
code available at https://github.com/fatemehpesaran310/lpoi.

---


### [Large Language Model-enhanced Reinforcement Learning for Low-Altitude Economy Networking](http://arxiv.org/abs/2505.21045v1)

Low-Altitude Economic Networking (LAENet) aims to support diverse flying
applications below 1,000 meters by deploying various aerial vehicles for
flexible and cost-effective aerial networking. However, complex
decision-making, resource constraints, and environmental uncertainty pose
significant challenges to the development of the LAENet. Reinforcement learning
(RL) offers a potential solution in response to these challenges but has
limitations in generalization, reward design, and model stability. The
emergence of large language models (LLMs) offers new opportunities for RL to
mitigate these limitations. In this paper, we first present a tutorial about
integrating LLMs into RL by using the capacities of generation, contextual
understanding, and structured reasoning of LLMs. We then propose an
LLM-enhanced RL framework for the LAENet in terms of serving the LLM as
information processor, reward designer, decision-maker, and generator.
Moreover, we conduct a case study by using LLMs to design a reward function to
improve the learning performance of RL in the LAENet. Finally, we provide a
conclusion and discuss future work.

---


### [Multi-Mode Process Control Using Multi-Task Inverse Reinforcement Learning](http://arxiv.org/abs/2505.21026v1)

In the era of Industry 4.0 and smart manufacturing, process systems
engineering must adapt to digital transformation. While reinforcement learning
offers a model-free approach to process control, its applications are limited
by the dependence on accurate digital twins and well-designed reward functions.
To address these limitations, this paper introduces a novel framework that
integrates inverse reinforcement learning (IRL) with multi-task learning for
data-driven, multi-mode control design. Using historical closed-loop data as
expert demonstrations, IRL extracts optimal reward functions and control
policies. A latent-context variable is incorporated to distinguish modes,
enabling the training of mode-specific controllers. Case studies on a
continuous stirred tank reactor and a fed-batch bioreactor validate the
effectiveness of this framework in handling multi-mode data and training
adaptable controllers.

---


### [Context-Aware Content Moderation for German Newspaper Comments](http://arxiv.org/abs/2505.20963v1)

The increasing volume of online discussions requires advanced automatic
content moderation to maintain responsible discourse. While hate speech
detection on social media is well-studied, research on German-language
newspaper forums remains limited. Existing studies often neglect
platform-specific context, such as user history and article themes. This paper
addresses this gap by developing and evaluating binary classification models
for automatic content moderation in German newspaper forums, incorporating
contextual information. Using LSTM, CNN, and ChatGPT-3.5 Turbo, and leveraging
the One Million Posts Corpus from the Austrian newspaper Der Standard, we
assess the impact of context-aware models. Results show that CNN and LSTM
models benefit from contextual information and perform competitively with
state-of-the-art approaches. In contrast, ChatGPT's zero-shot classification
does not improve with added context and underperforms.

---


### [Multi-objective Large Language Model Alignment with Hierarchical Experts](http://arxiv.org/abs/2505.20925v1)

Aligning large language models (LLMs) to simultaneously satisfy multiple
objectives remains a significant challenge, especially given the diverse and
often conflicting nature of human preferences. Existing alignment methods
struggle to balance trade-offs effectively, often requiring costly retraining
or yielding suboptimal results across the Pareto frontier of preferences. In
this paper, we introduce \textit{HoE}(Hierarchical Mixture-of-Experts), a
\textit{lightweight}, \textit{parameter-efficient}, and \textit{plug-and-play}
approach that eliminates the need for model training, while enabling LLMs to
adapt across the entire Pareto frontier and accommodate diverse user
preferences. In particular, \textit{HoE} consists of three hierarchical
components: LoRA Experts, Router Experts and Preference Routing, reaching
optimal Pareto frontiers and achieving a trade-off between parameter size,
training cost, and performance. We evaluate \textit{HoE} across various tasks
on 14 objectives and 200 different preferences among 6 benchmarks,
demonstrating superior performance over 15 recent baselines. Code is available
in the supplementary materials.

---


### [Revisiting Multi-Agent World Modeling from a Diffusion-Inspired Perspective](http://arxiv.org/abs/2505.20922v1)

World models have recently attracted growing interest in Multi-Agent
Reinforcement Learning (MARL) due to their ability to improve sample efficiency
for policy learning. However, accurately modeling environments in MARL is
challenging due to the exponentially large joint action space and highly
uncertain dynamics inherent in multi-agent systems. To address this, we reduce
modeling complexity by shifting from jointly modeling the entire state-action
transition dynamics to focusing on the state space alone at each timestep
through sequential agent modeling. Specifically, our approach enables the model
to progressively resolve uncertainty while capturing the structured
dependencies among agents, providing a more accurate representation of how
agents influence the state. Interestingly, this sequential revelation of
agents' actions in a multi-agent system aligns with the reverse process in
diffusion models--a class of powerful generative models known for their
expressiveness and training stability compared to autoregressive or latent
variable models. Leveraging this insight, we develop a flexible and robust
world model for MARL using diffusion models. Our method, Diffusion-Inspired
Multi-Agent world model (DIMA), achieves state-of-the-art performance across
multiple multi-agent control benchmarks, significantly outperforming prior
world models in terms of final return and sample efficiency, including MAMuJoCo
and Bi-DexHands. DIMA establishes a new paradigm for constructing multi-agent
world models, advancing the frontier of MARL research.

---


### [Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation](http://arxiv.org/abs/2505.20897v1)

Vision-and-Language Navigation (VLN) requires the agent to navigate by
following natural instructions under partial observability, making it difficult
to align perception with language. Recent methods mitigate this by imagining
future scenes, yet they rely on vision-based synthesis, leading to high
computational cost and redundant details. To this end, we propose to adaptively
imagine key environmental semantics via \textit{language} form, enabling a more
reliable and efficient strategy. Specifically, we introduce a novel Adaptive
Text Dreamer (ATD), a dual-branch self-guided imagination policy built upon a
large language model (LLM). ATD is designed with a human-like left-right brain
architecture, where the left brain focuses on logical integration, and the
right brain is responsible for imaginative prediction of future scenes. To
achieve this, we fine-tune only the Q-former within both brains to efficiently
activate domain-specific knowledge in the LLM, enabling dynamic updates of
logical reasoning and imagination during navigation. Furthermore, we introduce
a cross-interaction mechanism to regularize the imagined outputs and inject
them into a navigation expert module, allowing ATD to jointly exploit both the
reasoning capacity of the LLM and the expertise of the navigation model. We
conduct extensive experiments on the R2R benchmark, where ATD achieves
state-of-the-art performance with fewer parameters. The code is
\href{https://github.com/zhangpingrui/Adaptive-Text-Dreamer}{here}.

---


### [Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization](http://arxiv.org/abs/2505.20881v1)

Heuristic design with large language models (LLMs) has emerged as a promising
approach for tackling combinatorial optimization problems (COPs). However,
existing approaches often rely on manually predefined evolutionary computation
(EC) optimizers and single-task training schemes, which may constrain the
exploration of diverse heuristic algorithms and hinder the generalization of
the resulting heuristics. To address these issues, we propose Meta-Optimization
of Heuristics (MoH), a novel framework that operates at the optimizer level,
discovering effective optimizers through the principle of meta-learning.
Specifically, MoH leverages LLMs to iteratively refine a meta-optimizer that
autonomously constructs diverse optimizers through (self-)invocation, thereby
eliminating the reliance on a predefined EC optimizer. These constructed
optimizers subsequently evolve heuristics for downstream tasks, enabling
broader heuristic exploration. Moreover, MoH employs a multi-task training
scheme to promote its generalization capability. Experiments on classic COPs
demonstrate that MoH constructs an effective and interpretable meta-optimizer,
achieving state-of-the-art performance across various downstream tasks,
particularly in cross-size settings.

---


### [Respond to Change with Constancy: Instruction-tuning with LLM for Non-I.I.D. Network Traffic Classification](http://arxiv.org/abs/2505.20866v1)

Encrypted traffic classification is highly challenging in network security
due to the need for extracting robust features from content-agnostic traffic
data. Existing approaches face critical issues: (i) Distribution drift, caused
by reliance on the closedworld assumption, limits adaptability to realworld,
shifting patterns; (ii) Dependence on labeled data restricts applicability
where such data is scarce or unavailable. Large language models (LLMs) have
demonstrated remarkable potential in offering generalizable solutions across a
wide range of tasks, achieving notable success in various specialized fields.
However, their effectiveness in traffic analysis remains constrained by
challenges in adapting to the unique requirements of the traffic domain. In
this paper, we introduce a novel traffic representation model named Encrypted
Traffic Out-of-Distribution Instruction Tuning with LLM (ETooL), which
integrates LLMs with knowledge of traffic structures through a self-supervised
instruction tuning paradigm. This framework establishes connections between
textual information and traffic interactions. ETooL demonstrates more robust
classification performance and superior generalization in both supervised and
zero-shot traffic classification tasks. Notably, it achieves significant
improvements in F1 scores: APP53 (I.I.D.) to 93.19%(6.62%) and 92.11%(4.19%),
APP53 (O.O.D.) to 74.88%(18.17%) and 72.13%(15.15%), and ISCX-Botnet (O.O.D.)
to 95.03%(9.16%) and 81.95%(12.08%). Additionally, we construct NETD, a traffic
dataset designed to support dynamic distributional shifts, and use it to
validate ETooL's effectiveness under varying distributional conditions.
Furthermore, we evaluate the efficiency gains achieved through ETooL's
instruction tuning approach.

---


### [Rendering-Aware Reinforcement Learning for Vector Graphics Generation](http://arxiv.org/abs/2505.20793v1)

Scalable Vector Graphics (SVG) offer a powerful format for representing
visual designs as interpretable code. Recent advances in vision-language models
(VLMs) have enabled high-quality SVG generation by framing the problem as a
code generation task and leveraging large-scale pretraining. VLMs are
particularly suitable for this task as they capture both global semantics and
fine-grained visual patterns, while transferring knowledge across vision,
natural language, and code domains. However, existing VLM approaches often
struggle to produce faithful and efficient SVGs because they never observe the
rendered images during training. Although differentiable rendering for
autoregressive SVG code generation remains unavailable, rendered outputs can
still be compared to original inputs, enabling evaluative feedback suitable for
reinforcement learning (RL). We introduce RLRF(Reinforcement Learning from
Rendering Feedback), an RL method that enhances SVG generation in
autoregressive VLMs by leveraging feedback from rendered SVG outputs. Given an
input image, the model generates SVG roll-outs that are rendered and compared
to the original image to compute a reward. This visual fidelity feedback guides
the model toward producing more accurate, efficient, and semantically coherent
SVGs. RLRF significantly outperforms supervised fine-tuning, addressing common
failure modes and enabling precise, high-quality SVG generation with strong
structural understanding and generalization.

---


### [FM-Planner: Foundation Model Guided Path Planning for Autonomous Drone Navigation](http://arxiv.org/abs/2505.20783v1)

Path planning is a critical component in autonomous drone operations,
enabling safe and efficient navigation through complex environments. Recent
advances in foundation models, particularly large language models (LLMs) and
vision-language models (VLMs), have opened new opportunities for enhanced
perception and intelligent decision-making in robotics. However, their
practical applicability and effectiveness in global path planning remain
relatively unexplored. This paper proposes foundation model-guided path
planners (FM-Planner) and presents a comprehensive benchmarking study and
practical validation for drone path planning. Specifically, we first
systematically evaluate eight representative LLM and VLM approaches using
standardized simulation scenarios. To enable effective real-time navigation, we
then design an integrated LLM-Vision planner that combines semantic reasoning
with visual perception. Furthermore, we deploy and validate the proposed path
planner through real-world experiments under multiple configurations. Our
findings provide valuable insights into the strengths, limitations, and
feasibility of deploying foundation models in real-world drone applications and
providing practical implementations in autonomous flight. Project site:
https://github.com/NTU-ICG/FM-Planner.

---


### [PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](http://arxiv.org/abs/2505.20759v1)

Real-world objects are composed of distinctive, object-specific parts.
Identifying these parts is key to performing fine-grained, compositional
reasoning-yet, large multimodal models (LMMs) struggle to perform this
seemingly straightforward task. In this work, we introduce PARTONOMY, an LMM
benchmark designed for pixel-level part grounding. We construct PARTONOMY from
existing part datasets and our own rigorously annotated set of images,
encompassing 862 part labels and 534 object labels for evaluation. Unlike
existing datasets that simply ask models to identify generic parts, PARTONOMY
uses specialized concepts (e.g., agricultural airplane), and challenges models
to compare objects' parts, consider part-whole relationships, and justify
textual predictions with visual segmentations. Our experiments demonstrate
significant limitations in state-of-the-art LMMs (e.g., LISA-13B achieves only
5.9% gIoU), highlighting a critical gap in their part grounding abilities. We
note that existing segmentation-enabled LMMs (segmenting LMMs) have two key
architectural shortcomings: they use special [SEG] tokens not seen during
pretraining which induce distribution shift, and they discard predicted
segmentations instead of using past predictions to guide future ones. To
address these deficiencies, we train several part-centric LMMs and propose
PLUM, a novel segmenting LMM that uses span tagging instead of segmentation
tokens and that conditions on prior predictions in a feedback loop. We find
that pretrained PLUM outperforms existing segmenting LMMs on reasoning
segmentation, VQA, and visual hallucination benchmarks. In addition, PLUM
finetuned on our proposed Explanatory Part Segmentation task is competitive
with segmenting LMMs trained on significantly more segmentation data. Our work
opens up new avenues towards enabling fine-grained, grounded visual
understanding in LMMs.

---


### [Understand, Think, and Answer: Advancing Visual Reasoning with Large Multimodal Models](http://arxiv.org/abs/2505.20753v1)

Large Multimodal Models (LMMs) have recently demonstrated remarkable visual
understanding performance on both vision-language and vision-centric tasks.
However, they often fall short in integrating advanced, task-specific
capabilities for compositional reasoning, which hinders their progress toward
truly competent general vision models. To address this, we present a unified
visual reasoning mechanism that enables LMMs to solve complicated compositional
problems by leveraging their intrinsic capabilities (e.g. grounding and visual
understanding capabilities). Different from the previous shortcut learning
mechanism, our approach introduces a human-like
understanding-thinking-answering process, allowing the model to complete all
steps in a single pass forwarding without the need for multiple inferences or
external tools. This design bridges the gap between foundational visual
capabilities and general question answering, encouraging LMMs to generate
faithful and traceable responses for complex visual reasoning. Meanwhile, we
curate 334K visual instruction samples covering both general scenes and
text-rich scenes and involving multiple foundational visual capabilities. Our
trained model, Griffon-R, has the ability of end-to-end automatic
understanding, self-thinking, and reasoning answers. Comprehensive experiments
show that Griffon-R not only achieves advancing performance on complex visual
reasoning benchmarks including VSR and CLEVR, but also enhances multimodal
capabilities across various benchmarks like MMBench and ScienceQA. Data,
models, and codes will be release at
https://github.com/jefferyZhan/Griffon/tree/master/Griffon-R soon.

---


### [RRO: LLM Agent Optimization Through Rising Reward Trajectories](http://arxiv.org/abs/2505.20737v1)

Large language models (LLMs) have exhibited extraordinary performance in a
variety of tasks while it remains challenging for them to solve complex
multi-step tasks as agents. In practice, agents sensitive to the outcome of
certain key steps which makes them likely to fail the task because of a subtle
mistake in the planning trajectory. Recent approaches resort to calibrating the
reasoning process through reinforcement learning. They reward or penalize every
reasoning step with process supervision, as known as Process Reward Models
(PRMs). However, PRMs are difficult and costly to scale up with a large number
of next action candidates since they require extensive computations to acquire
the training data through the per-step trajectory exploration. To mitigate this
issue, we focus on the relative reward trend across successive reasoning steps
and propose maintaining an increasing reward in the collected trajectories for
process supervision, which we term Reward Rising Optimization (RRO).
Specifically, we incrementally augment the process supervision until
identifying a step exhibiting positive reward differentials, i.e. rising
rewards, relative to its preceding iteration. This method dynamically expands
the search space for the next action candidates, efficiently capturing
high-quality data. We provide mathematical groundings and empirical results on
the WebShop and InterCode-SQL benchmarks, showing that our proposed RRO
achieves superior performance while requiring much less exploration cost.

---


### [What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals](http://arxiv.org/abs/2505.20730v1)

User-item interactions contain rich collaborative signals that form the
backbone of many successful recommender systems. While recent work has explored
the use of large language models (LLMs) for recommendation, it remains unclear
whether LLMs can effectively reason over this type of collaborative
information. In this paper, we conduct a systematic comparison between LLMs and
classical matrix factorization (MF) models to assess LLMs' ability to leverage
user-item interaction data. We further introduce a simple retrieval-augmented
generation (RAG) method that enhances LLMs by grounding their predictions in
structured interaction data. Our experiments reveal that current LLMs often
fall short in capturing collaborative patterns inherent to MF models, but that
our RAG-based approach substantially improves recommendation
quality-highlighting a promising direction for future LLM-based recommenders.

---


### [Jigsaw-Puzzles: From Seeing to Understanding to Reasoning in Vision-Language Models](http://arxiv.org/abs/2505.20728v1)

Spatial reasoning is a core component of human cognition, enabling
individuals to perceive, comprehend, and interact with the physical world. It
relies on a nuanced understanding of spatial structures and inter-object
relationships, serving as the foundation for complex reasoning and
decision-making. To investigate whether current vision-language models (VLMs)
exhibit similar capability, we introduce Jigsaw-Puzzles, a novel benchmark
consisting of 1,100 carefully curated real-world images with high spatial
complexity. Based on this dataset, we design five tasks to rigorously evaluate
VLMs' spatial perception, structural understanding, and reasoning capabilities,
while deliberately minimizing reliance on domain-specific knowledge to better
isolate and assess the general spatial reasoning capability. We conduct a
comprehensive evaluation across 24 state-of-the-art VLMs. The results show that
even the strongest model, Gemini-2.5-Pro, achieves only 77.14% overall accuracy
and performs particularly poorly on the Order Generation task, with only 30.00%
accuracy, far below the performance exceeding 90% achieved by human
participants. This persistent gap underscores the need for continued progress,
positioning Jigsaw-Puzzles as a challenging and diagnostic benchmark for
advancing spatial reasoning research in VLMs.

---


### [VLM Can Be a Good Assistant: Enhancing Embodied Visual Tracking with Self-Improving Visual-Language Models](http://arxiv.org/abs/2505.20718v1)

We introduce a novel self-improving framework that enhances Embodied Visual
Tracking (EVT) with Visual-Language Models (VLMs) to address the limitations of
current active visual tracking systems in recovering from tracking failure. Our
approach combines the off-the-shelf active tracking methods with VLMs'
reasoning capabilities, deploying a fast visual policy for normal tracking and
activating VLM reasoning only upon failure detection. The framework features a
memory-augmented self-reflection mechanism that enables the VLM to
progressively improve by learning from past experiences, effectively addressing
VLMs' limitations in 3D spatial reasoning. Experimental results demonstrate
significant performance improvements, with our framework boosting success rates
by $72\%$ with state-of-the-art RL-based approaches and $220\%$ with PID-based
methods in challenging environments. This work represents the first integration
of VLM-based reasoning to assist EVT agents in proactive failure recovery,
offering substantial advances for real-world robotic applications that require
continuous target monitoring in dynamic, unstructured environments. Project
website: https://sites.google.com/view/evt-recovery-assistant.

---


### [Dissecting Physics Reasoning in Small Language Models: A Multi-Dimensional Analysis from an Educational Perspective](http://arxiv.org/abs/2505.20707v1)

Small Language Models (SLMs) offer computational efficiency and
accessibility, making them promising for educational applications. However,
their capacity for complex reasoning, particularly in domains such as physics,
remains underexplored. This study investigates the high school physics
reasoning capabilities of state-of-the-art SLMs (under 4 billion parameters),
including instruct versions of Llama 3.2, Phi 4 Mini, Gemma 3, and Qwen series.
We developed a comprehensive physics dataset from the OpenStax High School
Physics textbook, annotated according to Bloom's Taxonomy, with LaTeX and
plaintext mathematical notations. A novel cultural contextualization approach
was applied to a subset, creating culturally adapted problems for Asian,
African, and South American/Australian contexts while preserving core physics
principles. Using an LLM-as-a-judge framework with Google's Gemini 2.5 Flash,
we evaluated answer and reasoning chain correctness, along with calculation
accuracy. The results reveal significant differences between the SLMs. Qwen 3
1.7B achieved high `answer accuracy' (85%), but `fully correct reasoning' was
substantially low (38%). The format of the mathematical notation had a
negligible impact on performance. SLMs exhibited varied performance across the
physics topics and showed a decline in reasoning quality with increasing
cognitive and knowledge complexity. In particular, the consistency of reasoning
was largely maintained in diverse cultural contexts, especially by better
performing models. These findings indicate that, while SLMs can often find
correct answers, their underlying reasoning is frequently flawed, suggesting an
overreliance on pattern recognition. For SLMs to become reliable educational
tools in physics, future development must prioritize enhancing genuine
understanding and the generation of sound, verifiable reasoning chains over
mere answer accuracy.

---


### [Can we Debias Social Stereotypes in AI-Generated Images? Examining Text-to-Image Outputs and User Perceptions](http://arxiv.org/abs/2505.20692v1)

Recent advances in generative AI have enabled visual content creation through
text-to-image (T2I) generation. However, despite their creative potential, T2I
models often replicate and amplify societal stereotypes -- particularly those
related to gender, race, and culture -- raising important ethical concerns.
This paper proposes a theory-driven bias detection rubric and a Social
Stereotype Index (SSI) to systematically evaluate social biases in T2I outputs.
We audited three major T2I model outputs -- DALL-E-3, Midjourney-6.1, and
Stability AI Core -- using 100 queries across three categories -- geocultural,
occupational, and adjectival. Our analysis reveals that initial outputs are
prone to include stereotypical visual cues, including gendered professions,
cultural markers, and western beauty norms. To address this, we adopted our
rubric to conduct targeted prompt refinement using LLMs, which significantly
reduced bias -- SSI dropped by 61% for geocultural, 69% for occupational, and
51% for adjectival queries. We complemented our quantitative analysis through a
user study examining perceptions, awareness, and preferences around
AI-generated biased imagery. Our findings reveal a key tension -- although
prompt refinement can mitigate stereotypes, it can limit contextual alignment.
Interestingly, users often perceived stereotypical images to be more aligned
with their expectations. We discuss the need to balance ethical debiasing with
contextual relevance and call for T2I systems that support global diversity and
inclusivity while not compromising the reflection of real-world social
complexity.

---


### [Accelerating RL for LLM Reasoning with Optimal Advantage Regression](http://arxiv.org/abs/2505.20686v1)

Reinforcement learning (RL) has emerged as a powerful tool for fine-tuning
large language models (LLMs) to improve complex reasoning abilities. However,
state-of-the-art policy optimization methods often suffer from high
computational overhead and memory consumption, primarily due to the need for
multiple generations per prompt and the reliance on critic networks or
advantage estimates of the current policy. In this paper, we propose $A$*-PO, a
novel two-stage policy optimization framework that directly approximates the
optimal advantage function and enables efficient training of LLMs for reasoning
tasks. In the first stage, we leverage offline sampling from a reference policy
to estimate the optimal value function $V$*, eliminating the need for costly
online value estimation. In the second stage, we perform on-policy updates
using a simple least-squares regression loss with only a single generation per
prompt. Theoretically, we establish performance guarantees and prove that the
KL-regularized RL objective can be optimized without requiring complex
exploration strategies. Empirically, $A$*-PO achieves competitive performance
across a wide range of mathematical reasoning benchmarks, while reducing
training time by up to 2$\times$ and peak memory usage by over 30% compared to
PPO, GRPO, and REBEL. Implementation of $A$*-PO can be found at
https://github.com/ZhaolinGao/A-PO.

---


### [Pretraining Language Models to Ponder in Continuous Space](http://arxiv.org/abs/2505.20674v1)

Humans ponder before articulating complex sentence elements, enabling deeper
cognitive processing through focused effort. In this work, we introduce this
pondering process into language models by repeatedly invoking the forward
process within a single token generation step. During pondering, instead of
generating an actual token sampled from the prediction distribution, the model
ponders by yielding a weighted sum of all token embeddings according to the
predicted token distribution. The generated embedding is then fed back as input
for another forward pass. We show that the model can learn to ponder in this
way through self-supervised learning, without any human annotations. Our method
is straightforward and can be seamlessly integrated with various existing
language models. Experiments across three widely used open-source
architectures-GPT-2, Pythia, and LLaMA-and extensive downstream task
evaluations demonstrate the effectiveness and generality of our method. For
language modeling tasks, pondering language models achieve performance
comparable to vanilla models with twice the number of parameters. On 9
downstream benchmarks, our pondering-enhanced Pythia models significantly
outperform the official Pythia models. Notably, pondering-enhanced Pythia-1B is
comparable to TinyLlama-1.1B, which is trained on 10 times more data. The code
is available at https://github.com/LUMIA-Group/PonderingLM.

---


### [GIFARC: Synthetic Dataset for Leveraging Human-Intuitive Analogies to Elevate AI Reasoning](http://arxiv.org/abs/2505.20672v1)

The Abstraction and Reasoning Corpus (ARC) poses a stringent test of general
AI capabilities, requiring solvers to infer abstract patterns from only a
handful of examples. Despite substantial progress in deep learning,
state-of-the-art models still achieve accuracy rates of merely 40-55% on 2024
ARC Competition, indicative of a significant gap between their performance and
human-level reasoning. In this work, we seek to bridge that gap by introducing
an analogy-inspired ARC dataset, GIFARC. Leveraging large language models
(LLMs) and vision-language models (VLMs), we synthesize new ARC-style tasks
from a variety of GIF images that include analogies. Each new task is paired
with ground-truth analogy, providing an explicit mapping between visual
transformations and everyday concepts. By embedding robust human-intuitive
analogies into ARC-style tasks, GIFARC guides AI agents to evaluate the task
analogically before engaging in brute-force pattern search, thus efficiently
reducing problem complexity and build a more concise and human-understandable
solution. We empirically validate that guiding LLM with analogic approach with
GIFARC affects task-solving approaches of LLMs to align with analogic approach
of human.

---


### [LLM-Guided Reinforcement Learning: Addressing Training Bottlenecks through Policy Modulation](http://arxiv.org/abs/2505.20671v1)

While reinforcement learning (RL) has achieved notable success in various
domains, training effective policies for complex tasks remains challenging.
Agents often converge to local optima and fail to maximize long-term rewards.
Existing approaches to mitigate training bottlenecks typically fall into two
categories: (i) Automated policy refinement, which identifies critical states
from past trajectories to guide policy updates, but suffers from costly and
uncertain model training; and (ii) Human-in-the-loop refinement, where human
feedback is used to correct agent behavior, but this does not scale well to
environments with large or continuous action spaces. In this work, we design a
large language model-guided policy modulation framework that leverages LLMs to
improve RL training without additional model training or human intervention. We
first prompt an LLM to identify critical states from a sub-optimal agent's
trajectories. Based on these states, the LLM then provides action suggestions
and assigns implicit rewards to guide policy refinement. Experiments across
standard RL benchmarks demonstrate that our method outperforms state-of-the-art
baselines, highlighting the effectiveness of LLM-based explanations in
addressing RL training bottlenecks.

---


### [MIRROR: Multi-agent Intra- and Inter-Reflection for Optimized Reasoning in Tool Learning](http://arxiv.org/abs/2505.20670v1)

Complex tasks involving tool integration pose significant challenges for
Large Language Models (LLMs), leading to the emergence of multi-agent workflows
as a promising solution. Reflection has emerged as an effective strategy for
correcting erroneous trajectories in agentic workflows. However, existing
approaches only exploit such capability in the post-action stage, where the
agent observes the execution outcomes. We argue that, like humans, LLMs can
also engage in reflection before action execution: the agent can anticipate
undesirable outcomes from its own decisions, which not only provides a
necessarily complementary perspective to evaluate the decision but also
prevents the propagation of errors throughout the trajectory. In this paper, we
propose MIRROR, a framework that consists of both intra-reflection, which
critically assesses intended actions before execution, and inter-reflection,
which further adjusts the trajectory based on observations. This design
systematically leverages LLM reflection capabilities to eliminate and rectify
erroneous actions on a more comprehensive scope. Evaluations on both the
StableToolBench and TravelPlanner benchmarks demonstrate MIRROR's superior
performance, achieving state-of-the-art results compared to existing
approaches.

---


### [Self-Route: Automatic Mode Switching via Capability Estimation for Efficient Reasoning](http://arxiv.org/abs/2505.20664v1)

While reasoning-augmented large language models (RLLMs) significantly enhance
complex task performance through extended reasoning chains, they inevitably
introduce substantial unnecessary token consumption, particularly for simpler
problems where Short Chain-of-Thought (Short CoT) suffices. This overthinking
phenomenon leads to inefficient resource usage without proportional accuracy
gains. To address this issue, we propose Self-Route, a dynamic reasoning
framework that automatically selects between general and reasoning modes based
on model capability estimation. Our approach introduces a lightweight
pre-inference stage to extract capability-aware embeddings from hidden layer
representations, enabling real-time evaluation of the model's ability to solve
problems. We further construct Gradient-10K, a model difficulty
estimation-based dataset with dense complexity sampling, to train the router
for precise capability boundary detection. Extensive experiments demonstrate
that Self-Route achieves comparable accuracy to reasoning models while reducing
token consumption by 30-55\% across diverse benchmarks. The proposed framework
demonstrates consistent effectiveness across models with different parameter
scales and reasoning paradigms, highlighting its general applicability and
practical value.

---


### [TeroSeek: An AI-Powered Knowledge Base and Retrieval Generation Platform for Terpenoid Research](http://arxiv.org/abs/2505.20663v1)

Terpenoids are a crucial class of natural products that have been studied for
over 150 years, but their interdisciplinary nature (spanning chemistry,
pharmacology, and biology) complicates knowledge integration. To address this,
the authors developed TeroSeek, a curated knowledge base (KB) built from two
decades of terpenoid literature, coupled with an AI-powered question-answering
chatbot and web service. Leveraging a retrieval-augmented generation (RAG)
framework, TeroSeek provides structured, high-quality information and
outperforms general-purpose large language models (LLMs) in terpenoid-related
queries. It serves as a domain-specific expert tool for multidisciplinary
research and is publicly available at http://teroseek.qmclab.com.

---


### [FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information](http://arxiv.org/abs/2505.20650v1)

We introduce FinTagging, the first full-scope, table-aware XBRL benchmark
designed to evaluate the structured information extraction and semantic
alignment capabilities of large language models (LLMs) in the context of
XBRL-based financial reporting. Unlike prior benchmarks that oversimplify XBRL
tagging as flat multi-class classification and focus solely on narrative text,
FinTagging decomposes the XBRL tagging problem into two subtasks: FinNI for
financial entity extraction and FinCL for taxonomy-driven concept alignment. It
requires models to jointly extract facts and align them with the full 10k+
US-GAAP taxonomy across both unstructured text and structured tables, enabling
realistic, fine-grained evaluation. We assess a diverse set of LLMs under
zero-shot settings, systematically analyzing their performance on both subtasks
and overall tagging accuracy. Our results reveal that, while LLMs demonstrate
strong generalization in information extraction, they struggle with
fine-grained concept alignment, particularly in disambiguating closely related
taxonomy entries. These findings highlight the limitations of existing LLMs in
fully automating XBRL tagging and underscore the need for improved semantic
reasoning and schema-aware modeling to meet the demands of accurate financial
disclosure. Code is available at our GitHub repository and data is at our
Hugging Face repository.

---


### [Test-Time Learning for Large Language Models](http://arxiv.org/abs/2505.20633v1)

While Large Language Models (LLMs) have exhibited remarkable emergent
capabilities through extensive pre-training, they still face critical
limitations in generalizing to specialized domains and handling diverse
linguistic variations, known as distribution shifts. In this paper, we propose
a Test-Time Learning (TTL) paradigm for LLMs, namely TLM, which dynamically
adapts LLMs to target domains using only unlabeled test data during testing.
Specifically, we first provide empirical evidence and theoretical insights to
reveal that more accurate predictions from LLMs can be achieved by minimizing
the input perplexity of the unlabeled test data. Based on this insight, we
formulate the Test-Time Learning process of LLMs as input perplexity
minimization, enabling self-supervised enhancement of LLM performance.
Furthermore, we observe that high-perplexity samples tend to be more
informative for model optimization. Accordingly, we introduce a Sample
Efficient Learning Strategy that actively selects and emphasizes these
high-perplexity samples for test-time updates. Lastly, to mitigate catastrophic
forgetting and ensure adaptation stability, we adopt Low-Rank Adaptation (LoRA)
instead of full-parameter optimization, which allows lightweight model updates
while preserving more original knowledge from the model. We introduce the
AdaptEval benchmark for TTL and demonstrate through experiments that TLM
improves performance by at least 20% compared to original LLMs on domain
knowledge adaptation.

---


### [SeqPO-SiMT: Sequential Policy Optimization for Simultaneous Machine Translation](http://arxiv.org/abs/2505.20622v1)

We present Sequential Policy Optimization for Simultaneous Machine
Translation (SeqPO-SiMT), a new policy optimization framework that defines the
simultaneous machine translation (SiMT) task as a sequential decision making
problem, incorporating a tailored reward to enhance translation quality while
reducing latency. In contrast to popular Reinforcement Learning from Human
Feedback (RLHF) methods, such as PPO and DPO, which are typically applied in
single-step tasks, SeqPO-SiMT effectively tackles the multi-step SiMT task.
This intuitive framework allows the SiMT LLMs to simulate and refine the SiMT
process using a tailored reward. We conduct experiments on six datasets from
diverse domains for En to Zh and Zh to En SiMT tasks, demonstrating that
SeqPO-SiMT consistently achieves significantly higher translation quality with
lower latency. In particular, SeqPO-SiMT outperforms the supervised fine-tuning
(SFT) model by 1.13 points in COMET, while reducing the Average Lagging by 6.17
in the NEWSTEST2021 En to Zh dataset. While SiMT operates with far less context
than offline translation, the SiMT results of SeqPO-SiMT on 7B LLM surprisingly
rival the offline translation of high-performing LLMs, including
Qwen-2.5-7B-Instruct and LLaMA-3-8B-Instruct.

---


### [Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning](http://arxiv.org/abs/2505.20621v1)

Similar to other machine learning frameworks, Offline Reinforcement Learning
(RL) is shown to be vulnerable to poisoning attacks, due to its reliance on
externally sourced datasets, a vulnerability that is exacerbated by its
sequential nature. To mitigate the risks posed by RL poisoning, we extend
certified defenses to provide larger guarantees against adversarial
manipulation, ensuring robustness for both per-state actions, and the overall
expected cumulative reward. Our approach leverages properties of Differential
Privacy, in a manner that allows this work to span both continuous and discrete
spaces, as well as stochastic and deterministic environments -- significantly
expanding the scope and applicability of achievable guarantees. Empirical
evaluations demonstrate that our approach ensures the performance drops to no
more than $50\%$ with up to $7\%$ of the training data poisoned, significantly
improving over the $0.008\%$ in prior work~\citep{wu_copa_2022}, while
producing certified radii that is $5$ times larger as well. This highlights the
potential of our framework to enhance safety and reliability in offline RL.

---


### [Reinforcing General Reasoning without Verifiers](http://arxiv.org/abs/2505.21493v1)

The recent paradigm shift towards training large language models (LLMs) using
DeepSeek-R1-Zero-style reinforcement learning (RL) on verifiable rewards has
led to impressive advancements in code and mathematical reasoning. However,
this methodology is limited to tasks where rule-based answer verification is
possible and does not naturally extend to real-world domains such as chemistry,
healthcare, engineering, law, biology, business, and economics. Current
practical workarounds use an additional LLM as a model-based verifier; however,
this introduces issues such as reliance on a strong verifier LLM,
susceptibility to reward hacking, and the practical burden of maintaining the
verifier model in memory during training. To address this and extend
DeepSeek-R1-Zero-style training to general reasoning domains, we propose a
verifier-free method (VeriFree) that bypasses answer verification and instead
uses RL to directly maximize the probability of generating the reference
answer. We compare VeriFree with verifier-based methods and demonstrate that,
in addition to its significant practical benefits and reduced compute
requirements, VeriFree matches and even surpasses verifier-based methods on
extensive evaluations across MMLU-Pro, GPQA, SuperGPQA, and math-related
benchmarks. Moreover, we provide insights into this method from multiple
perspectives: as an elegant integration of training both the policy and
implicit verifier in a unified model, and as a variational optimization
approach. Code is available at https://github.com/sail-sg/VeriFree.

---


### [Leveraging Large Language Models for Bengali Math Word Problem Solving with Chain of Thought Reasoning](http://arxiv.org/abs/2505.21354v1)

Solving Bengali Math Word Problems (MWPs) remains a major challenge in
natural language processing (NLP) due to the language's low-resource status and
the multi-step reasoning required. Existing models struggle with complex
Bengali MWPs, largely because no human-annotated Bengali dataset has previously
addressed this task. This gap has limited progress in Bengali mathematical
reasoning. To address this, we created SOMADHAN, a dataset of 8792 complex
Bengali MWPs with manually written, step-by-step solutions. We designed this
dataset to support reasoning-focused evaluation and model development in a
linguistically underrepresented context. Using SOMADHAN, we evaluated a range
of large language models (LLMs) - including GPT-4o, GPT-3.5 Turbo, LLaMA series
models, Deepseek, and Qwen - through both zero-shot and few-shot prompting with
and without Chain of Thought (CoT) reasoning. CoT prompting consistently
improved performance over standard prompting, especially in tasks requiring
multi-step logic. LLaMA-3.3 70B achieved the highest accuracy of 88% with
few-shot CoT prompting. We also applied Low-Rank Adaptation (LoRA) to fine-tune
models efficiently, enabling them to adapt to Bengali MWPs with minimal
computational cost. Our work fills a critical gap in Bengali NLP by providing a
high-quality reasoning dataset and a scalable framework for solving complex
MWPs. We aim to advance equitable research in low-resource languages and
enhance reasoning capabilities in educational and language technologies.

---


### [Leveraging large language models and traditional machine learning ensembles for ADHD detection from narrative transcripts](http://arxiv.org/abs/2505.21324v1)

Despite rapid advances in large language models (LLMs), their integration
with traditional supervised machine learning (ML) techniques that have proven
applicability to medical data remains underexplored. This is particularly true
for psychiatric applications, where narrative data often exhibit nuanced
linguistic and contextual complexity, and can benefit from the combination of
multiple models with differing characteristics. In this study, we introduce an
ensemble framework for automatically classifying
Attention-Deficit/Hyperactivity Disorder (ADHD) diagnosis (binary) using
narrative transcripts. Our approach integrates three complementary models:
LLaMA3, an open-source LLM that captures long-range semantic structure;
RoBERTa, a pre-trained transformer model fine-tuned on labeled clinical
narratives; and a Support Vector Machine (SVM) classifier trained using
TF-IDF-based lexical features. These models are aggregated through a majority
voting mechanism to enhance predictive robustness. The dataset includes 441
instances, including 352 for training and 89 for validation. Empirical results
show that the ensemble outperforms individual models, achieving an F$_1$ score
of 0.71 (95\% CI: [0.60-0.80]). Compared to the best-performing individual
model (SVM), the ensemble improved recall while maintaining competitive
precision. This indicates the strong sensitivity of the ensemble in identifying
ADHD-related linguistic cues. These findings demonstrate the promise of hybrid
architectures that leverage the semantic richness of LLMs alongside the
interpretability and pattern recognition capabilities of traditional supervised
ML, offering a new direction for robust and generalizable psychiatric text
classification.

---


### [Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings](http://arxiv.org/abs/2505.21242v1)

Large Language Models (LLMs) recently achieved great success in medical text
summarization by simply using in-context learning. However, these recent
efforts do not perform fine-grained evaluations under difficult settings where
LLMs might fail. They typically report performance scores over the entire
dataset. Through our benchmarking study, we show that LLMs show a significant
performance drop for data points with high concentration of out-of-vocabulary
(OOV) words or with high novelty. Vocabulary adaptation is an intuitive
solution to this vocabulary mismatch issue where the LLM vocabulary gets
updated with certain expert domain (here, medical) words or subwords. An
interesting finding from our study is that Llama-3.1, even with a vocabulary
size of around 128K tokens, still faces over-fragmentation issue with medical
words. To that end, we show vocabulary adaptation helps improve the LLM
summarization performance even in difficult settings. Through extensive
experimentation of multiple vocabulary adaptation strategies, two continual
pretraining strategies, and three benchmark medical summarization datasets, we
gain valuable insights into the role of vocabulary adaptation strategies for
customizing LLMs to the medical domain. We also performed a human evaluation
study with medical experts where they found that vocabulary adaptation results
in more relevant and faithful summaries. Our codebase is made publicly
available at https://github.com/gb-kgp/LLM-MedicalSummarization-Benchmark.

---


### [Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning](http://arxiv.org/abs/2505.21178v1)

As test-time scaling becomes a pivotal research frontier in Large Language
Models (LLMs) development, contemporary and advanced post-training
methodologies increasingly focus on extending the generation length of long
Chain-of-Thought (CoT) responses to enhance reasoning capabilities toward
DeepSeek R1-like performance. However, recent studies reveal a persistent
overthinking phenomenon in state-of-the-art reasoning models, manifesting as
excessive redundancy or repetitive thinking patterns in long CoT responses. To
address this issue, in this paper, we propose a simple yet effective two-stage
reinforcement learning framework for achieving concise reasoning in LLMs, named
ConciseR. Specifically, the first stage, using more training steps, aims to
incentivize the model's reasoning capabilities via Group Relative Policy
Optimization with clip-higher and dynamic sampling components (GRPO++), and the
second stage, using fewer training steps, explicitly enforces conciseness and
improves efficiency via Length-aware Group Relative Policy Optimization
(L-GRPO). Significantly, ConciseR only optimizes response length once all
rollouts of a sample are correct, following the "walk before you run"
principle. Extensive experimental results demonstrate that our ConciseR model,
which generates more concise CoT reasoning responses, outperforms recent
state-of-the-art reasoning models with zero RL paradigm across AIME 2024,
MATH-500, AMC 2023, Minerva, and Olympiad benchmarks.

---


### [TAT-R1: Terminology-Aware Translation with Reinforcement Learning and Word Alignment](http://arxiv.org/abs/2505.21172v1)

Recently, deep reasoning large language models(LLMs) like DeepSeek-R1 have
made significant progress in tasks such as mathematics and coding. Inspired by
this, several studies have employed reinforcement learning(RL) to enhance
models' deep reasoning capabilities and improve machine translation(MT)
quality. However, the terminology translation, an essential task in MT, remains
unexplored in deep reasoning LLMs. In this paper, we propose \textbf{TAT-R1}, a
terminology-aware translation model trained with reinforcement learning and
word alignment. Specifically, we first extract the keyword translation pairs
using a word alignment model. Then we carefully design three types of
rule-based alignment rewards with the extracted alignment relationships. With
those alignment rewards, the RL-trained translation model can learn to focus on
the accurate translation of key information, including terminology in the
source text. Experimental results show the effectiveness of TAT-R1. Our model
significantly improves terminology translation accuracy compared to the
baseline models while maintaining comparable performance on general translation
tasks. In addition, we conduct detailed ablation studies of the
DeepSeek-R1-like training paradigm for machine translation and reveal several
key findings.

---


### [Will It Still Be True Tomorrow? Multilingual Evergreen Question Classification to Improve Trustworthy QA](http://arxiv.org/abs/2505.21115v1)

Large Language Models (LLMs) often hallucinate in question answering (QA)
tasks. A key yet underexplored factor contributing to this is the temporality
of questions -- whether they are evergreen (answers remain stable over time) or
mutable (answers change). In this work, we introduce EverGreenQA, the first
multilingual QA dataset with evergreen labels, supporting both evaluation and
training. Using EverGreenQA, we benchmark 12 modern LLMs to assess whether they
encode question temporality explicitly (via verbalized judgments) or implicitly
(via uncertainty signals). We also train EG-E5, a lightweight multilingual
classifier that achieves SoTA performance on this task. Finally, we demonstrate
the practical utility of evergreen classification across three applications:
improving self-knowledge estimation, filtering QA datasets, and explaining
GPT-4o retrieval behavior.

---


### [Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation](http://arxiv.org/abs/2505.21072v1)

Large Language Models (LLMs) enhanced with external knowledge retrieval, an
approach known as Retrieval-Augmented Generation (RAG), have shown strong
performance in open-domain question answering. However, RAG systems remain
susceptible to hallucinations: factually incorrect outputs that may arise
either from inconsistencies in the model's internal knowledge or incorrect use
of the retrieved context. Existing approaches often conflate factuality with
faithfulness to the retrieved context, misclassifying factually correct
statements as hallucinations if they are not directly supported by the
retrieval. In this paper, we introduce FRANQ (Faithfulness-based Retrieval
Augmented UNcertainty Quantification), a novel method for hallucination
detection in RAG outputs. FRANQ applies different Uncertainty Quantification
(UQ) techniques to estimate factuality based on whether a statement is faithful
to the retrieved context or not. To evaluate FRANQ and other UQ techniques for
RAG, we present a new long-form Question Answering (QA) dataset annotated for
both factuality and faithfulness, combining automated labeling with manual
validation of challenging examples. Extensive experiments on long- and
short-form QA across multiple datasets and LLMs show that FRANQ achieves more
accurate detection of factual errors in RAG-generated responses compared to
existing methods.

---


### [Predicting Implicit Arguments in Procedural Video Instructions](http://arxiv.org/abs/2505.21068v1)

Procedural texts help AI enhance reasoning about context and action
sequences. Transforming these into Semantic Role Labeling (SRL) improves
understanding of individual steps by identifying predicate-argument structure
like {verb,what,where/with}. Procedural instructions are highly elliptic, for
instance, (i) add cucumber to the bowl and (ii) add sliced tomatoes, the second
step's where argument is inferred from the context, referring to where the
cucumber was placed. Prior SRL benchmarks often miss implicit arguments,
leading to incomplete understanding. To address this, we introduce
Implicit-VidSRL, a dataset that necessitates inferring implicit and explicit
arguments from contextual information in multimodal cooking procedures. Our
proposed dataset benchmarks multimodal models' contextual reasoning, requiring
entity tracking through visual changes in recipes. We study recent multimodal
LLMs and reveal that they struggle to predict implicit arguments of what and
where/with from multi-modal procedural data given the verb. Lastly, we propose
iSRL-Qwen2-VL, which achieves a 17% relative improvement in F1-score for
what-implicit and a 14.7% for where/with-implicit semantic roles over GPT-4o.

---


### [Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](http://arxiv.org/abs/2505.21033v1)

Dialogue Topic Segmentation (DTS) aims to divide dialogues into coherent
segments. DTS plays a crucial role in various NLP downstream tasks, but suffers
from chronic problems: data shortage, labeling ambiguity, and incremental
complexity of recently proposed solutions. On the other hand, Despite advances
in Large Language Models (LLMs) and reasoning strategies, these have rarely
been applied to DTS. This paper introduces Def-DTS: Deductive Reasoning for
Open-domain Dialogue Topic Segmentation, which utilizes LLM-based multi-step
deductive reasoning to enhance DTS performance and enable case study using
intermediate result. Our method employs a structured prompting approach for
bidirectional context summarization, utterance intent classification, and
deductive topic shift detection. In the intent classification process, we
propose the generalizable intent list for domain-agnostic dialogue intent
classification. Experiments in various dialogue settings demonstrate that
Def-DTS consistently outperforms traditional and state-of-the-art approaches,
with each subtask contributing to improved performance, particularly in
reducing type 2 error. We also explore the potential for autolabeling,
emphasizing the importance of LLM reasoning techniques in DTS.

---


### [On VLMs for Diverse Tasks in Multimodal Meme Classification](http://arxiv.org/abs/2505.20937v1)

In this paper, we present a comprehensive and systematic analysis of
vision-language models (VLMs) for disparate meme classification tasks. We
introduced a novel approach that generates a VLM-based understanding of meme
images and fine-tunes the LLMs on textual understanding of the embedded meme
text for improving the performance. Our contributions are threefold: (1)
Benchmarking VLMs with diverse prompting strategies purposely to each sub-task;
(2) Evaluating LoRA fine-tuning across all VLM components to assess performance
gains; and (3) Proposing a novel approach where detailed meme interpretations
generated by VLMs are used to train smaller language models (LLMs),
significantly improving classification. The strategy of combining VLMs with
LLMs improved the baseline performance by 8.34%, 3.52% and 26.24% for sarcasm,
offensive and sentiment classification, respectively. Our results reveal the
strengths and limitations of VLMs and present a novel strategy for meme
understanding.

---


### [Automated Privacy Information Annotation in Large Language Model Interactions](http://arxiv.org/abs/2505.20910v1)

Users interacting with large language models (LLMs) under their real
identifiers often unknowingly risk disclosing private information.
Automatically notifying users whether their queries leak privacy and which
phrases leak what private information has therefore become a practical need.
Existing privacy detection methods, however, were designed for different
objectives and application scenarios, typically tagging personally identifiable
information (PII) in anonymous content. In this work, to support the
development and evaluation of privacy detection models for LLM interactions
that are deployable on local user devices, we construct a large-scale
multilingual dataset with 249K user queries and 154K annotated privacy phrases.
In particular, we build an automated privacy annotation pipeline with
cloud-based strong LLMs to automatically extract privacy phrases from dialogue
datasets and annotate leaked information. We also design evaluation metrics at
the levels of privacy leakage, extracted privacy phrase, and privacy
information. We further establish baseline methods using light-weight LLMs with
both tuning-free and tuning-based methods, and report a comprehensive
evaluation of their performance. Evaluation results reveal a gap between
current performance and the requirements of real-world LLM applications,
motivating future research into more effective local privacy detection methods
grounded in our dataset.

---


### [MSA at SemEval-2025 Task 3: High Quality Weak Labeling and LLM Ensemble Verification for Multilingual Hallucination Detection](http://arxiv.org/abs/2505.20880v1)

This paper describes our submission for SemEval-2025 Task 3: Mu-SHROOM, the
Multilingual Shared-task on Hallucinations and Related Observable
Overgeneration Mistakes. The task involves detecting hallucinated spans in text
generated by instruction-tuned Large Language Models (LLMs) across multiple
languages. Our approach combines task-specific prompt engineering with an LLM
ensemble verification mechanism, where a primary model extracts hallucination
spans and three independent LLMs adjudicate their validity through
probability-based voting. This framework simulates the human annotation
workflow used in the shared task validation and test data. Additionally, fuzzy
matching refines span alignment. Our system ranked 1st in Arabic and Basque,
2nd in German, Swedish, and Finnish, and 3rd in Czech, Farsi, and French.

---


### [Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG](http://arxiv.org/abs/2505.20871v1)

Large language models (LLMs) augmented with retrieval systems have
significantly advanced natural language processing tasks by integrating
external knowledge sources, enabling more accurate and contextually rich
responses. To improve the robustness of such systems against noisy retrievals,
Retrieval-Augmented Fine-Tuning (RAFT) has emerged as a widely adopted method.
However, RAFT conditions models to generate answers even in the absence of
reliable knowledge. This behavior undermines their reliability in high-stakes
domains, where acknowledging uncertainty is critical. To address this issue, we
propose Divide-Then-Align (DTA), a post-training approach designed to endow RAG
systems with the ability to respond with "I don't know" when the query is out
of the knowledge boundary of both the retrieved passages and the model's
internal knowledge. DTA divides data samples into four knowledge quadrants and
constructs tailored preference data for each quadrant, resulting in a curated
dataset for Direct Preference Optimization (DPO). Experimental results on three
benchmark datasets demonstrate that DTA effectively balances accuracy with
appropriate abstention, enhancing the reliability and trustworthiness of
retrieval-augmented systems.

---


### [Reinforced Informativeness Optimization for Long-Form Retrieval-Augmented Generation](http://arxiv.org/abs/2505.20825v1)

Long-form question answering (LFQA) presents unique challenges for large
language models, requiring the synthesis of coherent, paragraph-length answers.
While retrieval-augmented generation (RAG) systems have emerged as a promising
solution, existing research struggles with key limitations: the scarcity of
high-quality training data for long-form generation, the compounding risk of
hallucination in extended outputs, and the absence of reliable evaluation
metrics for factual completeness. In this paper, we propose RioRAG, a novel
reinforcement learning (RL) framework that advances long-form RAG through
reinforced informativeness optimization. Our approach introduces two
fundamental innovations to address the core challenges. First, we develop an RL
training paradigm of reinforced informativeness optimization that directly
optimizes informativeness and effectively addresses the slow-thinking deficit
in conventional RAG systems, bypassing the need for expensive supervised data.
Second, we propose a nugget-centric hierarchical reward modeling approach that
enables precise assessment of long-form answers through a three-stage process:
extracting the nugget from every source webpage, constructing a nugget claim
checklist, and computing rewards based on factual alignment. Extensive
experiments on two LFQA benchmarks LongFact and RAGChecker demonstrate the
effectiveness of the proposed method. Our codes are available at
https://github.com/RUCAIBox/RioRAG.

---


### [Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective](http://arxiv.org/abs/2505.20816v1)

Recent advances in multimodal question answering have primarily focused on
combining heterogeneous modalities or fine-tuning multimodal large language
models. While these approaches have shown strong performance, they often rely
on a single, generalized reasoning strategy, overlooking the unique
characteristics of each modality ultimately limiting both accuracy and
interpretability. To address these limitations, we propose MAMMQA, a
multi-agent QA framework for multimodal inputs spanning text, tables, and
images. Our system includes two Visual Language Model (VLM) agents and one
text-based Large Language Model (LLM) agent. The first VLM decomposes the user
query into sub-questions and sequentially retrieves partial answers from each
modality. The second VLM synthesizes and refines these results through
cross-modal reasoning. Finally, the LLM integrates the insights into a cohesive
answer. This modular design enhances interpretability by making the reasoning
process transparent and allows each agent to operate within its domain of
expertise. Experiments on diverse multimodal QA benchmarks demonstrate that our
cooperative, multi-agent framework consistently outperforms existing baselines
in both accuracy and robustness.

---


### [SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution](http://arxiv.org/abs/2505.20732v1)

Reinforcement learning (RL) holds significant promise for training LLM agents
to handle complex, goal-oriented tasks that require multi-step interactions
with external environments. However, a critical challenge when applying RL to
these agentic tasks arises from delayed rewards: feedback signals are typically
available only after the entire task is completed. This makes it non-trivial to
assign delayed rewards to earlier actions, providing insufficient guidance
regarding environmental constraints and hindering agent training. In this work,
we draw on the insight that the ultimate completion of a task emerges from the
cumulative progress an agent makes across individual steps. We propose Stepwise
Progress Attribution (SPA), a general reward redistribution framework that
decomposes the final reward into stepwise contributions, each reflecting its
incremental progress toward overall task completion. To achieve this, we train
a progress estimator that accumulates stepwise contributions over a trajectory
to match the task completion. During policy optimization, we combine the
estimated per-step contribution with a grounding signal for actions executed in
the environment as the fine-grained, intermediate reward for effective agent
training. Extensive experiments on common agent benchmarks (including Webshop,
ALFWorld, and VirtualHome) demonstrate that SPA consistently outperforms the
state-of-the-art method in both success rate (+2.5\% on average) and grounding
accuracy (+1.9\% on average). Further analyses demonstrate that our method
remarkably provides more effective intermediate rewards for RL training. Our
code is available at https://github.com/WangHanLinHenry/SPA-RL-Agent.

---


### [MUSEG: Reinforcing Video Temporal Understanding via Timestamp-Aware Multi-Segment Grounding](http://arxiv.org/abs/2505.20715v1)

Video temporal understanding is crucial for multimodal large language models
(MLLMs) to reason over events in videos. Despite recent advances in general
video understanding, current MLLMs still struggle with fine-grained temporal
reasoning. While reinforcement learning (RL) has been explored to address this
issue recently, existing RL approaches remain limited in effectiveness. In this
work, we propose MUSEG, a novel RL-based method that enhances temporal
understanding by introducing timestamp-aware multi-segment grounding. MUSEG
enables MLLMs to align queries with multiple relevant video segments, promoting
more comprehensive temporal reasoning. To facilitate effective learning, we
design a customized RL training recipe with phased rewards that progressively
guides the model toward temporally grounded reasoning. Extensive experiments on
temporal grounding and time-sensitive video QA tasks demonstrate that MUSEG
significantly outperforms existing methods and generalizes well across diverse
temporal understanding scenarios. View our project at
https://github.com/THUNLP-MT/MUSEG.

---


### [Beyond Templates: Dynamic Adaptation of Reasoning Demonstrations via Feasibility-Aware Exploration](http://arxiv.org/abs/2505.20700v1)

Large language models (LLMs) have shown remarkable reasoning capabilities,
yet aligning such abilities to small language models (SLMs) remains a challenge
due to distributional mismatches and limited model capacity. Existing reasoning
datasets, typically designed for powerful LLMs, often lead to degraded
performance when directly applied to weaker models. In this work, we introduce
Dynamic Adaptation of Reasoning Trajectories (DART), a novel data adaptation
framework that bridges the capability gap between expert reasoning trajectories
and diverse SLMs. Instead of uniformly imitating expert steps, DART employs a
selective imitation strategy guided by step-wise adaptability estimation via
solution simulation. When expert steps surpass the student's capacity --
signaled by an Imitation Gap -- the student autonomously explores alternative
reasoning paths, constrained by outcome consistency. We validate DART across
multiple reasoning benchmarks and model scales, demonstrating that it
significantly improves generalization and data efficiency over static
fine-tuning. Our method enhances supervision quality by aligning training
signals with the student's reasoning capabilities, offering a scalable solution
for reasoning alignment in resource-constrained models.

---


### [SELF-PERCEPT: Introspection Improves Large Language Models' Detection of Multi-Person Mental Manipulation in Conversations](http://arxiv.org/abs/2505.20679v1)

Mental manipulation is a subtle yet pervasive form of abuse in interpersonal
communication, making its detection critical for safeguarding potential
victims. However, due to manipulation's nuanced and context-specific nature,
identifying manipulative language in complex, multi-turn, and multi-person
conversations remains a significant challenge for large language models (LLMs).
To address this gap, we introduce the MultiManip dataset, comprising 220
multi-turn, multi-person dialogues balanced between manipulative and
non-manipulative interactions, all drawn from reality shows that mimic
real-world scenarios. For manipulative interactions, it includes 11 distinct
manipulations depicting real-life scenarios. We conduct extensive evaluations
of state-of-the-art LLMs, such as GPT-4o and Llama-3.1-8B, employing various
prompting strategies. Despite their capabilities, these models often struggle
to detect manipulation effectively. To overcome this limitation, we propose
SELF-PERCEPT, a novel, two-stage prompting framework inspired by
Self-Perception Theory, demonstrating strong performance in detecting
multi-person, multi-turn mental manipulation. Our code and data are publicly
available at https://github.com/danushkhanna/self-percept .

---


### [Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration](http://arxiv.org/abs/2505.20625v1)

Processing long contexts has become a critical capability for modern large
language models (LLMs). Existing works leverage agent-based divide-and-conquer
methods for processing long contexts. But these methods face crucial
limitations, including prohibitive accumulated latency and amplified
information loss from excessive agent invocations, and the disruption of
inherent textual dependencies by immoderate partitioning. In this paper, we
propose a novel multi-agent framework XpandA (Expand-Agent) coupled with
question-driven workflow and dynamic partitioning for robust long-context
processing. XpandA overcomes these limitations through: 1) dynamic partitioning
of long texts, which adaptively modulates the filling rate of context windows
for input sequences of vastly varying lengths; 2) question-guided protocol to
update flat information ensembles within centralized shared memory,
constructing consistent inter-agent knowledge across partitions; and 3)
selectively replaying specific partitions based on the state-tracking of
question-information couples to promote the resolution of inverted-order
structures across partitions (e.g., flashbacks). We perform a comprehensive
evaluation of XpandA on multiple long-context benchmarks with length varying
from 1k to 1M, demonstrating XpandA's feasibility for processing ultra-long
sequences and its significant effectiveness in enhancing the long-context
capabilities of various LLMs by achieving 20\% improvements and 1.5x inference
speedup over baselines of full-context, RAG and previous agent-based methods.

---


### [GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution](http://arxiv.org/abs/2505.21375v1)

Ultra-high-resolution (UHR) remote sensing (RS) imagery offers valuable data
for Earth observation but pose challenges for existing multimodal foundation
models due to two key bottlenecks: (1) limited availability of UHR training
data, and (2) token explosion caused by the large image size. To address data
scarcity, we introduce SuperRS-VQA (avg. 8,376$\times$8,376) and HighRS-VQA
(avg. 2,000$\times$1,912), the highest-resolution vision-language datasets in
RS to date, covering 22 real-world dialogue tasks. To mitigate token explosion,
our pilot studies reveal significant redundancy in RS images: crucial
information is concentrated in a small subset of object-centric tokens, while
pruning background tokens (e.g., ocean or forest) can even improve performance.
Motivated by these findings, we propose two strategies: Background Token
Pruning and Anchored Token Selection, to reduce the memory footprint while
preserving key semantics.Integrating these techniques, we introduce
GeoLLaVA-8K, the first RS-focused multimodal large language model capable of
handling inputs up to 8K$\times$8K resolution, built on the LLaVA framework.
Trained on SuperRS-VQA and HighRS-VQA, GeoLLaVA-8K sets a new state-of-the-art
on the XLRS-Bench.

---


### [Video-Holmes: Can MLLM Think Like Holmes for Complex Video Reasoning?](http://arxiv.org/abs/2505.21374v1)

Recent advances in CoT reasoning and RL post-training have been reported to
enhance video reasoning capabilities of MLLMs. This progress naturally raises a
question: can these models perform complex video reasoning in a manner
comparable to human experts? However, existing video benchmarks primarily
evaluate visual perception and grounding abilities, with questions that can be
answered based on explicit prompts or isolated visual cues. Such benchmarks do
not fully capture the intricacies of real-world reasoning, where humans must
actively search for, integrate, and analyze multiple clues before reaching a
conclusion. To address this issue, we present Video-Holmes, a benchmark
inspired by the reasoning process of Sherlock Holmes, designed to evaluate the
complex video reasoning capabilities of MLLMs. Video-Holmes consists of 1,837
questions derived from 270 manually annotated suspense short films, which spans
seven carefully designed tasks. Each task is constructed by first identifying
key events and causal relationships within films, and then designing questions
that require models to actively locate and connect multiple relevant visual
clues scattered across different video segments. Our comprehensive evaluation
of state-of-the-art MLLMs reveals that, while these models generally excel at
visual perception, they encounter substantial difficulties with integrating
information and often miss critical clues. For example, the best-performing
model, Gemini-2.5-Pro, achieves an accuracy of only 45%, with most models
scoring below 40%. We aim that Video-Holmes can serve as a "Holmes-test" for
multimodal reasoning, motivating models to reason more like humans and
emphasizing the ongoing challenges in this field. The benchmark is released in
https://github.com/TencentARC/Video-Holmes.

---


### [HoliTom: Holistic Token Merging for Fast Video Large Language Models](http://arxiv.org/abs/2505.21334v1)

Video large language models (video LLMs) excel at video comprehension but
face significant computational inefficiency due to redundant video tokens.
Existing token pruning methods offer solutions. However, approaches operating
within the LLM (inner-LLM pruning), such as FastV, incur intrinsic
computational overhead in shallow layers. In contrast, methods performing token
pruning before the LLM (outer-LLM pruning) primarily address spatial redundancy
within individual frames or limited temporal windows, neglecting the crucial
global temporal dynamics and correlations across longer video sequences. This
leads to sub-optimal spatio-temporal reduction and does not leverage video
compressibility fully. Crucially, the synergistic potential and mutual
influence of combining these strategies remain unexplored. To further reduce
redundancy, we introduce HoliTom, a novel training-free holistic token merging
framework. HoliTom employs outer-LLM pruning through global redundancy-aware
temporal segmentation, followed by spatial-temporal merging to reduce visual
tokens by over 90%, significantly alleviating the LLM's computational burden.
Complementing this, we introduce a robust inner-LLM token similarity-based
merging approach, designed for superior performance and compatibility with
outer-LLM pruning. Evaluations demonstrate our method's promising
efficiency-performance trade-off on LLaVA-OneVision-7B, reducing computational
costs to 6.9% of FLOPs while maintaining 99.1% of the original performance.
Furthermore, we achieve a 2.28x reduction in Time-To-First-Token (TTFT) and a
1.32x acceleration in decoding throughput, highlighting the practical benefits
of our integrated pruning approach for efficient video LLMs inference.

---


### [MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](http://arxiv.org/abs/2505.21333v1)

Multimodal Large Language Models (MLLMs) have achieved considerable accuracy
in Optical Character Recognition (OCR) from static images. However, their
efficacy in video OCR is significantly diminished due to factors such as motion
blur, temporal variations, and visual effects inherent in video content. To
provide clearer guidance for training practical MLLMs, we introduce the
MME-VideoOCR benchmark, which encompasses a comprehensive range of video OCR
application scenarios. MME-VideoOCR features 10 task categories comprising 25
individual tasks and spans 44 diverse scenarios. These tasks extend beyond text
recognition to incorporate deeper comprehension and reasoning of textual
content within videos. The benchmark consists of 1,464 videos with varying
resolutions, aspect ratios, and durations, along with 2,000 meticulously
curated, manually annotated question-answer pairs. We evaluate 18
state-of-the-art MLLMs on MME-VideoOCR, revealing that even the best-performing
model (Gemini-2.5 Pro) achieves an accuracy of only 73.7%. Fine-grained
analysis indicates that while existing MLLMs demonstrate strong performance on
tasks where relevant texts are contained within a single or few frames, they
exhibit limited capability in effectively handling tasks that demand holistic
video comprehension. These limitations are especially evident in scenarios that
require spatio-temporal reasoning, cross-frame information integration, or
resistance to language prior bias. Our findings also highlight the importance
of high-resolution visual input and sufficient temporal coverage for reliable
OCR in dynamic video scenarios.

---


### [CROP: Contextual Region-Oriented Visual Token Pruning](http://arxiv.org/abs/2505.21233v1)

Current VLM-based VQA methods often process entire images, leading to
excessive visual tokens that include redundant information irrelevant to the
posed question. This abundance of unnecessary image details creates numerous
visual tokens, drastically increasing memory and computational requirements in
VLMs. To address this, we propose Contextual Region-Oriented Visual Token
Pruning (CROP), a novel framework to compress visual tokens through a two-step
process: Localization and Pruning. Specifically, CROP first employs an
efficient model to identify the contextual region relevant to the input query.
Subsequently, two distinct strategies are introduced for pruning: (1) Pre-LLM
Compression (PLC), which adaptively compresses different image regions with
varying ratios, and (2) Inner-LLM Pruning (ILP), a training-free method that
prunes tokens within early LLM layers guided by the identified contextual
region. Extensive experiments on a wide range of VQA tasks demonstrate that
CROP significantly outperforms existing visual token pruning methods and
achieves state-of-the-art performance. Our code and datasets will be made
available.

---


### [IKMo: Image-Keyframed Motion Generation with Trajectory-Pose Conditioned Motion Diffusion Model](http://arxiv.org/abs/2505.21146v1)

Existing human motion generation methods with trajectory and pose inputs
operate global processing on both modalities, leading to suboptimal outputs. In
this paper, we propose IKMo, an image-keyframed motion generation method based
on the diffusion model with trajectory and pose being decoupled. The trajectory
and pose inputs go through a two-stage conditioning framework. In the first
stage, the dedicated optimization module is applied to refine inputs. In the
second stage, trajectory and pose are encoded via a Trajectory Encoder and a
Pose Encoder in parallel. Then, motion with high spatial and semantic fidelity
is guided by a motion ControlNet, which processes the fused trajectory and pose
data. Experiment results based on HumanML3D and KIT-ML datasets demonstrate
that the proposed method outperforms state-of-the-art on all metrics under
trajectory-keyframe constraints. In addition, MLLM-based agents are implemented
to pre-process model inputs. Given texts and keyframe images from users, the
agents extract motion descriptions, keyframe poses, and trajectories as the
optimized inputs into the motion generation model. We conducts a user study
with 10 participants. The experiment results prove that the MLLM-based agents
pre-processing makes generated motion more in line with users' expectation. We
believe that the proposed method improves both the fidelity and controllability
of motion generation by the diffusion model.

---


### [CityGo: Lightweight Urban Modeling and Rendering with Proxy Buildings and Residual Gaussians](http://arxiv.org/abs/2505.21041v1)

Accurate and efficient modeling of large-scale urban scenes is critical for
applications such as AR navigation, UAV based inspection, and smart city
digital twins. While aerial imagery offers broad coverage and complements
limitations of ground-based data, reconstructing city-scale environments from
such views remains challenging due to occlusions, incomplete geometry, and high
memory demands. Recent advances like 3D Gaussian Splatting (3DGS) improve
scalability and visual quality but remain limited by dense primitive usage,
long training times, and poor suit ability for edge devices. We propose CityGo,
a hybrid framework that combines textured proxy geometry with residual and
surrounding 3D Gaussians for lightweight, photorealistic rendering of urban
scenes from aerial perspectives. Our approach first extracts compact building
proxy meshes from MVS point clouds, then uses zero order SH Gaussians to
generate occlusion-free textures via image-based rendering and back-projection.
To capture high-frequency details, we introduce residual Gaussians placed based
on proxy-photo discrepancies and guided by depth priors. Broader urban context
is represented by surrounding Gaussians, with importance-aware downsampling
applied to non-critical regions to reduce redundancy. A tailored optimization
strategy jointly refines proxy textures and Gaussian parameters, enabling
real-time rendering of complex urban scenes on mobile GPUs with significantly
reduced training and memory requirements. Extensive experiments on real-world
aerial datasets demonstrate that our hybrid representation significantly
reduces training time, achieving on average 1.4x speedup, while delivering
comparable visual fidelity to pure 3D Gaussian Splatting approaches.
Furthermore, CityGo enables real-time rendering of large-scale urban scenes on
mobile consumer GPUs, with substantially reduced memory usage and energy
consumption.

---


### [Unified Alignment Protocol: Making Sense of the Unlabeled Data in New Domains](http://arxiv.org/abs/2505.21010v1)

Semi-Supervised Federated Learning (SSFL) is gaining popularity over
conventional Federated Learning in many real-world applications. Due to the
practical limitation of limited labeled data on the client side, SSFL considers
that participating clients train with unlabeled data, and only the central
server has the necessary resources to access limited labeled data, making it an
ideal fit for real-world applications (e.g., healthcare). However, traditional
SSFL assumes that the data distributions in the training phase and testing
phase are the same. In practice, however, domain shifts frequently occur,
making it essential for SSFL to incorporate generalization capabilities and
enhance their practicality. The core challenge is improving model
generalization to new, unseen domains while the client participate in SSFL.
However, the decentralized setup of SSFL and unsupervised client training
necessitates innovation to achieve improved generalization across domains. To
achieve this, we propose a novel framework called the Unified Alignment
Protocol (UAP), which consists of an alternating two-stage training process.
The first stage involves training the server model to learn and align the
features with a parametric distribution, which is subsequently communicated to
clients without additional communication overhead. The second stage proposes a
novel training algorithm that utilizes the server feature distribution to align
client features accordingly. Our extensive experiments on standard domain
generalization benchmark datasets across multiple model architectures reveal
that proposed UAP successfully achieves SOTA generalization performance in SSFL
setting.

---


### [DreamBoothDPO: Improving Personalized Generation using Direct Preference Optimization](http://arxiv.org/abs/2505.20975v1)

Personalized diffusion models have shown remarkable success in Text-to-Image
(T2I) generation by enabling the injection of user-defined concepts into
diverse contexts. However, balancing concept fidelity with contextual alignment
remains a challenging open problem. In this work, we propose an RL-based
approach that leverages the diverse outputs of T2I models to address this
issue. Our method eliminates the need for human-annotated scores by generating
a synthetic paired dataset for DPO-like training using external quality
metrics. These better-worse pairs are specifically constructed to improve both
concept fidelity and prompt adherence. Moreover, our approach supports flexible
adjustment of the trade-off between image fidelity and textual alignment.
Through multi-step training, our approach outperforms a naive baseline in
convergence speed and output quality. We conduct extensive qualitative and
quantitative analysis, demonstrating the effectiveness of our method across
various architectures and fine-tuning techniques. The source code can be found
at https://github.com/ControlGenAI/DreamBoothDPO.

---


### [ISAC: Training-Free Instance-to-Semantic Attention Control for Improving Multi-Instance Generation](http://arxiv.org/abs/2505.20935v1)

Text-to-image diffusion models excel at generating single-instance scenes but
struggle with multi-instance scenarios, often merging or omitting objects.
Unlike previous training-free approaches that rely solely on semantic-level
guidance without addressing instance individuation, our training-free method,
Instance-to-Semantic Attention Control (ISAC), explicitly resolves incomplete
instance formation and semantic entanglement through an instance-first modeling
approach. This enables ISAC to effectively leverage a hierarchical,
tree-structured prompt mechanism, disentangling multiple object instances and
individually aligning them with their corresponding semantic labels. Without
employing any external models, ISAC achieves up to 52% average multi-class
accuracy and 83% average multi-instance accuracy by effectively forming
disentangled instances. The code will be made available upon publication.

---


### [Fork-Merge Decoding: Enhancing Multimodal Understanding in Audio-Visual Large Language Models](http://arxiv.org/abs/2505.20873v1)

The goal of this work is to enhance balanced multimodal understanding in
audio-visual large language models (AV-LLMs) by addressing modality bias
without requiring additional training. In current AV-LLMs, audio and video
features are typically processed jointly in the decoder. While this strategy
facilitates unified multimodal understanding, it may introduce modality bias,
where the model tends to over-rely on one modality due to imbalanced training
signals. To mitigate this, we propose Fork-Merge Decoding (FMD), a simple yet
effective inference-time strategy that requires no additional training or
architectural modifications. FMD first performs modality-specific reasoning by
processing audio-only and video-only inputs through the early decoder layers (a
fork phase), and then merges the resulting hidden states for joint reasoning in
the remaining layers (a merge phase). This approach promotes balanced modality
contributions and leverages complementary information across modalities. We
evaluate our method on two representative AV-LLMs, VideoLLaMA2 and
video-SALMONN, using three benchmark datasets. Experimental results demonstrate
consistent performance improvements on tasks focused on audio, video, and
combined audio-visual reasoning, demonstrating the effectiveness of
inference-time interventions for robust multimodal understanding.

---


### [AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding](http://arxiv.org/abs/2505.20862v1)

Hallucination remains a major challenge in multimodal large language models
(MLLMs). To address this, various contrastive decoding (CD) methods have been
proposed that contrasts original logits with hallucinated logits generated from
perturbed inputs. While CD has shown promise in vision-language models (VLMs),
it is not well-suited for AV-LLMs, where hallucinations often emerge from both
unimodal and cross-modal combinations involving audio, video, and language.
These intricate interactions call for a more adaptive and modality-aware
decoding strategy. In this paper, we propose Audio-Visual Contrastive Decoding
(AVCD)-a novel, training-free decoding framework designed to model trimodal
interactions and suppress modality-induced hallucinations in AV-LLMs. Unlike
previous CD methods in VLMs that corrupt a fixed modality, AVCD leverages
attention distributions to dynamically identify less dominant modalities and
applies attentive masking to generate perturbed output logits. To support CD in
a trimodal setting, we also reformulate the original CD framework to jointly
handle audio, visual, and textual inputs. Finally, to improve efficiency, we
introduce entropy-guided adaptive decoding, which selectively skips unnecessary
decoding steps based on the model's confidence in its predictions. Extensive
experiments demonstrate that AVCD consistently outperforms existing decoding
methods. Especially, on the AVHBench dataset, it improves accuracy by 6% for
VideoLLaMA2 and 11% for video-SALMONN, demonstrating strong robustness and
generalizability.

---


### [TACO: Think-Answer Consistency for Optimized Long-Chain Reasoning and Efficient Data Learning via Reinforcement Learning in LVLMs](http://arxiv.org/abs/2505.20777v1)

DeepSeek R1 has significantly advanced complex reasoning for large language
models (LLMs). While recent methods have attempted to replicate R1's reasoning
capabilities in multimodal settings, they face limitations, including
inconsistencies between reasoning and final answers, model instability and
crashes during long-chain exploration, and low data learning efficiency. To
address these challenges, we propose TACO, a novel reinforcement learning
algorithm for visual reasoning. Building on Generalized Reinforcement Policy
Optimization (GRPO), TACO introduces Think-Answer Consistency, which tightly
couples reasoning with answer consistency to ensure answers are grounded in
thoughtful reasoning. We also introduce the Rollback Resample Strategy, which
adaptively removes problematic samples and reintroduces them to the sampler,
enabling stable long-chain exploration and future learning opportunities.
Additionally, TACO employs an adaptive learning schedule that focuses on
moderate difficulty samples to optimize data efficiency. Furthermore, we
propose the Test-Time-Resolution-Scaling scheme to address performance
degradation due to varying resolutions during reasoning while balancing
computational overhead. Extensive experiments on in-distribution and
out-of-distribution benchmarks for REC and VQA tasks show that fine-tuning
LVLMs leads to significant performance improvements.

---


### [Hierarchical Instruction-aware Embodied Visual Tracking](http://arxiv.org/abs/2505.20710v1)

User-Centric Embodied Visual Tracking (UC-EVT) presents a novel challenge for
reinforcement learning-based models due to the substantial gap between
high-level user instructions and low-level agent actions. While recent
advancements in language models (e.g., LLMs, VLMs, VLAs) have improved
instruction comprehension, these models face critical limitations in either
inference speed (LLMs, VLMs) or generalizability (VLAs) for UC-EVT tasks. To
address these challenges, we propose \textbf{Hierarchical Instruction-aware
Embodied Visual Tracking (HIEVT)} agent, which bridges instruction
comprehension and action generation using \textit{spatial goals} as
intermediaries. HIEVT first introduces \textit{LLM-based Semantic-Spatial Goal
Aligner} to translate diverse human instructions into spatial goals that
directly annotate the desired spatial position. Then the \textit{RL-based
Adaptive Goal-Aligned Policy}, a general offline policy, enables the tracker to
position the target as specified by the spatial goal. To benchmark UC-EVT
tasks, we collect over ten million trajectories for training and evaluate
across one seen environment and nine unseen challenging environments. Extensive
experiments and real-world deployments demonstrate the robustness and
generalizability of HIEVT across diverse environments, varying target dynamics,
and complex instruction combinations. The complete project is available at
https://sites.google.com/view/hievt.

---


### [DriveRX: A Vision-Language Reasoning Model for Cross-Task Autonomous Driving](http://arxiv.org/abs/2505.20665v1)

Autonomous driving requires real-time, robust reasoning across perception,
prediction, planning, and behavior. However, conventional end-to-end models
fail to generalize in complex scenarios due to the lack of structured
reasoning. Recent vision-language models (VLMs) have been applied to driving
tasks, but they typically rely on isolated modules and static supervision,
limiting their ability to support multi-stage decision-making. We present
AutoDriveRL, a unified training framework that formulates autonomous driving as
a structured reasoning process over four core tasks. Each task is independently
modeled as a vision-language question-answering problem and optimized using
task-specific reward models, enabling fine-grained reinforcement signals at
different reasoning stages. Within this framework, we train DriveRX, a
cross-task reasoning VLM designed for real-time decision-making. DriveRX
achieves strong performance on a public benchmark, outperforming GPT-4o in
behavior reasoning and demonstrating robustness under complex or corrupted
driving conditions. Our analysis further highlights the impact of vision
encoder design and reward-guided reasoning compression. We will release the
AutoDriveRL framework and the DriveRX model to support future research.

---


### [Open-Det: An Efficient Learning Framework for Open-Ended Detection](http://arxiv.org/abs/2505.20639v1)

Open-Ended object Detection (OED) is a novel and challenging task that
detects objects and generates their category names in a free-form manner,
without requiring additional vocabularies during inference. However, the
existing OED models, such as GenerateU, require large-scale datasets for
training, suffer from slow convergence, and exhibit limited performance. To
address these issues, we present a novel and efficient Open-Det framework,
consisting of four collaborative parts. Specifically, Open-Det accelerates
model training in both the bounding box and object name generation process by
reconstructing the Object Detector and the Object Name Generator. To bridge the
semantic gap between Vision and Language modalities, we propose a
Vision-Language Aligner with V-to-L and L-to-V alignment mechanisms,
incorporating with the Prompts Distiller to transfer knowledge from the VLM
into VL-prompts, enabling accurate object name generation for the LLM. In
addition, we design a Masked Alignment Loss to eliminate contradictory
supervision and introduce a Joint Loss to enhance classification, resulting in
more efficient training. Compared to GenerateU, Open-Det, using only 1.5% of
the training data (0.077M vs. 5.077M), 20.8% of the training epochs (31 vs.
149), and fewer GPU resources (4 V100 vs. 16 A100), achieves even higher
performance (+1.0% in APr). The source codes are available at:
https://github.com/Med-Process/Open-Det.

---


### [Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks](http://arxiv.org/abs/2505.21404v1)

Natural-gradient methods markedly accelerate the training of Physics-Informed
Neural Networks (PINNs), yet their Gauss--Newton update must be solved in the
parameter space, incurring a prohibitive $O(n^3)$ time complexity, where $n$ is
the number of network trainable weights. We show that exactly the same step can
instead be formulated in a generally smaller residual space of size $m =
\sum_{\gamma} N_{\gamma} d_{\gamma}$, where each residual class $\gamma$ (e.g.
PDE interior, boundary, initial data) contributes $N_{\gamma}$ collocation
points of output dimension $d_{\gamma}$.
  Building on this insight, we introduce \textit{Dual Natural Gradient Descent}
(D-NGD). D-NGD computes the Gauss--Newton step in residual space, augments it
with a geodesic-acceleration correction at negligible extra cost, and provides
both a dense direct solver for modest $m$ and a Nystrom-preconditioned
conjugate-gradient solver for larger $m$.
  Experimentally, D-NGD scales second-order PINN optimization to networks with
up to 12.8 million parameters, delivers one- to three-order-of-magnitude lower
final error $L^2$ than first-order methods (Adam, SGD) and quasi-Newton
methods, and -- crucially -- enables natural-gradient training of PINNs at this
scale on a single GPU.

---


### [DeCAF: Decentralized Consensus-And-Factorization for Low-Rank Adaptation of Foundation Models](http://arxiv.org/abs/2505.21382v1)

Low-Rank Adaptation (LoRA) has emerged as one of the most effective,
computationally tractable fine-tuning approaches for training Vision-Language
Models (VLMs) and Large Language Models (LLMs). LoRA accomplishes this by
freezing the pre-trained model weights and injecting trainable low-rank
matrices, allowing for efficient learning of these foundation models even on
edge devices. However, LoRA in decentralized settings still remains under
explored, particularly for the theoretical underpinnings due to the lack of
smoothness guarantee and model consensus interference (defined formally below).
This work improves the convergence rate of decentralized LoRA (DLoRA) to match
the rate of decentralized SGD by ensuring gradient smoothness. We also
introduce DeCAF, a novel algorithm integrating DLoRA with truncated singular
value decomposition (TSVD)-based matrix factorization to resolve consensus
interference. Theoretical analysis shows TSVD's approximation error is bounded
and consensus differences between DLoRA and DeCAF vanish as rank increases,
yielding DeCAF's matching convergence rate. Extensive experiments across
vision/language tasks demonstrate our algorithms outperform local training and
rivals federated learning under both IID and non-IID data distributions.

---


### [LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](http://arxiv.org/abs/2505.21289v1)

Large pre-trained models are commonly adapted to downstream tasks using
parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA),
which injects small trainable low-rank matrices instead of updating all
weights. While LoRA dramatically reduces trainable parameters with little
overhead, it can still underperform full fine-tuning in accuracy and often
converges more slowly. We introduce LoFT, a novel low-rank adaptation method
that behaves like full fine-tuning by aligning the optimizer's internal
dynamics with those of updating all model weights. LoFT not only learns weight
updates in a low-rank subspace (like LoRA) but also properly projects the
optimizer's first and second moments (Adam's momentum and variance) into the
same subspace, mirroring full-model updates. By aligning the low-rank update
itself with the full update, LoFT eliminates the need for tuning extra
hyperparameters, e.g., LoRA scaling factor $\alpha$. Empirically, this approach
substantially narrows the performance gap between adapter-based tuning and full
fine-tuning and consistently outperforms standard LoRA-style methods, all
without increasing inference cost.

---


### [LLaMEA-BO: A Large Language Model Evolutionary Algorithm for Automatically Generating Bayesian Optimization Algorithms](http://arxiv.org/abs/2505.21034v1)

Bayesian optimization (BO) is a powerful class of algorithms for optimizing
expensive black-box functions, but designing effective BO algorithms remains a
manual, expertise-driven task. Recent advancements in Large Language Models
(LLMs) have opened new avenues for automating scientific discovery, including
the automatic design of optimization algorithms. While prior work has used LLMs
within optimization loops or to generate non-BO algorithms, we tackle a new
challenge: Using LLMs to automatically generate full BO algorithm code. Our
framework uses an evolution strategy to guide an LLM in generating Python code
that preserves the key components of BO algorithms: An initial design, a
surrogate model, and an acquisition function. The LLM is prompted to produce
multiple candidate algorithms, which are evaluated on the established Black-Box
Optimization Benchmarking (BBOB) test suite from the COmparing Continuous
Optimizers (COCO) platform. Based on their performance, top candidates are
selected, combined, and mutated via controlled prompt variations, enabling
iterative refinement. Despite no additional fine-tuning, the LLM-generated
algorithms outperform state-of-the-art BO baselines in 19 (out of 24) BBOB
functions in dimension 5 and generalize well to higher dimensions, and
different tasks (from the Bayesmark framework). This work demonstrates that
LLMs can serve as algorithmic co-designers, offering a new paradigm for
automating BO development and accelerating the discovery of novel algorithmic
combinations. The source code is provided at
https://github.com/Ewendawi/LLaMEA-BO.

---


### [FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration](http://arxiv.org/abs/2505.20839v1)

As large language models become increasingly prevalent, memory bandwidth
constraints significantly limit inference throughput, motivating post-training
quantization (PTQ). In this paper, we propose FireQ, a co-designed PTQ
framework and an INT4-FP8 matrix multiplication kernel that accelerates LLM
inference across all linear layers. Specifically, FireQ quantizes linear layer
weights and key-values to INT4, and activations and queries to FP8,
significantly enhancing throughput. Additionally, we introduce a three-stage
pipelining for the prefill phase, which modifies the FlashAttention-3 kernel,
effectively reducing time-to-first-token in the prefill phase. To minimize
accuracy loss from quantization, we develop novel outlier smoothing techniques
tailored separately for linear and attention layers. In linear layers, we
explicitly use per-tensor scaling to prevent underflow caused by the FP8
quantization scaling factor of INT4 quantization, and channel-wise scaling to
compensate for coarse granularity of INT4. In attention layers, we address
quantization challenges posed by rotary positional embeddings (RoPE) by
combining pre-RoPE and post-RoPE scaling strategies. FireQ significantly
outperforms state-of-the-art methods, achieving 1.68x faster inference in
feed-forward network layers on Llama2-7B and 1.26x faster prefill phase
performance on Llama3-8B compared to QServe, with negligible accuracy loss.

---


### [Convergence of Clipped-SGD for Convex $(L_0,L_1)$-Smooth Optimization with Heavy-Tailed Noise](http://arxiv.org/abs/2505.20817v1)

Gradient clipping is a widely used technique in Machine Learning and Deep
Learning (DL), known for its effectiveness in mitigating the impact of
heavy-tailed noise, which frequently arises in the training of large language
models. Additionally, first-order methods with clipping, such as Clip-SGD,
exhibit stronger convergence guarantees than SGD under the
$(L_0,L_1)$-smoothness assumption, a property observed in many DL tasks.
However, the high-probability convergence of Clip-SGD under both assumptions --
heavy-tailed noise and $(L_0,L_1)$-smoothness -- has not been fully addressed
in the literature. In this paper, we bridge this critical gap by establishing
the first high-probability convergence bounds for Clip-SGD applied to convex
$(L_0,L_1)$-smooth optimization with heavy-tailed noise. Our analysis extends
prior results by recovering known bounds for the deterministic case and the
stochastic setting with $L_1 = 0$ as special cases. Notably, our rates avoid
exponentially large factors and do not rely on restrictive sub-Gaussian noise
assumptions, significantly broadening the applicability of gradient clipping.

---


### [GET: Goal-directed Exploration and Targeting for Large-Scale Unknown Environments](http://arxiv.org/abs/2505.20828v1)

Object search in large-scale, unstructured environments remains a fundamental
challenge in robotics, particularly in dynamic or expansive settings such as
outdoor autonomous exploration. This task requires robust spatial reasoning and
the ability to leverage prior experiences. While Large Language Models (LLMs)
offer strong semantic capabilities, their application in embodied contexts is
limited by a grounding gap in spatial reasoning and insufficient mechanisms for
memory integration and decision consistency.To address these challenges, we
propose GET (Goal-directed Exploration and Targeting), a framework that
enhances object search by combining LLM-based reasoning with experience-guided
exploration. At its core is DoUT (Diagram of Unified Thought), a reasoning
module that facilitates real-time decision-making through a role-based feedback
loop, integrating task-specific criteria and external memory. For repeated
tasks, GET maintains a probabilistic task map based on a Gaussian Mixture
Model, allowing for continual updates to object-location priors as environments
evolve.Experiments conducted in real-world, large-scale environments
demonstrate that GET improves search efficiency and robustness across multiple
LLMs and task settings, significantly outperforming heuristic and LLM-only
baselines. These results suggest that structured LLM integration provides a
scalable and generalizable approach to embodied decision-making in complex
environments.

---


### [Automating eHMI Action Design with LLMs for Automated Vehicle Communication](http://arxiv.org/abs/2505.20711v1)

The absence of explicit communication channels between automated vehicles
(AVs) and other road users requires the use of external Human-Machine
Interfaces (eHMIs) to convey messages effectively in uncertain scenarios.
Currently, most eHMI studies employ predefined text messages and manually
designed actions to perform these messages, which limits the real-world
deployment of eHMIs, where adaptability in dynamic scenarios is essential.
Given the generalizability and versatility of large language models (LLMs),
they could potentially serve as automated action designers for the
message-action design task. To validate this idea, we make three contributions:
(1) We propose a pipeline that integrates LLMs and 3D renderers, using LLMs as
action designers to generate executable actions for controlling eHMIs and
rendering action clips. (2) We collect a user-rated Action-Design Scoring
dataset comprising a total of 320 action sequences for eight intended messages
and four representative eHMI modalities. The dataset validates that LLMs can
translate intended messages into actions close to a human level, particularly
for reasoning-enabled LLMs. (3) We introduce two automated raters, Action
Reference Score (ARS) and Vision-Language Models (VLMs), to benchmark 18 LLMs,
finding that the VLM aligns with human preferences yet varies across eHMI
modalities.

---


### [SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](http://arxiv.org/abs/2505.21051v1)

Federated fine-tuning of large language models (LLMs) is critical for
improving their performance in handling domain-specific tasks. However, prior
work has shown that clients' private data can actually be recovered via
gradient inversion attacks. Existing privacy preservation techniques against
such attacks typically entail performance degradation and high costs, making
them ill-suited for clients with heterogeneous data distributions and device
capabilities. In this paper, we propose SHE-LoRA, which integrates selective
homomorphic encryption (HE) and low-rank adaptation (LoRA) to enable efficient
and privacy-preserving federated tuning of LLMs in cross-device environment.
Heterogeneous clients adaptively select partial model parameters for
homomorphic encryption based on parameter sensitivity assessment, with the
encryption subset obtained via negotiation. To ensure accurate model
aggregation, we design a column-aware secure aggregation method and customized
reparameterization techniques to align the aggregation results with the
heterogeneous device capabilities of clients. Extensive experiments demonstrate
that SHE-LoRA maintains performance comparable to non-private baselines,
achieves strong resistance to the state-of-the-art attacks, and significantly
reduces communication overhead by 94.901\% and encryption computation overhead
by 99.829\%, compared to baseline. Our code is accessible at
https://anonymous.4open.science/r/SHE-LoRA-8D84.

---


