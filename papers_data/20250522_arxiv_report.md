### [GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents](http://arxiv.org/abs/2505.15810v1)

Recent Graphical User Interface (GUI) agents replicate the R1-Zero paradigm,
coupling online Reinforcement Learning (RL) with explicit chain-of-thought
reasoning prior to object grounding and thereby achieving substantial
performance gains. In this paper, we first conduct extensive analysis
experiments of three key components of that training pipeline: input design,
output evaluation, and policy update-each revealing distinct challenges arising
from blindly applying general-purpose RL without adapting to GUI grounding
tasks. Input design: Current templates encourage the model to generate
chain-of-thought reasoning, but longer chains unexpectedly lead to worse
grounding performance. Output evaluation: Reward functions based on hit signals
or box area allow models to exploit box size, leading to reward hacking and
poor localization quality. Policy update: Online RL tends to overfit easy
examples due to biases in length and sample difficulty, leading to
under-optimization on harder cases. To address these issues, we propose three
targeted solutions. First, we adopt a Fast Thinking Template that encourages
direct answer generation, reducing excessive reasoning during training. Second,
we incorporate a box size constraint into the reward function to mitigate
reward hacking. Third, we revise the RL objective by adjusting length
normalization and adding a difficulty-aware scaling factor, enabling better
optimization on hard samples. Our GUI-G1-3B, trained on 17K public samples with
Qwen2.5-VL-3B-Instruct, achieves 90.3% accuracy on ScreenSpot and 37.1% on
ScreenSpot-Pro. This surpasses all prior models of similar size and even
outperforms the larger UI-TARS-7B, establishing a new state-of-the-art in GUI
agent grounding. The project repository is available at
https://github.com/Yuqi-Zhou/GUI-G1.

---


### [IA-T2I: Internet-Augmented Text-to-Image Generation](http://arxiv.org/abs/2505.15779v1)

Current text-to-image (T2I) generation models achieve promising results, but
they fail on the scenarios where the knowledge implied in the text prompt is
uncertain. For example, a T2I model released in February would struggle to
generate a suitable poster for a movie premiering in April, because the
character designs and styles are uncertain to the model. To solve this problem,
we propose an Internet-Augmented text-to-image generation (IA-T2I) framework to
compel T2I models clear about such uncertain knowledge by providing them with
reference images. Specifically, an active retrieval module is designed to
determine whether a reference image is needed based on the given text prompt; a
hierarchical image selection module is introduced to find the most suitable
image returned by an image search engine to enhance the T2I model; a
self-reflection mechanism is presented to continuously evaluate and refine the
generated image to ensure faithful alignment with the text prompt. To evaluate
the proposed framework's performance, we collect a dataset named Img-Ref-T2I,
where text prompts include three types of uncertain knowledge: (1) known but
rare. (2) unknown. (3) ambiguous. Moreover, we carefully craft a complex prompt
to guide GPT-4o in making preference evaluation, which has been shown to have
an evaluation accuracy similar to that of human preference evaluation.
Experimental results demonstrate the effectiveness of our framework,
outperforming GPT-4o by about 30% in human evaluation.

---


### [Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval](http://arxiv.org/abs/2505.15753v1)

Large Language Models (LLMs) are known to be vulnerable to jailbreaking
attacks, wherein adversaries exploit carefully engineered prompts to induce
harmful or unethical responses. Such threats have raised critical concerns
about the safety and reliability of LLMs in real-world deployment. While
existing defense mechanisms partially mitigate such risks, subsequent
advancements in adversarial techniques have enabled novel jailbreaking methods
to circumvent these protections, exposing the limitations of static defense
frameworks. In this work, we explore defending against evolving jailbreaking
threats through the lens of context retrieval. First, we conduct a preliminary
study demonstrating that even a minimal set of safety-aligned examples against
a particular jailbreak can significantly enhance robustness against this attack
pattern. Building on this insight, we further leverage the retrieval-augmented
generation (RAG) techniques and propose Safety Context Retrieval (SCR), a
scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our
comprehensive experiments demonstrate how SCR achieves superior defensive
performance against both established and emerging jailbreaking tactics,
contributing a new paradigm to LLM safety. Our code will be available upon
publication.

---


### [HybridProver: Augmenting Theorem Proving with LLM-Driven Proof Synthesis and Refinement](http://arxiv.org/abs/2505.15740v1)

Formal methods is pivotal for verifying the reliability of critical systems
through rigorous mathematical proofs. However, its adoption is hindered by
labor-intensive manual proofs and the expertise required to use theorem
provers. Recent advancements in large language models (LLMs) offer new
opportunities for automated theorem proving. Two promising approaches are
generating tactics step by step and generating a whole proof directly with an
LLM. However, existing work makes no attempt to combine the two approaches. In
this work, we introduce HybridProver, a dual-model proof synthesis framework
that combines tactic-based generation and whole-proof synthesis to harness the
benefits of both approaches. HybridProver generates whole proof candidates for
evaluation directly, then extracts proof sketches from those candidates. It
then uses a tactic-based generation model that integrates automated tools to
complete the sketches via stepwise refinement. We implement HybridProver for
the Isabelle theorem prover and fine-tune LLMs on our optimized Isabelle
datasets. Evaluation on the miniF2F dataset illustrates HybridProver's
effectiveness. We achieve a 59.4% success rate on miniF2F, where the previous
SOTA is 56.1%. Our ablation studies show that this SOTA result is attributable
to combining whole-proof and tactic-based generation. Additionally, we show how
the dataset quality, training parameters, and sampling diversity affect the
final result during automated theorem proving with LLMs. All of our code,
datasets, and LLMs are open source.

---


### [Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses](http://arxiv.org/abs/2505.15738v1)

Large language models (LLMs) are rapidly deployed in real-world applications
ranging from chatbots to agentic systems. Alignment is one of the main
approaches used to defend against attacks such as prompt injection and
jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even
against Greedy Coordinate Gradient (GCG), a white-box attack that generates
adversarial suffixes to induce attacker-desired outputs. However, this search
space over discrete tokens is extremely large, making the task of finding
successful attacks difficult. GCG has, for instance, been shown to converge to
local minima, making it sensitive to initialization choices. In this paper, we
assess the future-proof robustness of these defenses using a more informed
threat model: attackers who have access to some information about the alignment
process. Specifically, we propose an informed white-box attack leveraging the
intermediate model checkpoints to initialize GCG, with each checkpoint acting
as a stepping stone for the next one. We show this approach to be highly
effective across state-of-the-art (SOTA) defenses and models. We further show
our informed initialization to outperform other initialization methods and show
a gradient-informed checkpoint selection strategy to greatly improve attack
performance and efficiency. Importantly, we also show our method to
successfully find universal adversarial suffixes -- single suffixes effective
across diverse inputs. Our results show that, contrary to previous beliefs,
effective adversarial suffixes do exist against SOTA alignment-based defenses,
that these can be found by existing attack methods when adversaries exploit
alignment knowledge, and that even universal suffixes exist. Taken together,
our results highlight the brittleness of current alignment-based methods and
the need to consider stronger threat models when testing the safety of LLMs.

---


### [A Unified Theoretical Analysis of Private and Robust Offline Alignment: from RLHF to DPO](http://arxiv.org/abs/2505.15694v1)

In this paper, we theoretically investigate the effects of noisy labels in
offline alignment, with a focus on the interplay between privacy and robustness
against adversarial corruption. Specifically, under linear modeling
assumptions, we present a unified analysis covering both reinforcement learning
from human feedback (RLHF) and direct preference optimization (DPO) under
different privacy-corruption scenarios, such as Local differential
privacy-then-Corruption (LTC), where human preference labels are privatized
before being corrupted by an adversary, and Corruption-then-Local differential
privacy (CTL), where labels are corrupted before privacy protection. Our
analysis leverages a reduction framework that reduces the offline alignment
problem under linear modeling assumptions to parameter estimation in logistic
regression. This framework allows us to establish an interesting separation
result between LTC and CTL, demonstrating that LTC presents a greater challenge
than CTL in offline alignment, even under linear models. As important
by-products, our findings also advance the state-of-the-art theoretical results
in offline alignment under privacy-only or corruption-only scenarios.

---


### [Average Reward Reinforcement Learning for Omega-Regular and Mean-Payoff Objectives](http://arxiv.org/abs/2505.15693v1)

Recent advances in reinforcement learning (RL) have renewed focus on the
design of reward functions that shape agent behavior. Manually designing reward
functions is tedious and error-prone. A principled alternative is to specify
behaviors in a formal language that can be automatically translated into
rewards. Omega-regular languages are a natural choice for this purpose, given
their established role in formal verification and synthesis. However, existing
methods using omega-regular specifications typically rely on discounted reward
RL in episodic settings, with periodic resets. This setup misaligns with the
semantics of omega-regular specifications, which describe properties over
infinite behavior traces. In such cases, the average reward criterion and the
continuing setting -- where the agent interacts with the environment over a
single, uninterrupted lifetime -- are more appropriate.
  To address the challenges of infinite-horizon, continuing tasks, we focus on
absolute liveness specifications -- a subclass of omega-regular languages that
cannot be violated by any finite behavior prefix, making them well-suited to
the continuing setting. We present the first model-free RL framework that
translates absolute liveness specifications to average-reward objectives. Our
approach enables learning in communicating MDPs without episodic resetting. We
also introduce a reward structure for lexicographic multi-objective
optimization, aiming to maximize an external average-reward objective among the
policies that also maximize the satisfaction probability of a given
omega-regular specification. Our method guarantees convergence in unknown
communicating MDPs and supports on-the-fly reductions that do not require full
knowledge of the environment, thus enabling model-free RL. Empirical results
show our average-reward approach in continuing setting outperforms
discount-based methods across benchmarks.

---


### [A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability](http://arxiv.org/abs/2505.15683v1)

Private data is typically larger and of higher quality than public data,
offering great potential to improve LLM. However, its scattered distribution
across data silos and the high computational demands of LLMs limit their
deployment in federated environments. To address this, the transformer-based
split learning model has emerged, offloading most model parameters to the
server while retaining only the embedding and output layers on clients to
ensure privacy. However, it still faces significant challenges in security,
efficiency, and adaptability: 1) embedding gradients are vulnerable to attacks,
leading to reverse engineering of private data; 2) the autoregressive nature of
LLMs means that federated split learning can only train and infer sequentially,
causing high communication overhead; 3) fixed partition points lack
adaptability to downstream tasks. In this paper, we introduce FL-LLaMA, a
secure, efficient, and adaptive federated split framework based on LLaMA2.
First, we place some input and output blocks on the local client and inject
Gaussian noise into forward-pass hidden states, enabling secure end-to-end
propagation. Second, we employ client-batch and server-hierarchical strategies
to achieve parallel training, along with attention-mask compression and KV
cache mechanisms to accelerate inference, reducing communication costs
effectively. Third, we allow users to dynamically adjust the partition points
for input/output blocks based on specific task requirements and hardware
limitations. Experiments on NLU, summarization and conversational QA tasks show
that FL-LLaMA maintains performance comparable to centralized LLaMA2, and
achieves up to 2x train speedups and 8x inference speedups. Further analysis of
privacy attacks and different partition points also demonstrates the
effectiveness of FL-LLaMA in security and adaptability.

---


### [Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](http://arxiv.org/abs/2505.15612v1)

Large Reasoning Models (LRMs) have shown remarkable capabilities in solving
complex problems through reinforcement learning (RL), particularly by
generating long reasoning traces. However, these extended outputs often exhibit
substantial redundancy, which limits the efficiency of LRMs. In this paper, we
investigate RL-based approaches to promote reasoning efficiency. Specifically,
we first present a unified framework that formulates various efficient
reasoning methods through the lens of length-based reward shaping. Building on
this perspective, we propose a novel Length-bAsed StEp Reward shaping method
(LASER), which employs a step function as the reward, controlled by a target
length. LASER surpasses previous methods, achieving a superior Pareto-optimal
balance between performance and efficiency. Next, we further extend LASER based
on two key intuitions: (1) The reasoning behavior of the model evolves during
training, necessitating reward specifications that are also adaptive and
dynamic; (2) Rather than uniformly encouraging shorter or longer chains of
thought (CoT), we posit that length-based reward shaping should be
difficulty-aware i.e., it should penalize lengthy CoTs more for easy queries.
This approach is expected to facilitate a combination of fast and slow
thinking, leading to a better overall tradeoff. The resulting method is termed
LASER-D (Dynamic and Difficulty-aware). Experiments on
DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, and
DeepSeek-R1-Distill-Qwen-32B show that our approach significantly enhances both
reasoning performance and response length efficiency. For instance, LASER-D and
its variant achieve a +6.1 improvement on AIME2024 while reducing token usage
by 63%. Further analysis reveals our RL-based compression produces more concise
reasoning patterns with less redundant "self-reflections". Resources are at
https://github.com/hkust-nlp/Laser.

---


### [From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning](http://arxiv.org/abs/2505.15607v1)

Large language models (LLMs) can transform education, but their optimization
for direct question-answering often undermines effective pedagogy which
requires strategically withholding answers. To mitigate this, we propose an
online reinforcement learning (RL)-based alignment framework that can quickly
adapt LLMs into effective tutors using simulated student-tutor interactions by
emphasizing pedagogical quality and guided problem-solving over simply giving
away answers. We use our method to train a 7B parameter tutor model without
human annotations which reaches similar performance to larger proprietary
models like LearnLM. We introduce a controllable reward weighting to balance
pedagogical support and student solving accuracy, allowing us to trace the
Pareto frontier between these two objectives. Our models better preserve
reasoning capabilities than single-turn SFT baselines and can optionally
enhance interpretability through thinking tags that expose the model's
instructional planning.

---


### [Evaluate Bias without Manual Test Sets: A Concept Representation Perspective for LLMs](http://arxiv.org/abs/2505.15524v1)

Bias in Large Language Models (LLMs) significantly undermines their
reliability and fairness. We focus on a common form of bias: when two reference
concepts in the model's concept space, such as sentiment polarities (e.g.,
"positive" and "negative"), are asymmetrically correlated with a third, target
concept, such as a reviewing aspect, the model exhibits unintended bias. For
instance, the understanding of "food" should not skew toward any particular
sentiment. Existing bias evaluation methods assess behavioral differences of
LLMs by constructing labeled data for different social groups and measuring
model responses across them, a process that requires substantial human effort
and captures only a limited set of social concepts. To overcome these
limitations, we propose BiasLens, a test-set-free bias analysis framework based
on the structure of the model's vector space. BiasLens combines Concept
Activation Vectors (CAVs) with Sparse Autoencoders (SAEs) to extract
interpretable concept representations, and quantifies bias by measuring the
variation in representational similarity between the target concept and each of
the reference concepts. Even without labeled data, BiasLens shows strong
agreement with traditional bias evaluation metrics (Spearman correlation r >
0.85). Moreover, BiasLens reveals forms of bias that are difficult to detect
using existing methods. For example, in simulated clinical scenarios, a
patient's insurance status can cause the LLM to produce biased diagnostic
assessments. Overall, BiasLens offers a scalable, interpretable, and efficient
paradigm for bias discovery, paving the way for improving fairness and
transparency in LLMs.

---


### [Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets](http://arxiv.org/abs/2505.15517v1)

Vision-Language Models (VLMs) acquire real-world knowledge and general
reasoning ability through Internet-scale image-text corpora. They can augment
robotic systems with scene understanding and task planning, and assist
visuomotor policies that are trained on robot trajectory data. We explore the
reverse paradigm - using rich, real, multi-modal robot trajectory data to
enhance and evaluate VLMs. In this paper, we present Robo2VLM, a Visual
Question Answering (VQA) dataset generation framework for VLMs. Given a human
tele-operated robot trajectory, Robo2VLM derives ground-truth from non-visual
and non-descriptive sensory modalities, such as end-effector pose, gripper
aperture, and force sensing. Based on these modalities, it segments the robot
trajectory into a sequence of manipulation phases. At each phase, Robo2VLM uses
scene and interaction understanding to identify 3D properties of the robot,
task goal, and the target object. The properties are used to generate
representative VQA queries - images with textural multiple-choice questions -
based on spatial, goal-conditioned, and interaction reasoning question
templates. We curate Robo2VLM-1, a large-scale in-the-wild dataset with 684,710
questions covering 463 distinct scenes and 3,396 robotic manipulation tasks
from 176k real robot trajectories. Results suggest that Robo2VLM-1 can
benchmark and improve VLM capabilities in spatial and interaction reasoning.

---


### [AM-PPO: (Advantage) Alpha-Modulation with Proximal Policy Optimization](http://arxiv.org/abs/2505.15514v1)

Proximal Policy Optimization (PPO) is a widely used reinforcement learning
algorithm that heavily relies on accurate advantage estimates for stable and
efficient training. However, raw advantage signals can exhibit significant
variance, noise, and scale-related issues, impeding optimal learning
performance. To address this challenge, we introduce Advantage Modulation PPO
(AM-PPO), a novel enhancement of PPO that adaptively modulates advantage
estimates using a dynamic, non-linear scaling mechanism. This adaptive
modulation employs an alpha controller that dynamically adjusts the scaling
factor based on evolving statistical properties of the advantage signals, such
as their norm, variance, and a predefined target saturation level. By
incorporating a tanh-based gating function driven by these adaptively scaled
advantages, AM-PPO reshapes the advantage signals to stabilize gradient updates
and improve the conditioning of the policy gradient landscape. Crucially, this
modulation also influences value function training by providing consistent and
adaptively conditioned learning targets. Empirical evaluations across standard
continuous control benchmarks demonstrate that AM-PPO achieves superior reward
trajectories, exhibits sustained learning progression, and significantly
reduces the clipping required by adaptive optimizers. These findings underscore
the potential of advantage modulation as a broadly applicable technique for
enhancing reinforcement learning optimization.

---


### [LFTF: Locating First and Then Fine-Tuning for Mitigating Gender Bias in Large Language Models](http://arxiv.org/abs/2505.15475v1)

Nowadays, Large Language Models (LLMs) have attracted widespread attention
due to their powerful performance. However, due to the unavoidable exposure to
socially biased data during training, LLMs tend to exhibit social biases,
particularly gender bias. To better explore and quantifying the degree of
gender bias in LLMs, we propose a pair of datasets named GenBiasEval and
GenHintEval, respectively. The GenBiasEval is responsible for evaluating the
degree of gender bias in LLMs, accompanied by an evaluation metric named
AFGB-Score (Absolutely Fair Gender Bias Score). Meanwhile, the GenHintEval is
used to assess whether LLMs can provide responses consistent with prompts that
contain gender hints, along with the accompanying evaluation metric UB-Score
(UnBias Score). Besides, in order to mitigate gender bias in LLMs more
effectively, we present the LFTF (Locating First and Then Fine-Tuning)
algorithm.The algorithm first ranks specific LLM blocks by their relevance to
gender bias in descending order using a metric called BMI (Block Mitigating
Importance Score). Based on this ranking, the block most strongly associated
with gender bias is then fine-tuned using a carefully designed loss function.
Numerous experiments have shown that our proposed LFTF algorithm can
significantly mitigate gender bias in LLMs while maintaining their general
capabilities.

---


### [ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning](http://arxiv.org/abs/2505.15447v1)

Video understanding is inherently intention-driven-humans naturally focus on
relevant frames based on their goals. Recent advancements in multimodal large
language models (MLLMs) have enabled flexible query-driven reasoning; however,
video-based frameworks like Video Chain-of-Thought lack direct training signals
to effectively identify relevant frames. Current approaches often rely on
heuristic methods or pseudo-label supervised annotations, which are both costly
and limited in scalability across diverse scenarios. To overcome these
challenges, we introduce ViaRL, the first framework to leverage rule-based
reinforcement learning (RL) for optimizing frame selection in intention-driven
video understanding. An iterated amplification strategy is adopted to perform
alternating cyclic training in the video CoT system, where each component
undergoes iterative cycles of refinement to improve its capabilities. ViaRL
utilizes the answer accuracy of a downstream model as a reward signal to train
a frame selector through trial-and-error, eliminating the need for expensive
annotations while closely aligning with human-like learning processes.
Comprehensive experiments across multiple benchmarks, including VideoMME,
LVBench, and MLVU, demonstrate that ViaRL consistently delivers superior
temporal grounding performance and robust generalization across diverse video
understanding tasks, highlighting its effectiveness and scalability. Notably,
ViaRL achieves a nearly 15\% improvement on Needle QA, a subset of MLVU, which
is required to search a specific needle within a long video and regarded as one
of the most suitable benchmarks for evaluating temporal grounding.

---


### [Single LLM, Multiple Roles: A Unified Retrieval-Augmented Generation Framework Using Role-Specific Token Optimization](http://arxiv.org/abs/2505.15444v1)

Existing studies have optimized retrieval-augmented generation (RAG) across
various sub-tasks, such as query understanding and retrieval refinement, but
integrating these optimizations into a unified framework remains challenging.
To tackle this problem, this work proposes RoleRAG, a unified RAG framework
that achieves efficient multi-task processing through role-specific token
optimization. RoleRAG comprises six modules, each handling a specific sub-task
within the RAG process. Additionally, we introduce a query graph to represent
the decomposition of the query, which can be dynamically resolved according to
the decomposing state. All modules are driven by the same underlying LLM,
distinguished by task-specific role tokens that are individually optimized.
This design allows RoleRAG to dynamically activate different modules within a
single LLM instance, thereby streamlining deployment and reducing resource
consumption. Experimental results on five open-domain question-answering
datasets demonstrate the effectiveness, generalizability, and flexibility of
our framework.

---


### [Silent Leaks: Implicit Knowledge Extraction Attack on RAG Systems through Benign Queries](http://arxiv.org/abs/2505.15420v1)

Retrieval-Augmented Generation (RAG) systems enhance large language models
(LLMs) by incorporating external knowledge bases, but they are vulnerable to
privacy risks from data extraction attacks. Existing extraction methods
typically rely on malicious inputs such as prompt injection or jailbreaking,
making them easily detectable via input- or output-level detection. In this
paper, we introduce Implicit Knowledge Extraction Attack (IKEA), which conducts
knowledge extraction on RAG systems through benign queries. IKEA first
leverages anchor concepts to generate queries with the natural appearance, and
then designs two mechanisms to lead to anchor concept thoroughly 'explore' the
RAG's privacy knowledge: (1) Experience Reflection Sampling, which samples
anchor concepts based on past query-response patterns to ensure the queries'
relevance to RAG documents; (2) Trust Region Directed Mutation, which
iteratively mutates anchor concepts under similarity constraints to further
exploit the embedding space. Extensive experiments demonstrate IKEA's
effectiveness under various defenses, surpassing baselines by over 80% in
extraction efficiency and 90% in attack success rate. Moreover, the substitute
RAG system built from IKEA's extractions consistently outperforms those based
on baseline methods across multiple evaluation tasks, underscoring the
significant privacy risk in RAG systems.

---


### [Guided Policy Optimization under Partial Observability](http://arxiv.org/abs/2505.15418v1)

Reinforcement Learning (RL) in partially observable environments poses
significant challenges due to the complexity of learning under uncertainty.
While additional information, such as that available in simulations, can
enhance training, effectively leveraging it remains an open problem. To address
this, we introduce Guided Policy Optimization (GPO), a framework that co-trains
a guider and a learner. The guider takes advantage of privileged information
while ensuring alignment with the learner's policy that is primarily trained
via imitation learning. We theoretically demonstrate that this learning scheme
achieves optimality comparable to direct RL, thereby overcoming key limitations
inherent in existing approaches. Empirical evaluations show strong performance
of GPO across various tasks, including continuous control with partial
observability and noise, and memory-based challenges, significantly
outperforming existing methods.

---


### [Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning](http://arxiv.org/abs/2505.15311v1)

Policy-based methods currently dominate reinforcement learning (RL) pipelines
for large language model (LLM) reasoning, leaving value-based approaches
largely unexplored. We revisit the classical paradigm of Bellman Residual
Minimization and introduce Trajectory Bellman Residual Minimization (TBRM), an
algorithm that naturally adapts this idea to LLMs, yielding a simple yet
effective off-policy algorithm that optimizes a single trajectory-level Bellman
objective using the model's own logits as $Q$-values. TBRM removes the need for
critics, importance-sampling ratios, or clipping, and operates with only one
rollout per prompt. We prove convergence to the near-optimal KL-regularized
policy from arbitrary off-policy data via an improved
change-of-trajectory-measure analysis. Experiments on standard
mathematical-reasoning benchmarks show that TBRM consistently outperforms
policy-based baselines, like PPO and GRPO, with comparable or lower
computational and memory overhead. Our results indicate that value-based RL
might be a principled and efficient alternative for enhancing reasoning
capabilities in LLMs.

---


### [Multiple Weaks Win Single Strong: Large Language Models Ensemble Weak Reinforcement Learning Agents into a Supreme One](http://arxiv.org/abs/2505.15306v1)

Model ensemble is a useful approach in reinforcement learning (RL) for
training effective agents. Despite wide success of RL, training effective
agents remains difficult due to the multitude of factors requiring careful
tuning, such as algorithm selection, hyperparameter settings, and even random
seed choices, all of which can significantly influence an agent's performance.
Model ensemble helps overcome this challenge by combining multiple weak agents
into a single, more powerful one, enhancing overall performance. However,
existing ensemble methods, such as majority voting and Boltzmann addition, are
designed as fixed strategies and lack a semantic understanding of specific
tasks, limiting their adaptability and effectiveness. To address this, we
propose LLM-Ens, a novel approach that enhances RL model ensemble with
task-specific semantic understandings driven by large language models (LLMs).
Given a task, we first design an LLM to categorize states in this task into
distinct 'situations', incorporating high-level descriptions of the task
conditions. Then, we statistically analyze the strengths and weaknesses of each
individual agent to be used in the ensemble in each situation. During the
inference time, LLM-Ens dynamically identifies the changing task situation and
switches to the agent that performs best in the current situation, ensuring
dynamic model selection in the evolving task condition. Our approach is
designed to be compatible with agents trained with different random seeds,
hyperparameter settings, and various RL algorithms. Extensive experiments on
the Atari benchmark show that LLM-Ens significantly improves the RL model
ensemble, surpassing well-known baselines by up to 20.9%. For reproducibility,
our code is open-source at
https://anonymous.4open.science/r/LLM4RLensemble-F7EE.

---


### [LLM-Explorer: A Plug-in Reinforcement Learning Policy Exploration Enhancement Driven by Large Language Models](http://arxiv.org/abs/2505.15293v1)

Policy exploration is critical in reinforcement learning (RL), where existing
approaches include greedy, Gaussian process, etc. However, these approaches
utilize preset stochastic processes and are indiscriminately applied in all
kinds of RL tasks without considering task-specific features that influence
policy exploration. Moreover, during RL training, the evolution of such
stochastic processes is rigid, which typically only incorporates a decay in the
variance, failing to adjust flexibly according to the agent's real-time
learning status. Inspired by the analyzing and reasoning capability of large
language models (LLMs), we design LLM-Explorer to adaptively generate
task-specific exploration strategies with LLMs, enhancing the policy
exploration in RL. In our design, we sample the learning trajectory of the
agent during the RL training in a given task and prompt the LLM to analyze the
agent's current policy learning status and then generate a probability
distribution for future policy exploration. Updating the probability
distribution periodically, we derive a stochastic process specialized for the
particular task and dynamically adjusted to adapt to the learning process. Our
design is a plug-in module compatible with various widely applied RL
algorithms, including the DQN series, DDPG, TD3, and any possible variants
developed based on them. Through extensive experiments on the Atari and MuJoCo
benchmarks, we demonstrate LLM-Explorer's capability to enhance RL policy
exploration, achieving an average performance improvement up to 37.27%. Our
code is open-source at https://anonymous.4open.science/r/LLM-Explorer-19BE for
reproducibility.

---


### [Learning-based Autonomous Oversteer Control and Collision Avoidance](http://arxiv.org/abs/2505.15275v1)

Oversteer, wherein a vehicle's rear tires lose traction and induce
unintentional excessive yaw, poses critical safety challenges. Failing to
control oversteer often leads to severe traffic accidents. Although recent
autonomous driving efforts have attempted to handle oversteer through
stabilizing maneuvers, the majority rely on expert-defined trajectories or
assume obstacle-free environments, limiting real-world applicability. This
paper introduces a novel end-to-end (E2E) autonomous driving approach that
tackles oversteer control and collision avoidance simultaneously. Existing E2E
techniques, including Imitation Learning (IL), Reinforcement Learning (RL), and
Hybrid Learning (HL), generally require near-optimal demonstrations or
extensive experience. Yet even skilled human drivers struggle to provide
perfect demonstrations under oversteer, and high transition variance hinders
accumulating sufficient data. Hence, we present Q-Compared Soft Actor-Critic
(QC-SAC), a new HL algorithm that effectively learns from suboptimal
demonstration data and adapts rapidly to new conditions. To evaluate QC-SAC, we
introduce a benchmark inspired by real-world driver training: a vehicle
encounters sudden oversteer on a slippery surface and must avoid randomly
placed obstacles ahead. Experimental results show QC-SAC attains near-optimal
driving policies, significantly surpassing state-of-the-art IL, RL, and HL
baselines. Our method demonstrates the world's first safe autonomous oversteer
control with obstacle avoidance.

---


### [Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs](http://arxiv.org/abs/2505.15265v1)

Adversarial attacks aim to generate malicious inputs that mislead deep
models, but beyond causing model failure, they cannot provide certain
interpretable information such as ``\textit{What content in inputs make models
more likely to fail?}'' However, this information is crucial for researchers to
specifically improve model robustness. Recent research suggests that models may
be particularly sensitive to certain semantics in visual inputs (such as
``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this
paper we conducted the first exploration on large vision-language models
(LVLMs) and found that LVLMs indeed are susceptible to hallucinations and
various errors when facing specific semantic concepts in images. To efficiently
search for these sensitive concepts, we integrated large language models (LLMs)
and text-to-image (T2I) models to propose a novel semantic evolution framework.
Randomly initialized semantic concepts undergo LLM-based crossover and mutation
operations to form image descriptions, which are then converted by T2I models
into visual inputs for LVLMs. The task-specific performance of LVLMs on each
input is quantified as fitness scores for the involved semantics and serves as
reward signals to further guide LLMs in exploring concepts that induce LVLMs.
Extensive experiments on seven mainstream LVLMs and two multimodal tasks
demonstrate the effectiveness of our method. Additionally, we provide
interesting findings about the sensitive semantics of LVLMs, aiming to inspire
further in-depth research.

---


### [Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](http://arxiv.org/abs/2505.15245v1)

While large language models (LLMs) show great potential in temporal
reasoning, most existing work focuses heavily on enhancing performance, often
neglecting the explainable reasoning processes underlying the results. To
address this gap, we introduce a comprehensive benchmark covering a wide range
of temporal granularities, designed to systematically evaluate LLMs'
capabilities in explainable temporal reasoning. Furthermore, our findings
reveal that LLMs struggle to deliver convincing explanations when relying
solely on textual information. To address challenge, we propose GETER, a novel
structure-aware generative framework that integrates Graph structures with text
for Explainable TEmporal Reasoning. Specifically, we first leverage temporal
knowledge graphs to develop a temporal encoder that captures structural
information for the query. Subsequently, we introduce a structure-text prefix
adapter to map graph structure features into the text embedding space. Finally,
LLMs generate explanation text by seamlessly integrating the soft graph token
with instruction-tuning prompt tokens. Experimental results indicate that GETER
achieves state-of-the-art performance while also demonstrating its
effectiveness as well as strong generalization capabilities. Our dataset and
code are available at https://github.com/carryTatum/GETER.

---


### [Adaptive Plan-Execute Framework for Smart Contract Security Auditing](http://arxiv.org/abs/2505.15242v1)

Large Language Models (LLMs) have shown great promise in code analysis and
auditing; however, they still struggle with hallucinations and limited
context-aware reasoning. We introduce SmartAuditFlow, a novel Plan-Execute
framework that enhances smart contract security analysis through dynamic audit
planning and structured execution. Unlike conventional LLM-based auditing
approaches that follow fixed workflows and predefined steps, SmartAuditFlow
dynamically generates and refines audit plans based on the unique
characteristics of each smart contract. It continuously adjusts its auditing
strategy in response to intermediate LLM outputs and newly detected
vulnerabilities, ensuring a more adaptive and precise security assessment. The
framework then executes these plans step by step, applying a structured
reasoning process to enhance vulnerability detection accuracy while minimizing
hallucinations and false positives. To further improve audit precision,
SmartAuditFlow integrates iterative prompt optimization and external knowledge
sources, such as static analysis tools and Retrieval-Augmented Generation
(RAG). This ensures audit decisions are contextually informed and backed by
real-world security knowledge, producing comprehensive security reports.
Extensive evaluations across multiple benchmarks demonstrate that
SmartAuditFlow outperforms existing methods, achieving 100 percent accuracy on
common and critical vulnerabilities, 41.2 percent accuracy for comprehensive
coverage of known smart contract weaknesses in real-world projects, and
successfully identifying all 13 tested CVEs. These results highlight
SmartAuditFlow's scalability, cost-effectiveness, and superior adaptability
over traditional static analysis tools and contemporary LLM-based approaches,
establishing it as a robust solution for automated smart contract auditing.

---


### [BountyBench: Dollar Impact of AI Agent Attackers and Defenders on Real-World Cybersecurity Systems](http://arxiv.org/abs/2505.15216v1)

AI agents have the potential to significantly alter the cybersecurity
landscape. To help us understand this change, we introduce the first framework
to capture offensive and defensive cyber-capabilities in evolving real-world
systems. Instantiating this framework with BountyBench, we set up 25 systems
with complex, real-world codebases. To capture the vulnerability lifecycle, we
define three task types: Detect (detecting a new vulnerability), Exploit
(exploiting a specific vulnerability), and Patch (patching a specific
vulnerability). For Detect, we construct a new success indicator, which is
general across vulnerability types and provides localized evaluation. We
manually set up the environment for each system, including installing packages,
setting up server(s), and hydrating database(s). We add 40 bug bounties, which
are vulnerabilities with monetary awards from \$10 to \$30,485, and cover 9 of
the OWASP Top 10 Risks. To modulate task difficulty, we devise a new strategy
based on information to guide detection, interpolating from identifying a zero
day to exploiting a specific vulnerability. We evaluate 5 agents: Claude Code,
OpenAI Codex CLI, and custom agents with GPT-4.1, Gemini 2.5 Pro Preview, and
Claude 3.7 Sonnet Thinking. Given up to three attempts, the top-performing
agents are Claude Code (5% on Detect, mapping to \$1,350), Custom Agent with
Claude 3.7 Sonnet Thinking (5% on Detect, mapping to \$1,025; 67.5% on
Exploit), and OpenAI Codex CLI (5% on Detect, mapping to \$2,400; 90% on Patch,
mapping to \$14,422). OpenAI Codex CLI and Claude Code are more capable at
defense, achieving higher Patch scores of 90% and 87.5%, compared to Exploit
scores of 32.5% and 57.5% respectively; in contrast, the custom agents are
relatively balanced between offense and defense, achieving Exploit scores of
40-67.5% and Patch scores of 45-60%.

---


### [Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems](http://arxiv.org/abs/2505.15201v1)

Reinforcement Learning (RL) algorithms sample multiple n>1 solution attempts
for each problem and reward them independently. This optimizes for pass@1
performance and prioritizes the strength of isolated samples at the expense of
the diversity and collective utility of sets of samples. This under-utilizes
the sampling capacity, limiting exploration and eventual improvement on harder
examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a
transformation on the final rewards which leads to direct optimization of
pass@k performance, thus optimizing for sets of samples that maximize reward
when considered jointly. Our contribution is to derive novel low variance
unbiased estimators for pass@k and its gradient, in both the binary and
continuous reward settings. We show optimization with our estimators reduces to
standard RL with rewards that have been jointly transformed by a stable and
efficient transformation function.
  While previous efforts are restricted to k=n, ours is the first to enable
robust optimization of pass@k for any arbitrary k <= n. Moreover, instead of
trading off pass@1 performance for pass@k gains, our method allows annealing k
during training, optimizing both metrics and often achieving strong pass@1
numbers alongside significant pass@k gains.
  We validate our reward transformations on toy experiments, which reveal the
variance reducing properties of our formulations. We also include real-world
examples using the open-source LLM, GEMMA-2. We find that our transformation
effectively optimizes for the target k. Furthermore, higher k values enable
solving more and harder problems, while annealing k boosts both the pass@1 and
pass@k . Crucially, for challenging task sets where conventional pass@1
optimization stalls, our pass@k approach unblocks learning, likely due to
better exploration by prioritizing joint utility over the utility of individual
samples.

---


### [AvatarShield: Visual Reinforcement Learning for Human-Centric Video Forgery Detection](http://arxiv.org/abs/2505.15173v1)

The rapid advancement of Artificial Intelligence Generated Content (AIGC)
technologies, particularly in video generation, has led to unprecedented
creative capabilities but also increased threats to information integrity,
identity security, and public trust. Existing detection methods, while
effective in general scenarios, lack robust solutions for human-centric videos,
which pose greater risks due to their realism and potential for legal and
ethical misuse. Moreover, current detection approaches often suffer from poor
generalization, limited scalability, and reliance on labor-intensive supervised
fine-tuning. To address these challenges, we propose AvatarShield, the first
interpretable MLLM-based framework for detecting human-centric fake videos,
enhanced via Group Relative Policy Optimization (GRPO). Through our carefully
designed accuracy detection reward and temporal compensation reward, it
effectively avoids the use of high-cost text annotation data, enabling precise
temporal modeling and forgery detection. Meanwhile, we design a dual-encoder
architecture, combining high-level semantic reasoning and low-level artifact
amplification to guide MLLMs in effective forgery detection. We further collect
FakeHumanVid, a large-scale human-centric video benchmark that includes
synthesis methods guided by pose, audio, and text inputs, enabling rigorous
evaluation of detection methods in real-world scenes. Extensive experiments
show that AvatarShield significantly outperforms existing approaches in both
in-domain and cross-domain detection, setting a new standard for human-centric
video forensics.

---


### [Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning](http://arxiv.org/abs/2505.15154v1)

Recent advancements in reasoning have significantly enhanced the capabilities
of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs)
across diverse tasks. However, excessive reliance on chain-of-thought (CoT)
reasoning can impair model performance and brings unnecessarily lengthened
outputs, reducing efficiency. Our work reveals that prolonged reasoning does
not universally improve accuracy and even degrade performance on simpler tasks.
To address this, we propose Certainty-based Adaptive Reasoning (CAR), a novel
framework that dynamically switches between short answers and long-form
reasoning based on the model perplexity. CAR first generates a short answer and
evaluates its perplexity, triggering reasoning only when the model exhibits low
confidence (i.e., high perplexity). Experiments across diverse multimodal
VQA/KIE benchmarks and text reasoning datasets show that CAR outperforms both
short-answer and long-form reasoning approaches, striking an optimal balance
between accuracy and efficiency.

---


### [BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms](http://arxiv.org/abs/2505.15141v1)

Speculative decoding has emerged as a popular method to accelerate the
inference of Large Language Models (LLMs) while retaining their superior text
generation performance. Previous methods either adopt a fixed speculative
decoding configuration regardless of the prefix tokens, or train draft models
in an offline or online manner to align them with the context. This paper
proposes a training-free online learning framework to adaptively choose the
configuration of the hyperparameters for speculative decoding as text is being
generated. We first formulate this hyperparameter selection problem as a
Multi-Armed Bandit problem and provide a general speculative decoding framework
BanditSpec. Furthermore, two bandit-based hyperparameter selection algorithms,
UCBSpec and EXP3Spec, are designed and analyzed in terms of a novel quantity,
the stopping time regret. We upper bound this regret under both stochastic and
adversarial reward settings. By deriving an information-theoretic impossibility
result, it is shown that the regret performance of UCBSpec is optimal up to
universal constants. Finally, extensive empirical experiments with LLaMA3 and
Qwen2 demonstrate that our algorithms are effective compared to existing
methods, and the throughput is close to the oracle best hyperparameter in
simulated real-life LLM serving scenarios with diverse input prompts.

---


### [Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor Critic Algorithm](http://arxiv.org/abs/2505.15138v1)

This paper investigates infinite-horizon average reward Constrained Markov
Decision Processes (CMDPs) with general parametrization. We propose a
Primal-Dual Natural Actor-Critic algorithm that adeptly manages constraints
while ensuring a high convergence rate. In particular, our algorithm achieves
global convergence and constraint violation rates of
$\tilde{\mathcal{O}}(1/\sqrt{T})$ over a horizon of length $T$ when the mixing
time, $\tau_{\mathrm{mix}}$, is known to the learner. In absence of knowledge
of $\tau_{\mathrm{mix}}$, the achievable rates change to
$\tilde{\mathcal{O}}(1/T^{0.5-\epsilon})$ provided that $T \geq
\tilde{\mathcal{O}}\left(\tau_{\mathrm{mix}}^{2/\epsilon}\right)$. Our results
match the theoretical lower bound for Markov Decision Processes and establish a
new benchmark in the theoretical exploration of average reward CMDPs.

---


### [The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning](http://arxiv.org/abs/2505.15134v1)

Entropy minimization (EM) trains the model to concentrate even more
probability mass on its most confident outputs. We show that this simple
objective alone, without any labeled data, can substantially improve large
language models' (LLMs) performance on challenging math, physics, and coding
tasks. We explore three approaches: (1) EM-FT minimizes token-level entropy
similarly to instruction finetuning, but on unlabeled outputs drawn from the
model; (2) EM-RL: reinforcement learning with negative entropy as the only
reward to maximize; (3) EM-INF: inference-time logit adjustment to reduce
entropy without any training data or parameter updates. On Qwen-7B, EM-RL,
without any labeled data, achieves comparable or better performance than strong
RL baselines such as GRPO and RLOO that are trained on 60K labeled examples.
Furthermore, EM-INF enables Qwen-32B to match or exceed the performance of
proprietary models like GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro on the
challenging SciCode benchmark, while being 3x more efficient than
self-consistency and sequential refinement. Our findings reveal that many
pretrained LLMs possess previously underappreciated reasoning capabilities that
can be effectively elicited through entropy minimization alone, without any
labeled data or even any parameter updates.

---


### [An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents](http://arxiv.org/abs/2505.15117v1)

Reinforcement learning (RL) has demonstrated strong potential in training
large language models (LLMs) capable of complex reasoning for real-world
problem solving. More recently, RL has been leveraged to create sophisticated
LLM-based search agents that adeptly combine reasoning with search engine use.
While the use of RL for training search agents is promising, the optimal design
of such agents remains not fully understood. In particular, key factors -- such
as (1) reward formulation, (2) the choice and characteristics of the underlying
LLM, and (3) the role of the search engine in the RL process -- require further
investigation. In this work, we conduct comprehensive empirical studies to
systematically investigate these and offer actionable insights. We highlight
several key findings: format rewards are effective in improving final
performance, whereas intermediate retrieval rewards have limited impact; the
scale and initialization of the LLM (general-purpose vs. reasoning-specialized)
significantly influence RL outcomes; and the choice of search engine plays a
critical role in shaping RL training dynamics and the robustness of the trained
agent during inference. These establish important guidelines for successfully
building and deploying LLM-based search agents in real-world applications. Code
is available at https://github.com/PeterGriffinJin/Search-R1.

---


### [StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization](http://arxiv.org/abs/2505.15107v1)

Efficient multi-hop reasoning requires Large Language Models (LLMs) based
agents to acquire high-value external knowledge iteratively. Previous work has
explored reinforcement learning (RL) to train LLMs to perform search-based
document retrieval, achieving notable improvements in QA performance, but
underperform on complex, multi-hop QA resulting from the sparse rewards from
global signal only. To address this gap in existing research, we introduce
StepSearch, a framework for search LLMs that trained with step-wise proximal
policy optimization method. It consists of richer and more detailed
intermediate search rewards and token-level process supervision based on
information gain and redundancy penalties to better guide each search step. We
constructed a fine-grained question-answering dataset containing
sub-question-level search trajectories based on open source datasets through a
set of data pipeline method. On standard multi-hop QA benchmarks, it
significantly outperforms global-reward baselines, achieving 11.2% and 4.2%
absolute improvements for 3B and 7B models over various search with RL
baselines using only 19k training data, demonstrating the effectiveness of
fine-grained, stepwise supervision in optimizing deep search LLMs. Our
implementation is publicly available at
https://github.com/zxh20001117/StepSearch.

---


### [Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects](http://arxiv.org/abs/2505.15088v1)

Command injection vulnerabilities are a significant security threat in
dynamic languages like Python, particularly in widely used open-source projects
where security issues can have extensive impact. With the proven effectiveness
of Large Language Models(LLMs) in code-related tasks, such as testing,
researchers have explored their potential for vulnerabilities analysis. This
study evaluates the potential of large language models (LLMs), such as GPT-4,
as an alternative approach for automated testing for vulnerability detection.
In particular, LLMs have demonstrated advanced contextual understanding and
adaptability, making them promising candidates for identifying nuanced security
vulnerabilities within code. To evaluate this potential, we applied LLM-based
analysis to six high-profile GitHub projects-Django, Flask, TensorFlow,
Scikit-learn, PyTorch, and Langchain-each with over 50,000 stars and extensive
adoption across software development and academic research. Our analysis
assesses both the strengths and limitations of LLMs in detecting command
injection vulnerabilities, evaluating factors such as detection accuracy,
efficiency, and practical integration into development workflows. In addition,
we provide a comparative analysis of different LLM tools to identify those most
suitable for security applications. Our findings offer guidance for developers
and security researchers on leveraging LLMs as innovative and automated
approaches to enhance software security.

---


### [DISCO Balances the Scales: Adaptive Domain- and Difficulty-Aware Reinforcement Learning on Imbalanced Data](http://arxiv.org/abs/2505.15074v1)

Large Language Models (LLMs) are increasingly aligned with human preferences
through Reinforcement Learning from Human Feedback (RLHF). Among RLHF methods,
Group Relative Policy Optimization (GRPO) has gained attention for its
simplicity and strong performance, notably eliminating the need for a learned
value function. However, GRPO implicitly assumes a balanced domain distribution
and uniform semantic alignment across groups - assumptions that rarely hold in
real-world datasets. When applied to multi-domain, imbalanced data, GRPO
disproportionately optimizes for dominant domains, neglecting underrepresented
ones and resulting in poor generalization and fairness. We propose
Domain-Informed Self-Consistency Policy Optimization (DISCO), a principled
extension to GRPO that addresses inter-group imbalance with two key
innovations. Domain-aware reward scaling counteracts frequency bias by
reweighting optimization based on domain prevalence. Difficulty-aware reward
scaling leverages prompt-level self-consistency to identify and prioritize
uncertain prompts that offer greater learning value. Together, these strategies
promote more equitable and effective policy learning across domains. Extensive
experiments across multiple LLMs and skewed training distributions show that
DISCO improves generalization, outperforms existing GRPO variants by 5% on
Qwen3 models, and sets new state-of-the-art results on multi-domain alignment
benchmarks.

---


### [Self-GIVE: Associative Thinking from Limited Structured Knowledge for Enhanced Large Language Model Reasoning](http://arxiv.org/abs/2505.15062v1)

When addressing complex questions that require new information, people often
associate the question with existing knowledge to derive a sensible answer. For
instance, when evaluating whether melatonin aids insomnia, one might associate
"hormones helping mental disorders" with "melatonin being a hormone and
insomnia a mental disorder" to complete the reasoning. Large Language Models
(LLMs) also require such associative thinking, particularly in resolving
scientific inquiries when retrieved knowledge is insufficient and does not
directly answer the question. Graph Inspired Veracity Extrapolation (GIVE)
addresses this by using a knowledge graph (KG) to extrapolate structured
knowledge. However, it involves the construction and pruning of many
hypothetical triplets, which limits efficiency and generalizability. We propose
Self-GIVE, a retrieve-RL framework that enhances LLMs with automatic
associative thinking through reinforcement learning. Self-GIVE extracts
structured information and entity sets to assist the model in linking to the
queried concepts. We address GIVE's key limitations: (1) extensive LLM calls
and token overhead for knowledge extrapolation, (2) difficulty in deploying on
smaller LLMs (3B or 7B) due to complex instructions, and (3) inaccurate
knowledge from LLM pruning. Specifically, after fine-tuning using self-GIVE
with a 135 node UMLS KG, it improves the performance of the Qwen2.5 3B and 7B
models by up to $\textbf{28.5%$\rightarrow$71.4%}$ and
$\textbf{78.6$\rightarrow$90.5%}$ in samples $\textbf{unseen}$ in challenging
biomedical QA tasks. In particular, Self-GIVE allows the 7B model to match or
outperform GPT3.5 turbo with GIVE, while cutting token usage by over 90\%.
Self-GIVE enhances the scalable integration of structured retrieval and
reasoning with associative thinking.

---


### [ChartCards: A Chart-Metadata Generation Framework for Multi-Task Chart Understanding](http://arxiv.org/abs/2505.15046v1)

The emergence of Multi-modal Large Language Models (MLLMs) presents new
opportunities for chart understanding. However, due to the fine-grained nature
of these tasks, applying MLLMs typically requires large, high-quality datasets
for task-specific fine-tuning, leading to high data collection and training
costs. To address this, we propose ChartCards, a unified chart-metadata
generation framework for multi-task chart understanding. ChartCards
systematically synthesizes various chart information, including data tables,
visualization code, visual elements, and multi-dimensional semantic captions.
By structuring this information into organized metadata, ChartCards enables a
single chart to support multiple downstream tasks, such as text-to-chart
retrieval, chart summarization, chart-to-table conversion, chart description,
and chart question answering. Using ChartCards, we further construct MetaChart,
a large-scale high-quality dataset containing 10,862 data tables, 85K charts,
and 170 K high-quality chart captions. We validate the dataset through
qualitative crowdsourcing evaluations and quantitative fine-tuning experiments
across various chart understanding tasks. Fine-tuning six different models on
MetaChart resulted in an average performance improvement of 5% across all
tasks. The most notable improvements are seen in text-to-chart retrieval and
chart-to-table tasks, with Long-CLIP and Llama 3.2-11B achieving improvements
of 17% and 28%, respectively.

---


### [RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning](http://arxiv.org/abs/2505.15034v1)

Reinforcement learning (RL) has recently emerged as a compelling approach for
enhancing the reasoning capabilities of large language models (LLMs), where an
LLM generator serves as a policy guided by a verifier (reward model). However,
current RL post-training methods for LLMs typically use verifiers that are
fixed (rule-based or frozen pretrained) or trained discriminatively via
supervised fine-tuning (SFT). Such designs are susceptible to reward hacking
and generalize poorly beyond their training distributions. To overcome these
limitations, we propose Tango, a novel framework that uses RL to concurrently
train both an LLM generator and a verifier in an interleaved manner. A central
innovation of Tango is its generative, process-level LLM verifier, which is
trained via RL and co-evolves with the generator. Importantly, the verifier is
trained solely based on outcome-level verification correctness rewards without
requiring explicit process-level annotations. This generative RL-trained
verifier exhibits improved robustness and superior generalization compared to
deterministic or SFT-trained verifiers, fostering effective mutual
reinforcement with the generator. Extensive experiments demonstrate that both
components of Tango achieve state-of-the-art results among 7B/8B-scale models:
the generator attains best-in-class performance across five competition-level
math benchmarks and four challenging out-of-domain reasoning tasks, while the
verifier leads on the ProcessBench dataset. Remarkably, both components exhibit
particularly substantial improvements on the most difficult mathematical
reasoning problems. Code is at: https://github.com/kaiwenzha/rl-tango.

---


### [Learning to Rank Chain-of-Thought: An Energy-Based Approach with Outcome Supervision](http://arxiv.org/abs/2505.14999v1)

Mathematical reasoning presents a significant challenge for Large Language
Models (LLMs), often requiring robust multi step logical consistency. While
Chain of Thought (CoT) prompting elicits reasoning steps, it doesn't guarantee
correctness, and improving reliability via extensive sampling is
computationally costly. This paper introduces the Energy Outcome Reward Model
(EORM), an effective, lightweight, post hoc verifier. EORM leverages Energy
Based Models (EBMs) to simplify the training of reward models by learning to
assign a scalar energy score to CoT solutions using only outcome labels,
thereby avoiding detailed annotations. It achieves this by interpreting
discriminator output logits as negative energies, effectively ranking
candidates where lower energy is assigned to solutions leading to correct final
outcomes implicitly favoring coherent reasoning. On mathematical benchmarks
(GSM8k, MATH), EORM significantly improves final answer accuracy (e.g., with
Llama 3 8B, achieving 90.7% on GSM8k and 63.7% on MATH). EORM effectively
leverages a given pool of candidate solutions to match or exceed the
performance of brute force sampling, thereby enhancing LLM reasoning outcome
reliability through its streamlined post hoc verification process.

---


### [Advancing LLM Safe Alignment with Safety Representation Ranking](http://arxiv.org/abs/2505.15710v1)

The rapid advancement of large language models (LLMs) has demonstrated
milestone success in a variety of tasks, yet their potential for generating
harmful content has raised significant safety concerns. Existing safety
evaluation approaches typically operate directly on textual responses,
overlooking the rich information embedded in the model's internal
representations. In this paper, we propose Safety Representation Ranking (SRR),
a listwise ranking framework that selects safe responses using hidden states
from the LLM itself. SRR encodes both instructions and candidate completions
using intermediate transformer representations and ranks candidates via a
lightweight similarity-based scorer. Our approach directly leverages internal
model states and supervision at the list level to capture subtle safety
signals. Experiments across multiple benchmarks show that SRR significantly
improves robustness to adversarial prompts. Our code will be available upon
publication.

---


### [HDLxGraph: Bridging Large Language Models and HDL Repositories via HDL Graph Databases](http://arxiv.org/abs/2505.15701v1)

Large Language Models (LLMs) have demonstrated their potential in hardware
design tasks, such as Hardware Description Language (HDL) generation and
debugging. Yet, their performance in real-world, repository-level HDL projects
with thousands or even tens of thousands of code lines is hindered. To this
end, we propose HDLxGraph, a novel framework that integrates Graph Retrieval
Augmented Generation (Graph RAG) with LLMs, introducing HDL-specific graph
representations by incorporating Abstract Syntax Trees (ASTs) and Data Flow
Graphs (DFGs) to capture both code graph view and hardware graph view.
HDLxGraph utilizes a dual-retrieval mechanism that not only mitigates the
limited recall issues inherent in similarity-based semantic retrieval by
incorporating structural information, but also enhances its extensibility to
various real-world tasks by a task-specific retrieval finetuning. Additionally,
to address the lack of comprehensive HDL search benchmarks, we introduce
HDLSearch, a multi-granularity evaluation dataset derived from real-world
repository-level projects. Experimental results demonstrate that HDLxGraph
significantly improves average search accuracy, debugging efficiency and
completion quality by 12.04%, 12.22% and 5.04% compared to similarity-based
RAG, respectively. The code of HDLxGraph and collected HDLSearch benchmark are
available at https://github.com/Nick-Zheng-Q/HDLxGraph.

---


### [Thought-Augmented Policy Optimization: Bridging External Guidance and Internal Capabilities](http://arxiv.org/abs/2505.15692v1)

Reinforcement learning (RL) has emerged as an effective method for training
reasoning models. However, existing RL approaches typically bias the model's
output distribution toward reward-maximizing paths without introducing external
knowledge. This limits their exploration capacity and results in a narrower
reasoning capability boundary compared to base models. To address this
limitation, we propose TAPO (Thought-Augmented Policy Optimization), a novel
framework that augments RL by incorporating external high-level guidance
("thought patterns"). By adaptively integrating structured thoughts during
training, TAPO effectively balances model-internal exploration and external
guidance exploitation. Extensive experiments show that our approach
significantly outperforms GRPO by 99% on AIME, 41% on AMC, and 17% on Minerva
Math. Notably, these high-level thought patterns, abstracted from only 500
prior samples, generalize effectively across various tasks and models. This
highlights TAPO's potential for broader applications across multiple tasks and
domains. Our further analysis reveals that introducing external guidance
produces powerful reasoning models with superior explainability of inference
behavior and enhanced output readability.

---


### [ThinkLess: A Training-Free Inference-Efficient Method for Reducing Reasoning Redundancy](http://arxiv.org/abs/2505.15684v1)

While Chain-of-Thought (CoT) prompting improves reasoning in large language
models (LLMs), the excessive length of reasoning tokens increases latency and
KV cache memory usage, and may even truncate final answers under context
limits. We propose ThinkLess, an inference-efficient framework that terminates
reasoning generation early and maintains output quality without modifying the
model. Atttention analysis reveals that answer tokens focus minimally on
earlier reasoning steps and primarily attend to the reasoning terminator token,
due to information migration under causal masking. Building on this insight,
ThinkLess inserts the terminator token at earlier positions to skip redundant
reasoning while preserving the underlying knowledge transfer. To prevent format
discruption casued by early termination, ThinkLess employs a lightweight
post-regulation mechanism, relying on the model's natural instruction-following
ability to produce well-structured answers. Without fine-tuning or auxiliary
data, ThinkLess achieves comparable accuracy to full-length CoT decoding while
greatly reducing decoding time and memory consumption.

---


### [Segmentation-Variant Codebooks for Preservation of Paralinguistic and Prosodic Information](http://arxiv.org/abs/2505.15667v1)

Quantization in SSL speech models (e.g., HuBERT) improves compression and
performance in tasks like language modeling, resynthesis, and text-to-speech
but often discards prosodic and paralinguistic information (e.g., emotion,
prominence). While increasing codebook size mitigates some loss, it
inefficiently raises bitrates. We propose Segmentation-Variant Codebooks
(SVCs), which quantize speech at distinct linguistic units (frame, phone, word,
utterance), factorizing it into multiple streams of segment-specific discrete
features. Our results show that SVCs are significantly more effective at
preserving prosodic and paralinguistic information across probing tasks.
Additionally, we find that pooling before rather than after discretization
better retains segment-level information. Resynthesis experiments further
confirm improved style realization and slightly improved quality while
preserving intelligibility.

---


### [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](http://arxiv.org/abs/2505.15634v1)

Large Language Models (LLMs) demonstrate the ability to solve reasoning and
mathematical problems using the Chain-of-Thought (CoT) technique. Expanding CoT
length, as seen in models such as DeepSeek-R1, significantly enhances this
reasoning for complex problems, but requires costly and high-quality long CoT
data and fine-tuning. This work, inspired by the deep thinking paradigm of
DeepSeek-R1, utilizes a steering technique to enhance the reasoning ability of
an LLM without external datasets. Our method first employs Sparse Autoencoders
(SAEs) to extract interpretable features from vanilla CoT. These features are
then used to steer the LLM's internal states during generation. Recognizing
that many LLMs do not have corresponding pre-trained SAEs, we further introduce
a novel SAE-free steering algorithm, which directly computes steering
directions from the residual activations of an LLM, obviating the need for an
explicit SAE. Experimental results demonstrate that both our SAE-based and
subsequent SAE-free steering algorithms significantly enhance the reasoning
capabilities of LLMs.

---


### [CoLA: Collaborative Low-Rank Adaptation](http://arxiv.org/abs/2505.15471v1)

The scaling law of Large Language Models (LLMs) reveals a power-law
relationship, showing diminishing return on performance as model scale
increases. While training LLMs from scratch is resource-intensive, fine-tuning
a pre-trained model for specific tasks has become a practical alternative. Full
fine-tuning (FFT) achieves strong performance; however, it is computationally
expensive and inefficient. Parameter-efficient fine-tuning (PEFT) methods, like
LoRA, have been proposed to address these challenges by freezing the
pre-trained model and adding lightweight task-specific modules. LoRA, in
particular, has proven effective, but its application to multi-task scenarios
is limited by interference between tasks. Recent approaches, such as
Mixture-of-Experts (MOE) and asymmetric LoRA, have aimed to mitigate these
issues but still struggle with sample scarcity and noise interference due to
their fixed structure. In response, we propose CoLA, a more flexible LoRA
architecture with an efficient initialization scheme, and introduces three
collaborative strategies to enhance performance by better utilizing the
quantitative relationships between matrices $A$ and $B$. Our experiments
demonstrate the effectiveness and robustness of CoLA, outperforming existing
PEFT methods, especially in low-sample scenarios. Our data and code are fully
publicly available at https://github.com/zyy-2001/CoLA.

---


### [Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment](http://arxiv.org/abs/2505.15456v1)

Personalized alignment is essential for enabling large language models (LLMs)
to engage effectively in user-centric dialogue. While recent prompt-based and
offline optimization methods offer preliminary solutions, they fall short in
cold-start scenarios and long-term personalization due to their inherently
static and shallow designs. In this work, we introduce the Reinforcement
Learning for Personalized Alignment (RLPA) framework, in which an LLM interacts
with a simulated user model to iteratively infer and refine user profiles
through dialogue. The training process is guided by a dual-level reward
structure: the Profile Reward encourages accurate construction of user
representations, while the Response Reward incentivizes generation of responses
consistent with the inferred profile. We instantiate RLPA by fine-tuning
Qwen-2.5-3B-Instruct, resulting in Qwen-RLPA, which achieves state-of-the-art
performance in personalized dialogue. Empirical evaluations demonstrate that
Qwen-RLPA consistently outperforms prompting and offline fine-tuning baselines,
and even surpasses advanced commercial models such as Claude-3.5 and GPT-4o.
Further analysis highlights Qwen-RLPA's robustness in reconciling conflicting
user preferences, sustaining long-term personalization and delivering more
efficient inference compared to recent reasoning-focused LLMs. These results
emphasize the potential of dynamic profile inference as a more effective
paradigm for building personalized dialogue systems.

---


### [AdUE: Improving uncertainty estimation head for LoRA adapters in LLMs](http://arxiv.org/abs/2505.15443v1)

Uncertainty estimation remains a critical challenge in adapting pre-trained
language models to classification tasks, particularly under parameter-efficient
fine-tuning approaches such as adapters. We introduce AdUE1, an efficient
post-hoc uncertainty estimation (UE) method, to enhance softmax-based
estimates. Our approach (1) uses a differentiable approximation of the maximum
function and (2) applies additional regularization through L2-SP, anchoring the
fine-tuned head weights and regularizing the model. Evaluations on five NLP
classification datasets across four language models (RoBERTa, ELECTRA, LLaMA-2,
Qwen) demonstrate that our method consistently outperforms established
baselines such as Mahalanobis distance and softmax response. Our approach is
lightweight (no base-model changes) and produces better-calibrated confidence.

---


### [Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought](http://arxiv.org/abs/2505.15431v1)

As Large Language Models (LLMs) rapidly advance, we introduce Hunyuan-TurboS,
a novel large hybrid Transformer-Mamba Mixture of Experts (MoE) model. It
synergistically combines Mamba's long-sequence processing efficiency with
Transformer's superior contextual understanding. Hunyuan-TurboS features an
adaptive long-short chain-of-thought (CoT) mechanism, dynamically switching
between rapid responses for simple queries and deep "thinking" modes for
complex problems, optimizing computational resources. Architecturally, this 56B
activated (560B total) parameter model employs 128 layers (Mamba2, Attention,
FFN) with an innovative AMF/MF block pattern. Faster Mamba2 ensures linear
complexity, Grouped-Query Attention minimizes KV cache, and FFNs use an MoE
structure. Pre-trained on 16T high-quality tokens, it supports a 256K context
length and is the first industry-deployed large-scale Mamba model. Our
comprehensive post-training strategy enhances capabilities via Supervised
Fine-Tuning (3M instructions), a novel Adaptive Long-short CoT Fusion method,
Multi-round Deliberation Learning for iterative improvement, and a two-stage
Large-scale Reinforcement Learning process targeting STEM and general
instruction-following. Evaluations show strong performance: overall top 7 rank
on LMSYS Chatbot Arena with a score of 1356, outperforming leading models like
Gemini-2.0-Flash-001 (1352) and o4-mini-2025-04-16 (1345). TurboS also achieves
an average of 77.9% across 23 automated benchmarks. Hunyuan-TurboS balances
high performance and efficiency, offering substantial capabilities at lower
inference costs than many reasoning models, establishing a new paradigm for
efficient large-scale pre-trained models.

---


### [NeoN: A Tool for Automated Detection, Linguistic and LLM-Driven Analysis of Neologisms in Polish](http://arxiv.org/abs/2505.15426v1)

NeoN, a tool for detecting and analyzing Polish neologisms. Unlike
traditional dictionary-based methods requiring extensive manual review, NeoN
combines reference corpora, Polish-specific linguistic filters, an LLM-driven
precision-boosting filter, and daily RSS monitoring in a multi-layered
pipeline. The system uses context-aware lemmatization, frequency analysis, and
orthographic normalization to extract candidate neologisms while consolidating
inflectional variants. Researchers can verify candidates through an intuitive
interface with visualizations and filtering controls. An integrated LLM module
automatically generates definitions and categorizes neologisms by domain and
sentiment. Evaluations show NeoN maintains high accuracy while significantly
reducing manual effort, providing an accessible solution for tracking lexical
innovation in Polish.

---


### [Gated Integration of Low-Rank Adaptation for Continual Learning of Language Models](http://arxiv.org/abs/2505.15424v1)

Continual learning (CL), which requires the model to learn multiple tasks
sequentially, is crucial for language models (LMs). Recently, low-rank
adaptation (LoRA), one of the most representative parameter-efficient
fine-tuning (PEFT) methods, has gained increasing attention in CL of LMs.
However, most existing CL methods based on LoRA typically expand a new LoRA
branch to learn each new task and force the new and old LoRA branches to
contribute equally to old tasks, potentially leading to forgetting. In this
work, we propose a new method, called gated integration of low-rank adaptation
(GainLoRA), for CL of LMs. GainLoRA expands a new LoRA branch for each new task
and introduces gating modules to integrate the new and old LoRA branches.
Furthermore, GainLoRA leverages the new gating module to minimize the
contribution from the new LoRA branch to old tasks, effectively mitigating
forgetting and improving the model's overall performance. Experimental results
on CL benchmarks demonstrate that GainLoRA outperforms existing
state-of-the-art methods.

---


### [How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](http://arxiv.org/abs/2505.15404v1)

Large Reasoning Models (LRMs) have achieved remarkable success on
reasoning-intensive tasks such as mathematics and programming. However, their
enhanced reasoning capabilities do not necessarily translate to improved safety
performance-and in some cases, may even degrade it. This raises an important
research question: how can we enhance the safety of LRMs? In this paper, we
present a comprehensive empirical study on how to enhance the safety of LRMs
through Supervised Fine-Tuning (SFT). Our investigation begins with an
unexpected observation: directly distilling safe responses from DeepSeek-R1
fails to significantly enhance safety. We analyze this phenomenon and identify
three key failure patterns that contribute to it. We then demonstrate that
explicitly addressing these issues during the data distillation process can
lead to substantial safety improvements. Next, we explore whether a long and
complex reasoning process is necessary for achieving safety. Interestingly, we
find that simply using short or template-based reasoning process can attain
comparable safety performance-and are significantly easier for models to learn
than more intricate reasoning chains. These findings prompt a deeper reflection
on the role of reasoning in ensuring safety. Finally, we find that mixing math
reasoning data during safety fine-tuning is helpful to balance safety and
over-refusal. Overall, we hope our empirical study could provide a more
holistic picture on enhancing the safety of LRMs. The code and data used in our
experiments are released in https://github.com/thu-coai/LRM-Safety-Study.

---


### [An Empirical Study of the Anchoring Effect in LLMs: Existence, Mechanism, and Potential Mitigations](http://arxiv.org/abs/2505.15392v1)

The rise of Large Language Models (LLMs) like ChatGPT has advanced natural
language processing, yet concerns about cognitive biases are growing. In this
paper, we investigate the anchoring effect, a cognitive bias where the mind
relies heavily on the first information as anchors to make affected judgments.
We explore whether LLMs are affected by anchoring, the underlying mechanisms,
and potential mitigation strategies. To facilitate studies at scale on the
anchoring effect, we introduce a new dataset, SynAnchors. Combining refined
evaluation metrics, we benchmark current widely used LLMs. Our findings show
that LLMs' anchoring bias exists commonly with shallow-layer acting and is not
eliminated by conventional strategies, while reasoning can offer some
mitigation. This recontextualization via cognitive psychology urges that LLM
evaluations focus not on standard benchmarks or over-optimized robustness
tests, but on cognitive-bias-aware trustworthy evaluation.

---


### [Are Vision-Language Models Safe in the Wild? A Meme-Based Benchmark Study](http://arxiv.org/abs/2505.15389v1)

Rapid deployment of vision-language models (VLMs) magnifies safety risks, yet
most evaluations rely on artificial images. This study asks: How safe are
current VLMs when confronted with meme images that ordinary users share? To
investigate this question, we introduce MemeSafetyBench, a 50,430-instance
benchmark pairing real meme images with both harmful and benign instructions.
Using a comprehensive safety taxonomy and LLM-based instruction generation, we
assess multiple VLMs across single and multi-turn interactions. We investigate
how real-world memes influence harmful outputs, the mitigating effects of
conversational context, and the relationship between model scale and safety
metrics. Our findings demonstrate that VLMs show greater vulnerability to
meme-based harmful prompts than to synthetic or typographic images. Memes
significantly increase harmful responses and decrease refusals compared to
text-only inputs. Though multi-turn interactions provide partial mitigation,
elevated vulnerability persists. These results highlight the need for
ecologically valid evaluations and stronger safety mechanisms.

---


### [X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System](http://arxiv.org/abs/2505.15372v1)

Recently, large language model (LLM)-based agents have achieved significant
success in interactive environments, attracting significant academic and
industrial attention. Despite these advancements, current research
predominantly focuses on English scenarios. In reality, there are over 7,000
languages worldwide, all of which demand access to comparable agentic services.
Nevertheless, the development of language agents remains inadequate for meeting
the diverse requirements of multilingual agentic applications. To fill this
gap, we introduce X-WebAgentBench, a novel multilingual agent benchmark in an
interactive web environment, which evaluates the planning and interaction
performance of language agents across multiple languages, thereby contributing
to the advancement of global agent intelligence. Additionally, we assess the
performance of various LLMs and cross-lingual alignment methods, examining
their effectiveness in enhancing agents. Our findings reveal that even advanced
models like GPT-4o, when combined with cross-lingual techniques, fail to
achieve satisfactory results. We hope that X-WebAgentBench can serve as a
valuable benchmark for multilingual agent scenario in real-world applications.

---


### [AI vs. Human Judgment of Content Moderation: LLM-as-a-Judge and Ethics-Based Response Refusals](http://arxiv.org/abs/2505.15365v1)

As large language models (LLMs) are increasingly deployed in high-stakes
settings, their ability to refuse ethically sensitive prompts-such as those
involving hate speech or illegal activities-has become central to content
moderation and responsible AI practices. While refusal responses can be viewed
as evidence of ethical alignment and safety-conscious behavior, recent research
suggests that users may perceive them negatively. At the same time, automated
assessments of model outputs are playing a growing role in both evaluation and
training. In particular, LLM-as-a-Judge frameworks-in which one model is used
to evaluate the output of another-are now widely adopted to guide benchmarking
and fine-tuning. This paper examines whether such model-based evaluators assess
refusal responses differently than human users. Drawing on data from Chatbot
Arena and judgments from two AI judges (GPT-4o and Llama 3 70B), we compare how
different types of refusals are rated. We distinguish ethical refusals, which
explicitly cite safety or normative concerns (e.g., "I can't help with that
because it may be harmful"), and technical refusals, which reflect system
limitations (e.g., "I can't answer because I lack real-time data"). We find
that LLM-as-a-Judge systems evaluate ethical refusals significantly more
favorably than human users, a divergence not observed for technical refusals.
We refer to this divergence as a moderation bias-a systematic tendency for
model-based evaluators to reward refusal behaviors more than human users do.
This raises broader questions about transparency, value alignment, and the
normative assumptions embedded in automated evaluation systems.

---


### [AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving](http://arxiv.org/abs/2505.15298v1)

Vision-Language Models (VLMs) show promise for autonomous driving, yet their
struggle with hallucinations, inefficient reasoning, and limited real-world
validation hinders accurate perception and robust step-by-step reasoning. To
overcome this, we introduce \textbf{AgentThink}, a pioneering unified framework
that, for the first time, integrates Chain-of-Thought (CoT) reasoning with
dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's
core innovations include: \textbf{(i) Structured Data Generation}, by
establishing an autonomous driving tool library to automatically construct
structured, self-verified reasoning data explicitly incorporating tool usage
for diverse driving scenarios; \textbf{(ii) A Two-stage Training Pipeline},
employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization
(GRPO) to equip VLMs with the capability for autonomous tool invocation; and
\textbf{(iii) Agent-style Tool-Usage Evaluation}, introducing a novel
multi-tool assessment protocol to rigorously evaluate the model's tool
invocation and utilization. Experiments on the DriveLMM-o1 benchmark
demonstrate AgentThink significantly boosts overall reasoning scores by
\textbf{53.91\%} and enhances answer accuracy by \textbf{33.54\%}, while
markedly improving reasoning quality and consistency. Furthermore, ablation
studies and robust zero-shot/few-shot generalization experiments across various
benchmarks underscore its powerful capabilities. These findings highlight a
promising trajectory for developing trustworthy and tool-aware autonomous
driving models.

---


### [Web-Shepherd: Advancing PRMs for Reinforcing Web Agents](http://arxiv.org/abs/2505.15277v1)

Web navigation is a unique domain that can automate many repetitive real-life
tasks and is challenging as it requires long-horizon sequential decision making
beyond typical multimodal large language model (MLLM) tasks. Yet, specialized
reward models for web navigation that can be utilized during both training and
test-time have been absent until now. Despite the importance of speed and
cost-effectiveness, prior works have utilized MLLMs as reward models, which
poses significant constraints for real-world deployment. To address this, in
this work, we propose the first process reward model (PRM) called Web-Shepherd
which could assess web navigation trajectories in a step-level. To achieve
this, we first construct the WebPRM Collection, a large-scale dataset with 40K
step-level preference pairs and annotated checklists spanning diverse domains
and difficulty levels. Next, we also introduce the WebRewardBench, the first
meta-evaluation benchmark for evaluating PRMs. In our experiments, we observe
that our Web-Shepherd achieves about 30 points better accuracy compared to
using GPT-4o on WebRewardBench. Furthermore, when testing on WebArena-lite by
using GPT-4o-mini as the policy and Web-Shepherd as the verifier, we achieve
10.9 points better performance, in 10 less cost compared to using GPT-4o-mini
as the verifier. Our model, dataset, and code are publicly available at LINK.

---


### [ReGUIDE: Data Efficient GUI Grounding via Spatial Reasoning and Search](http://arxiv.org/abs/2505.15259v1)

Recent advances in Multimodal Large Language Models (MLLMs) have enabled
autonomous agents to interact with computers via Graphical User Interfaces
(GUIs), where accurately localizing the coordinates of interface elements
(e.g., buttons) is often required for fine-grained actions. However, this
remains significantly challenging, leading prior works to rely on large-scale
web datasets to improve the grounding accuracy. In this work, we propose
Reasoning Graphical User Interface Grounding for Data Efficiency (ReGUIDE), a
novel and effective framework for web grounding that enables MLLMs to learn
data efficiently through self-generated reasoning and spatial-aware criticism.
More specifically, ReGUIDE learns to (i) self-generate a language reasoning
process for the localization via online reinforcement learning, and (ii)
criticize the prediction using spatial priors that enforce equivariance under
input transformations. At inference time, ReGUIDE further boosts performance
through a test-time scaling strategy, which combines spatial search with
coordinate aggregation. Our experiments demonstrate that ReGUIDE significantly
advances web grounding performance across multiple benchmarks, outperforming
baselines with substantially fewer training data points (e.g., only 0.2%
samples compared to the best open-sourced baselines).

---


### [Multilingual Prompting for Improving LLM Generation Diversity](http://arxiv.org/abs/2505.15229v1)

Large Language Models (LLMs) are known to lack cultural representation and
overall diversity in their generations, from expressing opinions to answering
factual questions. To mitigate this problem, we propose multilingual prompting:
a prompting method which generates several variations of a base prompt with
added cultural and linguistic cues from several cultures, generates responses,
and then combines the results. Building on evidence that LLMs have
language-specific knowledge, multilingual prompting seeks to increase diversity
by activating a broader range of cultural knowledge embedded in model training
data. Through experiments across multiple models (GPT-4o, GPT-4o-mini, LLaMA
70B, and LLaMA 8B), we show that multilingual prompting consistently
outperforms existing diversity-enhancing techniques such as high-temperature
sampling, step-by-step recall, and personas prompting. Further analyses show
that the benefits of multilingual prompting vary with language resource level
and model size, and that aligning the prompting language with the cultural cues
reduces hallucination about culturally-specific information.

---


### [R-TOFU: Unlearning in Large Reasoning Models](http://arxiv.org/abs/2505.15214v1)

Large Reasoning Models (LRMs) embed private or copyrighted information not
only in their final answers but also throughout multi-step chain-of-thought
(CoT) traces, making reliable unlearning far more demanding than in standard
LLMs. We introduce Reasoning-TOFU (R-TOFU), the first benchmark tailored to
this setting. R-TOFU augments existing unlearning tasks with realistic CoT
annotations and provides step-wise metrics that expose residual knowledge
invisible to answer-level checks. Using R-TOFU, we carry out a comprehensive
comparison of gradient-based and preference-optimization baselines and show
that conventional answer-only objectives leave substantial forget traces in
reasoning. We further propose Reasoned IDK, a preference-optimization variant
that preserves coherent yet inconclusive reasoning, achieving a stronger
balance between forgetting efficacy and model utility than earlier refusal
styles. Finally, we identify a failure mode: decoding variants such as
ZeroThink and LessThink can still reveal forgotten content despite seemingly
successful unlearning, emphasizing the need to evaluate models under diverse
decoding settings. Together, the benchmark, analysis, and new baseline
establish a systematic foundation for studying and improving unlearning in LRMs
while preserving their reasoning capabilities.

---


### [EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](http://arxiv.org/abs/2505.15196v1)

Goal-oriented script planning, or the ability to devise coherent sequences of
actions toward specific goals, is commonly employed by humans to plan for
typical activities. In e-commerce, customers increasingly seek LLM-based
assistants to generate scripts and recommend products at each step, thereby
facilitating convenient and efficient shopping experiences. However, this
capability remains underexplored due to several challenges, including the
inability of LLMs to simultaneously conduct script planning and product
retrieval, difficulties in matching products caused by semantic discrepancies
between planned actions and search queries, and a lack of methods and benchmark
data for evaluation. In this paper, we step forward by formally defining the
task of E-commerce Script Planning (EcomScript) as three sequential subtasks.
We propose a novel framework that enables the scalable generation of
product-enriched scripts by associating products with each step based on the
semantic similarity between the actions and their purchase intentions. By
applying our framework to real-world e-commerce data, we construct the very
first large-scale EcomScript dataset, EcomScriptBench, which includes 605,229
scripts sourced from 2.4 million products. Human annotations are then conducted
to provide gold labels for a sampled subset, forming an evaluation benchmark.
Extensive experiments reveal that current (L)LMs face significant challenges
with EcomScript tasks, even after fine-tuning, while injecting product purchase
intentions improves their performance.

---


### [ALN-P3: Unified Language Alignment for Perception, Prediction, and Planning in Autonomous Driving](http://arxiv.org/abs/2505.15158v1)

Recent advances have explored integrating large language models (LLMs) into
end-to-end autonomous driving systems to enhance generalization and
interpretability. However, most existing approaches are limited to either
driving performance or vision-language reasoning, making it difficult to
achieve both simultaneously. In this paper, we propose ALN-P3, a unified
co-distillation framework that introduces cross-modal alignment between "fast"
vision-based autonomous driving systems and "slow" language-driven reasoning
modules. ALN-P3 incorporates three novel alignment mechanisms: Perception
Alignment (P1A), Prediction Alignment (P2A), and Planning Alignment (P3A),
which explicitly align visual tokens with corresponding linguistic outputs
across the full perception, prediction, and planning stack. All alignment
modules are applied only during training and incur no additional costs during
inference. Extensive experiments on four challenging benchmarks-nuScenes, Nu-X,
TOD3Cap, and nuScenes QA-demonstrate that ALN-P3 significantly improves both
driving decisions and language reasoning, achieving state-of-the-art results.

---


### [RoT: Enhancing Table Reasoning with Iterative Row-Wise Traversals](http://arxiv.org/abs/2505.15110v1)

The table reasoning task, crucial for efficient data acquisition, aims to
answer questions based on the given table. Recently, reasoning large language
models (RLLMs) with Long Chain-of-Thought (Long CoT) significantly enhance
reasoning capabilities, leading to brilliant performance on table reasoning.
However, Long CoT suffers from high cost for training and exhibits low
reliability due to table content hallucinations. Therefore, we propose
Row-of-Thought (RoT), which performs iteratively row-wise table traversal,
allowing for reasoning extension and reflection-based refinement at each
traversal. Scaling reasoning length by row-wise traversal and leveraging
reflection capabilities of LLMs, RoT is training-free. The sequential traversal
encourages greater attention to the table, thus reducing hallucinations.
Experiments show that RoT, using non-reasoning models, outperforms RLLMs by an
average of 4.3%, and achieves state-of-the-art results on WikiTableQuestions
and TableBench with comparable models, proving its effectiveness. Also, RoT
outperforms Long CoT with fewer reasoning tokens, indicating higher efficiency.

---


### [Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective](http://arxiv.org/abs/2505.15045v1)

Large language model (LLM)-based embedding models, benefiting from large
scale pre-training and post-training, have begun to surpass BERT and T5-based
models on general-purpose text embedding tasks such as document retrieval.
However, a fundamental limitation of LLM embeddings lies in the unidirectional
attention used during autoregressive pre-training, which misaligns with the
bidirectional nature of text embedding tasks. To this end, We propose adopting
diffusion language models for text embeddings, motivated by their inherent
bidirectional architecture and recent success in matching or surpassing LLMs
especially on reasoning tasks. We present the first systematic study of the
diffusion language embedding model, which outperforms the LLM-based embedding
model by 20% on long-document retrieval, 8% on reasoning-intensive retrieval,
2% on instruction-following retrieval, and achieve competitive performance on
traditional text embedding benchmarks. Our analysis verifies that bidirectional
attention is crucial for encoding global context in long and complex text.

---


### [Effective and Efficient Schema-aware Information Extraction Using On-Device Large Language Models](http://arxiv.org/abs/2505.14992v1)

Information extraction (IE) plays a crucial role in natural language
processing (NLP) by converting unstructured text into structured knowledge.
Deploying computationally intensive large language models (LLMs) on
resource-constrained devices for information extraction is challenging,
particularly due to issues like hallucinations, limited context length, and
high latency-especially when handling diverse extraction schemas. To address
these challenges, we propose a two-stage information extraction approach
adapted for on-device LLMs, called Dual-LoRA with Incremental Schema Caching
(DLISC), which enhances both schema identification and schema-aware extraction
in terms of effectiveness and efficiency. In particular, DLISC adopts an
Identification LoRA module for retrieving the most relevant schemas to a given
query, and an Extraction LoRA module for performing information extraction
based on the previously selected schemas. To accelerate extraction inference,
Incremental Schema Caching is incorporated to reduce redundant computation,
substantially improving efficiency. Extensive experiments across multiple
information extraction datasets demonstrate notable improvements in both
effectiveness and efficiency.

---


### [CRAFT: Training-Free Cascaded Retrieval for Tabular QA](http://arxiv.org/abs/2505.14984v1)

Table Question Answering (TQA) involves retrieving relevant tables from a
large corpus to answer natural language queries. Traditional dense retrieval
models, such as DTR and ColBERT, not only incur high computational costs for
large-scale retrieval tasks but also require retraining or fine-tuning on new
datasets, limiting their adaptability to evolving domains and knowledge. In
this work, we propose $\textbf{CRAFT}$, a cascaded retrieval approach that
first uses a sparse retrieval model to filter a subset of candidate tables
before applying more computationally expensive dense models and neural
re-rankers. Our approach achieves better retrieval performance than
state-of-the-art (SOTA) sparse, dense, and hybrid retrievers. We further
enhance table representations by generating table descriptions and titles using
Gemini Flash 1.5. End-to-end TQA results using various Large Language Models
(LLMs) on NQ-Tables, a subset of the Natural Questions Dataset, demonstrate
$\textbf{CRAFT}$ effectiveness.

---


### [Streamline Without Sacrifice -- Squeeze out Computation Redundancy in LMM](http://arxiv.org/abs/2505.15816v1)

Large multimodal models excel in multimodal tasks but face significant
computational challenges due to excessive computation on visual tokens. Unlike
token reduction methods that focus on token-level redundancy, we identify and
study the computation-level redundancy on vision tokens to ensure no
information loss. Our key insight is that vision tokens from the pretrained
vision encoder do not necessarily require all the heavy operations (e.g.,
self-attention, FFNs) in decoder-only LMMs and could be processed more lightly
with proper designs. We designed a series of experiments to discover and
progressively squeeze out the vision-related computation redundancy. Based on
our findings, we propose ProxyV, a novel approach that utilizes proxy vision
tokens to alleviate the computational burden on original vision tokens. ProxyV
enhances efficiency without compromising performance and can even yield notable
performance gains in scenarios with more moderate efficiency improvements.
Furthermore, the flexibility of ProxyV is demonstrated through its combination
with token reduction methods to boost efficiency further. The code will be made
public at this https://github.com/penghao-wu/ProxyV URL.

---


### [MMaDA: Multimodal Large Diffusion Language Models](http://arxiv.org/abs/2505.15809v1)

We introduce MMaDA, a novel class of multimodal diffusion foundation models
designed to achieve superior performance across diverse domains such as textual
reasoning, multimodal understanding, and text-to-image generation. The approach
is distinguished by three key innovations: (i) MMaDA adopts a unified diffusion
architecture with a shared probabilistic formulation and a modality-agnostic
design, eliminating the need for modality-specific components. This
architecture ensures seamless integration and processing across different data
types. (ii) We implement a mixed long chain-of-thought (CoT) fine-tuning
strategy that curates a unified CoT format across modalities. By aligning
reasoning processes between textual and visual domains, this strategy
facilitates cold-start training for the final reinforcement learning (RL)
stage, thereby enhancing the model's ability to handle complex tasks from the
outset. (iii) We propose UniGRPO, a unified policy-gradient-based RL algorithm
specifically tailored for diffusion foundation models. Utilizing diversified
reward modeling, UniGRPO unifies post-training across both reasoning and
generation tasks, ensuring consistent performance improvements. Experimental
results demonstrate that MMaDA-8B exhibits strong generalization capabilities
as a unified multimodal foundation model. It surpasses powerful models like
LLaMA-3-7B and Qwen2-7B in textual reasoning, outperforms Show-o and SEED-X in
multimodal understanding, and excels over SDXL and Janus in text-to-image
generation. These achievements highlight MMaDA's effectiveness in bridging the
gap between pretraining and post-training within unified diffusion
architectures, providing a comprehensive framework for future research and
development. We open-source our code and trained models at:
https://github.com/Gen-Verse/MMaDA

---


### [STAR-R1: Spacial TrAnsformation Reasoning by Reinforcing Multimodal LLMs](http://arxiv.org/abs/2505.15804v1)

Multimodal Large Language Models (MLLMs) have demonstrated remarkable
capabilities across diverse tasks, yet they lag significantly behind humans in
spatial reasoning. We investigate this gap through Transformation-Driven Visual
Reasoning (TVR), a challenging task requiring identification of object
transformations across images under varying viewpoints. While traditional
Supervised Fine-Tuning (SFT) fails to generate coherent reasoning paths in
cross-view settings, sparse-reward Reinforcement Learning (RL) suffers from
inefficient exploration and slow convergence. To address these limitations, we
propose STAR-R1, a novel framework that integrates a single-stage RL paradigm
with a fine-grained reward mechanism tailored for TVR. Specifically, STAR-R1
rewards partial correctness while penalizing excessive enumeration and passive
inaction, enabling efficient exploration and precise reasoning. Comprehensive
evaluations demonstrate that STAR-R1 achieves state-of-the-art performance
across all 11 metrics, outperforming SFT by 23% in cross-view scenarios.
Further analysis reveals STAR-R1's anthropomorphic behavior and highlights its
unique ability to compare all objects for improving spatial reasoning. Our work
provides critical insights in advancing the research of MLLMs and reasoning
models. The codes, model weights, and data will be publicly available at
https://github.com/zongzhao23/STAR-R1.

---


### [Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization](http://arxiv.org/abs/2505.15660v1)

The generalization capabilities of vision-language-action (VLA) models to
unseen tasks are crucial to achieving general-purpose robotic manipulation in
open-world settings. However, the cross-task generalization capabilities of
existing VLA models remain significantly underexplored. To address this gap, we
introduce AGNOSTOS, a novel simulation benchmark designed to rigorously
evaluate cross-task zero-shot generalization in manipulation. AGNOSTOS
comprises 23 unseen manipulation tasks for testing, distinct from common
training task distributions, and incorporates two levels of generalization
difficulty to assess robustness. Our systematic evaluation reveals that current
VLA models, despite being trained on diverse datasets, struggle to generalize
effectively to these unseen tasks. To overcome this limitation, we propose
Cross-Task In-Context Manipulation (X-ICM), a method that conditions large
language models (LLMs) on in-context demonstrations from seen tasks to predict
action sequences for unseen tasks. Additionally, we introduce a dynamics-guided
sample selection strategy that identifies relevant demonstrations by capturing
cross-task dynamics. On AGNOSTOS, X-ICM significantly improves cross-task
zero-shot generalization performance over leading VLAs. We believe AGNOSTOS and
X-ICM will serve as valuable tools for advancing general-purpose robotic
manipulation.

---


### [SNAP: A Benchmark for Testing the Effects of Capture Conditions on Fundamental Vision Tasks](http://arxiv.org/abs/2505.15628v1)

Generalization of deep-learning-based (DL) computer vision algorithms to
various image perturbations is hard to establish and remains an active area of
research. The majority of past analyses focused on the images already captured,
whereas effects of the image formation pipeline and environment are less
studied. In this paper, we address this issue by analyzing the impact of
capture conditions, such as camera parameters and lighting, on DL model
performance on 3 vision tasks -- image classification, object detection, and
visual question answering (VQA). To this end, we assess capture bias in common
vision datasets and create a new benchmark, SNAP (for $\textbf{S}$hutter speed,
ISO se$\textbf{N}$sitivity, and $\textbf{AP}$erture), consisting of images of
objects taken under controlled lighting conditions and with densely sampled
camera settings. We then evaluate a large number of DL vision models and show
the effects of capture conditions on each selected vision task. Lastly, we
conduct an experiment to establish a human baseline for the VQA task. Our
results show that computer vision datasets are significantly biased, the models
trained on this data do not reach human accuracy even on the well-exposed
images, and are susceptible to both major exposure changes and minute
variations of camera settings. Code and data can be found at
https://github.com/ykotseruba/SNAP

---


### [LENS: Multi-level Evaluation of Multimodal Reasoning with Large Language Models](http://arxiv.org/abs/2505.15616v1)

Multimodal Large Language Models (MLLMs) have achieved significant advances
in integrating visual and linguistic information, yet their ability to reason
about complex and real-world scenarios remains limited. The existing benchmarks
are usually constructed in the task-oriented manner without guarantee that
different task samples come from the same data distribution, thus they often
fall short in evaluating the synergistic effects of lower-level perceptual
capabilities on higher-order reasoning. To lift this limitation, we contribute
Lens, a multi-level benchmark with 3.4K contemporary images and 60K+
human-authored questions covering eight tasks and 12 daily scenarios, forming
three progressive task tiers, i.e., perception, understanding, and reasoning.
One feature is that each image is equipped with rich annotations for all tasks.
Thus, this dataset intrinsically supports to evaluate MLLMs to handle
image-invariable prompts, from basic perception to compositional reasoning. In
addition, our images are manully collected from the social media, in which 53%
were published later than Jan. 2025. We evaluate 15+ frontier MLLMs such as
Qwen2.5-VL-72B, InternVL3-78B, GPT-4o and two reasoning models QVQ-72B-preview
and Kimi-VL. These models are released later than Dec. 2024, and none of them
achieve an accuracy greater than 60% in the reasoning tasks. Project page:
https://github.com/Lens4MLLMs/lens. ICCV 2025 workshop page:
https://lens4mllms.github.io/mars2-workshop-iccv2025/

---


### [TinyDrive: Multiscale Visual Question Answering with Selective Token Routing for Autonomous Driving](http://arxiv.org/abs/2505.15564v1)

Vision Language Models (VLMs) employed for visual question-answering (VQA) in
autonomous driving often require substantial computational resources that pose
a challenge for their deployment in resource-constrained vehicles. To address
this challenge, we introduce TinyDrive, a lightweight yet effective VLM for
multi-view VQA in driving scenarios. Our model comprises two key components
including a multiscale vision encoder and a dual-level prioritization mechanism
for tokens and sequences. The multiscale encoder facilitates the processing of
multi-view images at diverse resolutions through scale injection and
cross-scale gating to generate enhanced visual representations. At the token
level, we design a token routing mechanism that dynamically selects and process
the most informative tokens based on learned importance scores. At the sequence
level, we propose integrating normalized loss, uncertainty estimates, and a
diversity metric to formulate sequence scores that rank and preserve samples
within a sequence priority buffer. Samples with higher scores are more
frequently selected for training. TinyDrive is first evaluated on our
custom-curated VQA dataset, and it is subsequently tested on the public DriveLM
benchmark, where it achieves state-of-the-art language understanding
performance. Notably, it achieves relative improvements of 11.1% and 35.4% in
BLEU-4 and METEOR scores, respectively, despite having a significantly smaller
parameter count.

---


### [Chain-of-Focus: Adaptive Visual Search and Zooming for Multimodal Reasoning via RL](http://arxiv.org/abs/2505.15436v1)

Vision language models (VLMs) have achieved impressive performance across a
variety of computer vision tasks. However, the multimodal reasoning capability
has not been fully explored in existing models. In this paper, we propose a
Chain-of-Focus (CoF) method that allows VLMs to perform adaptive focusing and
zooming in on key image regions based on obtained visual cues and the given
questions, achieving efficient multimodal reasoning. To enable this CoF
capability, we present a two-stage training pipeline, including supervised
fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we
construct the MM-CoF dataset, comprising 3K samples derived from a visual agent
designed to adaptively identify key regions to solve visual tasks with
different image resolutions and questions. We use MM-CoF to fine-tune the
Qwen2.5-VL model for cold start. In the RL stage, we leverage the outcome
accuracies and formats as rewards to update the Qwen2.5-VL model, enabling
further refining the search and reasoning strategy of models without human
priors. Our model achieves significant improvements on multiple benchmarks. On
the V* benchmark that requires strong visual reasoning capability, our model
outperforms existing VLMs by 5% among 8 image resolutions ranging from 224 to
4K, demonstrating the effectiveness of the proposed CoF method and facilitating
the more efficient deployment of VLMs in practical applications.

---


### [TimeCausality: Evaluating the Causal Ability in Time Dimension for Vision Language Models](http://arxiv.org/abs/2505.15435v1)

Reasoning about temporal causality, particularly irreversible transformations
of objects governed by real-world knowledge (e.g., fruit decay and human
aging), is a fundamental aspect of human visual understanding. Unlike temporal
perception based on simple event sequences, this form of reasoning requires a
deeper comprehension of how object states change over time. Although the
current powerful Vision-Language Models (VLMs) have demonstrated impressive
performance on a wide range of downstream tasks, their capacity to reason about
temporal causality remains underexplored. To address this gap, we introduce
\textbf{TimeCausality}, a novel benchmark specifically designed to evaluate the
causal reasoning ability of VLMs in the temporal dimension. Based on our
TimeCausality, we find that while the current SOTA open-source VLMs have
achieved performance levels comparable to closed-source models like GPT-4o on
various standard visual question answering tasks, they fall significantly
behind on our benchmark compared with their closed-source competitors.
Furthermore, even GPT-4o exhibits a marked drop in performance on TimeCausality
compared to its results on other tasks. These findings underscore the critical
need to incorporate temporal causality into the evaluation and development of
VLMs, and they highlight an important challenge for the open-source VLM
community moving forward. Code and Data are available at
\href{https://github.com/Zeqing-Wang/TimeCausality }{TimeCausality}.

---


### [Visual Question Answering on Multiple Remote Sensing Image Modalities](http://arxiv.org/abs/2505.15401v1)

The extraction of visual features is an essential step in Visual Question
Answering (VQA). Building a good visual representation of the analyzed scene is
indeed one of the essential keys for the system to be able to correctly
understand the latter in order to answer complex questions. In many fields such
as remote sensing, the visual feature extraction step could benefit
significantly from leveraging different image modalities carrying complementary
spectral, spatial and contextual information. In this work, we propose to add
multiple image modalities to VQA in the particular context of remote sensing,
leading to a novel task for the computer vision community. To this end, we
introduce a new VQA dataset, named TAMMI (Text and Multi-Modal Imagery) with
diverse questions on scenes described by three different modalities (very high
resolution RGB, multi-spectral imaging data and synthetic aperture radar).
Thanks to an automated pipeline, this dataset can be easily extended according
to experimental needs. We also propose the MM-RSVQA (Multi-modal
Multi-resolution Remote Sensing Visual Question Answering) model, based on
VisualBERT, a vision-language transformer, to effectively combine the multiple
image modalities and text through a trainable fusion process. A preliminary
experimental study shows promising results of our methodology on this
challenging dataset, with an accuracy of 65.56% on the targeted VQA task. This
pioneering work paves the way for the community to a new multi-modal
multi-resolution VQA task that can be applied in other imaging domains (such as
medical imaging) where multi-modality can enrich the visual representation of a
scene. The dataset and code are available at https://tammi.sylvainlobry.com/.

---


### [Parameter-Efficient Fine-Tuning of Multispectral Foundation Models for Hyperspectral Image Classification](http://arxiv.org/abs/2505.15334v1)

Foundation models have achieved great success across diverse domains,
including remote sensing (RS), thanks to their versatility and strong
generalization abilities. However, most RS foundation models are designed for
multispectral data, while hyperspectral imagery (HSI) - with its hundreds of
spectral bands - remains less explored. Fine-tuning such models for downstream
tasks is also challenging, often demanding considerable memory and storage. In
this paper, we propose an efficient framework to fine-tune SpectralGPT, a
multispectral foundation model, for hyperspectral image classification (HSIC).
We explore several Parameter-Efficient Fine-Tuning (PEFT) methods, including
Low-Rank Adaptation (LoRA), Kronecker-based adaptation (KronA), Low-Rank
Kronecker (LoKr), and the recent LoRA+, which uses distinct learning rates for
low-rank adapters scaled by a factor lambda. Inspired by LoRA+, we introduce
KronA+, which applies a similar mechanism to the Kronecker matrices. We
evaluate our approach on five datasets from different sensors, showing
competitive performance with state-of-the-art HSI models. Our full fine-tuning
(FFT) setup for SpectralGPT even outperforms a dedicated hyperspectral
foundation model on some datasets while requiring only a quarter of the
training epochs. Under the same number of epochs, KronA+ reaches similar
performance with far fewer trainable parameters - just 0.056 percent - and adds
only approximately 0.2 megabytes of storage, making it the most effective PEFT
method tested.

---


### [Towards Zero-Shot Differential Morphing Attack Detection with Multimodal Large Language Models](http://arxiv.org/abs/2505.15332v1)

Leveraging the power of multimodal large language models (LLMs) offers a
promising approach to enhancing the accuracy and interpretability of morphing
attack detection (MAD), especially in real-world biometric applications. This
work introduces the use of LLMs for differential morphing attack detection
(D-MAD). To the best of our knowledge, this is the first study to employ
multimodal LLMs to D-MAD using real biometric data. To effectively utilize
these models, we design Chain-of-Thought (CoT)-based prompts to reduce
failure-to-answer rates and enhance the reasoning behind decisions. Our
contributions include: (1) the first application of multimodal LLMs for D-MAD
using real data subjects, (2) CoT-based prompt engineering to improve response
reliability and explainability, (3) comprehensive qualitative and quantitative
benchmarking of LLM performance using data from 54 individuals captured in
passport enrollment scenarios, and (4) comparative analysis of two multimodal
LLMs: ChatGPT-4o and Gemini providing insights into their morphing attack
detection accuracy and decision transparency. Experimental results show that
ChatGPT-4o outperforms Gemini in detection accuracy, especially against
GAN-based morphs, though both models struggle under challenging conditions.
While Gemini offers more consistent explanations, ChatGPT-4o is more resilient
but prone to a higher failure-to-answer rate.

---


### [LiveVLM: Efficient Online Video Understanding via Streaming-Oriented KV Cache and Retrieval](http://arxiv.org/abs/2505.15269v1)

Recent developments in Video Large Language Models (Video LLMs) have enabled
models to process long video sequences and demonstrate remarkable performance.
Nonetheless, studies predominantly focus on offline video question answering,
neglecting memory usage and response speed that are essential in various
real-world applications, such as Deepseek services, autonomous driving, and
robotics. To mitigate these challenges, we propose $\textbf{LiveVLM}$, a
training-free framework specifically designed for streaming, online video
understanding and real-time interaction. Unlike existing works that process
videos only after one question is posed, LiveVLM constructs an innovative
streaming-oriented KV cache to process video streams in real-time, retain
long-term video details and eliminate redundant KVs, ensuring prompt responses
to user queries. For continuous video streams, LiveVLM generates and compresses
video key-value tensors (video KVs) to reserve visual information while
improving memory efficiency. Furthermore, when a new question is proposed,
LiveVLM incorporates an online question-answering process that efficiently
fetches both short-term and long-term visual information, while minimizing
interference from redundant context. Extensive experiments demonstrate that
LiveVLM enables the foundation LLaVA-OneVision model to process 44$\times$
number of frames on the same device, and achieves up to 5$\times$ speedup in
response speed compared with SoTA online methods at an input of 256 frames,
while maintaining the same or better model performance.

---


### [Flashback: Memory-Driven Zero-shot, Real-time Video Anomaly Detection](http://arxiv.org/abs/2505.15205v1)

Video Anomaly Detection (VAD) automatically identifies anomalous events from
video, mitigating the need for human operators in large-scale surveillance
deployments. However, three fundamental obstacles hinder real-world adoption:
domain dependency and real-time constraints -- requiring near-instantaneous
processing of incoming video. To this end, we propose Flashback, a zero-shot
and real-time video anomaly detection paradigm. Inspired by the human cognitive
mechanism of instantly judging anomalies and reasoning in current scenes based
on past experience, Flashback operates in two stages: Recall and Respond. In
the offline recall stage, an off-the-shelf LLM builds a pseudo-scene memory of
both normal and anomalous captions without any reliance on real anomaly data.
In the online respond stage, incoming video segments are embedded and matched
against this memory via similarity search. By eliminating all LLM calls at
inference time, Flashback delivers real-time VAD even on a consumer-grade GPU.
On two large datasets from real-world surveillance scenarios, UCF-Crime and
XD-Violence, we achieve 87.3 AUC (+7.0 pp) and 75.1 AP (+13.1 pp),
respectively, outperforming prior zero-shot VAD methods by large margins.

---


### [HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving](http://arxiv.org/abs/2505.15793v1)

Integrating Large Language Models (LLMs) with Reinforcement Learning (RL) can
enhance autonomous driving (AD) performance in complex scenarios. However,
current LLM-Dominated RL methods over-rely on LLM outputs, which are prone to
hallucinations.Evaluations show that state-of-the-art LLM indicates a
non-hallucination rate of only approximately 57.95% when assessed on essential
driving-related tasks. Thus, in these methods, hallucinations from the LLM can
directly jeopardize the performance of driving policies. This paper argues that
maintaining relative independence between the LLM and the RL is vital for
solving the hallucinations problem. Consequently, this paper is devoted to
propose a novel LLM-Hinted RL paradigm. The LLM is used to generate semantic
hints for state augmentation and policy optimization to assist RL agent in
motion planning, while the RL agent counteracts potential erroneous semantic
indications through policy learning to achieve excellent driving performance.
Based on this paradigm, we propose the HCRMP (LLM-Hinted Contextual
Reinforcement Learning Motion Planner) architecture, which is designed that
includes Augmented Semantic Representation Module to extend state space.
Contextual Stability Anchor Module enhances the reliability of multi-critic
weight hints by utilizing information from the knowledge base. Semantic Cache
Module is employed to seamlessly integrate LLM low-frequency guidance with RL
high-frequency control. Extensive experiments in CARLA validate HCRMP's strong
overall driving performance. HCRMP achieves a task success rate of up to 80.3%
under diverse driving conditions with different traffic densities. Under
safety-critical driving conditions, HCRMP significantly reduces the collision
rate by 11.4%, which effectively improves the driving performance in complex
scenarios.

---


### [FLARE: Robot Learning with Implicit World Modeling](http://arxiv.org/abs/2505.15659v1)

We introduce $\textbf{F}$uture $\textbf{LA}$tent $\textbf{RE}$presentation
Alignment ($\textbf{FLARE}$), a novel framework that integrates predictive
latent world modeling into robot policy learning. By aligning features from a
diffusion transformer with latent embeddings of future observations,
$\textbf{FLARE}$ enables a diffusion transformer policy to anticipate latent
representations of future observations, allowing it to reason about long-term
consequences while generating actions. Remarkably lightweight, $\textbf{FLARE}$
requires only minimal architectural modifications -- adding a few tokens to
standard vision-language-action (VLA) models -- yet delivers substantial
performance gains. Across two challenging multitask simulation imitation
learning benchmarks spanning single-arm and humanoid tabletop manipulation,
$\textbf{FLARE}$ achieves state-of-the-art performance, outperforming prior
policy learning baselines by up to 26%. Moreover, $\textbf{FLARE}$ unlocks the
ability to co-train with human egocentric video demonstrations without action
labels, significantly boosting policy generalization to a novel object with
unseen geometry with as few as a single robot demonstration. Our results
establish $\textbf{FLARE}$ as a general and scalable approach for combining
implicit world modeling with high-frequency robotic control.

---


### [SSR: Speculative Parallel Scaling Reasoning in Test-time](http://arxiv.org/abs/2505.15340v1)

Large language models (LLMs) have achieved impressive results on multi-step
mathematical reasoning, yet at the cost of high computational overhead. This
challenge is particularly acute for test-time scaling methods such as parallel
decoding, which increase answer diversity but scale poorly in efficiency. To
address this efficiency-accuracy trade-off, we propose SSR (Speculative
Parallel Scaling Reasoning), a training-free framework that leverages a key
insight: by introducing speculative decoding at the step level, we can
accelerate reasoning without sacrificing correctness. SSR integrates two
components: a Selective Parallel Module (SPM) that identifies a small set of
promising reasoning strategies via model-internal scoring, and Step-level
Speculative Decoding (SSD), which enables efficient draft-target collaboration
for fine-grained reasoning acceleration. Experiments on three mathematical
benchmarks-AIME 2024, MATH-500, and LiveMathBench - demonstrate that SSR
achieves strong gains over baselines. For instance, on LiveMathBench, SSR
improves pass@1 accuracy by 13.84% while reducing computation to 80.5% of the
baseline FLOPs. On MATH-500, SSR reduces compute to only 30% with no loss in
accuracy.

---


### [Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models](http://arxiv.org/abs/2505.15130v1)

Vision-Language Models (VLMs) such as CLIP have shown remarkable performance
in cross-modal tasks through large-scale contrastive pre-training. To adapt
these large transformer-based models efficiently for downstream tasks,
Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as
scalable alternatives to full fine-tuning, especially in few-shot scenarios.
However, like traditional deep neural networks, VLMs are highly vulnerable to
adversarial attacks, where imperceptible perturbations can significantly
degrade model performance. Adversarial training remains the most effective
strategy for improving model robustness in PEFT. In this work, we propose
AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial
robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method
formulates adversarial fine-tuning as a minimax optimization problem and
provides theoretical guarantees for convergence under smoothness and
nonconvex-strong-concavity assumptions. Empirical results across eight datasets
using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly
improves robustness against common adversarial attacks (e.g., FGSM, PGD),
without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA
as a practical and theoretically grounded approach for robust adaptation of
VLMs in resource-constrained settings.

---


### [Agentic Feature Augmentation: Unifying Selection and Generation with Teaming, Planning, and Memories](http://arxiv.org/abs/2505.15076v1)

As a widely-used and practical tool, feature engineering transforms raw data
into discriminative features to advance AI model performance. However, existing
methods usually apply feature selection and generation separately, failing to
strive a balance between reducing redundancy and adding meaningful dimensions.
To fill this gap, we propose an agentic feature augmentation concept, where the
unification of feature generation and selection is modeled as agentic teaming
and planning. Specifically, we develop a Multi-Agent System with Long and
Short-Term Memory (MAGS), comprising a selector agent to eliminate redundant
features, a generator agent to produce informative new dimensions, and a router
agent that strategically coordinates their actions. We leverage in-context
learning with short-term memory for immediate feedback refinement and long-term
memory for globally optimal guidance. Additionally, we employ offline Proximal
Policy Optimization (PPO) reinforcement fine-tuning to train the router agent
for effective decision-making to navigate a vast discrete feature space.
Extensive experiments demonstrate that this unified agentic framework
consistently achieves superior task performance by intelligently orchestrating
feature selection and generation.

---


### [Harnessing Large Language Models Locally: Empirical Results and Implications for AI PC](http://arxiv.org/abs/2505.15030v1)

The increasing deployment of Large Language Models (LLMs) on edge devices,
driven by model advancements and hardware improvements, offers significant
privacy benefits. However, these on-device LLMs inherently face performance
limitations due to reduced model capacity and necessary compression techniques.
To address this, we introduce a systematic methodology -- encompassing model
capability, development efficiency, and system resources -- for evaluating
on-device LLMs. Our comprehensive evaluation, encompassing models from 0.5B to
14B parameters and seven post-training quantization (PTQ) methods on commodity
laptops, yields several critical insights: 1) System-level metrics exhibit
near-linear scaling with effective bits-per-weight (BPW). 2) A practical
threshold exists around $\sim$3.5 effective BPW, larger models subjected to
low-bit quantization consistently outperform smaller models utilizing higher
bit-precision. 3) Quantization with low BPW incurs marginal accuracy loss but
significant memory savings. 4) Determined by low-level implementation specifics
power consumption on CPU, where computation-intensive operations spend more
power than memory-intensive ones. These findings offer crucial insights and
practical guidelines for the efficient deployment and optimized configuration
of LLMs on resource-constrained edge devices. Our codebase is available at
https://github.com/simmonssong/LLMOnDevice.

---


### [From Grounding to Manipulation: Case Studies of Foundation Model Integration in Embodied Robotic Systems](http://arxiv.org/abs/2505.15685v1)

Foundation models (FMs) are increasingly used to bridge language and action
in embodied agents, yet the operational characteristics of different FM
integration strategies remain under-explored -- particularly for complex
instruction following and versatile action generation in changing environments.
This paper examines three paradigms for building robotic systems: end-to-end
vision-language-action (VLA) models that implicitly integrate perception and
planning, and modular pipelines incorporating either vision-language models
(VLMs) or multimodal large language models (LLMs). We evaluate these paradigms
through two focused case studies: a complex instruction grounding task
assessing fine-grained instruction understanding and cross-modal
disambiguation, and an object manipulation task targeting skill transfer via
VLA finetuning. Our experiments in zero-shot and few-shot settings reveal
trade-offs in generalization and data efficiency. By exploring performance
limits, we distill design implications for developing language-driven physical
agents and outline emerging challenges and opportunities for FM-powered
robotics in real-world conditions.

---


### [Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control](http://arxiv.org/abs/2505.15304v1)

Deep neural network (DNN)-based policy models, such as vision-language-action
(VLA) models, excel at automating complex decision-making from multi-modal
inputs. However, scaling these models greatly increases computational overhead,
complicating deployment in resource-constrained settings like robot
manipulation and autonomous driving. To address this, we propose Saliency-Aware
Quantized Imitation Learning (SQIL), which combines quantization-aware training
with a selective loss-weighting strategy for mission-critical states. By
identifying these states via saliency scores and emphasizing them in the
training loss, SQIL preserves decision fidelity under low-bit precision. We
validate SQIL's generalization capability across extensive simulation
benchmarks with environment variations, real-world tasks, and cross-domain
tasks (self-driving, physics simulation), consistently recovering
full-precision performance. Notably, a 4-bit weight-quantized VLA model for
robotic manipulation achieves up to 2.5x speedup and 2.5x energy savings on an
edge GPU with minimal accuracy loss. These results underline SQIL's potential
for efficiently deploying large IL-based policy models on resource-limited
devices.

---


### [Enhancing Cloud Task Scheduling Using a Hybrid Particle Swarm and Grey Wolf Optimization Approach](http://arxiv.org/abs/2505.15171v1)

Assigning tasks efficiently in cloud computing is a challenging problem and
is considered an NP-hard problem. Many researchers have used metaheuristic
algorithms to solve it, but these often struggle to handle dynamic workloads
and explore all possible options effectively. Therefore, this paper presents a
new hybrid method that combines two popular algorithms, Grey Wolf Optimizer
(GWO) and Particle Swarm Optimization (PSO). GWO offers strong global search
capabilities (exploration), while PSO enhances local refinement (exploitation).
The hybrid approach, called HybridPSOGWO, is compared with other existing
methods like MPSOSA, RL-GWO, CCGP, and HybridPSOMinMin, using key performance
indicators such as makespan, throughput, and load balancing. We tested our
approach using both a simulation tool (CloudSim Plus) and real-world data. The
results show that HybridPSOGWO outperforms other methods, with up to 15\%
improvement in makespan and 10\% better throughput, while also distributing
tasks more evenly across virtual machines. Our implementation achieves
consistent convergence within a few iterations, highlighting its potential for
efficient and adaptive cloud scheduling.

---


