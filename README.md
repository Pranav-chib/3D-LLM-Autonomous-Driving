# <p align=center>`3D-LLM-Autonomous-Driving`<br>
**LLM/VLLM Agent in Autonomous-Driving:** Integrates varied input modalities from different driving environments to perform specific tasks. Depending on the function, the model outputs perception, planning, and control signals, leveraging the reasoning and cognitive capabilities of the LLM through question answering, counterfactual reasoning, and description.


</p>

<p align="center">
<img src="/Paradigm.drawio.png" width="900" height="200"/>
<p>


<hr />

# <p align=center>3D Perception and LLM-based Autonomous Driving: A Survey of State-of-the-art <a href="https://medium.com/@pranavs_chib/end-to-end-autonomous-driving-using-deep-learning-8a94ecb3bb6b">

Authors: [Nitin Dwivedi](https://github.com/), [Pravendra Singh](https://scholar.google.com/citations?user=YwDTxJMAAAAJ&hl=en)</p>
The ongoing advancements in 3D perception and the integration of Large Language Models (LLMs) in Autonomous Driving (AD) have significantly enhanced the capabilities of intelligent vehicles, building on the achievements of 2D perception in tasks like object detection, scene analysis, and language-guided reasoning. This paper presents a comprehensive survey of 3D perception-based LLM agents, examining their methodologies, applications, and potential to revolutionize autonomous systems. Uniquely, our work bridges a critical gap in the literature by offering the first meta-analysis exclusively focused on the synergy between 3D perception and LLMs, addressing emerging challenges such as 3D tokenization, spatial reasoning, and computational scalability. Unlike prior surveys centered on 2D tasks, this study provides an in-depth exploration of 3D-specific advancements, highlighting the transformative potential of these systems. The paper highlights the significance of this integration for fostering safer, human-centric AD, identifying opportunities to overcome current limitations, and driving innovation in intelligent mobility solutions.


### <p align=center>**Timeline of existing LLMs integrated with 3D perception for AD in recent years**


<p align="center">
<img src="/drivellm_timeline.drawio.png" width="900" height="550"/>
<p>




## Table of Contents
This repo contains a curated list of resources on 3D LLM-based autonomous driving research, arranged chronologically. We regularly update it with the latest papers and their corresponding open-source implementations.
1. [LLM driving Agents](#LLM-Driving-Agents)
2. [3D Perception based LLM driving Agents](#EXPLAINABILITY)
3. [Generative World Models](#EVALUATION)
4. [Language Based AD Datasets (QA, Captioning etc.)](#SAFETY)
<hr />


# LLM Driving Agents


[**Mtd-gpt: A multi-task
decision-making gpt model for autonomous driving at unsignalized in-
tersections**](https://arxiv.org/abs/2307.16118) [ITSC-2023]<br> Jiaqi Liu, Peng Hang, Xiao Qi, Jianqiang Wang, Jian Sun <br>

- LLM Arch: [GPT 2](https://huggingface.co/openai-community/gpt2)

- Task: Perception, Planning, Decision Making

- Metrics: RLHF, GPT Score

- Datasets: [Expert Dataset](https://github.com/Farama-Foundation/HighwayEnv)


[**GPT-Driver: Learning to Drive with GPT**](https://arxiv.org/abs/2307.16118) [NeurIPS-2023 Workshop]<br> Jiageng Mao, Yuxi Qian, Junjie Ye, Hang Zhao, Yue Wang <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PointsCoder/GPT-Driver)


- LLM Arch: [GPT-3.5](https://platform.openai.com/docs/models)

- Task: Planning 

- Metrics: Avg. L2 and Collision Rate

- Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)


[**Drivegpt4: Interpretable end-to-end autonomous driving via
large language mode**](https://arxiv.org/abs/2310.01412) [RAL-2024]<br> Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee. K. Wong, Zhenguo Li, Hengshuang Zhao <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://drive.google.com/drive/folders/1PsGL7ZxMMz1ZPDS5dZSjzjfPjuPHxVL5?usp=sharing) 

- LLM Arch: [Llama 2](https://www.llama.com/llama2/), [CLIP](https://github.com/openai/CLIP)

- Task: Planning, Control

- Metrics: BLEU4, METEOR, ChatGPT score, RMSE, CIDEr

- Datasets: [BDD-X](https://github.com/JinkyuKimUCB/BDD-X-dataset)

