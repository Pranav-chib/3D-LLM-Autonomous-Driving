# <p align=center>`3D-LLM-Autonomous-Driving`<br>
**LLM/VLLM Agent in Autonomous-Driving:** Integrates varied input modalities from different driving environments to perform specific tasks. Depending on the function, the model outputs perception, planning, and control signals, leveraging the reasoning and cognitive capabilities of the LLM through question answering, counterfactual reasoning, and description.


</p>

<p align="center">
<img src="/Paradigm.drawio.png" width="900" height="200"/>
<p>


<hr />

# <p align=center>3D Perception and LLM-based Autonomous Driving: A Survey of State-of-the-art <a href="https://medium.com/@pranavs_chib/end-to-end-autonomous-driving-using-deep-learning-8a94ecb3bb6b">



Authors: [Nitin Dwivedi](https://github.com/), [Pranav Singh Chib](https://github.com/Pranav-chib), [Pravendra Singh](https://scholar.google.com/citations?user=YwDTxJMAAAAJ&hl=en)</p>
The ongoing advancements in 3D perception and the integration of Large Language Models (LLMs) in Autonomous Driving (AD) have significantly enhanced the capabilities of intelligent vehicles, building on the achievements of 2D perception in tasks like object detection, scene analysis, and language-guided reasoning. This paper presents a comprehensive survey of 3D perception-based LLM agents, examining their methodologies, applications, and potential to revolutionize autonomous systems. Uniquely, our work bridges a critical gap in the literature by offering the first meta-analysis exclusively focused on the synergy between 3D perception and LLMs, addressing emerging challenges such as 3D tokenization, spatial reasoning, and computational scalability. Unlike prior surveys centered on 2D tasks, this study provides an in-depth exploration of 3D-specific advancements, highlighting the transformative potential of these systems. The paper highlights the significance of this integration for fostering safer, human-centric AD, identifying opportunities to overcome current limitations, and driving innovation in intelligent mobility solutions.


### <p align=center>**Timeline of existing LLMs integrated with 3D perception for AD in recent years**


<p align="center">
<img src="/drivellm_timeline.drawio.png" width="900" height="550"/>
<p>




## Table of Contents
This repo contains a curated list of resources on 3D LLM-based autonomous driving research, arranged chronologically. We regularly update it with the latest papers and their corresponding open-source implementations.
1. [LLM driving Agents](#LLM-Driving-Agents)
2. [3D Perception based LLM driving Agents](#3D-Perception-based-LLM-Driving-Agents)
3. [Generative World Models](#Generative-World-Models)
4. [Language Based AD Datasets (QA, Captioning etc.)](#Language-Based-AD-Datasets)
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



[**Drive as You Speak: Enabling Human-Like Interaction with Large Language Models in Autonomous Vehicles**](https://openaccess.thecvf.com/content/WACV2024W/LLVM-AD/papers/Cui_Drive_As_You_Speak_Enabling_Human-Like_Interaction_With_Large_Language_WACVW_2024_paper.pdf) [WACV-2024 Workshop]<br> Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Ziran Wang <br>

- LLM Arch: [GPT4](https://platform.openai.com/docs/models)

- Task: Motion Planning
  
- Metrics: GPT Score



[**Drive Like a Human: Rethinking Autonomous Driving with Large Language Models**](https://arxiv.org/abs/2307.07162) [WACV-2024 Workshop]<br> Daocheng Fu, Xin Li, Licheng Wen, Min Dou, Pinlong Cai, Botian Shi, Yu Qiao <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PJLab-ADG/DriveLikeAHuman) 

- LLM Arch: [GPT-3.5](https://platform.openai.com/docs/models)
  
- Task: Planning, Control

- Datasets: [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)


[**Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving**](https://browse.arxiv.org/abs/2310.01957) [ICRA-2024]  <br>Long Chen, Oleg Sinavski, Jan Hünermann, Alice Karnsund, Andrew James Willmott, Danny Birch, Daniel Maund, Jamie Shotton <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/wayveai/Driving-with-LLMs/tree/main/data)

- LLM Arch: [GPT-3.5](https://platform.openai.com/docs/models)
  
- Task: Perception, Planning
  
- Metrics: MAE, GPT Grading

- Datasets: [Driving QA](https://github.com/wayveai/Driving-with-LLMs/tree/main/data)




[**LANGUAGEMPC: LARGE LANGUAGE MODELS AS DECISION MAKERS FOR AUTONOMOUS DRIVING**](https://arxiv.org/pdf/2310.03026) [arXiv-2023]  <br>Hao Sha, Yao Mu, Yuxuan Jiang, Li Chen, Chenfeng Xu, Ping Luo, Shengbo Eben Li, Masayoshi Tomizuka, Wei Zhan, Mingyu Ding <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://sites.google.com/view/llm-mpc)

- LLM Arch: [GPT-3.5](https://platform.openai.com/docs/models)
  
- Task: Planning, Control
  
- Metrics: Observation matrix, Weight matrix, Action bias, Collision Rate, Inefficiency, Time
  
- Datasets: [IdSim](https://github.com/liuyuqi123/ComplexUrbanScenarios)



[**A language agent for autonomous driving**](https://openreview.net/forum?id=UPE6WYE8vg#discussion) [COLM-2024]  <br>Jiageng Mao, Jiageng_Mao1, Junjie Ye, Yuxi Qian, Marco Pavone, Yue Wang<br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/USC-GVL/Agent-Driver)

- LLM Arch: [GPT-3.5](https://platform.openai.com/docs/models)
  
- Task: Perception, Prediction, Planning
  
- Metrics: Avg. L2, Collision Rate
  
- Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)


[**“Asynchronous Large Language Model Enhanced Planner for Autonomous Driving**](https://arxiv.org/abs/2406.14556) [ECCV-2024]  <br>Yuan Chen, Zi-han Ding, Ziqin Wang, Yan Wang, Lijun Zhang, Si Liu<br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/memberRE/AsyncDriver)

- LLM Arch: [Llama2-13B](https://huggingface.co/meta-llama/Llama-2-13b)
  
- Task: Planning
  
- Metrics: Driving direction compliance, Ego comfortable, Ego progress along the expert route, No ego at-fault collisions, Speed limit compliance, Time to collision 
  
- Datasets: [nuPlan](https://www.nuscenes.org/nuplan)

  
[🔼 Back to top](#Table-of-Contents)
<hr />


# 3D Perception based LLM Driving Agents


[**Omnidrive: A holistic llm-agent framework
for autonomous driving with 3d perception, reasoning and planninge**](https://arxiv.org/abs/2405.01533) [arXiv-2024]<br> Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, Jose M. Alvarez <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NVlabs/OmniDrive) 

- LLM Arch: [GPT4](https://platform.openai.com/docs/models), [EVA-02-L](https://github.com/baaivision/EVA) 

- Task: 3D perception, VQA

- Metrics: CR, IR, METEOR, ROUGE and CIDEr

- Datasets: [OmniDrive](https://github.com/NVlabs/OmniDrive/tree/main), [nuScenes](https://www.nuscenes.org/nuscenes)


[**Is a 3D-Tokenized LLM the Key to Reliable Autonomous Driving?**](https://arxiv.org/abs/2405.18361) [arXiv-2024]<br> Yifan Bai, Dongming Wu, Yingfei Liu, Fan Jia, Weixin Mao, Ziheng Zhang, Yucheng Zhao, Jianbing Shen, Xing Wei, Tiancai Wang, Xiangyu Zhang <br>

- LLM Arch: [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [StreamPETR](https://github.com/exiawsh/StreamPETR), [TopoMLP](https://github.com/wudongming97/TopoMLP)

- Task: 3D perception

- Metrics: Avg. L2 and Collision Rate

- Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)



[**DME-Driver: Integrating Human Decision Logic and 3D Scene Perception in Autonomous Driving**](https://arxiv.org/abs/2401.03641) [arXiv-2024]<br> Wencheng Han, Dongqian Guo, Cheng-Zhong Xu, Jianbing Shen <br>


- LLM Arch: [GPT4](https://platform.openai.com/docs/models), [LLaVA](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

- Task: Perception, Planning

- Metrics: Avg. L2 and Collision Rate

- Datasets: [Human-Driver Behavior and Decision-Making](https://arxiv.org/abs/2401.03641)



[**Talk2bev: Language-enhanced bird’s-eye view maps for autonomous driving**](https://arxiv.org/abs/2310.02251) [ICRA-2024]<br> Tushar Choudhary, Vikrant Dewangan, Shivam Chandhok, Shubham Priyadarshan, Anushka Jain, Arun K. Singh, Siddharth Srivastava, Krishna Murthy Jatavallabhula, K. Madhava Krishna <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://llmbev.github.io/talk2bev/) 


- LLM Arch: [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/), [Flan5XXL](https://huggingface.co/google/flan-t5-xxl)

- Task: Perception, Planning

- Metrics: Accuracy on MCQ, Jaccard Index, Distance Error

- Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)


[**DriveLM: Driving with Graph Visual Question Answering**](https://arxiv.org/abs/2312.14150) [ECCV-2024]<br> Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, Hongyang Li <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/OpenDriveLab/DriveLM) 


- LLM Arch: [Vicuna-13B](https://lmsys.org/blog/2023-03-30-vicuna/)

- Task: Graph VQA, Perception, Prediction, Planning

- Metrics: GPTScore, Meteor, BLEU1, CIDEr, ROUGE, Classification accuracy

- Datasets:  [DriveLM-nuScenes](https://github.com/OpenDriveLab/DriveLM), [DriveLM-CARLA](https://github.com/OpenDriveLab/DriveLM)




[**LLMI3D: MLLM-based 3D Perception from a Single 2D Image**](https://arxiv.org/abs/2408.07422) [arXiv-2025]<br> Fan Yang, Sicheng Zhao, Yanhao Zhang, Hui Chen, Haonan Lu, Jungong Han, Guiguang Ding Li <br>


- LLM Arch: [GPT4](https://platform.openai.com/docs/models)

- Task: Perception, Reasoning.

- Metrics:  Acc@0.25, Acc@0.5 (IoU thresholds of 25% and 50%), DepthError, LengthError, WidthError, and HeightError.

- Datasets:  [IG3D](https://arxiv.org/abs/2408.07422)


[🔼 Back to top](#Table-of-Contents)
<hr />

# Generative World Models


[**Dolphins: Multimodal Language Model for Driving**](https://arxiv.org/abs/2312.00438) [ECCV-2024]<br> Yingzi Ma, Yulong Cao, Jiachen Sun, Marco Pavone, Chaowei Xiao <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SaFoLab-WISC/Dolphins) 

- LLM Arch: [Flamingo](https://github.com/mlfoundations/open_flamingo)
[MPT](https://huggingface.co/docs/transformers/en/model_doc/mpt), [BLIP](https://huggingface.co/docs/transformers/model_doc/blip), 
[CLIP](https://github.com/openai/CLIP)

- Task: Prediction and Planning VQA

- Metrics: VQA accuracy

- Datasets:     [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html), [MSCOCO](https://cocodataset.org/#home), [VQAv2](https://visualqa.org/), [OK-VQA](https://okvqa.allenai.org/), [TDIUC](https://kushalkafle.com/projects/tdiuc.html), [Visual Genome dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), [BDD-X](https://github.com/JinkyuKimUCB/BDD-X-dataset)



[**SurrealDriver: Designing Generative Driver Agent Simulation Framework in Urban Contexts based on Large Language Model**](https://arxiv.org/abs/2309.13193) [arXiv-2024]<br> Ye Jin, Ruoxuan Yang, Zhijie Yi, Xiaoxi Shen, Huiling Peng, Xiaoan Liu, Jingli Qin, Jiayang Li, Jintao Xie, Peizhong Gao, Guyue Zhou, Jiangtao Gong <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AIR-DISCOVER/Driving-Thinking-Dataset) 

- LLM Arch: [GPT4](https://platform.openai.com/docs/models) 

- Task: Planning, Control

- Metrics: Collision Rate, ANOVAs

- Datasets: [Driving-Thinking-Dataset based on CARLA](https://github.com/carla-simulator)


[**DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving**](https://arxiv.org/abs/2309.09777) [ECCV-2024]<br> Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, Jiwen Lu <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://drivedreamer.github.io/) 

- LLM Arch:  [CLIP](https://github.com/openai/CLIP), [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4)

- Task: Generation

- Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)

[**DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation**](https://arxiv.org/abs/2403.06845) [arXiv-2024]<br> Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, Xingang Wang <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://drivedreamer2.github.io) 


- LLM Arch:  [GPT-3.5](https://platform.openai.com/docs/models) 

- Task: Generation

- Datasets: [nuScenes](https://www.nuscenes.org/nuscenes)

[**GAIA-1: A Generative World Model for Autonomous Driving**](https://arxiv.org/abs/2309.17080) [arXiv-2024]<br> Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Gianluca Corrado <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://wayve.ai/thinking/scaling-gaia-1/) 


- LLM Arch:  [DINO](https://github.com/facebookresearch/dino), [T5-large](https://huggingface.co/google-t5/t5-large)

- Task: Planning Generation

- Datasets: [GAIA-1](https://wayve.ai/thinking/scaling-gaia-1/)




[**Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving**](https://arxiv.org/pdf/2311.17918.pdf) [CVPR-2024]<br> AYuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, Zhaoxiang Zhang <br>
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BraveGroup/Drive-WM) 


- LLM Arch:  [CLIP](https://github.com/openai/CLIP), [GPT4-V](https://platform.openai.com/docs/models)
- 
- Task: Perception, Planning

- Datasets: [nuScenes](https://www.nuscenes.org/nuscenes), [Waymo Open Dataset](https://waymo.com/open/)





[🔼 Back to top](#Table-of-Contents)
<hr />


# Language Based AD Datasets


  | Dataset  |   Reasoning    |   Outlook | Size     | 
|:---------:|:-------------:|:------:|:--------------------------------------------:|
| [BDD-X 2018](https://github.com/JinkyuKimUCB/explainable-deep-driving)  | Description | Planning Description & Justification    | 8M frames, 20k text strings   
| [HAD HRI Advice 2019](https://usa.honda-ri.com/had)   | Advice | Goal-oriented & stimulus-driven advice | 5,675 video clips, 45k text strings  
| [Talk2Car 2019](https://github.com/talk2car/Talk2Car)  | Description |  Goal Point Description | 30k frames, 10k text strings 
| [DRAMA 2022](https://usa.honda-ri.com/drama)  | Description |  QA + Captions | 18k frames, 100k text strings 
| [nuScenes-QA 2023](https://arxiv.org/abs/2305.14836)   |  QA |  Perception Result     | 30k frames, 460k QA pairs
| [DriveLM-2023](https://github.com/OpenDriveLab/DriveLM) |  QA + Scene Description | Perception, Prediction and Planning with Logic | 30k frames, 600k QA pairs 
| [Rank2Tell-2023](https://usa.honda-ri.com/rank2tell) |  Captioning and Reasoning | Localization and Ranking | 118 frames
| [DRAMA-2023](http://usa.honda-ri.com/drama) |  Captioning and Reasoning | Perception and Prediction | 17785 frames
| [LingoQA-2024](https://github.com/wayveai/LingoQA) |  Captioning and Reasoning | Perception and Planning | 28k frames, 419.9k Annotations
| [MAPLM-2024](https://github.com/LLVM-AD/MAPLM) |  QA, Captioning and Reasoning | Perception and Prediction | 2M frames



[🔼 Back to top](#Table-of-Contents)
<hr />
