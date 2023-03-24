# CCIM
This repository contains the PyTorch implementation of the core contribution *Contextual Causal Intervention Module (CCIM)* from the CVPR2023 paper:

[Context De-confounded Emotion Recognition](https://arxiv.org/pdf/2303.11921.pdf)

## Abstract
Context-Aware Emotion Recognition (CAER) is a crucial and challenging task that aims to perceive the emotional states of
the target person with contextual information. Recent approaches invariably focus on designing sophisticated architectures 
or mechanisms to extract seemingly meaningful representations from subjects and contexts. However, a long-overlooked issue 
is that a context bias in existing datasets leads to a significantly unbalanced distribution of emotional states among different 
context scenarios. Concretely, the harmful bias is a confounder that misleads existing models to learn spurious correlations based 
on conventional likelihood estimation, significantly limiting the models' performance. To tackle the issue, this paper provides a 
causality-based perspective to disentangle the models from the impact of such bias, and formulate the causalities among variables in 
the CAER task via a tailored causal graph. Then, we propose a Contextual Causal Intervention Module (CCIM) based on the backdoor adjustment 
to de-confound the confounder and exploit the true causal effect for model training. CCIM is plug-in and model-agnostic, which improves diverse 
state-of-the-art approaches by considerable margins. Extensive experiments on three benchmark datasets demonstrate the effectiveness of our CCIM 
and the significance of causal insight.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/25423296/163456776-7f95b81a-f1ed-45f7-b7ab-8fa810d529fa.png">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
  <img alt="Shows an illustrated sun in light mode and a moon with stars in dark mode." src="https://user-images.githubusercontent.com/25423296/163456779-a8556205-d0a5-45e2-ac17-42d089e3c3f8.png">
</picture>
