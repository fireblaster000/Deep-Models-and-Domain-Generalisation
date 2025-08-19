# Deep Models and Domain Generalization

This repository contains research and experiments on domain generalization in machine learning, focusing on the performance of Vision Transformer (ViT), ResNet, and CLIP models when faced with Out-of-Domain (OOD) data.

## Abstract

This study addresses the critical challenge of domain generalization in machine learning, focusing on the performance of three prominent deep learning models—Vision Transformer (ViT), ResNet, and CLIP—when faced with Out-of-Domain (OOD) data. Through a series of evaluations, we identify distinct biases and generalization capabilities that impact each model's robustness to semantic and covariate shifts.

Our findings reveal that ResNet has the highest texture and colour bias, while Clip ViT has higher shape bias, and CLIP exhibits adaptability across various styles. By highlighting the influence of spurious features and cue conflicts, this research provides valuable insights into the inherent biases of deep learning models and their implications for generalization to unseen data distributions.

## Introduction

Domain generalization is a significant challenge in machine learning, especially for real-world applications where models must perform on unseen data distributions. Unlike traditional models that are trained and tested on the same data distributions, domain generalization aims to develop models capable of handling Out-of-Domain (OOD) data, which can vary significantly from the training data due to semantic and covariate shifts.

Semantic shifts involve changes in the meanings of features, while covariate shifts refer to alterations in the input data distribution. These shifts test a model's robustness and accuracy when confronted with unfamiliar inputs.

A crucial factor influencing performance on OOD data is the model's reliance on superficial features—such as color, texture, or background—which may not be consistent across different domains. This issue is compounded by cue conflicts, where models must choose between conflicting visual cues, often revealing their biases.

## Datasets

The following datasets were used in our experiments:

- **CIFAR-10**: Used for general classification tasks
- **PACS**: Used to demonstrate covariate shift (Photos, Art paintings, Cartoons, Sketches)
- **SVHN**: Used to explore both covariate and semantic shifts
- **Caltech-101**: Used to examine shape and color bias
- **90-Different Animals**: Used to study texture bias
- **Flickr Material Database**: Source of textures for style transfer
- **Oxford Pets**: Used for high-resolution feature and attention map generation

## Models

We evaluated three categories of models:

- **Discriminative Models**: Vision Transformer (ViT) and ResNet101
- **Contrastive Models**: CLIP (openai/clip-vit-large-patch14)
- **Generative Models**: Stable Diffusion (CompVis/stable-diffusion-v1-4)

## Text-to-Image Generative Model as Zero-Shot Classifier

We implemented the Stable Diffusion model as a zero-shot classifier for image classification on the CIFAR-10 dataset. The approach involved:

- Resizing images to 512x512 for compatibility with the SD model
- Sampling timesteps from a uniform distribution
- Applying forward diffusion and text-conditioned reverse diffusion
- Calculating Weighted Mean-Squared Error between original and predicted latent representations
- Selecting the class with the lowest cumulative MSE across all timesteps

### Results

We achieved an accuracy of 30% with 100 samples per label and 10% with 50 samples per label. The relatively low performance was attributed to:

- CIFAR-10's low resolution (32x32) compared to the SD model's requirement (512x512)
- Computational constraints limiting the number of samples
- Potential for improved accuracy with higher-quality input images and reduced class numbers

## Evaluation on IID Dataset

We evaluated the discriminative and contrastive models on the CIFAR-10 dataset with an independent and identical distribution between train and test splits.

### Results

| Model              | Accuracy |
| ------------------ | -------- |
| Vision Transformer | 96.71%   |
| ResNet             | 78.32%   |
| CLIP               | 94.40%   |

The training loss decreased from 0.4361 to 0.0983 for ViT and from 1.2047 to 0.7110 for ResNet over three epochs.

## Evaluation for Domain Generalization

We evaluated model performance on datasets with domain shifts:

- **PACS dataset**: Demonstrates covariate shift with different artistic styles
- **SVHN dataset**: Demonstrates both covariate and concept shifts

### Results

| Model              | PACS Accuracy | SVHN Accuracy |
| ------------------ | ------------- | ------------- |
| Vision Transformer | 50.32%        | 60.97%        |
| ResNet             | 46.31%        | 43.11%        |
| CLIP               | 97.66%        | 41.23%        |

CLIP demonstrated superior performance on PACS due to its training on diverse data sources, while all models struggled with SVHN due to its real-world characteristics differing significantly from pre-training data.

## Inductive Biases: Semantic Biases

We evaluated shape, color, and texture biases in the models using modified versions of the Caltech-101 and animal datasets.

### Results

| Model          | Shape Bias | Color Bias | Texture Bias |
| -------------- | ---------- | ---------- | ------------ |
| ViT            | 0.548      | 0.0134     | 0.2957       |
| ResNet101      | 0.1038     | 0.1982     | 0.8823       |
| CLIP-ViT-Large | 0.7501     | 0.0334     | 0.348        |

Key findings:

- CLIP showed the highest shape bias, followed by ViT
- ResNet exhibited the highest color and texture biases
- Transformer-based models (ViT and CLIP) showed lower sensitivity to color and texture changes

## Inductive Biases: Locality Biases

We evaluated model robustness to various image perturbations on CIFAR-10:

- Localized noise injection
- Global style changes
- Image scrambling with different patch sizes

### Results

| Model  | Original | Noisy | Style | S-16  | S-32  |
| ------ | -------- | ----- | ----- | ----- | ----- |
| ViT    | 97.3%    | 93.4% | 96.2% | 27%   | 53.5% |
| ResNet | 76.6%    | 10.1% | 19.9% | 12.7% | 23.7% |
| CLIP   | 95.4%    | 79.9% | 25.8% | 26.5% | 46%   |

Key findings:

- ViT showed strong robustness to noise and style changes due to its global attention mechanism
- ResNet was highly sensitive to all perturbations due to its reliance on local features
- CLIP demonstrated resilience to noise and scrambling but struggled with style changes

## Combining Convolution and Self-Attention

We explored the integration of convolutional operations with self-attention mechanisms:

- **Depthwise Convolution**: Focuses on local pixel neighborhoods
- **Self-Attention**: Captures global context by weighing contributions from all pixels
- **Attention Modulated Convolution**: Convolution process modulated by attention weights
- **Convolution Modulated Attention**: Depthwise convolution applied before attention

We visualized feature and attention maps using the Oxford Pets dataset, demonstrating how different operations emphasize various aspects of the input image.

## Discussion

Our comprehensive evaluation reveals distinct strengths and weaknesses across the three model architectures:

- ViT's global attention mechanism provides robustness to various image perturbations
- ResNet excels at capturing texture details but is highly sensitive to domain shifts and perturbations
- CLIP demonstrates strong zero-shot capabilities and shape recognition but struggles with certain domain shifts

These findings highlight the importance of understanding model biases and their implications for real-world applications where domain shifts are common.

## Conclusion

This study highlights the critical challenge of domain generalization in machine learning, particularly when deploying models in real-world scenarios characterized by Out-of-Domain data. By evaluating the performance of Vision Transformer, ResNet, and CLIP, we have identified distinct biases and generalization capabilities that influence their robustness to semantic and covariate shifts.

Addressing these challenges is essential for improving model robustness and ensuring effective performance in diverse real-world applications, ultimately guiding future research and development strategies in the field.

## References

- Clark, K. and Jaini, P. Text-to-image diffusion models are zero-shot classifiers. Google DeepMind, 2024.
- Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. M. Deeper, broader and artier domain generalization. arXiv preprint arXiv:1710.03077, 2017.
- Tian, J., Hsu, Y.-C., Shen, Y., Jin, H., and Kira, Z. Exploring covariate and concept shift for detection and calibration of out-of-distribution data. arXiv preprint arXiv:2110.15231, 2021.

## Authors

| Author             | Contributions                                                             |
| ------------------ | ------------------------------------------------------------------------- |
| Mustafa Abbas      | Model Selection, IID Evaluation, Semantic Biases                          |
| Muhammad Safiullah | Model Selection, Text-to-Image Classification, Locality Biases            |
| Ibrahim Farrukh    | Model Selection, Domain Generalization, Convolution-Attention Integration |
