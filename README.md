<!DOCTYPE html>
<html>
<head>
    <title>Deep Models and Domain Generalization</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #24292e;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }
        h1 {
            font-size: 2em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #eaecef;
        }
        h2 {
            font-size: 1.5em;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #eaecef;
        }
        h3 {
            font-size: 1.25em;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 16px;
        }
        th, td {
            padding: 6px 13px;
            border: 1px solid #dfe2e5;
        }
        th {
            background-color: #f6f8fa;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f6f8fa;
        }
        code {
            padding: 0.2em 0.4em;
            margin: 0;
            font-size: 85%;
            background-color: rgba(27,31,35,0.05);
            border-radius: 3px;
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        }
        pre {
            padding: 16px;
            overflow: auto;
            font-size: 85%;
            line-height: 1.45;
            background-color: #f6f8fa;
            border-radius: 3px;
            margin-bottom: 16px;
        }
        pre code {
            display: inline;
            padding: 0;
            margin: 0;
            overflow: visible;
            line-height: inherit;
            word-wrap: normal;
            background-color: transparent;
            border: 0;
        }
        blockquote {
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0 0 16px 0;
        }
        img {
            max-width: 100%;
            box-sizing: content-box;
        }
        .badge {
            display: inline-block;
            padding: 3px 6px;
            font-size: 12px;
            font-weight: 600;
            line-height: 1;
            color: #fff;
            background-color: #6c757d;
            border-radius: 2px;
            margin-right: 5px;
        }
        .badge-primary {
            background-color: #007bff;
        }
        .badge-success {
            background-color: #28a745;
        }
        .badge-info {
            background-color: #17a2b8;
        }
        .results-table {
            width: 100%;
            margin-bottom: 1rem;
            color: #212529;
        }
        .results-table th, .results-table td {
            padding: 0.75rem;
            vertical-align: top;
            border-top: 1px solid #dee2e6;
        }
        .results-table thead th {
            vertical-align: bottom;
            border-bottom: 2px solid #dee2e6;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        .figure-caption {
            font-style: italic;
            color: #6c757d;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <h1>Deep Models and Domain Generalization</h1>

    <p>This repository contains research and experiments on domain generalization in machine learning, focusing on the performance of Vision Transformer (ViT), ResNet, and CLIP models when faced with Out-of-Domain (OOD) data.</p>

    <h2>Abstract</h2>
    <p>This study addresses the critical challenge of domain generalization in machine learning, focusing on the performance of three prominent deep learning models—Vision Transformer (ViT), ResNet, and CLIP—when faced with Out-of-Domain (OOD) data. Through a series of evaluations, we identify distinct biases and generalization capabilities that impact each model's robustness to semantic and covariate shifts.</p>
    <p>Our findings reveal that ResNet has the highest texture and colour bias, while Clip ViT has higher shape bias, and CLIP exhibits adaptability across various styles. By highlighting the influence of spurious features and cue conflicts, this research provides valuable insights into the inherent biases of deep learning models and their implications for generalization to unseen data distributions.</p>

    <h2>Introduction</h2>
    <p>Domain generalization is a significant challenge in machine learning, especially for real-world applications where models must perform on unseen data distributions. Unlike traditional models that are trained and tested on the same data distributions, domain generalization aims to develop models capable of handling Out-of-Domain (OOD) data, which can vary significantly from the training data due to semantic and covariate shifts.</p>
    <p>Semantic shifts involve changes in the meanings of features, while covariate shifts refer to alterations in the input data distribution. These shifts test a model's robustness and accuracy when confronted with unfamiliar inputs.</p>
    <p>A crucial factor influencing performance on OOD data is the model's reliance on superficial features—such as color, texture, or background—which may not be consistent across different domains. This issue is compounded by cue conflicts, where models must choose between conflicting visual cues, often revealing their biases.</p>

    <h2>Datasets</h2>
    <p>The following datasets were used in our experiments:</p>
    <ul>
        <li><strong>CIFAR-10</strong>: Used for general classification tasks</li>
        <li><strong>PACS</strong>: Used to demonstrate covariate shift (Photos, Art paintings, Cartoons, Sketches)</li>
        <li><strong>SVHN</strong>: Used to explore both covariate and semantic shifts</li>
        <li><strong>Caltech-101</strong>: Used to examine shape and color bias</li>
        <li><strong>90-Different Animals</strong>: Used to study texture bias</li>
        <li><strong>Flickr Material Database</strong>: Source of textures for style transfer</li>
        <li><strong>Oxford Pets</strong>: Used for high-resolution feature and attention map generation</li>
    </ul>

    <h2>Models</h2>
    <p>We evaluated three categories of models:</p>
    <ul>
        <li><strong>Discriminative Models</strong>: Vision Transformer (ViT) and ResNet101</li>
        <li><strong>Contrastive Models</strong>: CLIP (openai/clip-vit-large-patch14)</li>
        <li><strong>Generative Models</strong>: Stable Diffusion (CompVis/stable-diffusion-v1-4)</li>
    </ul>

    <h2>Text-to-Image Generative Model as Zero-Shot Classifier</h2>
    <p>We implemented the Stable Diffusion model as a zero-shot classifier for image classification on the CIFAR-10 dataset. The approach involved:</p>
    <ul>
        <li>Resizing images to 512x512 for compatibility with the SD model</li>
        <li>Sampling timesteps from a uniform distribution</li>
        <li>Applying forward diffusion and text-conditioned reverse diffusion</li>
        <li>Calculating Weighted Mean-Squared Error between original and predicted latent representations</li>
        <li>Selecting the class with the lowest cumulative MSE across all timesteps</li>
    </ul>

    <h3>Results</h3>
    <p>We achieved an accuracy of 30% with 100 samples per label and 10% with 50 samples per label. The relatively low performance was attributed to:</p>
    <ul>
        <li>CIFAR-10's low resolution (32x32) compared to the SD model's requirement (512x512)</li>
        <li>Computational constraints limiting the number of samples</li>
        <li>Potential for improved accuracy with higher-quality input images and reduced class numbers</li>
    </ul>

    <h2>Evaluation on IID Dataset</h2>
    <p>We evaluated the discriminative and contrastive models on the CIFAR-10 dataset with an independent and identical distribution between train and test splits.</p>

    <h3>Results</h3>
    <table class="results-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Vision Transformer</td>
                <td>96.71%</td>
            </tr>
            <tr>
                <td>ResNet</td>
                <td>78.32%</td>
            </tr>
            <tr>
                <td>CLIP</td>
                <td>94.40%</td>
            </tr>
        </tbody>
    </table>

    <p>The training loss decreased from 0.4361 to 0.0983 for ViT and from 1.2047 to 0.7110 for ResNet over three epochs.</p>

    <h2>Evaluation for Domain Generalization</h2>
    <p>We evaluated model performance on datasets with domain shifts:</p>
    <ul>
        <li><strong>PACS dataset</strong>: Demonstrates covariate shift with different artistic styles</li>
        <li><strong>SVHN dataset</strong>: Demonstrates both covariate and concept shifts</li>
    </ul>

    <h3>Results</h3>
    <table class="results-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>PACS Accuracy</th>
                <th>SVHN Accuracy</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Vision Transformer</td>
                <td>50.32%</td>
                <td>60.97%</td>
            </tr>
            <tr>
                <td>ResNet</td>
                <td>46.31%</td>
                <td>43.11%</td>
            </tr>
            <tr>
                <td>CLIP</td>
                <td>97.66%</td>
                <td>41.23%</td>
            </tr>
        </tbody>
    </table>

    <p>CLIP demonstrated superior performance on PACS due to its training on diverse data sources, while all models struggled with SVHN due to its real-world characteristics differing significantly from pre-training data.</p>

    <h2>Inductive Biases: Semantic Biases</h2>
    <p>We evaluated shape, color, and texture biases in the models using modified versions of the Caltech-101 and animal datasets.</p>

    <h3>Results</h3>
    <table class="results-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Shape Bias</th>
                <th>Color Bias</th>
                <th>Texture Bias</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>ViT</td>
                <td>0.548</td>
                <td>0.0134</td>
                <td>0.2957</td>
            </tr>
            <tr>
                <td>ResNet101</td>
                <td>0.1038</td>
                <td>0.1982</td>
                <td>0.8823</td>
            </tr>
            <tr>
                <td>CLIP-ViT-Large</td>
                <td>0.7501</td>
                <td>0.0334</td>
                <td>0.348</td>
            </tr>
        </tbody>
    </table>

    <p>Key findings:</p>
    <ul>
        <li>CLIP showed the highest shape bias, followed by ViT</li>
        <li>ResNet exhibited the highest color and texture biases</li>
        <li>Transformer-based models (ViT and CLIP) showed lower sensitivity to color and texture changes</li>
    </ul>

    <h2>Inductive Biases: Locality Biases</h2>
    <p>We evaluated model robustness to various image perturbations on CIFAR-10:</p>
    <ul>
        <li>Localized noise injection</li>
        <li>Global style changes</li>
        <li>Image scrambling with different patch sizes</li>
    </ul>

    <h3>Results</h3>
    <table class="results-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Original</th>
                <th>Noisy</th>
                <th>Style</th>
                <th>S-16</th>
                <th>S-32</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>ViT</td>
                <td>97.3%</td>
                <td>93.4%</td>
                <td>96.2%</td>
                <td>27%</td>
                <td>53.5%</td>
            </tr>
            <tr>
                <td>ResNet</td>
                <td>76.6%</td>
                <td>10.1%</td>
                <td>19.9%</td>
                <td>12.7%</td>
                <td>23.7%</td>
            </tr>
            <tr>
                <td>CLIP</td>
                <td>95.4%</td>
                <td>79.9%</td>
                <td>25.8%</td>
                <td>26.5%</td>
                <td>46%</td>
            </tr>
        </tbody>
    </table>

    <p>Key findings:</p>
    <ul>
        <li>ViT showed strong robustness to noise and style changes due to its global attention mechanism</li>
        <li>ResNet was highly sensitive to all perturbations due to its reliance on local features</li>
        <li>CLIP demonstrated resilience to noise and scrambling but struggled with style changes</li>
    </ul>

    <h2>Combining Convolution and Self-Attention</h2>
    <p>We explored the integration of convolutional operations with self-attention mechanisms:</p>
    <ul>
        <li><strong>Depthwise Convolution</strong>: Focuses on local pixel neighborhoods</li>
        <li><strong>Self-Attention</strong>: Captures global context by weighing contributions from all pixels</li>
        <li><strong>Attention Modulated Convolution</strong>: Convolution process modulated by attention weights</li>
        <li><strong>Convolution Modulated Attention</strong>: Depthwise convolution applied before attention</li>
    </ul>

    <p>We visualized feature and attention maps using the Oxford Pets dataset, demonstrating how different operations emphasize various aspects of the input image.</p>

    <h2>Discussion</h2>
    <p>Our comprehensive evaluation reveals distinct strengths and weaknesses across the three model architectures:</p>
    <ul>
        <li>ViT's global attention mechanism provides robustness to various image perturbations</li>
        <li>ResNet excels at capturing texture details but is highly sensitive to domain shifts and perturbations</li>
        <li>CLIP demonstrates strong zero-shot capabilities and shape recognition but struggles with certain domain shifts</li>
    </ul>

    <p>These findings highlight the importance of understanding model biases and their implications for real-world applications where domain shifts are common.</p>

    <h2>Conclusion</h2>
    <p>This study highlights the critical challenge of domain generalization in machine learning, particularly when deploying models in real-world scenarios characterized by Out-of-Domain data. By evaluating the performance of Vision Transformer, ResNet, and CLIP, we have identified distinct biases and generalization capabilities that influence their robustness to semantic and covariate shifts.</p>
    <p>Addressing these challenges is essential for improving model robustness and ensuring effective performance in diverse real-world applications, ultimately guiding future research and development strategies in the field.</p>

    <h2>References</h2>
    <ul>
        <li>Clark, K. and Jaini, P. Text-to-image diffusion models are zero-shot classifiers. Google DeepMind, 2024.</li>
        <li>Li, D., Yang, Y., Song, Y.-Z., and Hospedales, T. M. Deeper, broader and artier domain generalization. arXiv preprint arXiv:1710.03077, 2017.</li>
        <li>Tian, J., Hsu, Y.-C., Shen, Y., Jin, H., and Kira, Z. Exploring covariate and concept shift for detection and calibration of out-of-distribution data. arXiv preprint arXiv:2110.15231, 2021.</li>
    </ul>

    <h2>Authors</h2>
    <table class="results-table">
        <thead>
            <tr>
                <th>Author</th>
                <th>Contributions</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Mustafa Abbas</td>
                <td>Model Selection, IID Evaluation, Semantic Biases</td>
            </tr>
            <tr>
                <td>Muhammad Safiullah</td>
                <td>Model Selection, Text-to-Image Classification, Locality Biases</td>
            </tr>
            <tr>
                <td>Ibrahim Farrukh</td>
                <td>Model Selection, Domain Generalization, Convolution-Attention Integration</td>
            </tr>
        </tbody>
    </table>

</body>
</html>
