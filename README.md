# MicroAI Repository

Welcome to the MicroAI repository! 🤖 This project aims to provide minimal implementations ⚒️ of influential papers 📜 in the field of machine learning (ML) and artificial intelligence (AI). 

The implementations are designed to be concise yet illustrative, focusing on key concepts across various subareas, including computer vision, natural language processing, reinforcement learning, graph machine learning, etc.

**Note**: this repository is meant to be a constant work in progress 🏗️, new implementations will be added as time permits ⌛

![MicroAI Logo](./assets/cover.jpg)

## Implemented Papers

The following table summarizes currently implemented papers.

| Paper/Method                                 | Description                                                                                               | Notebook |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------- |
| Concepts of “**Automatic Differentiation Engine**”                                    | Recursive implementation of the backpropagation algorithm.                                                | [autograd.ipynb](./notebooks/autograd.ipynb) |
| Building blocks of a “**Deep Learning Library**”                     | Minimal implementation of basic deep learning building blocks inspired by PyTorch.                        | [nn.ipynb](./notebooks/nn.ipynb) |
| A. Vaswani et al., “**Attention Is All You Need**”                    | Minimal implementation of the transformer model, including _decoder-only_ and _encoder-decoder_ variants. | [attention_is_all_you_need.ipynb](./notebooks/papers/attention_is_all_you_need.ipynb) |
| Y. Bengio et al., “**A Neural Probabilistic Language Model**” | Simple implementation of a neural probabilistic language model, trained to generate song titles.                                           | [neural_probabilistic_language_model.ipynb](./notebooks/papers/neural_probabilistic_language_model.ipynb) |
| K. He et al., “**Deep Residual Learning for Image Recognition**” | Implementation of residual networks, with a classification example using the CIFAR-10 dataset.                                        | [deep_residual_learning_for_image_recognition.ipynb](./notebooks/papers/deep_residual_learning_for_image_recognition.ipynb) |
| I. J. Goodfellow et al., “**Generative Adversarial Networks**” and M. Mirza et al., “**Conditional Generative Adversarial Nets**” | Implementation of GAN and conditional GAN, with a MNIST digit generation example.                   | [generative_adversarial_networks.ipynb](./notebooks/papers/generative_adversarial_networks.ipynb) |

## Getting Started

To explore the implementations and run example notebooks, follow the steps:
1. Clone this repository to your local machine;
2. Install [Poetry](https://python-poetry.org/);
3. Install dependencies by running `poetry install`;
4. Explore the available notebooks.

## Contributing

If you'd like to contribute or have suggestions for additional papers to implement, please feel free to create an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
