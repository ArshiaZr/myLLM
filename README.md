# myLLM

Welcome to myLLM! This project is an educational initiative to build a Language Model (LLM) from scratch, exploring various transformer techniques along the way. The goal is to gain a deep understanding of how modern language models work and to experiment with different approaches in the transformer architecture.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Transformer Techniques](#transformer-techniques)
- [Contributing](#contributing)
- [License](#license)

## Introduction

myLLM is designed for educational purposes to understand and implement the core components of language models based on transformer architectures. The project covers:

- Tokenization
- Embeddings
- Encoder-Decoder models
- Attention mechanisms
- Training and fine-tuning techniques

## Features

- **Custom Tokenizer**: Build and train your own tokenizer.
- **Embedding Layer**: Learn how to create and use embedding layers.
- **Transformer Architecture**: Implement encoder and decoder components.
- **Attention Mechanisms**: Explore different attention mechanisms, including self-attention and cross-attention.
- **Training Pipeline**: Set up a training pipeline for your model.
- **Fine-Tuning**: Techniques to fine-tune the model on specific tasks.

## Installation

To get started with myLLM, you need to have Python 3.8 or higher installed. Follow the steps below to set up the project:

1. Clone the repository:

   ```sh
   git clone https://github.com/ArshiaZr/myLLM.git
   cd myLLM
   ```

2. Create a virtual environment and activate it:

   ```sh
   python -m venv cuda
   source cuda/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To train and test your LLM, follow these steps:

1. **Tokenization**: Tokenize your dataset using the custom tokenizer.

   ```python
   from tokenizer import CustomTokenizer
   tokenizer = CustomTokenizer()
   tokenizer.train('path_to_dataset')
   ```

2. **Model Training**: Train the transformer model.

   ```python
   from models.GPT import GPTLanguageModel
   model = GPTLanguageModel(vocab_size=vocab_size, device=device)
   model._train(epochs=200, learning_rate=3e-4, eval_iters=100)
   ```

3. **Inference**: Use the trained model for inference.
   ```python
   from utils.helpers import load_model
   model = load_model('path_to_trained_model')
   result = model.generate('your_encoded_input_text')
   print(result)
   ```

## Project Structure

The project structure is organized as follows:

```
myLLM/
├── data/
├── models/
│   ├── transformer.py
│   └── ...
├── tokenization/
│   ├── tokenizer.py
│   └── ...
├── utils/
│   ├── helpers.py
│   └── ...
├── README.md
└── requirements.txt
```

## Transformer Techniques

This project explores various techniques in transformer models, including:

- **Positional Encoding**: Adding positional information to the embeddings.
- **Multi-Head Attention**: Implementing and understanding multi-head attention mechanisms.
- **Layer Normalization**: Using layer normalization to stabilize training.
- **Feed-Forward Networks**: Incorporating feed-forward neural networks within the transformer blocks.
- **Residual Connections**: Implementing residual connections to improve gradient flow.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

```
Feel free to customize this README to better fit your project's specifics and your preferences!
```
