# 🦙 Llama 3.2 Fine-tuning with Unsloth

A comprehensive guide and implementation for fine-tuning Meta's Llama 3.2 models using Unsloth's efficient LoRA (Low-Rank Adaptation) technique. This project demonstrates how to fine-tune large language models efficiently with minimal computational resources, perfect for Google Colab and local development.

## 🌟 Features

### Core Functionality
- **Efficient Fine-tuning**: LoRA adaptation for memory-efficient training
- **Multiple Model Sizes**: Support for 1B and 3B parameter models
- **High-Quality Dataset**: FineTome-100k ShareGPT-style multi-turn conversations
- **Parameter-Efficient**: LoRA adapters instead of full fine-tuning
- **GPU Optimized**: CUDA-optimized training for faster processing
- **Colab Ready**: Optimized for Google Colab free tier

### Advanced Capabilities
- **Memory Optimization**: Efficient memory usage for large models
- **Fast Training**: Unsloth's optimized training pipeline
- **Flexible Architecture**: Easy model and parameter customization
- **Production Ready**: Trained models can be deployed and used
- **Cost Effective**: Minimal computational resources required
- **Scalable**: Works across different model sizes and datasets

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Google Colab account (for free GPU access)
- Sufficient storage for model weights

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rchhabra13/portfolio-projects.git
   cd portfolio-projects/llama3.2_finetuning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the fine-tuning**
   ```bash
   python finetune_llama3.2.py
   ```

4. **Access the results**
   - Fine-tuned model saved to `finetuned_model/`
   - Training logs and metrics available
   - Model ready for inference

## 💡 Usage Examples

### Basic Fine-tuning
```python
# Default configuration
model_name = "unsloth/Llama-3.2-3B-Instruct"
max_seq_length = 2048
r = 16  # LoRA rank
```

### Custom Model Configuration
```python
# For 1B model
model_name = "unsloth/Llama-3.2-1B-Instruct"
max_seq_length = 2048
r = 8  # Lower rank for smaller model
```

### Training Parameters
```python
# Custom training settings
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=60,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="finetuned_model",
)
```

## 🛠️ Technical Architecture

### Core Technologies
- **Framework**: Unsloth for efficient fine-tuning
- **Model**: Meta Llama 3.2 Instruct models
- **Adaptation**: LoRA (Low-Rank Adaptation) for parameter efficiency
- **Training**: TRL's SFTTrainer for supervised fine-tuning
- **Dataset**: FineTome-100k for high-quality training data
- **Platform**: Google Colab for free GPU access

### Fine-tuning Pipeline
1. **Model Loading**: Llama 3.2 model loaded with Unsloth's FastLanguageModel
2. **LoRA Setup**: LoRA adapters attached to specific model layers
3. **Data Preparation**: FineTome-100k dataset processed with chat template
4. **Training Configuration**: SFTTrainer configured with optimal parameters
5. **Fine-tuning**: Model trained on prepared dataset
6. **Model Saving**: Fine-tuned weights saved for inference

## 📊 Supported Models

### Llama 3.2 1B
- **Use Case**: Quick experimentation and prototyping
- **Memory**: 4-6GB VRAM required
- **Training Time**: 15-30 minutes
- **Best For**: Learning and testing

### Llama 3.2 3B
- **Use Case**: Production applications and research
- **Memory**: 8-12GB VRAM required
- **Training Time**: 30-60 minutes
- **Best For**: Serious applications

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0
```

### Model Parameters
- **model_name**: Choose from available Llama 3.2 models
- **max_seq_length**: Maximum sequence length (2048, 4096)
- **r**: LoRA rank (8, 16, 32)
- **target_modules**: Layers to apply LoRA to

### Training Parameters
- **per_device_train_batch_size**: Batch size per device
- **gradient_accumulation_steps**: Gradient accumulation steps
- **learning_rate**: Learning rate for training
- **max_steps**: Maximum training steps
- **warmup_steps**: Warmup steps for learning rate

## 📈 Performance Features

- **Memory Efficiency**: LoRA reduces trainable parameters by 99%
- **Fast Training**: Unsloth's optimized training pipeline
- **Scalable**: Works across different model sizes
- **Cost Effective**: Minimal computational resources required
- **Production Ready**: Trained models ready for deployment

## 🔒 Security & Privacy

- **Data Privacy**: Training data processed locally
- **Model Security**: Fine-tuned models stored securely
- **API Security**: No external API calls during training
- **Privacy Compliance**: Follows data privacy best practices

## 🤝 Contributing

We welcome contributions from developers and AI researchers:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Contribution
- Additional model configurations
- Improved training algorithms
- Better memory optimization
- Enhanced documentation
- Performance optimizations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation
- Review the FAQ section

## 🔮 Roadmap

- [ ] Support for more model architectures
- [ ] Advanced quantization techniques
- [ ] Multi-GPU training support
- [ ] Automated hyperparameter tuning
- [ ] Model compression techniques
- [ ] Advanced evaluation metrics
- [ ] Cloud training support
- [ ] Model deployment tools

## 📊 Use Cases

### Research and Development
- Model fine-tuning experiments
- Performance benchmarking
- Architecture research
- Dataset evaluation

### Production Applications
- Custom model training
- Domain-specific fine-tuning
- Performance optimization
- Model deployment

### Education
- Learning fine-tuning techniques
- Understanding LoRA adaptation
- Hands-on AI training
- Model architecture study

## 🙏 Acknowledgments

- Meta for the Llama 3.2 models
- Unsloth for the efficient fine-tuning framework
- TRL for the training infrastructure
- Google Colab for free GPU access
- The open-source community
- FineTome for the high-quality dataset

---

**Note**: This project is designed for educational and research purposes. Always ensure you have the right to use the training data and respect model licensing terms.
<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->

<!-- Updated: 2025-09-16 -->
