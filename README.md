# Memory Network Chatbot for Context-Aware Question Answering

## 🎯 Project Overview
An intelligent chatbot system built with Memory Networks that can understand story contexts and answer questions about entity locations and relationships. Achieves **80.7% accuracy** on context-aware question-answering tasks.

## 🚀 Key Features
- **Memory Network Architecture**: Explicit memory storage and attention mechanisms
- **Context Awareness**: Tracks entity movements across multiple sentences
- **Advanced Regularization**: L2, Dropout, BatchNorm for robust performance
- **Comprehensive Evaluation**: Multiple metrics including confidence analysis

## 📊 Performance Metrics
- **Accuracy**: 80.7%
- **F1-Score**: 0.81
- **Average Confidence**: 80.5%
- **Training Efficiency**: Converges in ~32 epochs with early stopping

## 🏗️ Architecture Highlights

### Core Components
1. **Dual Input Encoders**: Separate memory and context representations
2. **Attention Mechanism**: Dynamic focus on relevant story segments
3. **Sequential Processing**: Bidirectional LSTM + GRU stack
4. **Advanced Regularization**: Multi-layer overfitting prevention

### Model Architecture

Story Input → Memory Encoder (128d) ↘
→ Attention → Response Generation
Question Input → Question Encoder (128d) ↗ ↓
BiLSTM + GRU → Dense → Answer


## 🛠️ Installation & Usage

### Prerequisites

pip install tensorflow keras numpy pandas matplotlib scikit-learn seaborn

### Quick Start

Load and run the complete notebook
jupyter notebook chatbot_NLP.ipynb

Or run key components
from model import build_improved_memory_network
model = build_improved_memory_network(vocab_len, max_story_len, max_ques_len)


## 📈 Technical Specifications

### Model Configuration
- **Parameters**: 206,498 total (804KB)
- **Embedding Dimension**: 128
- **Sequence Lengths**: Stories (156), Questions (6)
- **Vocabulary Size**: 38 tokens
- **Architecture**: Memory Network + BiLSTM + GRU

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 64
- **Regularization**: L2 (0.001) + Dropout (0.3-0.5)
- **Callbacks**: EarlyStopping, ModelCheckpoint, LR Scheduling

## 🎓 Research Context

### Problem Statement
Traditional chatbots struggle with:
- Context retention across multiple sentences
- Spatial reasoning about entity locations
- Relationship understanding between entities

### Solution Approach
- Memory Networks for explicit fact storage
- Attention mechanisms for relevant information retrieval
- Sequential processing for temporal understanding

### Innovation Points
1. **Dual Encoding Strategy**: Separate memory and context representations
2. **Advanced Regularization**: Multiple overfitting prevention techniques
3. **Comprehensive Evaluation**: Multi-metric performance analysis

## 📊 Results & Analysis

### Performance Comparison
| Metric | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | 51.1% | 80.7% | +58% |
| F1-Score | 0.51 | 0.81 | +59% |
| Confidence | ~50% | 80.5% | +61% |

### Error Analysis
- **Common Errors**: Complex multi-entity scenarios
- **Strengths**: Simple spatial reasoning, entity tracking
- **Robustness**: High confidence on correct predictions

## 🔬 Future Enhancements
1. **Vocabulary Expansion**: Support larger, real-world vocabularies
2. **Multi-hop Reasoning**: Complex inference chains
3. **Transfer Learning**: Pre-trained embedding integration
4. **Real-world Deployment**: Production-ready optimizations
