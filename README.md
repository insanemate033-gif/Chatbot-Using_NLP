# ContextMind: Memory Network Chatbot for Context-Aware Question Answering

## 🎯 Project Overview
An intelligent chatbot system built with Memory Networks that understands story contexts and answers questions about entity locations and relationships. Implements **dual architecture comparison** between Memory Networks and Transformers, achieving **79.2% accuracy** with Memory Networks significantly outperforming Transformers by **28.9%**.

## 🚀 Key Features
- **Memory Network Architecture**: Explicit memory storage with attention mechanisms
- **Context Awareness**: Tracks entity movements across multiple sentences  
- **Transformer Comparison**: Head-to-head evaluation of two state-of-the-art architectures
- **Advanced Regularization**: L2, Dropout, BatchNorm for robust performance
- **Comprehensive Evaluation**: Multiple metrics including confidence analysis

## 📊 Performance Metrics

### Memory Network (Winner)
- **Accuracy**: 79.2%
- **F1-Score**: 0.792
- **Average Confidence**: 82.9%
- **Parameters**: 206,498 (67% fewer than Transformer)
- **Training Efficiency**: Converges in ~21 epochs

### Transformer Comparison
- **Accuracy**: 50.3%
- **F1-Score**: 0.337
- **Average Confidence**: 52.9%
- **Parameters**: 628,326
- **Training Efficiency**: Early stopping at 16 epochs

## 🏗️ Architecture Highlights

### Memory Network Components
1. **Dual Input Encoders**: Separate memory and context representations
2. **Attention Mechanism**: Dot-product attention with softmax
3. **Sequential Processing**: Bidirectional LSTM + GRU stack
4. **Advanced Regularization**: Multi-layer dropout and L2 regularization

### Model Architecture Flow
Story Input → Memory Encoder (128d) ↘
→ Attention → Response Generation
Question Input → Question Encoder (128d) ↗ ↓
BiLSTM + GRU → Dense → Answer


### Transformer Architecture
- **Multi-Head Attention**: 8 heads with cross-attention
- **Positional Encoding**: Sinusoidal position embeddings
- **Feed-Forward Networks**: 256-dimensional hidden layers
- **Layer Normalization**: Residual connections throughout

## 🛠️ Installation & Usage

### Prerequisites
pip install tensorflow keras numpy pandas matplotlib scikit-learn seaborn


### Quick Start

Load and run the complete notebook
jupyter notebook Memory-Network.ipynb
Or run key components
from model import build_improved_memory_network
model = build_improved_memory_network(vocab_len, max_story_len, max_ques_len)


## 📈 Technical Specifications

### Dataset Configuration
- **Training Samples**: 10,000
- **Test Samples**: 1,000
- **Vocabulary Size**: 37 unique words
- **Max Story Length**: 156 tokens
- **Max Question Length**: 6 tokens

### Memory Network Configuration
- **Total Parameters**: 206,498 (806KB)
- **Trainable Parameters**: 205,974
- **Embedding Dimension**: 128
- **LSTM Units**: 64 (bidirectional)
- **GRU Units**: 32

### Training Configuration
- **Optimizer**: Adam (lr=0.001 with exponential decay)
- **Batch Size**: 64
- **L2 Regularization**: 0.001
- **Dropout Rates**: 0.3-0.5
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## 🎓 Research Context

### Problem Statement
Traditional chatbots struggle with:
- Context retention across multiple sentences
- Spatial reasoning about entity locations
- Relationship understanding between entities
- Memory-efficient architecture selection

### Solution Approach
- **Memory Networks** for explicit fact storage and retrieval
- **Attention mechanisms** for relevant information focus
- **Comparative analysis** between Memory Networks vs Transformers
- **Statistical evaluation** with comprehensive metrics

### Innovation Points
1. **Architecture Comparison**: Empirical evaluation of Memory Networks vs Transformers
2. **Parameter Efficiency**: 67% reduction while maintaining superior performance
3. **Advanced Training**: Learning rate scheduling and early stopping
4. **Comprehensive Metrics**: Precision, recall, F1, confidence analysis

## 📊 Results & Analysis

### Performance Comparison
| Metric | Memory Network | Transformer | Improvement |
|--------|----------------|-------------|-------------|
| Accuracy | **79.2%** | 50.3% | **+57%** |
| F1-Score | **0.792** | 0.337 | **+135%** |
| Confidence | **82.9%** | 52.9% | **+57%** |
| Parameters | **206K** | 628K | **-67%** |

### Classification Performance
**Memory Network Results:**
- **Yes Precision**: 0.77, **Recall**: 0.82
- **No Precision**: 0.81, **Recall**: 0.76
- **Balanced Performance**: Strong on both classes

**Transformer Results:**
- **Yes Precision**: 0.00 (complete failure)
- **No Precision**: 0.50 (random performance)
- **Severe Overfitting**: Poor generalization

### Key Insights
1. **Memory Networks excel** at explicit reasoning tasks with small datasets
2. **Parameter efficiency matters** for reasoning-intensive applications
3. **Attention mechanisms** in Memory Networks more effective than self-attention
4. **Training stability** superior with Memory Network architecture

## 🔬 Future Enhancements
1. **Multi-hop Reasoning**: Complex inference chains across multiple facts
2. **Vocabulary Expansion**: Support for larger, real-world vocabularies
3. **Hybrid Architectures**: Combine Memory Networks with modern transformers
4. **Real-world Deployment**: Production optimizations and API integration
5. **Transfer Learning**: Pre-trained embedding integration

## 📝 CV Bullets
- Developed Memory Network achieving **79.2% accuracy** on context-aware QA, outperforming Transformer by **28.9%** with **67% fewer parameters**
- Engineered end-to-end NLP pipeline processing **11,000 samples** with attention mechanisms and regularization, achieving **0.792 F1-score**
- Architected and compared **two state-of-the-art models** with comprehensive statistical analysis, demonstrating Memory Network superiority for reasoning tasks

---
*This project demonstrates the effectiveness of Memory Networks for context-aware reasoning tasks and provides empirical evidence for architecture selection in resource-constrained environments.*
