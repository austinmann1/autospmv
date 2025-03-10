# Resume-Job Matcher

A hybrid machine learning model designed to match resumes to job descriptions with 74.67% accuracy using a combination of transformer-based embeddings, TF-IDF features, and exact matching metrics.

## Model Architecture

The model uses a three-pronged approach to evaluate the fit between a resume and a job description:

### 1. BERT Semantic Understanding
- Utilizes a lightweight BERT model (`prajjwal1/bert-tiny`) for efficient semantic understanding
- Captures contextual relationships between the resume and job text
- Extracts the [CLS] token embedding which represents the overall relationship between inputs

### 2. TF-IDF + SVD Feature Extraction
- Applies TF-IDF vectorization to identify important keywords in both documents
- Uses Truncated SVD for dimensionality reduction (50 components)
- Maintains information about important terms while reducing computational complexity

### 3. Exact Matching Metrics
- Computes direct similarity metrics between resume and job text:
  - Jaccard similarity: measures overlap between word sets
  - Length difference ratio: captures size difference between documents
  - Overlap ratio: quantifies the relative shared term count

### Classifier Components
- Features from all three approaches are concatenated into a unified representation
- Passes through a simple but effective neural network classifier
- Final layer outputs probabilities for three classes: "No Fit," "Potential Fit," and "Good Fit"

## Performance

After training on a doubled dataset size (300 samples per class, 900 total) for 40 epochs with weight decay regularization:

- **Final Test Accuracy**: 74.67%
- **Per-Class Accuracy**:
  - No Fit: 60.00%
  - Potential Fit: 68.00%
  - Good Fit: 96.00%

## Training Approach

- **Optimizer**: AdamW with weight decay (0.01) for regularization
- **Loss Function**: CrossEntropyLoss
- **Dataset Size**: 900 training samples (balanced across classes)
- **Learning Curve Tracking**: Loss and accuracy monitored over time
- **Model Selection**: Best model saved based on validation accuracy

## Requirements

- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- NumPy
- Datasets (Hugging Face)

## Usage

The model can be used to predict the fit between a resume and a job description, helping job seekers prioritize applications and recruiters identify promising candidates.

```python
# Example usage (after training):
model = HybridResumeJobMatcher()
model.load_state_dict(torch.load("best_model.pt"))
# Process resume and job description
# Get prediction: "No Fit", "Potential Fit", or "Good Fit"
```

## Improvements Over Previous Version

- Dataset size doubled from 450 to 900 samples
- Added L2 regularization (weight decay) to prevent overfitting
- Implemented detailed per-class accuracy reporting
- Learning curve tracking for performance analysis
- Achieved 74.67% test accuracy (significant improvement)
