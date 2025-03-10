import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from datasets import load_dataset, ClassLabel
import warnings
import re

# Silence the specific warning about overflowing tokens
logging.set_verbosity_error()
warnings.filterwarnings('ignore', message='.*overflowing tokens.*')

# ===============================================================================
# DATA HANDLING
# ===============================================================================
# ResumeJobDataset handles tokenization of resume and job description texts
# and converts string labels to integer labels for model training
# ===============================================================================
class ResumeJobDataset(Dataset):  # PyTorch: Dataset is an abstract class that should be inherited for custom datasets
    def __init__(self, samples, tokenizer, max_length=256):  
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Map string labels to integers
        self.label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
        
        # Get unique labels in the dataset
        label_set = set([sample['label'] for sample in samples])
        print(f"Unique labels in dataset: {label_set}")
        
        self.label_classes = ClassLabel(names=["No Fit", "Potential Fit", "Good Fit"])
        
    def __len__(self):  # PyTorch: Required Dataset method that returns the number of samples
        return len(self.samples)
    
    def __getitem__(self, idx):  # PyTorch: Required Dataset method that returns a sample at the given index
        item = self.samples[idx]
        resume = item['resume_text']
        job = item['job_description_text']
        label = item['label']
        
        # Limit text length to avoid memory issues
        max_chars = 8192
        resume = resume[:max_chars]
        job = job[:max_chars]
        
        # Tokenize using BERT tokenizer
        encoding = self.tokenizer(
            resume, 
            job, 
            padding="max_length", 
            truncation='longest_first',  # Changed to use longest_first strategy
            max_length=self.max_length, 
            return_tensors="pt"  # PyTorch: "pt" returns PyTorch tensors instead of numpy arrays
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add raw text for feature extraction
        encoding['resume_text'] = resume
        encoding['job_text'] = job
        
        # Convert label to integer
        encoding['label'] = torch.tensor(self.label_classes.str2int(label))
        
        return encoding

# ===============================================================================
# MODEL ARCHITECTURE
# ===============================================================================
# This hybrid model uses three key components:
# 1. BERT embeddings: Captures semantic understanding between resume and job text
# 2. TF-IDF + SVD: Extracts important keyword features reduced to manageable dimensions
# 3. Exact matching: Computes direct similarity metrics between resume and job
#
# These features are combined and passed through a classifier to predict fit level
# ===============================================================================
class HybridResumeJobMatcher(nn.Module):
    def __init__(self, bert_model="prajjwal1/bert-tiny", svd_components=50):
        super().__init__()  # PyTorch: Initialize the parent nn.Module class
        
        # ===============================================================================
        # 1. BERT COMPONENT
        # ===============================================================================
        # We use a small BERT model to efficiently encode semantic relationships
        # between resume and job description text
        # ===============================================================================
        self.bert = BertModel.from_pretrained(bert_model)
        self.bert_dim = self.bert.config.hidden_size  # 128 for bert-tiny
        
        # ===============================================================================
        # 2. TF-IDF + SVD COMPONENT
        # ===============================================================================
        # TF-IDF captures important keywords, SVD reduces dimensions while preserving
        # information about important terms in both documents
        # ===============================================================================
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2500)
        self.svd = TruncatedSVD(n_components=svd_components)
        self.svd_dim = svd_components * 2  # Resume + Job SVD features
        
        # ===============================================================================
        # 3. EXACT MATCHING COMPONENT
        # ===============================================================================
        # Direct similarity metrics that complement semantic understanding with
        # explicit matching of terms between resume and job
        # ===============================================================================
        self.exact_match_dim = 3  # Jaccard, length diff, overlap
        
        # ===============================================================================
        # COMBINED CLASSIFIER
        # ===============================================================================
        # Integrates all feature types for final classification decision
        # ===============================================================================
        total_features = self.bert_dim + self.svd_dim + self.exact_match_dim
        
        # PyTorch: nn.Sequential creates a container of layers that are applied in sequence
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 64),  # PyTorch: Linear layer applies W*x + b transformation
            nn.ReLU(),  # PyTorch: ReLU activation applies max(0,x) element-wise
            nn.Dropout(0.3),  # PyTorch: Dropout randomly zeros elements with probability p during training
            nn.Linear(64, 3)  # Three-class classification
        )
        
        self.tfidf_fitted = False
    
    def fit_tfidf_svd(self, texts):
        """Fit TF-IDF vectorizer and SVD on a corpus of texts"""
        # Fit TF-IDF vectorizer
        if not self.tfidf_fitted:
            self.tfidf_vectorizer.fit(texts)
            
            # Get sample vectors for SVD fitting
            sample_vectors = self.tfidf_vectorizer.transform(texts[:1000])  # Use subset for efficiency
            self.svd.fit(sample_vectors)
            
            self.tfidf_fitted = True
    
    def get_tfidf_svd_features(self, resume, job):
        """Extract TF-IDF features with SVD dimensionality reduction"""
        if not self.tfidf_fitted:
            raise ValueError("TF-IDF vectorizer and SVD must be fitted before feature extraction")
        
        # Transform texts to TF-IDF vectors
        resume_tfidf = self.tfidf_vectorizer.transform([resume])
        job_tfidf = self.tfidf_vectorizer.transform([job])
        
        # Apply SVD for dimensionality reduction
        resume_svd = self.svd.transform(resume_tfidf)
        job_svd = self.svd.transform(job_tfidf)
        
        # Concatenate resume and job SVD features
        combined_svd = np.concatenate([resume_svd, job_svd], axis=1)
        
        return torch.tensor(combined_svd, dtype=torch.float)
    
    def get_exact_match_features(self, resume, job):
        """Compute exact matching features between resume and job"""
        # Simple text normalization
        def normalize_text(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
            text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
            return text
        
        # Normalize texts
        resume_norm = normalize_text(resume)
        job_norm = normalize_text(job)
        
        # Tokenize
        resume_tokens = set(resume_norm.split())
        job_tokens = set(job_norm.split())
        
        # Calculate Jaccard similarity
        intersection = len(resume_tokens & job_tokens)
        union = len(resume_tokens | job_tokens)
        jaccard = intersection / max(union, 1)
        
        # Length difference ratio
        len_diff = abs(len(resume_norm) - len(job_norm)) / max(len(resume_norm), len(job_norm), 1)
        
        # Overlap ratio
        overlap = intersection / min(len(resume_tokens), len(job_tokens), 1)
        
        return torch.tensor([jaccard, len_diff, overlap], dtype=torch.float)
    
    def forward(self, input_ids, attention_mask, token_type_ids, resume_text, job_text):
        # PyTorch: forward method defines the computation performed at every call
        
        # ===============================================================================
        # STEP 1: BERT PROCESSING
        # ===============================================================================
        # Extract semantic relationships using BERT's [CLS] token embedding
        # ===============================================================================
        bert_output = self.bert(
            input_ids=input_ids.squeeze(1),  # PyTorch: squeeze removes dimensions of size 1
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids.squeeze(1)
        )
        cls_emb = bert_output.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        # ===============================================================================
        # STEP 2: TF-IDF + SVD FEATURE EXTRACTION
        # ===============================================================================
        # Extract keyword-based features with dimension reduction
        # ===============================================================================
        tfidf_svd_features_list = []
        for r, j in zip(resume_text, job_text):
            tfidf_svd = self.get_tfidf_svd_features(r, j)
            tfidf_svd_features_list.append(tfidf_svd)
        tfidf_svd_features = torch.stack(tfidf_svd_features_list)  # PyTorch: stack concatenates tensors along new dimension
        
        # ===============================================================================
        # STEP 3: EXACT MATCHING FEATURE EXTRACTION
        # ===============================================================================
        # Compute direct similarity metrics between resume and job
        # ===============================================================================
        exact_match_features_list = []
        for r, j in zip(resume_text, job_text):
            exact_match = self.get_exact_match_features(r, j)
            exact_match_features_list.append(exact_match)
        exact_match_features = torch.stack(exact_match_features_list)
        
        # Scale the features to give more weight to exact matches
        exact_match_features = exact_match_features * 2.0
        
        # ===============================================================================
        # STEP 4: FEATURE COMBINATION AND CLASSIFICATION
        # ===============================================================================
        # Combine all features and pass through the classifier for prediction
        # ===============================================================================
        # PyTorch: cat concatenates tensors along an existing dimension
        combined_features = torch.cat([cls_emb, tfidf_svd_features, exact_match_features], dim=1)
        
        # Pass through meta-classifier
        logits = self.classifier(combined_features)  # PyTorch: Model outputs raw logits, not probabilities
        return logits

# ===============================================================================
# TRAINING AND EVALUATION
# ===============================================================================
# These functions handle the model training loop, evaluation metrics, and data loading
# ===============================================================================
def train_model(model, train_loader, val_loader, device, epochs=40):
    """Train the model and track training losses and validation accuracies over time"""
    # Set optimizer with weight decay (L2 regularization) to prevent overfitting
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # Calculate class weights based on training data distribution to handle imbalance
    label_counts = [0, 0, 0]  # [No Fit, Potential Fit, Good Fit]
    for batch in train_loader:
        for label in batch['label']:
            label_counts[label.item()]  += 1
    
    # Compute inverse frequency weights
    total = sum(label_counts)
    class_weights = [total / (3 * count) if count > 0 else 1.0 for count in label_counts]
    print(f"Class weights: {class_weights}")
    
    # Convert to tensor and move to device
    weight_tensor = torch.FloatTensor(class_weights).to(device)  # PyTorch: .to(device) moves tensor to CPU/GPU
    # PyTorch: CrossEntropyLoss combines LogSoftmax and NLLLoss in one single class
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    best_accuracy = 0
    accumulation_steps = 4  # Accumulate gradients for 4 steps
    
    # Lists to track metrics for analysis
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()  # PyTorch: Sets model to training mode (enables dropout, batch norm updates)
        train_loss = 0
        batch_count = 0
        optimizer.zero_grad()  # PyTorch: Clears gradients of all optimized tensors
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Training progress: ", end="", flush=True)
        
        for i, batch in enumerate(train_loader):
            # Progress indicator
            if batch_count % 10 == 0:
                print(".", end="", flush=True)
            batch_count += 1
            
            # Move tensors to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Raw text stays on CPU (more efficient)
            resume_text = batch['resume_text']
            job_text = batch['job_text']
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                resume_text=resume_text,
                job_text=job_text
            )
            
            # Compute loss and scale it
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()  # PyTorch: Computes gradients of outputs with respect to inputs
            train_loss += loss.item() * accumulation_steps
            
            # Update weights every accumulation_steps batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # PyTorch: Updates parameters based on gradients
                optimizer.zero_grad()
        
        # Make sure to update with any remaining gradients
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate average loss for this epoch
        avg_train_loss = train_loss / batch_count
        train_losses.append(avg_train_loss)
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        print("Validating...", end="", flush=True)
        accuracy, class_accuracies = detailed_evaluate(model, val_loader, device)
        val_accuracies.append(accuracy)
        
        print(" done!")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model checkpoint
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best model saved! (accuracy: {accuracy:.4f})")
        
    # Print the learning curve
    print("\nLearning Curve:")
    print("Epoch | Train Loss | Val Accuracy")
    print("-" * 30)
    for i in range(len(train_losses)):
        print(f"{i+1:5d} | {train_losses[i]:.4f} | {val_accuracies[i]:.4f}")
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load("best_model.pt"))
    return model, train_losses, val_accuracies

def detailed_evaluate(model, data_loader, device):
    """Evaluate the model and provide detailed metrics per class"""
    model.eval()  # PyTorch: Sets model to evaluation mode (disables dropout, batch norm updates)
    correct = 0
    total = 0
    
    # Track per-class metrics
    class_correct = [0, 0, 0]  # No Fit, Potential Fit, Good Fit
    class_total = [0, 0, 0]
    
    with torch.no_grad():  # PyTorch: Context manager that disables gradient calculation for inference
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            # Raw text stays on CPU
            resume_text = batch['resume_text']
            job_text = batch['job_text']
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                resume_text=resume_text,
                job_text=job_text
            )
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Update metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update per-class metrics
            for i in range(len(predicted)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    # Calculate per-class accuracies
    class_accuracies = []
    print("Detailed Accuracy Report:")
    print("-" * 30)
    for i in range(3):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            class_accuracies.append(accuracy)
            label_name = ["No Fit", "Potential Fit", "Good Fit"][i]
            print(f"{label_name}: {accuracy:.4f} ({class_correct[i]}/{class_total[i]})")
        else:
            class_accuracies.append(0)
            label_name = ["No Fit", "Potential Fit", "Good Fit"][i]
            print(f"{label_name}: N/A (0/0)")
    print("-" * 30)
    
    # Return overall accuracy and per-class accuracies
    return correct / total, class_accuracies

# Main execution function
def main():
    # Load dataset
    dataset = load_dataset("cnamuangtoun/resume-job-description-fit")
    
    # First, separate data by label
    label_to_indices = {"No Fit": [], "Potential Fit": [], "Good Fit": []}
    for idx, item in enumerate(dataset["train"]):
        label_to_indices[item['label']].append(idx)
    
    # Sample equally from each class with doubled dataset size
    samples_per_class = 300  # Doubled from 150 to 300
    train_indices = []
    val_indices = []
    
    for label_indices in label_to_indices.values():
        if len(label_indices) >= samples_per_class + 50:
            # Shuffle the indices
            np.random.shuffle(label_indices)
            # Take first portion for training
            train_indices.extend(label_indices[:samples_per_class])
            # Take next portion for validation
            val_indices.extend(label_indices[samples_per_class:samples_per_class + 50])
    
    # Create balanced datasets
    train_data = dataset["train"].select(train_indices)
    val_data = dataset["train"].select(val_indices)
    
    # For test data, take a balanced sample
    test_indices = []
    for label_indices in label_to_indices.values():
        if len(label_indices) >= samples_per_class + 75:
            test_indices.extend(label_indices[samples_per_class + 50:samples_per_class + 75])
    test_data = dataset["train"].select(test_indices)
    
    print(f"\nDataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Initialize tokenizer and datasets
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = ResumeJobDataset(train_data, tokenizer)
    val_dataset = ResumeJobDataset(val_data, tokenizer)
    test_dataset = ResumeJobDataset(test_data, tokenizer)
    
    # DataLoaders with smaller batch size
    # PyTorch: DataLoader combines a dataset and a sampler, and provides iterable over the dataset
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)
    
    # Collect texts for TF-IDF
    all_texts = []
    for item in train_data:
        all_texts.append(item['resume_text'])
        all_texts.append(item['job_description_text'])
    
    # Initialize model with smaller SVD components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridResumeJobMatcher(svd_components=50)  
    model.fit_tfidf_svd(all_texts)
    model.to(device)
    
    # Train and evaluate with increased epochs
    model, train_losses, val_accuracies = train_model(model, train_loader, val_loader, device, epochs=40)
    
    # Test final model
    test_accuracy, test_class_accuracies = detailed_evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
