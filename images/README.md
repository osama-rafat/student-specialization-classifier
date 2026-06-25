
## Visualizations and Results

### Class Distribution

The dataset is well-balanced across all specializations, which helps the model learn each class fairly without significant bias.

![Class Distribution](images/class_distribution.png)

---

### Feature Correlation Heatmap

The heatmap illustrates relationships between features and highlights positive and negative correlations among students' skills and interests.

![Feature Correlation Heatmap](images/heatmap.png)

---

### Training Performance

The Neural Network converges successfully during training. The loss decreases steadily while validation accuracy remains consistently high.

![Training Performance](images/training_validation.png)

---

### Confusion Matrix

The confusion matrix shows excellent classification performance with very few misclassifications between specializations.

![Confusion Matrix](images/confusion_matrix.png)

---

### Per-Class Accuracy

The model achieves high accuracy across all classes, reaching 100% for Cyber Security and Frontend while maintaining strong performance in the remaining specializations.

![Per-Class Accuracy](images/per_class_accuracy.png)

---

## Model Performance

### Overall Accuracy

**97.8%**

### Class-wise Accuracy

| Specialization | Accuracy |
|---------------|-----------|
| AI | 94.5% |
| Backend | 95.6% |
| Cyber Security | 100.0% |
| Frontend | 100.0% |
| Software Testing | 97.8% |

The model demonstrates strong generalization performance and effectively distinguishes between the five Computer Science specializations.
