import nbformat
import sys

try:
    with open('Train_Brain_Tumor_Models.ipynb', 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    roc_code = """# ==========================================
# Model Evaluation & ROC Curves
# ==========================================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np

# Generate predictions for the test set
# We'll use the CNN model for this evaluation
print("Generating predictions on the test set...")
ts_gen.reset()
preds = cnn_model.predict(ts_gen, verbose=1)

# Get true labels
y_true = ts_gen.classes
# Binarize the output for multi-class ROC
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
n_classes = y_true_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], preds[:len(y_true), i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
class_names = list(ts_gen.class_indices.keys())

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()"""

    nb.cells.append(nbformat.v4.new_code_cell(roc_code))

    with open('Train_Brain_Tumor_Models.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Successfully added ROC code to notebook.")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
