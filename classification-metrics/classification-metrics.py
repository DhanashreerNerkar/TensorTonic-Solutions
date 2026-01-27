import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 1. Base Metric: Accuracy (Same for all modes)
    accuracy = np.mean(y_true == y_pred) if len(y_true) > 0 else 0.0

    # 2. Get Unique Classes & Map to Indices
    classes = np.unique(np.concatenate([y_true, y_pred]))
    class_map = {c: i for i, c in enumerate(classes)}
    
    # 3. Build Confusion Matrix (Rows=True, Cols=Pred)
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_map[t], class_map[p]] += 1

    # 4. Vectorized TP, FP, FN calculation for ALL classes at once
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp  # Column sums - diagonal
    fn = cm.sum(axis=1) - tp  # Row sums - diagonal
    
    # 5. Handle 'binary' mode specific logic
    if average == 'binary':
        if pos_label not in class_map: return {"accuracy": accuracy, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        idx = class_map[pos_label]
        tp, fp, fn = np.array([tp[idx]]), np.array([fp[idx]]), np.array([fn[idx]])

    # 6. Global Aggregation for 'micro'
    if average == 'micro':
        tp, fp, fn = np.array([tp.sum()]), np.array([fp.sum()]), np.array([fn.sum()])

    # 7. Calculate Metrics (Vectorized)
    # Use np.errstate to safely handle division by zero (0/0 = 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.nan_to_num(tp / (tp + fp))
        recall    = np.nan_to_num(tp / (tp + fn))
        f1        = np.nan_to_num(2 * precision * recall / (precision + recall))

    # 8. Final Averaging
    if average == 'weighted':
        support = cm.sum(axis=1) # Total true instances per class
        return {
            "accuracy": accuracy,
            "precision": np.average(precision, weights=support),
            "recall":    np.average(recall, weights=support),
            "f1":        np.average(f1, weights=support)
        }
    
    # Macro, Micro, Binary all just take the mean (for Micro/Binary, it's mean of 1 item)
    return {
        "accuracy": accuracy,
        "precision": np.mean(precision),
        "recall":    np.mean(recall),
        "f1":        np.mean(f1)
    }