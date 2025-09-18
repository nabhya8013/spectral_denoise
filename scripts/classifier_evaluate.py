import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_classifier(X, y):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    return acc, f1

# Example usage for denoising performance:
# clf_raw = train_classifier(raw_spectra, labels)
# acc_raw, f1_raw = evaluate_classifier(clf_raw, raw_spectra_test, labels_test)
# clf_denoised = train_classifier(denoised_spectra, labels)
# acc_denoised, f1_denoised = evaluate_classifier(clf_denoised, denoised_spectra_test, labels_test)
