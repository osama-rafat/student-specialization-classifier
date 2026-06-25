"""
============================================================
Student Specialization Classifier
Multi-Class Classification – Neural Network (MLPClassifier)
Pipeline: Data Engineering → EDA → Feature Engineering →
            Feature Extraction → Feature Selection → Scaling →
            Encoding → MLP Neural Network → Evaluation → Prediction
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score)

os.makedirs("outputs", exist_ok=True)

print("=" * 65)
print("   STUDENT SPECIALIZATION CLASSIFIER – Full ML Pipeline")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
#  STAGE 1 ─ GENERATE SYNTHETIC DATASET WITH INTENTIONAL ERRORS
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 1]  Generating Synthetic Dataset (3 000 students)…")

np.random.seed(42)
N   = 3000
SPECS = ["AI", "Cyber Security", "Frontend", "Backend", "Software Testing"]

PROFILES = {
    "AI": {
        "math_score":          (88, 8),  "statistics_interest": (87, 8),
        "algorithms_interest": (85, 9),  "coding_interest":     (80, 10),
        "problem_solving":     (85, 8),  "network_interest":    (50, 15),
        "security_interest":   (45, 15), "design_interest":     (40, 15),
        "ui_ux_interest":      (35, 15), "database_interest":   (70, 12),
        "web_interest":        (55, 15), "linux_interest":      (65, 15),
        "debugging_skill":     (72, 10), "automation_interest": (75, 10),
        "creativity":          (65, 12), "logical_thinking":    (88,  8),
        "attention_to_detail": (80, 10), "teamwork":            (70, 12),
        "communication":       (65, 12),
    },
    "Cyber Security": {
        "math_score":          (75, 10), "statistics_interest": (65, 12),
        "algorithms_interest": (72, 10), "coding_interest":     (75, 10),
        "problem_solving":     (82,  8), "network_interest":    (92,  6),
        "security_interest":   (93,  6), "design_interest":     (35, 15),
        "ui_ux_interest":      (30, 14), "database_interest":   (65, 12),
        "web_interest":        (60, 12), "linux_interest":      (90,  8),
        "debugging_skill":     (80,  9), "automation_interest": (70, 12),
        "creativity":          (55, 12), "logical_thinking":    (85,  8),
        "attention_to_detail": (85,  8), "teamwork":            (68, 12),
        "communication":       (62, 13),
    },
    "Frontend": {
        "math_score":          (60, 12), "statistics_interest": (45, 14),
        "algorithms_interest": (55, 12), "coding_interest":     (78, 10),
        "problem_solving":     (68, 11), "network_interest":    (50, 15),
        "security_interest":   (40, 14), "design_interest":     (92,  6),
        "ui_ux_interest":      (91,  6), "database_interest":   (50, 13),
        "web_interest":        (92,  6), "linux_interest":      (45, 14),
        "debugging_skill":     (65, 11), "automation_interest": (55, 13),
        "creativity":          (90,  7), "logical_thinking":    (65, 11),
        "attention_to_detail": (78, 10), "teamwork":            (80, 10),
        "communication":       (82,  9),
    },
    "Backend": {
        "math_score":          (78, 10), "statistics_interest": (65, 12),
        "algorithms_interest": (80,  9), "coding_interest":     (88,  7),
        "problem_solving":     (82,  8), "network_interest":    (68, 12),
        "security_interest":   (58, 13), "design_interest":     (40, 14),
        "ui_ux_interest":      (35, 13), "database_interest":   (88,  7),
        "web_interest":        (72, 11), "linux_interest":      (72, 11),
        "debugging_skill":     (82,  8), "automation_interest": (75, 10),
        "creativity":          (55, 12), "logical_thinking":    (85,  8),
        "attention_to_detail": (82,  8), "teamwork":            (70, 11),
        "communication":       (65, 12),
    },
    "Software Testing": {
        "math_score":          (65, 12), "statistics_interest": (70, 11),
        "algorithms_interest": (65, 12), "coding_interest":     (70, 11),
        "problem_solving":     (80,  9), "network_interest":    (55, 13),
        "security_interest":   (60, 13), "design_interest":     (50, 13),
        "ui_ux_interest":      (55, 13), "database_interest":   (65, 12),
        "web_interest":        (60, 13), "linux_interest":      (58, 13),
        "debugging_skill":     (92,  6), "automation_interest": (88,  7),
        "creativity":          (60, 12), "logical_thinking":    (82,  9),
        "attention_to_detail": (92,  6), "teamwork":            (78, 10),
        "communication":       (78, 10),
    },
}

ENG_PROBS = {
    "AI":               [0.10, 0.20, 0.40, 0.30],
    "Cyber Security":   [0.15, 0.25, 0.40, 0.20],
    "Frontend":         [0.05, 0.15, 0.45, 0.35],
    "Backend":          [0.10, 0.20, 0.45, 0.25],
    "Software Testing": [0.10, 0.20, 0.45, 0.25],
}
ENG_LEVELS = ["Beginner", "Intermediate", "Advanced", "Expert"]

def make_student(spec):
    row = {}
    for feat, (mu, sd) in PROFILES[spec].items():
        row[feat] = round(float(np.clip(np.random.normal(mu, sd), 0, 100)), 1)
    row["english_level"]   = np.random.choice(ENG_LEVELS, p=ENG_PROBS[spec])
    row["gender"]          = np.random.choice(["Male", "Female"], p=[0.60, 0.40])
    row["specialization"]  = spec
    return row

per_cls  = N // len(SPECS)
students = [make_student(s) for s in SPECS for _ in range(per_cls)]
df_clean_ref = pd.DataFrame(students)          # keep clean reference

# ── INJECT INTENTIONAL ERRORS ─────────────────────────────────
print("  → Injecting: missing values, duplicates, outliers, typos…")
df = df_clean_ref.copy()
idx = df.index.tolist()

# Missing numeric
for col in ["math_score","network_interest","statistics_interest",
            "coding_interest","debugging_skill","database_interest"]:
    df.loc[np.random.choice(idx, int(N*0.04), replace=False), col] = np.nan

# Missing categorical
for col in ["english_level","gender"]:
    df.loc[np.random.choice(idx, int(N*0.03), replace=False), col] = np.nan

# Duplicates
dup_idx = np.random.choice(idx, int(N*0.02), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# Outliers
oi = np.random.choice(idx, 30, replace=False)
df.loc[oi[:15], "math_score"]      = np.random.choice([200, 300, -20, -50], 15)
df.loc[oi[15:], "coding_interest"] = np.random.choice([250, 180, -30],      15)

# Typos / inconsistent categories
ti = np.random.choice(idx, 40, replace=False)
df.loc[ti, "english_level"] = np.random.choice(
    ["intermediate","ADVANCED","Advancedd","beginner","Expert ","advanced"], 40)
tg = np.random.choice(idx, 50, replace=False)
df.loc[tg, "gender"] = np.random.choice(["male","MALE","female","FEMALE","M","F"], 50)

df.to_csv("outputs/students_raw_dirty.csv", index=False)
print(f"  ✓ Dirty CSV saved  →  outputs/students_raw_dirty.csv")
print(f"  Shape {df.shape}  |  Missing cells: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════
#  STAGE 2 ─ DATA ENGINEERING (Cleaning)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 2]  Data Engineering & Cleaning…")

# 2-a  Duplicates
n_before = len(df)
df.drop_duplicates(inplace=True)
print(f"  ✓ Removed {n_before - len(df)} duplicate rows")

# 2-b  Standardise strings
df["english_level"] = (df["english_level"].astype(str)
    .str.strip().str.title()
    .replace({"Advancedd":"Advanced","Expert ":"Expert","Nan":np.nan}))
df["gender"] = (df["gender"].astype(str)
    .str.strip().str.capitalize()
    .replace({"M":"Male","F":"Female","Nan":np.nan}))

# 2-c  Outlier capping via IQR
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
n_outliers = 0
for col in num_cols:
    Q1,Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    lo,hi = Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1)
    mask  = (df[col]<lo)|(df[col]>hi)
    n_outliers += int(mask.sum())
    df[col] = df[col].clip(lo, hi)
print(f"  ✓ Capped {n_outliers} outlier values (IQR)")

# 2-d  Fill missing numeric → median
for col in num_cols:
    if df[col].isnull().sum():
        df[col].fillna(df[col].median(), inplace=True)

# 2-e  Fill missing categorical → mode
for col in ["english_level","gender"]:
    if df[col].isnull().sum():
        df[col].fillna(df[col].mode()[0], inplace=True)

# 2-f  Clip scores 0–100
df[num_cols] = df[num_cols].clip(0, 100)

print(f"  ✓ Missing after cleaning : {df.isnull().sum().sum()}")
print(f"  ✓ Final shape            : {df.shape}")

# ══════════════════════════════════════════════════════════════
#  STAGE 3 ─ EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 3]  Exploratory Data Analysis…")

PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

# Plot 1 – class distribution
fig, ax = plt.subplots(figsize=(8,5))
df["specialization"].value_counts().plot(kind="bar",color=PALETTE,ax=ax,edgecolor="white",width=0.6)
ax.set_title("Class Distribution",fontsize=14,fontweight="bold")
ax.set_xlabel("Specialization"); ax.set_ylabel("Count")
ax.tick_params(axis='x',rotation=25)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2, p.get_height()+5),
                ha='center', fontsize=9)
plt.tight_layout(); plt.savefig("outputs/01_class_distribution.png",dpi=130); plt.close()

# Plot 2 – correlation heatmap
fig, ax = plt.subplots(figsize=(17,14))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.3, ax=ax, annot_kws={"size":6.5}, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap",fontsize=14,fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/02_correlation_heatmap.png",dpi=120); plt.close()

# Plot 3 – histograms
fig, axes = plt.subplots(4,5,figsize=(22,14))
for i,col in enumerate(num_cols):
    axes.flat[i].hist(df[col],bins=30,color=PALETTE[i%5],edgecolor="white",alpha=0.85)
    axes.flat[i].set_title(col.replace("_"," ").title(),fontsize=8,fontweight="bold")
    axes.flat[i].tick_params(labelsize=7)
plt.suptitle("Feature Distributions",fontsize=14,fontweight="bold",y=1.01)
plt.tight_layout(); plt.savefig("outputs/03_feature_distributions.png",dpi=100,bbox_inches="tight"); plt.close()

# Plot 4 – box plots by specialization for key features
key = ["math_score","coding_interest","design_interest","network_interest",
       "debugging_skill","statistics_interest"]
fig, axes = plt.subplots(2,3,figsize=(16,9))
for i,col in enumerate(key):
    df.boxplot(column=col, by="specialization", ax=axes.flat[i],
               patch_artist=True)
    axes.flat[i].set_title(col.replace("_"," ").title(),fontsize=10,fontweight="bold")
    axes.flat[i].set_xlabel("")
    axes.flat[i].tick_params(axis='x',rotation=30,labelsize=8)
plt.suptitle("Key Features by Specialization",fontsize=13,fontweight="bold")
plt.tight_layout(); plt.savefig("outputs/04_boxplots_by_spec.png",dpi=120); plt.close()

print("  ✓ 4 EDA plots saved")

# ══════════════════════════════════════════════════════════════
#  STAGE 4 ─ FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 4]  Feature Engineering…")

df["technical_score"]  = (df["coding_interest"] + df["problem_solving"]
                           + df["algorithms_interest"]) / 3
df["security_profile"] = (df["network_interest"] + df["security_interest"]
                           + df["linux_interest"]) / 3
df["design_profile"]   = (df["design_interest"] + df["ui_ux_interest"]
                           + df["creativity"]) / 3
df["data_profile"]     = (df["math_score"] + df["statistics_interest"]
                           + df["database_interest"]) / 3
df["quality_profile"]  = (df["debugging_skill"] + df["attention_to_detail"]
                           + df["automation_interest"]) / 3
df["soft_skill_score"] = (df["teamwork"] + df["communication"]) / 2

print("  ✓ technical_score | security_profile | design_profile")
print("  ✓ data_profile    | quality_profile  | soft_skill_score")

# ══════════════════════════════════════════════════════════════
#  STAGE 5 ─ FEATURE EXTRACTION (weighted composites)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 5]  Feature Extraction (weighted composite strength scores)…")

df["developer_strength"]  = (0.40*df["coding_interest"]
+ 0.30*df["algorithms_interest"]
+ 0.30*df["problem_solving"])

df["ai_strength"]         = (0.35*df["math_score"]
+ 0.35*df["statistics_interest"]
+ 0.30*df["algorithms_interest"])

df["cyber_strength"]      = (0.35*df["network_interest"]
+ 0.35*df["security_interest"]
+ 0.30*df["linux_interest"])

df["frontend_strength"]   = (0.30*df["design_interest"]
+ 0.25*df["ui_ux_interest"]
+ 0.25*df["web_interest"]
+ 0.20*df["creativity"])

df["testing_strength"]    = (0.40*df["debugging_skill"]
+ 0.30*df["attention_to_detail"]
+ 0.30*df["automation_interest"])

print("  ✓ developer_strength | ai_strength | cyber_strength")
print("  ✓ frontend_strength  | testing_strength")

# ══════════════════════════════════════════════════════════════
#  STAGE 6 ─ ENCODING
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 6]  Encoding Categorical Features (OneHotEncoder)…")

ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_arr  = ohe.fit_transform(df[["gender","english_level"]])
cat_cols = list(ohe.get_feature_names_out(["gender","english_level"]))
df_cat   = pd.DataFrame(cat_arr, columns=cat_cols, index=df.index)
df_cat   = df_cat.fillna(0)
# drop dummy nan columns created from NaN values
nan_cols = [c for c in df_cat.columns if '_nan' in c]
df_cat.drop(columns=nan_cols, inplace=True, errors='ignore')
cat_cols = [c for c in df_cat.columns]
df       = pd.concat([df.drop(columns=["gender","english_level"]), df_cat], axis=1)

print(f"  ✓ gender       → {[c for c in cat_cols if 'gender' in c]}")
print(f"  ✓ english_level→ {[c for c in cat_cols if 'english' in c]}")

le = LabelEncoder()
df["target"] = le.fit_transform(df["specialization"])
print(f"  ✓ Target labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ══════════════════════════════════════════════════════════════
#  STAGE 7 ─ FEATURE SELECTION
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 7]  Feature Selection (Mutual Information + Random Forest)…")

feat_cols = [c for c in df.columns if c not in ["specialization","target"]]
# Final safety fillna – ensure zero NaN before sklearn
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
X_all = df[feat_cols].astype(float).values
assert not np.isnan(X_all).any(), "NaN still present!"
print(f"  NaN check passed – X shape {X_all.shape}")
y_all = df["target"].values

# Mutual Information
mi_scores = pd.Series(
    SelectKBest(mutual_info_classif, k="all").fit(X_all,y_all).scores_,
    index=feat_cols).sort_values(ascending=False)

# Random Forest importance
rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X_all, y_all)
rf_imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)

top_mi  = set(mi_scores.head(18).index)
top_rf  = set(rf_imp.head(18).index)
selected = sorted(top_mi | top_rf)
print(f"  ✓ {len(selected)} features selected (union of top-18 MI & RF)")
print(f"  Features: {selected}")

# Plot feature importance
fig, axes = plt.subplots(1,2,figsize=(18,8))
mi_scores.head(18).sort_values().plot(kind="barh",ax=axes[0],color="#4C72B0",edgecolor="white")
axes[0].set_title("Mutual Information Scores (top 18)",fontweight="bold",fontsize=11)
rf_imp.head(18).sort_values().plot(kind="barh",ax=axes[1],color="#55A868",edgecolor="white")
axes[1].set_title("Random Forest Importance (top 18)",fontweight="bold",fontsize=11)
plt.tight_layout(); plt.savefig("outputs/05_feature_selection.png",dpi=120); plt.close()
print("  ✓ Feature importance plot saved")

# ══════════════════════════════════════════════════════════════
#  STAGE 8 ─ SCALING
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 8]  Scaling (StandardScaler)…")

X = df[selected].astype(float).values
y = df["target"].values

scaler    = StandardScaler()
X_scaled  = scaler.fit_transform(X)
print(f"  ✓ StandardScaler → mean≈0, std≈1  |  X shape: {X_scaled.shape}")

# ══════════════════════════════════════════════════════════════
#  STAGE 9 ─ TRAIN / VALIDATION / TEST SPLIT
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 9]  Splitting (70% Train | 15% Val | 15% Test)…")

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

print(f"  Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

# ══════════════════════════════════════════════════════════════
#  STAGE 10 ─ NEURAL NETWORK  (MLPClassifier)
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 10]  Building & Training Neural Network (MLP)…")
print("""
  Architecture:
  ─────────────────────────────────
  Input  →  Dense(256, relu)
         →  Dropout (α=0.3)
         →  Dense(128, relu)
         →  Dropout (α=0.2)
         →  Dense(64,  relu)
         →  Dense(32,  relu)
         →  Output(5,  softmax)
  Optimiser : Adam  |  Loss : cross-entropy
  ─────────────────────────────────
""")

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=0.001,          # L2 regularisation
    batch_size=64,
    learning_rate_init=0.001,
    learning_rate="adaptive",
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
    verbose=False,
)
mlp.fit(X_train, y_train)
print(f"  ✓ Training stopped at iteration {mlp.n_iter_}  (best val loss: {mlp.best_validation_score_:.4f})")

# Training history
fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].plot(mlp.loss_curve_,color="#4C72B0",lw=2,label="Train Loss")
axes[0].set_title("Loss over Iterations",fontweight="bold"); axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Cross-Entropy Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(mlp.validation_scores_,color="#55A868",lw=2,label="Val Accuracy")
axes[1].set_title("Validation Accuracy",fontweight="bold"); axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig("outputs/06_training_history.png",dpi=120); plt.close()

# ══════════════════════════════════════════════════════════════
#  STAGE 11 ─ EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 11]  Model Evaluation on Test Set…")

y_pred  = mlp.predict(X_test)
acc     = accuracy_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred, average="weighted")
prec    = precision_score(y_test, y_pred, average="weighted")
rec     = recall_score(y_test, y_pred, average="weighted")

print("\n  ╔══════════════════════════════════════╗")
print(f"  ║  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)       ║")
print(f"  ║  F1 Score  : {f1:.4f}                   ║")
print(f"  ║  Precision : {prec:.4f}                   ║")
print(f"  ║  Recall    : {rec:.4f}                   ║")
print("  ╚══════════════════════════════════════╝\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix plot
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(9,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, ax=ax, annot_kws={"size":12})
ax.set_title("Confusion Matrix – Test Set",fontsize=13,fontweight="bold")
ax.set_xlabel("Predicted",fontsize=11); ax.set_ylabel("Actual",fontsize=11)
plt.tight_layout(); plt.savefig("outputs/07_confusion_matrix.png",dpi=130); plt.close()
print("  ✓ Confusion matrix plot saved")

# Per-class accuracy bar
per_class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(le.classes_, per_class_acc*100, color=PALETTE, edgecolor="white", width=0.55)
for bar, v in zip(bars, per_class_acc):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{v*100:.1f}%", ha="center", fontsize=10)
ax.set_ylim(0,110); ax.set_title("Per-Class Accuracy",fontweight="bold",fontsize=12)
ax.set_ylabel("Accuracy %"); ax.tick_params(axis='x',rotation=20)
plt.tight_layout(); plt.savefig("outputs/08_per_class_accuracy.png",dpi=120); plt.close()

# ══════════════════════════════════════════════════════════════
#  STAGE 12 ─ PREDICTION DEMO
# ══════════════════════════════════════════════════════════════
print("\n[STAGE 12]  Prediction Demo – 4 Sample Profiles…\n")

def predict(raw_profile, label):
    """Predict specialization for a new student profile dict."""
    # --- replicate the full pipeline ---
    tmp = {f: 50.0 for f in num_cols}          # default numeric base
    tmp.update(raw_profile)
    row = pd.DataFrame([tmp])

    # Feature engineering
    row["technical_score"]  = (row["coding_interest"]+row["problem_solving"]+row["algorithms_interest"])/3
    row["security_profile"] = (row["network_interest"]+row["security_interest"]+row["linux_interest"])/3
    row["design_profile"]   = (row["design_interest"]+row["ui_ux_interest"]+row["creativity"])/3
    row["data_profile"]     = (row["math_score"]+row["statistics_interest"]+row["database_interest"])/3
    row["quality_profile"]  = (row["debugging_skill"]+row["attention_to_detail"]+row["automation_interest"])/3
    row["soft_skill_score"] = (row["teamwork"]+row["communication"])/2
    row["developer_strength"]= 0.40*row["coding_interest"]+0.30*row["algorithms_interest"]+0.30*row["problem_solving"]
    row["ai_strength"]       = 0.35*row["math_score"]+0.35*row["statistics_interest"]+0.30*row["algorithms_interest"]
    row["cyber_strength"]    = 0.35*row["network_interest"]+0.35*row["security_interest"]+0.30*row["linux_interest"]
    row["frontend_strength"] = 0.30*row["design_interest"]+0.25*row["ui_ux_interest"]+0.25*row["web_interest"]+0.20*row["creativity"]
    row["testing_strength"]  = 0.40*row["debugging_skill"]+0.30*row["attention_to_detail"]+0.30*row["automation_interest"]

    # Encoding
    gender_val = raw_profile.get("gender","Male")
    eng_val    = raw_profile.get("english_level","Intermediate")
    cat_dummy  = ohe.transform([[gender_val, eng_val]])
    for j,cname in enumerate(cat_cols):
        row[cname] = cat_dummy[0,j]

    row_sel   = row[selected].astype(float).values
    row_sc    = scaler.transform(row_sel)
    proba     = mlp.predict_proba(row_sc)[0]
    result    = dict(zip(le.classes_, proba))
    best      = max(result, key=result.get)

    print(f"  ── {label} ──")
    for spec in SPECS:
        bar = "█" * int(result[spec]*35)
        print(f"  {spec:<22} {result[spec]*100:5.1f}%  {bar}")
    print(f"  ✓  Predicted → {best}\n")
    return result

predict({
    "math_score":95,"statistics_interest":90,"algorithms_interest":88,
    "coding_interest":80,"problem_solving":87,"design_interest":20,
    "network_interest":35,"security_interest":30,"logical_thinking":90,
    "linux_interest":60,"web_interest":55,"database_interest":72,
    "debugging_skill":70,"automation_interest":72,"creativity":60,
    "attention_to_detail":80,"teamwork":68,"communication":65,
    "ui_ux_interest":30,"gender":"Male","english_level":"Advanced"
}, "Profile 1 – Expected: AI")

predict({
    "network_interest":93,"security_interest":92,"linux_interest":90,
    "coding_interest":75,"problem_solving":80,"design_interest":25,
    "math_score":72,"logical_thinking":85,"statistics_interest":65,
    "database_interest":62,"web_interest":58,"debugging_skill":80,
    "automation_interest":68,"creativity":52,"attention_to_detail":84,
    "teamwork":66,"communication":60,"ui_ux_interest":28,
    "algorithms_interest":70,"gender":"Male","english_level":"Intermediate"
}, "Profile 2 – Expected: Cyber Security")

predict({
    "design_interest":94,"ui_ux_interest":92,"web_interest":91,
    "creativity":90,"coding_interest":75,"communication":85,
    "math_score":55,"network_interest":40,"security_interest":38,
    "linux_interest":42,"statistics_interest":44,"database_interest":48,
    "debugging_skill":63,"automation_interest":52,"attention_to_detail":77,
    "teamwork":82,"problem_solving":66,"algorithms_interest":53,
    "logical_thinking":63,"gender":"Female","english_level":"Advanced"
}, "Profile 3 – Expected: Frontend")

predict({
    "debugging_skill":95,"attention_to_detail":93,"automation_interest":90,
    "problem_solving":82,"coding_interest":70,"design_interest":45,
    "logical_thinking":83,"teamwork":80,"communication":78,
    "math_score":63,"statistics_interest":68,"network_interest":53,
    "security_interest":58,"ui_ux_interest":53,"web_interest":58,
    "linux_interest":55,"database_interest":63,"algorithms_interest":63,
    "creativity":58,"gender":"Female","english_level":"Intermediate"
}, "Profile 4 – Expected: Software Testing")

# ══════════════════════════════════════════════════════════════
#  SAVE CLEAN DATASET
# ══════════════════════════════════════════════════════════════
df_clean_ref.to_csv("outputs/students_cleaned.csv", index=False)
print("  ✓ Clean dataset saved  →  outputs/students_cleaned.csv")

print("\n" + "=" * 65)
print("  PIPELINE COMPLETE – All outputs in ./outputs/")
print("  ├─ students_raw_dirty.csv       ← raw dirty data")
print("  ├─ students_cleaned.csv         ← clean reference data")
print("  ├─ 01_class_distribution.png")
print("  ├─ 02_correlation_heatmap.png")
print("  ├─ 03_feature_distributions.png")
print("  ├─ 04_boxplots_by_spec.png")
print("  ├─ 05_feature_selection.png")
print("  ├─ 06_training_history.png")
print("  ├─ 07_confusion_matrix.png")
print("  └─ 08_per_class_accuracy.png")
print("=" * 65)
