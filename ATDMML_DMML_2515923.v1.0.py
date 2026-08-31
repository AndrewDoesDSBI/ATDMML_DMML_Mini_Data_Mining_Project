# -*- coding: utf-8 -*-
"""
Project: ATDMML CW2 - Telco Customer Churn (mini DM project)
Dataset: Kaggle / IBM Telco Customer Churn
         https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Author: Andrew Cocking
Student ID: 2515923
Module: Applied Techniques of Data Mining and Machine Learning
Created: 18/08/2026

ANALYSIS PIPELINE

Block 1: Libraries, Configuration, Constants & Data Ingestion
    Purpose: Environment setup, constants, data ingestion, pristine copy
    Inputs:  WA_Fn-UseC_-Telco-Customer-Churn.csv
    Outputs: raw_df, pristineRaw_df (pristine ground truth)

Block 2: Global Functions
    Purpose: Reusable functions used across multiple blocks
    Inputs:  N/A
    Outputs: N/A

Block 3: Data Profiling (CRISP-DM: Data Understanding)
    Purpose: Tabular health check - identify issues, do not treat them (Read Only)
    Inputs:  raw_df
    Outputs: tables/ under ATDMML_CW2_Analysis/tables

Block 4: Phase 1 EDA (CRISP-DM: Data Understanding)
    Purpose: Visualise what Block 3 surfaced - inform prep, not replace it (Read Only)
    Inputs:  raw_df & Block 3 tables
    Outputs: plots/ & summary CSVs

Block 5: Data Preparation (CRISP-DM: Data Preparation)
    Purpose: Systematically treat quality issues surfaced in Blocks 3 & 4
    Inputs:  pristineRaw_df (pristine copy)
    Outputs: Encoded train/test matrices, fitted transformer, prep decision log

Block 6: Modelling and Evaluation (CRISP-DM: Modelling / Evaluation)
    Purpose: Random Forest (bagging) and XGBoost (boosting) - baseline and then tuned
    Inputs:  Block 5 artefacts (encoded matrices & y vectors)
    Outputs: Metrics tables, confusion matrices, importance plots, ROC overlay

"""

#%% Block 1: Libraries, Configuration and Constants

##### Purpose:
    #  - All library imports for entire pipeline
    #  - Diagnostic helpers (note/info/warn) replacing raw print statements
    #  - File paths, column groups, analytical parameters and colour palette defined once
    #  - Primary dataset ingested once, pristine copy preserved as ground truth throughout


##### 1a: Libraries and display settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
import xgboost as xgb
import joblib

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 80) # keeps long strings readable in console withoyt truncation

##### 1b: lightweight diagnostic helpers
def note(msg): print(f"NOTE  | {msg}") # used for design decisions/justification etc
def info(msg): print(f"INFO  | {msg}") # used for runtime fact/count/results etc
def warn(msg): print(f"WARN  | {msg}") # used for flagging anomalies/issues etc

##### 1c: File paths and output directoriess
sourceFile = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")

outputDir = Path("ATDMML_CW2_Analysis")
outputDirTables = outputDir / "tables"
outputDirPlots = outputDir / "plots"
outputDirTables.mkdir(parents=True, exist_ok=True)
outputDirPlots.mkdir(parents=True, exist_ok=True)

##### 1d: Column groups (as published, dtypes as loaded may disagree - that is a finding)
idCol = "customerID"
targetCol = "Churn"

# SeniorCitizen is 0/1 integer but semantically categorical (Block 3 will confirm this)
seniorCitizenCols = ["SeniorCitizen"]
numericFields = ["tenure", "MonthlyCharges", "TotalCharges"]

##### 1e: Analytical parameters
randomState = 42 # single seed applied throughout for reproducibility
churnLabel = 1 # Yes maps to 1 in Block 5e
testSize = 0.20 # 80/20 stratified split
innerFolds = 5 # Stratified so that each fold preserves ~26.54% Yes

##### 1f: Colour palette (fixed so report figures do not shift)
palette = {
    "churnYes": "#d62728",
    "churnNo": "#1f77b4",
    "neutral": "steelblue",
    "warn": "#ffbf00",
}

sns.set_theme(style="whitegrid")

##### 1g: Ingestion and pristine copy
if not sourceFile.exists():
    raise FileNotFoundError(
        f"Source file not found: {sourceFile}\n"
        f"Expected location: {sourceFile.resolve()}\n"
        f"Ensure the Telco Customer Churn .csv file is in the project root folder."
    )

raw_df = pd.read_csv(sourceFile)
pristineRaw_df = raw_df.copy()

info(f"Ingested {sourceFile} | shape={raw_df.shape}")
info(f"columns={list(raw_df.columns)}")

##### 1h: Confirmation print
print("Block 1: Libraries, Config/Constants and Ingestion loaded successfully")

#%% Block 2: Global Functions

##### Purpose:
    #  - Reusable functions called across two or more blocks
    #  - Single-use helpers remain inline at point of use
    #  - No mutation of raw_df or pristineRaw_df in any helper
    #  - Fully documented with Args / Returns

def saveTable(df, name, index=False):
    """
    Writes a DataFrame to CSV under tables/ and returns the path.
    Every decision-facing table is copy-pasteable into the report.

    Args:
        df - Dataframe to save
        name - output filename (eg. 'block3a_schema.csv')
        index - whether to write the row index (defualt false)
    Returns:
        Path object of saved file
    """
    path = outputDirTables / name
    df.to_csv(path, index=index)
    info(f"table saved: {path} | rows={len(df)}")
    return path

def coerceTotalCharges(series):
    """
    EDA-only numeric view of TotalCharges.
    Blank strings become NaN. Does not write back to raw_df.

    Args:
        series - TotalCharges column as loaded (object dtype)
    Returns:
        numeric series with NaN where coercion fails
    """
    return pd.to_numeric(series, errors="coerce")

def enhancedDescribeNumeric(df, cols):
    """
    Compact numeric profile for the report
    returns count, missing, mean, sd, min, quartiles and max per column.

    Args:
        df - source DataFrame
        cols - list of column names to profile
    Returns:
        DataFrame with one row per variable
    """
    rows = []
    for c in cols:
        s = df[c]
        rows.append({
            "variable": c,
            "dtype": str(s.dtype),
            "n": int(s.notna().sum()),
            "nMissing": int(s.isna().sum()),
            "pctMissing": round(100 * s.isna().mean(), 3),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "q25": s.quantile(0.25),
            "median": s.median(),
            "q75": s.quantile(0.75),
            "max": s.max(),
        })
    return pd.DataFrame(rows)

def freqTable(series, colName):
    """
    Absolute and relative frequencies including NaN (dropna=False).

    Args:
        series - pandas Series to tabulate
        colName - label for the value column in output
    Returns:
        DataFrame with columns [colName, n, pct]
    """
    vc = series.value_counts(dropna=False)
    out = vc.rename_axis(colName).reset_index(name="n")
    out["pct"] = (out["n"] / out["n"].sum() * 100).round(2)
    return out

def metricsRow(yTrue, yPred, yProba, modelName, stage):
    """
    Headline metrics for the report comparison table.

    Args:
        yTrue - true labels
        yPred - predicted labels
        yProba - P(class=1) for ROC-AUC
        modelName - string label for the model
        stage - 'baseline_test' or 'tuned_test'
    Returns:
        dict of metrics
    """
    return {
        "model": modelName,
        "stage": stage,
        "accuracy": accuracy_score(yTrue, yPred),
        "precision": precision_score(yTrue, yPred, pos_label=churnLabel, zero_division=0),
        "recall": recall_score(yTrue, yPred, pos_label=churnLabel, zero_division=0),
        "f1": f1_score(yTrue, yPred, pos_label=churnLabel, zero_division=0),
        "rocAuc": roc_auc_score(yTrue, yProba),
        "nPredYes": int((yPred == 1).sum()),
        "nTrueYes": int((yTrue == 1).sum()),
    }

def cmFrame(yTrue, yPred, modelName, stage):
    """
    Returns a one-row dataFrame of confusion matrix cells.

    Args:
        yTrue - true labels
        yPred - predicted labels
        modelName - string label for the model
        stage - 'baseline_test' or 'tuned_test'
    Returns:
        single-row DataFrame [model, stage, tn, fp, fn, tp]
    """
    cm = confusion_matrix(yTrue, yPred, labels=[0, 1]) # labels=[0,1] forces No/Yes order regardless of class frequency
    return pd.DataFrame(
        {
            "model": modelName,
            "stage": stage,
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        },
        index=[0],
    )

def saveCmPlot(yTrue, yPred, title, fname):
    """
    Saves a seaborn confusion matrix heatmap to plots/

    Args:
        yTrue - true labels
        yPred - predicted labels
        title - plot title string
        fname - output filename (eg. 'block6b_rfTuned_cm.png')
    """
    cm = confusion_matrix(yTrue, yPred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred No", "Pred Yes"],
                yticklabels=["Actual No", "Actual Yes"])
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outputDirPlots / fname, dpi=150, bbox_inches="tight")
    plt.show()
    note(f"saved {fname}")

def importanceFrame(names, values, modelName, topN=15):
    """
    Builds ranked feature importance DataFrame and returns full & top-n slices

    Args:
        names - feature names (iterable)
        values - importance scores (iterable, same order as names)
        modelName - string label for the model
        topN - number of features to include in the top slice (default = 15)
    Returns:
        (impAll_df, impTop_df) tuple
    """
    imp = (
        pd.DataFrame({"feature": names, "importance": values, "model": modelName})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp, imp.head(topN)

def logPrep(step, issue, evidence, action, nAffected, rationale):
    """
    Appends one row to the preparation decision log.

    Args:
        prepRows - list accumulating decision rows (mutated in place)
        step - block step reference (eg. '5c')
        issue - quality issue being addressed
        evidence - evidence table filename from Block 3
        action - treatment applied
        nAffected - number of rows/columns affected
        rationale - justification for the decision
    """
    prepRows.append({
        "step": step,
        "issue": issue,
        "evidenceTable": evidence,
        "action": action,
        "nAffected": nAffected,
        "rationale": rationale,
    })

##### Confirmation print
print("Block 2: Global functions defined successfully")

#%% Block 3: Data Profiling (READ ONLY)

##### Purpose:
    #  - Tabular health check to identify/quantify issues (not treat them)
    #  - raw_df is not mutated in this block
    #  - TotalCharges coerced ONLY into a temporary side series for arithmetic checks
    #  - Block ends with an explicit pitfall / handoff table linking findings to Block 5

##### Strategy:
    #  - Section A: Schema, identifier integrity, missingness, domain checks
    #  - Section B: Target balance, categorical frequencies, numeric profile, collinearity
    #  - Section C: Handoff - prepHandoff table maps each finding to its Block 5 treatment
    #  - All tables saved to outputDirTables for report use

### 3a: Shape and schema
schema_df = pd.DataFrame({
    "variable": raw_df.columns,
    "dtypeLoaded": raw_df.dtypes.astype(str).values,
    "nUnique": [raw_df[c].nunique(dropna=False) for c in raw_df.columns],
    "nMissingLoaded": raw_df.isna().sum().values,
})
# role tags for the report variable table
roleMap = {idCol: "ID", targetCol: "target"}
for c in raw_df.columns:
    if c not in roleMap:
        roleMap[c] = "feature"
schema_df["role"] = schema_df["variable"].map(roleMap)

# intended measurement type (semantic), distinct from dtypeLoaded
semantic = {}
for c in raw_df.columns:
    if c == idCol:
        semantic[c] = "identifier"
    elif c == targetCol:
        semantic[c] = "binary class"
    elif c == "SeniorCitizen":
        semantic[c] = "binary categorical (coded 0/1)"
    elif c in ["tenure"]:
        semantic[c] = "integer numeric (months)"
    elif c in ["MonthlyCharges", "TotalCharges"]:
        semantic[c] = "continuous numeric (currency)"
    else:
        semantic[c] = "nominal / ordinal categorical"
schema_df["semanticType"] = schema_df["variable"].map(semantic)

print("\n3a schema")
print(schema_df.to_string(index=False))
saveTable(schema_df, "block3a_schema.csv")

info(f"nRows={len(raw_df)} | nCols={raw_df.shape[1]}")
if raw_df.shape[0] != 7043:
    warn(f"row count {raw_df.shape[0]} differs from published 7043 - check file version")

### 3b: Identifier integrity
idDupe = raw_df[idCol].duplicated().sum()
idNull = raw_df[idCol].isna().sum()
idAudit_df = pd.DataFrame({
    "check": ["nID", "nUniqueID", "duplicateIDs", "nullIDs"],
    "value": [len(raw_df), raw_df[idCol].nunique(), int(idDupe), int(idNull)],
})
print("\n3b ID audit")
print(idAudit_df.to_string(index=False))
saveTable(idAudit_df, "block3b_idAudit.csv")
if idDupe or idNull:
    warn("customerID is not a clean unique key")
else:
    note("customerID confirmed as unique - no predictive value, drop prior to modelling")

### 3c: Missingness as loaded vs after TotalCharges coerce
totalChargesNumeric = coerceTotalCharges(raw_df["TotalCharges"])
# blanks in object TotalCharges
totalChargesBlank = (raw_df["TotalCharges"].astype(str).str.strip() == "").sum()
totalChargesNonNumeric = totalChargesNumeric.isna().sum()

missing_df = pd.DataFrame({
    "variable": raw_df.columns,
    "nMissingLoaded": raw_df.isna().sum().values,
    "pctMissingLoaded": (raw_df.isna().mean().values * 100).round(3),
})
# append TotalCharges coerce finding as extra rows in a dedicated table
totalChargesIssue_df = pd.DataFrame({
    "check": [
        "TotalCharges dtype as loaded",
        "TotalCharges blank/whitespace strings",
        "TotalCharges NaN after to_numeric(errors='coerce')",
        "tenure==0 rows",
        "tenure==0 AND TotalCharges not numeric",
    ],
    "value": [
        str(raw_df["TotalCharges"].dtype),
        int(totalChargesBlank),
        int(totalChargesNonNumeric),
        int((raw_df["tenure"] == 0).sum()),
        int(((raw_df["tenure"] == 0) & totalChargesNumeric.isna()).sum()),
    ],
})
print("\n3c missingness (loaded)")
print(missing_df[missing_df["nMissingLoaded"] > 0].to_string(index=False)
      if (missing_df["nMissingLoaded"] > 0).any()
      else "no pandas-NaN as loaded (blanks may still exist in object columns)")
print("\n3c TotalCharges / tenure=0")
print(totalChargesIssue_df.to_string(index=False))
saveTable(missing_df, "block3c_missingLoaded.csv")
saveTable(totalChargesIssue_df, "block3c_totalChargesIssue.csv")

if int(totalChargesNonNumeric) > 0:
    warn(f"{int(totalChargesNonNumeric)} TotalCharges values not numeric - all on tenure==0 new accounts, expected")
    note("Block 5: set TotalCharges=0 where tenure==0 - measured zero, not an imputation")

### 3d: SeniorCitizen domain (must not be treated as continuous)
seniorCitizen_df = freqTable(raw_df["SeniorCitizen"], "SeniorCitizen")
print("\n3d SeniorCitizen")
print(seniorCitizen_df.to_string(index=False))
saveTable(seniorCitizen_df, "block3d_seniorCitizen.csv")
if set(raw_df["SeniorCitizen"].dropna().unique()) <= {0, 1}:
    note("SeniorCitizen is binary flag coded 0/1 - recode from integer to categorical in Block 5")
else:
    warn("SeniorCitizen has values other than 0/1")

### 3e: Target balance
churn_df = freqTable(raw_df[targetCol], targetCol)
print("\n3e target")
print(churn_df.to_string(index=False))
saveTable(churn_df, "block3e_targetBalance.csv")
yesPct = float(churn_df.loc[churn_df[targetCol].astype(str).str.lower().eq("yes"), "pct"].sum())
info(f"churn Yes ~ {yesPct:.2f}% - mild imbalance, F1/recall/AUC are metrics that matter here")
note("Block 5/6: stratify the split, handle imbalance via model weights, report F1/recall/precision/AUC")

### 3f: Categorical frequencies (all object cols except ID and TotalCharges)
catCols = [c for c in raw_df.select_dtypes(include="object").columns
           if c not in [idCol, "TotalCharges"]]
catFreqFrames = []
for c in catCols:
    t = freqTable(raw_df[c], "level")
    t.insert(0, "variable", c)
    catFreqFrames.append(t)
catFreq_df = pd.concat(catFreqFrames, ignore_index=True)
print("\n3f categorical frequencies (head)")
print(catFreq_df.head(40).to_string(index=False))
saveTable(catFreq_df, "block3f_categoricalFrequencies.csv")

### 3g: Numeric profile (tenure, MonthlyCharges - TotalCharges coerced side-copy)
edaNumeric_df = raw_df[["tenure", "MonthlyCharges"]].copy()
edaNumeric_df["TotalCharges_coerced"] = totalChargesNumeric
numericDescribe_df = enhancedDescribeNumeric(
    edaNumeric_df, ["tenure", "MonthlyCharges", "TotalCharges_coerced"]
)
print("\n3g numeric describe (TotalCharges coerced for EDA only)")
print(numericDescribe_df.to_string(index=False))
saveTable(numericDescribe_df, "block3g_numericDescribe.csv")

### 3h: Derived-field / collinearity hint
# TotalCharges should track tenure * MonthlyCharges for established customers
corrCalc_df = pd.DataFrame({
    "tenure": raw_df["tenure"],
    "MonthlyCharges": raw_df["MonthlyCharges"],
    "TotalCharges_coerced": totalChargesNumeric,
})
corrCalc_df["implied"] = corrCalc_df["tenure"] * corrCalc_df["MonthlyCharges"]
corrValidRows = corrCalc_df.dropna(subset=["TotalCharges_coerced"])
corrTenureMonthly = corrValidRows["tenure"].corr(corrValidRows["MonthlyCharges"])
corrTenureTotal = corrValidRows["tenure"].corr(corrValidRows["TotalCharges_coerced"])
corrMonthlyTotal = corrValidRows["MonthlyCharges"].corr(corrValidRows["TotalCharges_coerced"])
corrImplied = corrValidRows["TotalCharges_coerced"].corr(corrValidRows["implied"])
collinearity_df = pd.DataFrame({
    "pair": [
        "tenure vs MonthlyCharges",
        "tenure vs TotalCharges_coerced",
        "MonthlyCharges vs TotalCharges_coerced",
        "TotalCharges_coerced vs tenure*MonthlyCharges",
    ],
    "pearsonR": [corrTenureMonthly, corrTenureTotal, corrMonthlyTotal, corrImplied],
})
print("\n3h collinearity hint")
print(collinearity_df.to_string(index=False))
saveTable(collinearity_df, "block3h_collinearityHint.csv")
note("TotalCharges is effectively tenure * MonthlyCharges - r~0.9996 confirms this. "
     "Trees handle this fine, flag for future linear baseline.")

### 3i: Impossible / suspicious values
suspect_df = pd.DataFrame({
    "check": [
        "tenure < 0",
        "MonthlyCharges < 0",
        "TotalCharges_coerced < 0",
        "PhoneService=No but MultipleLines not in {No phone service}",
        "InternetService=No but OnlineSecurity not in {No internet service}",
    ],
    "n": [
        int((raw_df["tenure"] < 0).sum()),
        int((raw_df["MonthlyCharges"] < 0).sum()),
        int((totalChargesNumeric < 0).sum()),
        int(((raw_df["PhoneService"] == "No") &
             (~raw_df["MultipleLines"].isin(["No phone service"]))).sum())
        if "MultipleLines" in raw_df.columns else np.nan,
        int(((raw_df["InternetService"] == "No") &
             (~raw_df["OnlineSecurity"].isin(["No internet service"]))).sum())
        if "OnlineSecurity" in raw_df.columns else np.nan,
    ],
})
print("\n3i domain / consistency")
print(suspect_df.to_string(index=False))
saveTable(suspect_df, "block3i_domainChecks.csv")

### 3j: Handoff - pitfalls for pre-processing and mining (rubric language)
handoff_df = pd.DataFrame([
    {
        "issue": "customerID unique key",
        "evidenceTable": "block3b_idAudit.csv",
        "pitfallIfIgnored": "ID used as a feature (no signal / accidental leakage)",
        "block5Candidate": "Drop customerID before modelling",
    },
    {
        "issue": "TotalCharges stored as object, ~11 non-numeric (tenure=0)",
        "evidenceTable": "block3c_totalChargesIssue.csv",
        "pitfallIfIgnored": "column unusable or silent row drop",
        "block5Candidate": "to_numeric, set 0 where tenure==0, document leftover NaNs",
    },
    {
        "issue": "SeniorCitizen coded 0/1 integer",
        "evidenceTable": "block3d_seniorCitizen.csv",
        "pitfallIfIgnored": "treated as continuous, meaningless 'mean seniorness'",
        "block5Candidate": "Cast to category / OHE with other categoricals",
    },
    {
        "issue": "Churn ~26% Yes",
        "evidenceTable": "block3e_targetBalance.csv",
        "pitfallIfIgnored": "accuracy looks strong while leavers are missed",
        "block5Candidate": "Stratified split, imbalance-aware metrics and weights",
    },
    {
        "issue": "Many nominal service/contract fields",
        "evidenceTable": "block3f_categoricalFrequencies.csv",
        "pitfallIfIgnored": "models cannot consume strings, ordinal assumed wrongly",
        "block5Candidate": "One-hot (or equivalent), keep 'No internet service' as its own level",
    },
    {
        "issue": "TotalCharges ~ f(tenure, MonthlyCharges)",
        "evidenceTable": "block3h_collinearityHint.csv",
        "pitfallIfIgnored": "unstable linear coefficients, double-counting in interpretation",
        "block5Candidate": "Retain for trees, optional drop/PCA only if a linear baseline needs it",
    },
    {
        "issue": "Snapshot table, no event date",
        "evidenceTable": "schema / source",
        "pitfallIfIgnored": "causal / 'when they will leave' claims",
        "block5Candidate": "Classification only, state proportion in section (b) and section (e)",
    },
])
print("\n3j Block 5 handoff (copy into report Section (b) pitfalls)")
print(handoff_df.to_string(index=False))
saveTable(handoff_df, "block3j_prepHandoff.csv")

##### Observations/Interpretations:
    #  - 7,043 rows x 21 columns confirmed, one row per customer, no duplicates
    #  - customerID unique and clean - zero predictive value, drop prior to modelling
    #  - TotalCharges: 11 blanks, all on tenure==0 representing new accounts with no bill yet
    #  - SeniorCitizen coded as 0/1 integer but is a categorical flag - requires recoding
    #  - Target: 26.54% Yes (churn), mild imbalance
    #  - Domain checks clean: no negative values, no structural dependency violations
    #  - TotalCharges r~0.9996 with tenure*MonthlyCharges - cumulative bill as expected, not a third independent driver
    #    Trees tolerate this fine, linear coefficients wouldn't be independently interpretable
    #  - All 7 handoff items confirmed, Block 5 treatments defined

print("Block 3: Data profiling complete (read only, no treatment applied)")

#%% Block 4: Phase 1 EDA (READ ONLY)

##### Purpose:
    #  - Each figure answers one Block 3 question
    #  - Companion .csv next to each plot so report can quote numbers
    #  - TotalCharges_coerced is for plots only, not written back to raw_df
    #  - Six figures in report narrative order (Figures 1-6)

##### Strategy:
    #  - plot_df is a working copy for EDA visualisation only
    #  - raw_df and pristineRaw_df remain unmodified throughout this block
    #  - All figures saved to outputDirPlots at standardised figsize (5, 3.5)
    #  - Exception: heatmap (also 5, 3.5) with rotated x-axis labels for readability
    #  - All companion .csv's saved to outputDirTables

plot_df = raw_df.copy()
plot_df["TotalCharges_coerced"] = coerceTotalCharges(plot_df["TotalCharges"])

### 4a: Figure 1 - Class balance
fig, ax = plt.subplots(figsize=(5, 3.5))
churnOrder = plot_df[targetCol].value_counts().index
sns.countplot(data=plot_df, x=targetCol, order=churnOrder,
              palette=[palette["churnNo"] if str(o).lower() == "no" else palette["churnYes"]
                       for o in churnOrder],
              ax=ax)
ax.set_title("Figure 1: Churn Distribution (n = 7,043)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(outputDirPlots / "block4a_classBalance.png", dpi=150, bbox_inches="tight")
plt.show()
note("Block 4a saved: block4a_classBalance.png")

### 4b: Churn rate by Contract (business-facing driver)
contractRates = (
    plot_df.groupby("Contract")[targetCol]
    .apply(lambda s: (s.astype(str).str.lower() == "yes").mean() * 100)
    .rename("churnRatePct")
    .reset_index()
    .sort_values("churnRatePct", ascending=False)
)
nByContract = plot_df.groupby("Contract").size().rename("n").reset_index()
contractRates = contractRates.merge(nByContract, on="Contract")
print("\n4b churn rate by Contract")
print(contractRates.to_string(index=False))
saveTable(contractRates, "block4b_churnByContract.csv")

fig, ax = plt.subplots(figsize=(5, 3.5))
sns.barplot(data=contractRates, x="Contract", y="churnRatePct", color=palette["neutral"], ax=ax)
ax.set_ylabel("Churn rate (%)")
ax.set_title("Figure 4: Churn Rate by Contract Type")
plt.tight_layout()
plt.savefig(outputDirPlots / "block4b_churnByContract.png", dpi=150, bbox_inches="tight")
plt.show()
note("Block 4b saved: block4b_churnByContract.png")

### 4c: Churn rate by InternetService
if "InternetService" in plot_df.columns:
    intRates = (
        plot_df.groupby("InternetService")[targetCol]
        .apply(lambda s: (s.astype(str).str.lower() == "yes").mean() * 100)
        .rename("churnRatePct")
        .reset_index()
        .sort_values("churnRatePct", ascending=False)
    )
    intRates = intRates.merge(plot_df.groupby("InternetService").size().rename("n"),
                              on="InternetService")
    print("\n4c churn rate by InternetService")
    print(intRates.to_string(index=False))
    saveTable(intRates, "block4c_churnByInternetService.csv")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(data=intRates, x="InternetService", y="churnRatePct",
                color=palette["neutral"], ax=ax)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Figure 5: Churn Rate by Internet Service Type")
    plt.tight_layout()
    plt.savefig(outputDirPlots / "block4c_churnByInternetService.png", dpi=150, bbox_inches="tight")
    plt.show()
    note("Block 4c saved: block4c_churnByInternetService.png")

### 4d: Tenure by churn (survival-ish, still not survival analysis)
fig, ax = plt.subplots(figsize=(5, 3.5))
sns.histplot(data=plot_df, x="tenure", hue=targetCol, bins=36,
             stat="density", common_norm=False, element="step", ax=ax)
ax.set_title("Figure 2: Tenure Distribution by Churn Status")
plt.tight_layout()
plt.savefig(outputDirPlots / "block4d_tenureByChurn.png", dpi=150, bbox_inches="tight")
plt.show()
note("Block 4d saved: block4d_tenureByChurn.png")

tenureByChurn = (
    plot_df.groupby(targetCol)["tenure"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)
print("\n4d tenure by churn")
print(tenureByChurn.to_string(index=False))
saveTable(tenureByChurn, "block4d_tenureByChurn.csv")

### 4e: MonthlyCharges by churn
fig, ax = plt.subplots(figsize=(5, 3.5))
sns.boxplot(data=plot_df, x=targetCol, y="MonthlyCharges", ax=ax)
ax.set_title("Figure 3: Monthly Charges by Churn Status")
plt.tight_layout()
plt.savefig(outputDirPlots / "block4e_monthlyByChurn.png", dpi=150, bbox_inches="tight")
plt.show()
note("Block 4e saved: block4e_monthlyByChurn.png")

monthlyByChurn = (
    plot_df.groupby(targetCol)["MonthlyCharges"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)
saveTable(monthlyByChurn, "block4e_monthlyByChurn.csv")

### 4f: Numeric correlation heatmap (coerced TotalCharges - documented)
corrCols = ["tenure", "MonthlyCharges", "TotalCharges_coerced"]
corrM = plot_df[corrCols].corr()
corrM.index = ["tenure", "MonthlyCharges", "TotalCharges"]
corrM.columns = ["tenure", "MonthlyCharges", "TotalCharges"]
print("\n4f numeric correlation")
print(corrM.round(3).to_string())
saveTable(corrM.reset_index().rename(columns={"index": "variable"}),
          "block4f_numericCorrelation.csv")

fig, ax = plt.subplots(figsize=(5, 3.5))
sns.heatmap(
    corrM, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax,
    xticklabels=corrM.columns, yticklabels=corrM.index
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
ax.set_title("Figure 6: Pearson correlations (numeric features)")
plt.tight_layout()
plt.savefig(outputDirPlots / "block4f_corrHeatmap.png", dpi=150, bbox_inches="tight")
plt.show()
note("Block 4f saved: block4f_corrHeatmap.png")

##### Observations/Interpretations:
    #  - Figure 1: 73.46% No / 26.54% Yes - mild class imbalance confirmed visually
    #  - Figure 2: leavers concentrate in early tenure (months 0-10), retained customers spread across 0-72 with a spike
    #    at 72 (loyal and long-tenured customers)
    #  - Figure 3: leavers pay higher monthly charges (median ~79) vs retained (~64)
    #  - Figure 4: month-to-month contracts show 42.% churn vs 2.8% for two-year
    #  - Figure 5: fibre optic customers churn at 41.9% vs 7.4% for no internet services
    #  - Figure 6: TotalCharges r~0.9996 with tenure*MonthlyCharges - cumulative bill and not a third independent 
    #    driver, r~0.83 tenure vs TotalCharges as expected
    #  - Forward action: all findings consistent with Block 3 profile, no surprises

print("Block 4: Phase 1 EDA completed - all visuals and tables written")
print(f"Tables: {outputDirTables}")
print(f"Plots: {outputDirPlots}")
note("STOP. Do not one-hot, scale, or fit models in this file. Block 5 is prep.")

#%% Block 5: Data Preparation (treatment - working copy only)

##### Purpose:
    #  - Systematically address all issues identified in Blocks 3 & 4
    #  - pristineRaw_df remains ground truth, all mutation on work_df
    #  - Every operation cites a Block 3 evidence table
    #  - Split FIRST on the pre-encoded table, then encode/scale fit on TRAIN only (no leakage)

##### Strategy:
    #  - Step 5a: Working copy - work_df derived from pristineRaw_df
    #  - Step 5b: TotalCharges - object coercion, tenure=0 set to 0
    #  - Step 5c: SeniorCitizen - recoding integer to categorical
    #  - Step 5d: Drop customerID before an encoding
    #  - Step 5e: Column role declaration (numeric vs categorical)
    #  - Step 5f: Stratified 80/20 split before fitting any encoder
    #  - Step 5g: OneHotEncoder fit on TRAIN only, transform applied to test
    #  - Step 5h: Shape audit and prep decision log for report section (c)
    #  - Tree-based models don't need scaling - no StandardScaler in theis pipeline
    #  - No SMOTE/undersample - class_weight deferred to Block 6 modelling

##### 5a: Working copy
work_df = pristineRaw_df.copy()
note("Block 5 operates on work_df, pristineRaw_df is never written to")
info(f"work_df start: {work_df.shape}")

prepRows = []  # accumulate decision log


##### 5b: TotalCharges - object & 11 blanks on tenure==0
# evidence: block3c_totalChargesIssue.csv (11 blanks == 11 tenure==0 == 11 coerce-NaN)
totalChargesBefore = pd.to_numeric(work_df["TotalCharges"], errors="coerce")
nBlank = int(totalChargesBefore.isna().sum())
nTenure0 = int((work_df["tenure"] == 0).sum())
nBoth = int(((work_df["tenure"] == 0) & totalChargesBefore.isna()).sum())

work_df["TotalCharges"] = totalChargesBefore
# new customers have no cumulative bill - 0 is a measurement, not an imputation guess
work_df.loc[work_df["tenure"] == 0, "TotalCharges"] = 0.0
nLeft = int(work_df["TotalCharges"].isna().sum())
if nLeft:
    warn(f"TotalCharges still NaN after tenure==0 rule: {nLeft} - inspect before continuing")
else:
    info("TotalCharges: no leftover NaN after fix - clean")

logPrep(
    "5b",
    "TotalCharges object / 11 non-numeric",
    "block3c_totalChargesIssue.csv",
    "to_numeric, set 0 where tenure==0",
    nBoth,
    "All 11 blanks are on tenure==0 new accounts - no bill yet so 0 is the correct value "
    "and not an imputation guess therefore not grounds for dropping of rows",
)

##### 5c: SeniorCitizen - integer-coded categorical
work_df["SeniorCitizen"] = work_df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype("object")
if work_df["SeniorCitizen"].isna().any():
    warn("SeniorCitizen mapping produced NaN - unexpected codes")
info(f"SeniorCitizen recoded to Yes/No | value_counts:\n{work_df['SeniorCitizen'].value_counts().to_string()}")
logPrep(
    "5c",
    "SeniorCitizen stored as 0/1 int",
    "block3d_seniorCitizen.csv",
    "Map 0/1 -> No/Yes, treat as nominal with other categoricals",
    len(work_df),
    "It's a demographic flag and not a number - leaving as integer would allow it to be scaled "
    "or averaged which makes no sense",
)

##### 5d: Drop identifier
assert work_df["customerID"].is_unique, "customerID not unique - stop"
work_df = work_df.drop(columns=["customerID"])
info(f"dropped customerID | shape now {work_df.shape}")
logPrep(
    "5d",
    "customerID unique key",
    "block3b_idAudit.csv",
    "Drop before split/encode",
    1,
    "No signal here - just a row identifier, drop it before anything touches the data.",
)

##### 5e: Column groups after 5b-5d (pre-split)
targetCol = "Churn"
y = work_df[targetCol].map({"No": 0, "Yes": 1}) # Yes=1 so churnLabel=1 is the positive class throughout
if y.isna().any():
    raise ValueError("Churn has unexpected levels")
X = work_df.drop(columns=[targetCol])

numericCols = ["tenure", "MonthlyCharges", "TotalCharges"]
categoricalCols = [c for c in X.columns if c not in numericCols]

# sanity: every leftover column is object/category
nonObjectCols = X[categoricalCols].select_dtypes(exclude=["object", "category"]).columns.tolist()
if nonObjectCols:
    warn(f"categoricalCols still non-object: {nonObjectCols}")

info(f"numericCols={numericCols}")
info(f"categoricalCols ({len(categoricalCols)}): {categoricalCols}")

colRole_df = pd.DataFrame({
    "variable": list(numericCols) + list(categoricalCols) + [targetCol],
    "roleAfterPrep": (["numeric"] * len(numericCols)
                      + ["categorical"] * len(categoricalCols)
                      + ["target (Yes=1)"]),
})
saveTable(colRole_df, "block5e_columnRoles.csv")

logPrep(
    "5e",
    "Mixed types for sklearn",
    "block3a_schema.csv / block3f",
    f"Declare {len(numericCols)} numeric & {len(categoricalCols)} categorical, target Yes=1",
    X.shape[1],
    "Only categoricals get encoded - numerics pass through as-is. "
    "Collinearity note in Block 3h but trees handle this fine.",
)

##### 5f: Stratified split BEFORE fitting any encoder
# evidence: block3e_targetBalance.csv - 26.54% Yes, random split can drift
xTrain, xTest, yTrain, yTest = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

def classMix(s, label):
    return pd.Series({
        "split": label,
        "n": int(len(s)),
        "nYes": int(s.sum()),
        "pctYes": round(100 * s.mean(), 2),
    })

splitMix_df = pd.DataFrame([
    classMix(y, "full"),
    classMix(yTrain, "train"),
    classMix(yTest, "test"),
])
print("\n5f stratified 80/20")
print(splitMix_df.to_string(index=False))
saveTable(splitMix_df, "block5f_splitBalance.csv")

logPrep(
    "5f",
    "Churn 26.54% Yes",
    "block3e_targetBalance.csv",
    "train_test_split test_size=0.20, stratify=y, random_state=42",
    len(yTest),
    "Retains the 26.54% churn rate consistent across both folds so metrics are comparable - "
    "encoder fitted on training dataset only.",
)

##### 5g: One-hot on TRAIN only, apply to test
# keep 'No internet service' / 'No phone service' as full levels (block3i / 3j)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False) # sparse_output=False gives a dense array, easier to inspect
# remainder numeric passthrough - no scaler here (RF/XGB)
columnTransformer = ColumnTransformer(
    transformers=[
        ("cat", ohe, categoricalCols),
        ("num", "passthrough", numericCols),
    ],
    remainder="drop",
    verbose_feature_names_out=True, # adds cat__ / num__ prefixes, stripped for display in featNamesDisplay
)

xTrainEnc = columnTransformer.fit_transform(xTrain)
xTestEnc = columnTransformer.transform(xTest)
featNames = columnTransformer.get_feature_names_out()
# display names for plots and report tables (strip cat__ / num__ prefixes)
featNamesDisplay = [f.replace("cat__", "").replace("num__", "") for f in featNames] # clean names for plots and report tables

xTrainEnc_df = pd.DataFrame(xTrainEnc, columns=featNames, index=xTrain.index)
xTestEnc_df = pd.DataFrame(xTestEnc, columns=featNames, index=xTest.index)

info(f"encoded train {xTrainEnc_df.shape} | test {xTestEnc_df.shape}")
info(f"nDummyFeatures={len(featNames)}")

encodedFeatures_df = pd.DataFrame({"feature": featNamesDisplay})
saveTable(encodedFeatures_df, "block5g_encodedFeatureNames.csv")

# optional persist for Block 6 modelling (same split, no re-draw)
xTrainEnc_df.to_csv(outputDirTables / "block5_xTrain.csv", index=False)
xTestEnc_df.to_csv(outputDirTables / "block5_xTestEnc.csv", index=False)
yTrain.to_csv(outputDirTables / "block5_yTrain.csv", index=False)
yTest.to_csv(outputDirTables / "block5_yTest.csv", index=False)
joblib.dump(columnTransformer, outputDirTables / "block5_columnTransformer.joblib")
note("encoder fitted on train only, dumped block5_columnTransformer.joblib")

logPrep(
    "5g",
    "Nominal service/contract fields (including structural 'no service' levels)",
    "block3f_categoricalFrequencies.csv, block3i_domainChecks.csv",
    "OneHotEncoder(handle_unknown='ignore') inside ColumnTransformer, fit on train only",
    len(featNames),
    "No ordinal assumption on Contract type. Structural 'No internet/phone service' levels kept as "
    "valid dummies. handle_unknown=ignore is safety net - all levels exist in this dataset.",
)

logPrep(
    "5g-scale",
    "Scaling not required for RF / gradient boosting",
    "block3j & method choice section (a)",
    "No StandardScaler on numericCols in this pipeline",
    0,
    "Tree-based models don't need scaling, so none applied. If a logistic regression baseline "
    "is added later, fit a scaler on train numerics only - keep separate from ensemble pipe.",
)

logPrep(
    "5g-imbalance-note",
    "Mild class imbalance (not a prep transform)",
    "block3e_targetBalance.csv",
    "No SMOTE/undersample in Block 5, defer class_weight / scale_pos_weight to Block 6",
    0,
    "Resampling before we have a baseline changes the playing field - weights are "
    "cleaner here and keep all 7,043 rows in play. Consistent with CW1 ensemble approach.",
)

##### 5h: Shape / closure audit (copy into section (c))
shape_df = pd.DataFrame([
    {"stage": "raw ingest", "nRows": pristineRaw_df.shape[0], "nCols": pristineRaw_df.shape[1],
     "note": "includes customerID & Churn"},
    {"stage": "after TotalCharges numeric", "nRows": len(work_df) + 0,
     "nCols": "-", "note": f"NaN leftover={nLeft}, tenure0 set to 0 (n={nBoth})"},
    {"stage": "after drop ID & isolate y", "nRows": len(X), "nCols": X.shape[1],
     "note": "features only"},
    {"stage": "train features (pre-OHE)", "nRows": len(xTrain), "nCols": xTrain.shape[1],
     "note": f"pctYes={splitMix_df.loc[splitMix_df['split']=='train','pctYes'].iloc[0]}"},
    {"stage": "test features (pre-OHE)", "nRows": len(xTest), "nCols": xTest.shape[1],
     "note": f"pctYes={splitMix_df.loc[splitMix_df['split']=='test','pctYes'].iloc[0]}"},
    {"stage": "train encoded", "nRows": xTrainEnc_df.shape[0], "nCols": xTrainEnc_df.shape[1],
     "note": "OHE & numeric passthrough"},
    {"stage": "test encoded", "nRows": xTestEnc_df.shape[0], "nCols": xTestEnc_df.shape[1],
     "note": "transform only"},
])
print("\n5h shape audit")
print(shape_df.to_string(index=False))
saveTable(shape_df, "block5h_shapeAudit.csv")

prepDecisions_df = pd.DataFrame(prepRows)
print("\n5 prep decisions (report table)")
print(prepDecisions_df.to_string(index=False))
saveTable(prepDecisions_df, "block5_prepDecisions.csv")

# unencoded modelling table (useful for tree models that accept categoricals later / audit)
work_df.to_csv(outputDirTables / "block5_work_df_preEncode.csv", index=False)

#### Observations/Interpretations:
    #  - TotalCharges: 11 blanks all on tenure==0, set to 0, zero leftover NaN confirmed
    #  - SeniorCitizen: recoded from 0/1 to No/Yes and grouped with categoricals
    #  - customerID dropped cleanly, uniqueness confirmed prior to drop
    #  - Stratified split confirmed: 26.54% Yes in full, train and test folds
    #  - OneHotEncoder (OHE) fitted on training data only, applied to test via transform ensuring no data leakage
    #  - 46 encoded features (dummies & 3 numeric passthroughs) in both matrices
    #  - No rows lost at any stage - all 7,043 rows carry through to training/test matrices
    #  - No scaling applied - not needed for tree-based methods

print("Block 5: Preprocessing complete - no estimator fitted")
note("STOP. Block 6 picks up from block5_xTrain/yTrain - RF first, then XGBoost, both with imbalance weights.")


#%% Block 6: Modelling and Model Evaluation

##### Purpose:
    #  - Two methods from section (a) and CW1: Random Forest (bagging) and XGBoost (boosting)
    #  - Consume Block 5 artefacts only, do not re-split or re-fit encoder
    #  - Baseline (defaults & imbalance weight) then GridSearchCV on TRAIN fold only
    #  - Single test evaluation per model after refit on full train with best params
    #  - Feature importances from tuned models for report interpretation

##### Strategy:
    #  - Step 6a: Load Block 5 matrices, compute scale_pos_weight from train counts
    #  - Step 6b: Random Forest, baseline then grid search (F1 inner scoring)
    #  - Step 6c: XGBoost, baseline then grid search (F1 inner scoring)
    #  - Step 6d: Comparison tables and ROC overlay (test set only)
    #  - No test set peeking during grid search
    #  - Stop tuning when gains are marginal, document decision and move on
    #  - Accuracy logged but not headlined - F1, recall, precision and ROC-AUC are primary

##### 6a: Load Block 5 matrices (frozen split)
xTrain = pd.read_csv(outputDirTables / "block5_xTrain.csv")
xTestEnc = pd.read_csv(outputDirTables / "block5_xTestEnc.csv")
yTrain = pd.read_csv(outputDirTables / "block5_yTrain.csv").squeeze()
yTest = pd.read_csv(outputDirTables / "block5_yTest.csv").squeeze()

# squeeze can leave a DataFrame if column name retained
if isinstance(yTrain, pd.DataFrame): yTrain = yTrain.iloc[:, 0]
if isinstance(yTest, pd.DataFrame): yTest = yTest.iloc[:, 0]
yTrain = yTrain.astype(int)
yTest = yTest.astype(int)

info(f"Block 6 load | train {xTrain.shape} test {xTestEnc.shape}")
info(f"train pctYes={100 * yTrain.mean():.2f} | test pctYes={100 * yTest.mean():.2f}")
if list(xTrain.columns) != list(xTestEnc.columns):
    warn("train/test feature names diverge - stop")
if xTrain.shape[1] != 46 or xTestEnc.shape[1] != 46:
    warn(f"expected 46 encoded cols from Block 5, got train={xTrain.shape[1]} test={xTestEnc.shape[1]}")

# scale_pos_weight from TRAIN only (neg/pos)
nNeg = int((yTrain == 0).sum())
nPos = int((yTrain == 1).sum())
scalePosWeight = nNeg / nPos
info(f"train nNeg={nNeg} nPos={nPos} | scale_pos_weight={scalePosWeight:.4f}")

innerCv = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState) # shuffle=True with fixed seed so fold membership is reproducible

metricRows = [] # one dict per model/stage, assembled into metrics_df in 6d
cmRows = []
rocStore = {}  # keyed by modelName (fpr, tpr, auc) for the overlay plot

### 6b: Random Forest - baseline then grid
##### Strategy:
    #  - class_weight='balanced' addresses the imbalance found in Block 3e - no resampling required
    #  - Baseline: sklearn defaults & class_weight & n_jobs
    #  - Grid kept deliberately small (time constraints). Scoring on F1 throughout to stay consistent with section (a)
    #  - n_estimators / max_depth / min_samples_leaf / max_features
    #  - If rank-1 and rank-2 are within ~0.001 F1 then the grid is flat - document and stop, don't keep searching

rfBase = RandomForestClassifier(
    class_weight="balanced",
    random_state=randomState,
    n_jobs=-1,
)
rfBase.fit(xTrain, yTrain)
rfBasePred = rfBase.predict(xTestEnc)
rfBaseProba = rfBase.predict_proba(xTestEnc)[:, 1] # column 1 = P(Yes), needed for ROC-AUC
metricRows.append(metricsRow(yTest, rfBasePred, rfBaseProba, "RandomForest", "baseline_test"))
cmRows.append(cmFrame(yTest, rfBasePred, "RandomForest", "baseline_test"))
info("RF baseline test metrics:")
print(pd.Series(metricRows[-1]).to_string())
print(classification_report(yTest, rfBasePred, target_names=["No", "Yes"], digits=3))

rfGrid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 8, 16],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.5],
}
note("RF grid size = "
     f"{np.prod([len(v) for v in rfGrid.values()])} candidates x 5 folds - stop if marginal")

rfSearch = GridSearchCV(
    estimator=clone(rfBase),
    param_grid=rfGrid,
    scoring="f1",
    cv=innerCv,
    n_jobs=-1,
    refit=True, # refit on full train with best params, this si the model used on test
    verbose=0,
)
rfSearch.fit(xTrain, yTrain)
info(f"RF best params: {rfSearch.best_params_}")
info(f"RF best inner CV F1: {rfSearch.best_score_:.4f}")

rfCv_df = pd.DataFrame(rfSearch.cv_results_)
rfGridSlim_df = rfCv_df[[
    "mean_test_score", "std_test_score", "rank_test_score",
    "param_n_estimators", "param_max_depth", "param_min_samples_leaf", "param_max_features",
]].sort_values("rank_test_score")
print("\n6b RF grid (ranked by inner F1)")
print(rfGridSlim_df.head(10).to_string(index=False))
saveTable(rfGridSlim_df, "block6b_rfGridResults.csv")

rfBest = rfSearch.best_estimator_
rfPred = rfBest.predict(xTestEnc)
rfProba = rfBest.predict_proba(xTestEnc)[:, 1]
metricRows.append(metricsRow(yTest, rfPred, rfProba, "RandomForest", "tuned_test"))
cmRows.append(cmFrame(yTest, rfPred, "RandomForest", "tuned_test"))
info("RF tuned test metrics:")
print(pd.Series(metricRows[-1]).to_string())
print(classification_report(yTest, rfPred, target_names=["No", "Yes"], digits=3))

saveCmPlot(yTest, rfBasePred, "Random Forest: Baseline (test set)",
           "block6b_rfBaseline_cm.png")
saveCmPlot(yTest, rfPred, "Figure 8: Confusion Matrix - Random Forest, tuned (test set)",
           "block6b_rfTuned_cm.png")

rfImpAll, rfImpTop = importanceFrame(
    featNamesDisplay, rfBest.feature_importances_, "RandomForest", topN=15
)
saveTable(rfImpAll, "block6b_rfImportanceAll.csv")
saveTable(rfImpTop, "block6b_rfImportanceTop15.csv")
print("\n6b RF top 15 importances (tuned)")
print(rfImpTop.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=rfImpTop, y="feature", x="importance", color=palette["neutral"], ax=ax)
ax.set_title("Figure 10: Feature Importance - Random Forest, tuned (top 15)")
plt.tight_layout()
plt.savefig(outputDirPlots / "block6b_rfImportanceTop15.png", dpi=150, bbox_inches="tight")
plt.show()
note("saved block6b_rfImportanceTop15.png")

fpr, tpr, _ = roc_curve(yTest, rfProba)
rocStore["RandomForest_tuned"] = (fpr, tpr, roc_auc_score(yTest, rfProba))

joblib.dump(rfBest, outputDirTables / "block6_rfTuned.joblib") # for future scoring without re-running the grid
note("RF tuned model persisted: block6_rfTuned.joblib")

### 6c: XGBoost - baseline then grid
##### Strategy:
    #  - scale_pos_weight calculated from training counts only - consistent with Block 5 imbalance approach
    #  - eval_metric logloss, random_state aligned to randomState constant
    #  - Grid: n_estimators, max_depth, learning_rate, subsample
    #  - Same F1 inner scoring as RF so the two grids are comparable
    #  - Note in report: boosting amplifies label noise where bagging dilutes it (Opitz & Maclin - covered in CW1)

xgbBase = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scalePosWeight,
    random_state=randomState,
    n_jobs=-1,
    tree_method="hist",
)
xgbBase.fit(xTrain, yTrain)
xgbBasePred = xgbBase.predict(xTestEnc)
xgbBaseProba = xgbBase.predict_proba(xTestEnc)[:, 1]
metricRows.append(metricsRow(yTest, xgbBasePred, xgbBaseProba, "XGBoost", "baseline_test"))
cmRows.append(cmFrame(yTest, xgbBasePred, "XGBoost", "baseline_test"))
info("XGB baseline test metrics:")
print(pd.Series(metricRows[-1]).to_string())
print(classification_report(yTest, xgbBasePred, target_names=["No", "Yes"], digits=3))

xgbGrid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 8],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}
note("XGB grid size = "
     f"{np.prod([len(v) for v in xgbGrid.values()])} candidates x 5 folds")

xgbSearch = GridSearchCV(
    estimator=clone(xgbBase),
    param_grid=xgbGrid,
    scoring="f1",
    cv=innerCv,
    n_jobs=-1,
    refit=True,
    verbose=0,
)
xgbSearch.fit(xTrain, yTrain)
info(f"XGB best params: {xgbSearch.best_params_}")
info(f"XGB best inner CV F1: {xgbSearch.best_score_:.4f}")

xgbCv_df = pd.DataFrame(xgbSearch.cv_results_)
xgbGridSlim_df = xgbCv_df[[
    "mean_test_score", "std_test_score", "rank_test_score",
    "param_n_estimators", "param_max_depth", "param_learning_rate", "param_subsample",
]].sort_values("rank_test_score")
print("\n6c XGB grid (ranked by inner F1)")
print(xgbGridSlim_df.head(10).to_string(index=False))
saveTable(xgbGridSlim_df, "block6c_xgbGridResults.csv")

xgbBest = xgbSearch.best_estimator_
xgbPred = xgbBest.predict(xTestEnc)
xgbProba = xgbBest.predict_proba(xTestEnc)[:, 1]
metricRows.append(metricsRow(yTest, xgbPred, xgbProba, "XGBoost", "tuned_test"))
cmRows.append(cmFrame(yTest, xgbPred, "XGBoost", "tuned_test"))
info("XGB tuned test metrics:")
print(pd.Series(metricRows[-1]).to_string())
print(classification_report(yTest, xgbPred, target_names=["No", "Yes"], digits=3))

saveCmPlot(yTest, xgbBasePred, "XGBoost: Baseline (test set)",
           "block6c_xgbBaseline_cm.png")
saveCmPlot(yTest, xgbPred, "Figure 9: Confusion Matrix - XGBoost, tuned (test set)",
           "block6c_xgbTuned_cm.png")

xgbImpAll, xgbImpTop = importanceFrame(
    featNamesDisplay, xgbBest.feature_importances_, "XGBoost", topN=15
)
saveTable(xgbImpAll, "block6c_xgbImportanceAll.csv")
saveTable(xgbImpTop, "block6c_xgbImportanceTop15.csv")
print("\n6c XGB top 15 importances (tuned)")
print(xgbImpTop.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=xgbImpTop, y="feature", x="importance", color=palette["churnYes"], ax=ax)
ax.set_title("Figure 11: Feature Importance - XGBoost, tuned (top 15)")
plt.tight_layout()
plt.savefig(outputDirPlots / "block6c_xgbImportanceTop15.png", dpi=150, bbox_inches="tight")
plt.show()
note("saved block6c_xgbImportanceTop15.png")

fpr, tpr, _ = roc_curve(yTest, xgbProba)
rocStore["XGBoost_tuned"] = (fpr, tpr, roc_auc_score(yTest, xgbProba))
joblib.dump(xgbBest, outputDirTables / "block6_xgbTuned.joblib")
note("XGB tuned model persisted: block6c_xgbTuned.joblib")

### 6d: Comparison tables & ROC overlay (TEST only)
metrics_df = pd.DataFrame(metricRows)
# round for the report paste
metricsReport_df = metrics_df.copy()
for c in ["accuracy", "precision", "recall", "f1", "rocAuc"]:
    metricsReport_df[c] = metricsReport_df[c].round(4)

print("\n6d test metrics comparison")
print(metricsReport_df.to_string(index=False))
saveTable(metrics_df, "block6d_testMetrics.csv")
saveTable(metricsReport_df, "block6d_testMetrics_rounded.csv")

confusionMatrix_df = pd.concat(cmRows, ignore_index=True)
print("\n6d confusion matrices")
print(confusionMatrix_df.to_string(index=False))
saveTable(confusionMatrix_df, "block6d_confusionMatrices.csv")

# tuned-only headline (the table that goes in sectrion (d))
tunedMetrics_df = metricsReport_df[metricsReport_df["stage"] == "tuned_test"].drop(columns=["stage"])
saveTable(tunedMetrics_df, "block6d_tunedHeadlines.csv")

# ROC overlay - tuned models only
fig, ax = plt.subplots(figsize=(6, 5))
for name, (fpr, tpr, aucV) in rocStore.items():
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={aucV:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="chance") # diagonal line, AUC=0.5 reference
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Figure 7: ROC Curves - Random Forest vs XGBoost (tuned models)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(outputDirPlots / "block6d_rocOverlay.png", dpi=150, bbox_inches="tight")
plt.show()
note("saved block6d_rocOverlay.png")

# best parameter summary for report section (d)
rfParams = {f"p_{k}": str(v) for k, v in rfSearch.best_params_.items()}
xgbParams = {f"p_{k}": str(v) for k, v in xgbSearch.best_params_.items()}

paramSummary_df = pd.DataFrame([
    {"model": "RandomForest", "where": "innerCV_best", **rfParams, "innerCvF1": rfSearch.best_score_},
    {"model": "XGBoost", "where": "innerCV_best", **xgbParams, "innerCvF1": xgbSearch.best_score_},
])
saveTable(paramSummary_df, "block6d_bestParams.csv")
print("\n6d best params")
print(paramSummary_df.to_string(index=False))

##### Observations/Interpretations:
    #  - RF baseline had the highest accuracy (0.789) bu tmissed nearly half the leavers (recall 0.48) confirming the
    #    accuracy trap
    #  - Tuning shifted both models toward recall: RF recall 0.77, XGB 0.79
    #  - RF and XGB essentially tie once both are tuned - F1 ~0.63, AUC ~0.84, no material winner
    #  - Both grids came back flat - rank 1 vs rank 2 less than 0.001 F1 difference. Stopping was correct call
    #  - RF importances spread across features, XGB concentrates on month-to-month (0.44)
    #  - Both models point to the same drivers found in Block 4 - contract type, tenure, charges
    #  - Default 0.5 threshold won't work in production. Any deployment needs a cost-weighted cut based on FN vs FP costs

print("Block 6: Modelling complete - test scored once per stage")
note("STOP. Draft report section (d) from block6d_testMetrics_rounded.csv, CMs, grids, importances.")
note("Do not add a third algorithm unless LR is a one-paragraph baseline with train-only scaling.")