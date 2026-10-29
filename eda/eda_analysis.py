"""
Exploratory Data Analysis (EDA) for RAG Document Q&A System
─────────────────────────────────────────────────────────────
Analyzes the provided dataset.csv to understand RAG system
performance characteristics.

HCLTech Career Shaper Programme — Industry Project
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Paths ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset.csv"
DATA_DICT_PATH = BASE_DIR / "data_dictionary.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
VIZ_DIR = OUTPUT_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style Configuration ─────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "accent": "#f093fb",
    "success": "#43e97b",
    "warning": "#FFC107",
    "palette": ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#43e97b"],
}
sns.set_palette(COLORS["palette"])


def load_data():
    """Load and return the dataset."""
    print("=" * 60)
    print("  RAG Document Q&A System — Exploratory Data Analysis")
    print("=" * 60)
    print()

    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded: {DATASET_PATH}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print()
    return df


def basic_inspection(df):
    """Perform basic data inspection."""
    print("─" * 60)
    print("1. BASIC DATA INSPECTION")
    print("─" * 60)

    print("\n📊 Data Types:")
    print(df.dtypes.to_string())

    print(f"\n📏 Dataset Shape: {df.shape}")

    print("\n📈 First 5 Rows:")
    print(df.head().to_string())

    print("\n📉 Statistical Summary:")
    print(df.describe().round(4).to_string())

    print("\n📋 Column Value Counts Overview:")
    for col in df.columns:
        nunique = df[col].nunique()
        print(f"   {col}: {nunique} unique values")

    return df


def check_data_quality(df):
    """Check for missing values, duplicates, and data issues."""
    print("\n" + "─" * 60)
    print("2. DATA QUALITY ASSESSMENT")
    print("─" * 60)

    # Missing values
    print("\n🔍 Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values found")
    else:
        for col, count in missing.items():
            if count > 0:
                pct = count / len(df) * 100
                print(f"   ⚠️  {col}: {count} missing ({pct:.1f}%)")

    # Duplicates
    print("\n🔍 Duplicate Rows:")
    dup_count = df.duplicated().sum()
    print(f"   {'⚠️  ' + str(dup_count) + ' duplicate rows found' if dup_count > 0 else '✅ No duplicate rows'}")

    dup_chunk = df.duplicated(subset=["chunk_id"]).sum()
    print(f"   Duplicate chunk_ids: {'⚠️  ' + str(dup_chunk) if dup_chunk > 0 else '✅ None'}")

    # Outlier detection using IQR for numeric columns
    print("\n🔍 Outlier Detection (IQR Method):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_info[col] = len(outliers)
        if len(outliers) > 0:
            print(f"   ⚠️  {col}: {len(outliers)} outliers (range: [{lower:.2f}, {upper:.2f}])")
        else:
            print(f"   ✅ {col}: No outliers detected")

    # Data type validation
    print("\n🔍 Data Validation:")
    if "retrieval_score" in df.columns:
        invalid_rs = df[(df["retrieval_score"] < 0) | (df["retrieval_score"] > 1)]
        print(f"   retrieval_score outside [0,1]: {'⚠️  ' + str(len(invalid_rs)) if len(invalid_rs) > 0 else '✅ All valid'}")

    if "answer_quality_score" in df.columns:
        invalid_aq = df[(df["answer_quality_score"] < 0) | (df["answer_quality_score"] > 1)]
        print(f"   answer_quality_score outside [0,1]: {'⚠️  ' + str(len(invalid_aq)) if len(invalid_aq) > 0 else '✅ All valid'}")

    if "word_count" in df.columns:
        invalid_wc = df[df["word_count"] <= 0]
        print(f"   word_count <= 0: {'⚠️  ' + str(len(invalid_wc)) if len(invalid_wc) > 0 else '✅ All valid'}")

    return outlier_info


def create_cleaned_dataset(df):
    """Create a cleaned version of the dataset without modifying the original."""
    print("\n" + "─" * 60)
    print("3. DATA CLEANING")
    print("─" * 60)

    df_clean = df.copy()
    changes = []

    # Remove exact duplicate rows
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_dups = before - len(df_clean)
    if removed_dups > 0:
        changes.append(f"Removed {removed_dups} duplicate rows")

    # Strip whitespace from string columns
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].str.strip()
        changes.append(f"Stripped whitespace from '{col}'")

    # Ensure numeric columns are proper types
    if "word_count" in df_clean.columns:
        df_clean["word_count"] = pd.to_numeric(df_clean["word_count"], errors="coerce")
    if "retrieval_score" in df_clean.columns:
        df_clean["retrieval_score"] = pd.to_numeric(df_clean["retrieval_score"], errors="coerce")
    if "answer_quality_score" in df_clean.columns:
        df_clean["answer_quality_score"] = pd.to_numeric(df_clean["answer_quality_score"], errors="coerce")

    # Drop rows where critical fields became NaN after coercion
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["doc_id", "chunk_id"])
    removed_na = before - len(df_clean)
    if removed_na > 0:
        changes.append(f"Removed {removed_na} rows with missing doc_id/chunk_id")

    # Save cleaned dataset
    cleaned_path = OUTPUT_DIR / "dataset_cleaned.csv"
    df_clean.to_csv(cleaned_path, index=False)

    print(f"\n   Original dataset: {len(df)} rows")
    print(f"   Cleaned dataset:  {len(df_clean)} rows")
    print(f"   Changes applied:")
    for c in changes:
        print(f"     • {c}")
    print(f"\n   ✅ Cleaned dataset saved to: {cleaned_path}")

    return df_clean


def create_visualizations(df):
    """Generate meaningful visualizations and save them."""
    print("\n" + "─" * 60)
    print("4. VISUALIZATIONS")
    print("─" * 60)

    # ─── Visualization 1: Document Type Distribution & Retrieval Score ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Document type distribution
    doc_type_counts = df["doc_type"].value_counts()
    wedges, texts, autotexts = axes[0].pie(
        doc_type_counts.values,
        labels=doc_type_counts.index,
        autopct="%1.1f%%",
        colors=COLORS["palette"],
        startangle=90,
        textprops={"fontsize": 10},
    )
    axes[0].set_title("Distribution of Document Types", fontsize=13, fontweight="bold", pad=15)

    # Right: Retrieval score by document type
    doc_type_order = df["doc_type"].value_counts().index
    sns.boxplot(
        data=df,
        x="doc_type",
        y="retrieval_score",
        order=doc_type_order,
        ax=axes[1],
        palette=COLORS["palette"],
    )
    axes[1].set_title("Retrieval Score by Document Type", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_xlabel("Document Type", fontsize=11)
    axes[1].set_ylabel("Retrieval Score", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path1 = VIZ_DIR / "01_doc_type_distribution_and_retrieval.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {path1}")

    # ─── Visualization 2: Query Type Analysis ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Query type counts
    query_counts = df["query_type"].value_counts()
    bars = axes[0].barh(
        query_counts.index,
        query_counts.values,
        color=COLORS["palette"][:len(query_counts)],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].set_title("Query Type Distribution", fontsize=13, fontweight="bold", pad=15)
    axes[0].set_xlabel("Count", fontsize=11)
    for bar, val in zip(bars, query_counts.values):
        axes[0].text(val + 1, bar.get_y() + bar.get_height() / 2, str(val),
                     va="center", fontsize=10, fontweight="bold")

    # Right: Answer quality by query type
    sns.violinplot(
        data=df,
        x="query_type",
        y="answer_quality_score",
        palette=COLORS["palette"],
        ax=axes[1],
        inner="quartile",
    )
    axes[1].set_title("Answer Quality by Query Type", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_xlabel("Query Type", fontsize=11)
    axes[1].set_ylabel("Answer Quality Score", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path2 = VIZ_DIR / "02_query_type_analysis.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {path2}")

    # ─── Visualization 3: Correlation & Score Distributions ──────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Scatter plot — retrieval score vs answer quality
    scatter = axes[0].scatter(
        df["retrieval_score"],
        df["answer_quality_score"],
        c=df["word_count"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="white",
        linewidth=0.3,
        s=30,
    )
    axes[0].set_xlabel("Retrieval Score", fontsize=11)
    axes[0].set_ylabel("Answer Quality Score", fontsize=11)
    axes[0].set_title("Retrieval Score vs Answer Quality\n(colored by word count)",
                      fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(scatter, ax=axes[0], label="Word Count")

    # Right: Correlation heatmap
    numeric_df = df[["word_count", "retrieval_score", "answer_quality_score"]]
    corr = numeric_df.corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".3f",
        ax=axes[1],
        linewidths=1,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    axes[1].set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    path3 = VIZ_DIR / "03_score_analysis_and_correlation.png"
    fig.savefig(path3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {path3}")

    # ─── Visualization 4: Word Count Distribution ────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram of word counts
    axes[0].hist(
        df["word_count"],
        bins=30,
        color=COLORS["primary"],
        edgecolor="white",
        alpha=0.8,
    )
    axes[0].axvline(
        df["word_count"].mean(),
        color=COLORS["accent"],
        linestyle="--",
        linewidth=2,
        label=f'Mean: {df["word_count"].mean():.0f}',
    )
    axes[0].axvline(
        df["word_count"].median(),
        color=COLORS["success"],
        linestyle="--",
        linewidth=2,
        label=f'Median: {df["word_count"].median():.0f}',
    )
    axes[0].set_title("Distribution of Chunk Word Counts", fontsize=13, fontweight="bold", pad=15)
    axes[0].set_xlabel("Word Count", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].legend(fontsize=10)

    # Right: Top documents by chunk count
    doc_chunks = df["doc_id"].value_counts().head(15)
    axes[1].barh(
        doc_chunks.index[::-1],
        doc_chunks.values[::-1],
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_title("Top 15 Documents by Chunk Count", fontsize=13, fontweight="bold", pad=15)
    axes[1].set_xlabel("Number of Chunks", fontsize=11)

    plt.tight_layout()
    path4 = VIZ_DIR / "04_word_count_and_document_chunks.png"
    fig.savefig(path4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {path4}")

    # ─── Visualization 5: Performance Quadrant Analysis ──────────────
    fig, ax = plt.subplots(figsize=(10, 7))

    # Classify chunks into quadrants
    median_ret = df["retrieval_score"].median()
    median_ans = df["answer_quality_score"].median()

    colors_map = []
    labels_map = []
    for _, row in df.iterrows():
        if row["retrieval_score"] >= median_ret and row["answer_quality_score"] >= median_ans:
            colors_map.append(COLORS["success"])
            labels_map.append("High Retrieval + High Quality")
        elif row["retrieval_score"] >= median_ret:
            colors_map.append(COLORS["warning"])
            labels_map.append("High Retrieval + Low Quality")
        elif row["answer_quality_score"] >= median_ans:
            colors_map.append("#4facfe")
            labels_map.append("Low Retrieval + High Quality")
        else:
            colors_map.append("#F44336")
            labels_map.append("Low Retrieval + Low Quality")

    ax.scatter(
        df["retrieval_score"],
        df["answer_quality_score"],
        c=colors_map,
        alpha=0.5,
        s=25,
        edgecolors="white",
        linewidth=0.3,
    )
    ax.axvline(median_ret, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(median_ans, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Retrieval Score", fontsize=12)
    ax.set_ylabel("Answer Quality Score", fontsize=12)
    ax.set_title("Performance Quadrant Analysis", fontsize=14, fontweight="bold", pad=15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["success"], label="High Ret + High Quality"),
        Patch(facecolor=COLORS["warning"], label="High Ret + Low Quality"),
        Patch(facecolor="#4facfe", label="Low Ret + High Quality"),
        Patch(facecolor="#F44336", label="Low Ret + Low Quality"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path5 = VIZ_DIR / "05_performance_quadrant.png"
    fig.savefig(path5, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   ✅ Saved: {path5}")

    return [path1, path2, path3, path4, path5]


def create_data_dictionary(df):
    """Generate a complete data dictionary with descriptions filled in."""
    print("\n" + "─" * 60)
    print("5. DATA DICTIONARY")
    print("─" * 60)

    descriptions = {
        "doc_id": (
            "Unique identifier for each source document in the corpus "
            "(e.g., DOC001, DOC002). Multiple chunks can share the same doc_id."
        ),
        "doc_type": (
            "Format/type of the source document from which the chunk was extracted. "
            "Values include: TXT (plain text), PDF (Portable Document Format), "
            "HTML (web page), Word (Microsoft Word), Markdown (Markdown format)."
        ),
        "chunk_id": (
            "Unique identifier for each text chunk created during document splitting "
            "(e.g., CHK00001). Each chunk_id maps to one specific text segment."
        ),
        "word_count": (
            "Number of words in the text chunk. Reflects chunk size after splitting. "
            f"Range: {df['word_count'].min()}–{df['word_count'].max()}, "
            f"Mean: {df['word_count'].mean():.0f}."
        ),
        "retrieval_score": (
            "Cosine similarity score between the query embedding and the chunk embedding "
            "during retrieval. Range: 0.0 (no similarity) to 1.0 (identical). "
            f"Mean: {df['retrieval_score'].mean():.4f}."
        ),
        "query_type": (
            "Category of the user query that retrieved this chunk. Values: "
            "factual (fact lookup), analytical (analysis/reasoning), "
            "comparative (comparison), procedural (how-to), summarization (summary request)."
        ),
        "answer_quality_score": (
            "Quality score of the generated answer using the retrieved chunk as context. "
            "Range: 0.0 (poor) to 1.0 (excellent). "
            f"Mean: {df['answer_quality_score'].mean():.4f}."
        ),
    }

    # Build the data dictionary DataFrame
    dict_rows = []
    for col in df.columns:
        sample_values = df[col].dropna().unique()[:3]
        dict_rows.append({
            "column_name": col,
            "data_type": str(df[col].dtype),
            "sample_value_1": str(sample_values[0]) if len(sample_values) > 0 else "",
            "sample_value_2": str(sample_values[1]) if len(sample_values) > 1 else "",
            "sample_value_3": str(sample_values[2]) if len(sample_values) > 2 else "",
            "nullable": "No" if df[col].isnull().sum() == 0 else "Yes",
            "description": descriptions.get(col, "[No description available]"),
        })

    dict_df = pd.DataFrame(dict_rows)
    dict_path = OUTPUT_DIR / "data_dictionary_completed.csv"
    dict_df.to_csv(dict_path, index=False)

    print(f"\n   ✅ Completed data dictionary saved to: {dict_path}")
    print("\n   Column Descriptions:")
    for col, desc in descriptions.items():
        print(f"   • {col}: {desc[:80]}...")

    return dict_df


def generate_eda_summary(df, outlier_info, viz_paths):
    """Generate a summary report of the EDA findings."""
    print("\n" + "─" * 60)
    print("6. EDA SUMMARY")
    print("─" * 60)

    summary = f"""
EDA Summary Report — RAG Document Q&A System
═══════════════════════════════════════════════

Dataset Overview:
  • Total records: {len(df)}
  • Total columns: {len(df.columns)}
  • Columns: {', '.join(df.columns)}

Numeric Summary:
  • word_count — Min: {df['word_count'].min()}, Max: {df['word_count'].max()}, Mean: {df['word_count'].mean():.1f}, Std: {df['word_count'].std():.1f}
  • retrieval_score — Min: {df['retrieval_score'].min()}, Max: {df['retrieval_score'].max()}, Mean: {df['retrieval_score'].mean():.4f}
  • answer_quality_score — Min: {df['answer_quality_score'].min()}, Max: {df['answer_quality_score'].max()}, Mean: {df['answer_quality_score'].mean():.4f}

Categorical Summary:
  • doc_type values: {', '.join(df['doc_type'].unique())}
  • query_type values: {', '.join(df['query_type'].unique())}
  • Unique documents: {df['doc_id'].nunique()}
  • Unique chunks: {df['chunk_id'].nunique()}

Data Quality:
  • Missing values: {df.isnull().sum().sum()}
  • Duplicate rows: {df.duplicated().sum()}
  • Outliers detected: {sum(outlier_info.values())} total across {sum(1 for v in outlier_info.values() if v > 0)} columns

Visualizations Generated:
"""
    for i, p in enumerate(viz_paths, 1):
        summary += f"  {i}. {p.name}\n"

    summary += """
Key Observations:
  1. The dataset represents metadata for a RAG system with {n_docs} unique documents.
  2. Document types are distributed across {n_types} formats: {types}.
  3. Retrieval scores range from {rs_min:.4f} to {rs_max:.4f}, suggesting varied retrieval quality.
  4. Answer quality scores show a mean of {aq_mean:.4f}, indicating generally good quality.
  5. Query types are {balanced_or_not} distributed across categories.
""".format(
        n_docs=df["doc_id"].nunique(),
        n_types=df["doc_type"].nunique(),
        types=", ".join(df["doc_type"].unique()),
        rs_min=df["retrieval_score"].min(),
        rs_max=df["retrieval_score"].max(),
        aq_mean=df["answer_quality_score"].mean(),
        balanced_or_not="relatively evenly" if df["query_type"].value_counts().std() < 15 else "unevenly",
    )

    summary_path = OUTPUT_DIR / "eda_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(summary)
    print(f"\n   ✅ Summary saved to: {summary_path}")

    return summary


def main():
    """Run the complete EDA pipeline."""
    # 1. Load data
    df = load_data()

    # 2. Basic inspection
    df = basic_inspection(df)

    # 3. Data quality
    outlier_info = check_data_quality(df)

    # 4. Clean data
    df_clean = create_cleaned_dataset(df)

    # 5. Visualizations
    viz_paths = create_visualizations(df_clean)

    # 6. Data dictionary
    create_data_dictionary(df_clean)

    # 7. Summary
    generate_eda_summary(df_clean, outlier_info, viz_paths)

    print("\n" + "=" * 60)
    print("  ✅ EDA COMPLETE — All outputs saved to eda/ directory")
    print("=" * 60)


if __name__ == "__main__":
    main()
