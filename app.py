# app.py - Advanced SAS Map Streamlit app
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, MACCSkeys
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Try to import matplotlib with fallback
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    st.warning("⚠️ matplotlib and seaborn are not available. Enhanced plots will be disabled.")

st.set_page_config(page_title="Advanced SAS Map (SALI) — Streamlit", layout="wide")
st.title("🧭 Advanced SAS Map Generator — SALI / Activity Cliffs")

# Show installation instructions if matplotlib is not available
if not MATPLOTLIB_AVAILABLE:
    st.error("""
    **Required packages missing!** 
    
    To enable all features, please install the required packages:
    
    ```bash
    pip install matplotlib seaborn
    ```
    
    For now, the app will run with basic Plotly visualizations only.
    """)

# ---------- Sidebar: Upload & Parameters ----------
st.sidebar.header("Input & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Fingerprint type selection
fingerprint_type = st.sidebar.selectbox(
    "Fingerprint Type", 
    ["ECFP4", "ECFP6", "MACCS"], 
    index=0
)

# Conditional parameters based on fingerprint type
if fingerprint_type.startswith("ECFP"):
    radius = st.sidebar.slider("Morgan radius", 1, 4, 2 if fingerprint_type == "ECFP4" else 3)
    n_bits = st.sidebar.selectbox("Fingerprint size (bits)", [512, 1024, 2048], index=2)
else:  # MACCS
    radius = None
    n_bits = 167  # MACCS has fixed size

color_by = st.sidebar.selectbox("Color by", ["SALI", "MaxActivity"])
max_pairs_plot = st.sidebar.number_input("Max pairs to plot", min_value=2000, max_value=200000, value=10000, step=1000)

# Enhanced visualization parameters (only show if matplotlib is available)
if MATPLOTLIB_AVAILABLE:
    st.sidebar.header("Enhanced Visualization")
    similarity_threshold = st.sidebar.slider("Similarity threshold", 0.1, 0.9, 0.5, 0.05)
    activity_threshold = st.sidebar.slider("Activity threshold", 0.1, 5.0, 1.0, 0.1)
    show_classification = st.sidebar.checkbox("Show pair classification", value=True)
    enhanced_plots = st.sidebar.checkbox("Enhanced matplotlib plots", value=True)
else:
    similarity_threshold = 0.5
    activity_threshold = 1.0
    show_classification = False
    enhanced_plots = False

# ---------- Functions ----------
def compute_fingerprints(smiles_list, fp_type, radius, n_bits):
    """Compute fingerprints for a list of SMILES"""
    fps = []
    valid_idx = []
    invalid_smiles = []
    
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            invalid_smiles.append(s)
            continue
            
        try:
            if fp_type == "ECFP4":
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=n_bits)
            elif fp_type == "ECFP6":
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=n_bits)
            elif fp_type == "MACCS":
                fp = MACCSkeys.GenMACCSKeys(m)
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)
            
            fps.append(fp)
            valid_idx.append(i)
        except Exception as e:
            invalid_smiles.append(f"{s} (error: {str(e)})")
            continue
    
    return fps, valid_idx, invalid_smiles

def compute_similarity_matrix(fps):
    """Compute pairwise Tanimoto similarity matrix"""
    n = len(fps)
    sim_matrix = np.zeros((n, n), dtype=float)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                try:
                    s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sim_matrix[i, j] = s
                    sim_matrix[j, i] = s
                except Exception:
                    sim_matrix[i, j] = 0.0
                    sim_matrix[j, i] = 0.0
    
    return sim_matrix

def create_enhanced_matplotlib_plot(pairs_df, color_by, similarity_threshold, activity_threshold):
    """Create enhanced matplotlib plot with professional styling"""
    if not MATPLOTLIB_AVAILABLE:
        return None
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for plotting
    plot_data = pairs_df.copy()
    
    # Color mapping
    if color_by == 'SALI':
        colors = plot_data['SALI']
        cmap = plt.cm.RdYlBu_r
        label = 'SALI'
    else:  # MaxActivity
        colors = plot_data['MaxActivity']
        cmap = 'viridis'
        label = 'Maximum Activity'
    
    # Create scatter plot
    sc = ax.scatter(plot_data['Similarity'], plot_data['Activity_Diff'],
                   c=colors, cmap=cmap, alpha=0.7, s=30)
    
    # Add threshold lines
    ax.axvline(x=similarity_threshold, color='red', linestyle='--', alpha=0.8, 
               label=f'Similarity threshold = {similarity_threshold}')
    ax.axhline(y=activity_threshold, color='blue', linestyle='--', alpha=0.8, 
               label=f'Activity threshold = {activity_threshold}')
    
    # Labels and title
    ax.set_xlabel('Structural Similarity (Tanimoto)')
    ax.set_ylabel('Activity Difference')
    ax.set_title(f'Enhanced SAS Map - Colored by {label}')
    
    # Colorbar
    plt.colorbar(sc, ax=ax, label=label)
    
    # Legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def classify_compound_pairs(pairs_df, similarity_threshold, activity_threshold):
    """Classify compound pairs into different categories"""
    data = pairs_df
    
    classifications = {
        'activity_cliffs': data[(data['Similarity'] > similarity_threshold) & 
                               (data['Activity_Diff'] > activity_threshold)],
        'smooth_sar': data[(data['Similarity'] > similarity_threshold) & 
                          (data['Activity_Diff'] <= activity_threshold)],
        'scaffold_hopping': data[(data['Similarity'] <= similarity_threshold) & 
                               (data['Activity_Diff'] <= activity_threshold)],
        'activity_gaps': data[(data['Similarity'] <= similarity_threshold) & 
                            (data['Activity_Diff'] > activity_threshold)]
    }
    
    return classifications

# ---------- Main UI ----------
if uploaded_file is None:
    st.info("📁 Upload a CSV file containing columns: SMILES and an activity (e.g. pIC50).")
    st.stop()

# Read file
try:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Molecules", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

# Column selectors
cols = list(df.columns)
smiles_col = st.selectbox("SMILES column", cols, index=0)
activity_col = st.selectbox("Activity column", cols, index=1 if len(cols)>1 else 0)
id_col_opt = ["None"] + cols
id_col = st.selectbox("Optional ID column", id_col_opt, index=0)

# Generate button
if st.button("🚀 Generate SAS map and analyze"):
    st.info("Processing — this may take a while for large datasets.")
    
    # Basic filter and validation
    df_clean = df.dropna(subset=[smiles_col, activity_col]).copy()
    if len(df_clean) == 0:
        st.error("No valid data after removing rows with missing SMILES or activity values.")
        st.stop()
        
    # Parse activities
    try:
        activities = df_clean[activity_col].astype(float).values
    except Exception as e:
        st.error(f"Activity column conversion to float failed: {e}")
        st.stop()

    ids = df_clean[id_col].astype(str).values if id_col != "None" else np.array([f"Mol_{i+1}" for i in range(len(df_clean))])
    smiles_list = df_clean[smiles_col].astype(str).values

    # Step 1: Compute fingerprints
    st.write(f"### Step 1: Computing {fingerprint_type} fingerprints...")
    fps, valid_idx, invalid_smiles = compute_fingerprints(smiles_list, fingerprint_type, radius, n_bits)
    
    if invalid_smiles:
        st.warning(f"{len(invalid_smiles)} invalid SMILES found and excluded.")
        with st.expander("Show invalid SMILES"):
            for bad_smiles in invalid_smiles[:10]:
                st.write(bad_smiles)
            if len(invalid_smiles) > 10:
                st.write(f"... and {len(invalid_smiles) - 10} more")
    
    # Keep only valid entries
    activities = activities[valid_idx]
    ids = ids[valid_idx]
    smiles_list = smiles_list[valid_idx]
    n = len(fps)
    
    if n < 2:
        st.error("Need at least 2 valid molecules to compute pairs.")
        st.stop()

    st.success(f"✅ Fingerprints computed for {n} molecules using {fingerprint_type}.")

    # Step 2: Compute similarity matrix
    st.write("### Step 2: Computing pairwise Tanimoto similarities...")
    sim_matrix = compute_similarity_matrix(fps)
    st.success(f"✅ Similarity matrix computed for {n} molecules.")

    # Step 3: Build pairs and compute SALI
    st.write("### Step 3: Building pair list and computing SALI...")
    pairs = []
    eps_distance = 1e-2
    
    total_pairs = n * (n - 1) // 2
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    pair_count = 0
    for i in range(n):
        for j in range(i+1, n):
            if pair_count % 1000 == 0:
                progress_text.text(f"Processing pairs: {pair_count:,}/{total_pairs:,}")
                progress_bar.progress(min(pair_count / total_pairs, 1.0))
            
            sim = sim_matrix[i, j]
            act_diff = float(abs(activities[i] - activities[j]))
            max_val = float(max(activities[i], activities[j]))
            distance = max(1.0 - sim, eps_distance)
            sali = act_diff / distance
            
            pairs.append({
                "Mol1_idx": i, "Mol2_idx": j,
                "Mol1_ID": ids[i], "Mol2_ID": ids[j],
                "SMILES1": smiles_list[i], "SMILES2": smiles_list[j],
                "Activity1": activities[i], "Activity2": activities[j],
                "Similarity": sim, "Activity_Diff": act_diff,
                "MaxActivity": max_val, "SALI": sali
            })
            pair_count += 1
    
    progress_text.empty()
    progress_bar.empty()
    
    if not pairs:
        st.error("No valid pairs were generated. Check your data.")
        st.stop()
        
    pairs_df = pd.DataFrame(pairs)
    st.success(f"✅ Created {len(pairs_df):,} molecular pairs.")

    # ---------- RESULTS VISUALIZATION ----------
    st.markdown("---")
    st.header("📊 Results Visualization")
    
    # Create tabs for different visualizations
    if MATPLOTLIB_AVAILABLE and enhanced_plots:
        tabs = st.tabs(["SAS Map (Plotly)", "Enhanced SAS Map", "Statistics"])
    else:
        tabs = st.tabs(["SAS Map", "Statistics"])
    
    with tabs[0]:  # SAS Map tab
        st.subheader("SAS Activity Landscape Map")
        
        # Optionally subsample for plotting
        plot_df = pairs_df
        if len(pairs_df) > max_pairs_plot:
            st.warning(f"Too many pairs ({len(pairs_df):,}) — subsampling {max_pairs_plot:,} for plotting.")
            plot_df = pairs_df.sample(n=max_pairs_plot, random_state=42)

        # Create the plot
        fig = px.scatter(
            plot_df,
            x="Similarity",
            y="Activity_Diff",
            color=color_by,
            opacity=0.7,
            hover_data=["Mol1_ID", "Mol2_ID", "Similarity", "Activity_Diff", "SALI"],
            title=f"SAS Map ({fingerprint_type}) — colored by {color_by}",
            width=1000,
            height=650,
        )
        fig.update_traces(marker=dict(size=8))
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics for the plot
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Pairs Plotted", len(plot_df))
        with col2:
            st.metric("Average Similarity", f"{plot_df['Similarity'].mean():.3f}")
        with col3:
            st.metric("Average Activity Diff", f"{plot_df['Activity_Diff'].mean():.3f}")
        with col4:
            st.metric("Max SALI", f"{plot_df['SALI'].max():.3f}")
    
    # Enhanced matplotlib plot tab
    if MATPLOTLIB_AVAILABLE and enhanced_plots and len(tabs) > 1:
        with tabs[1]:
            st.subheader("Enhanced SAS Map (Matplotlib)")
            
            fig = create_enhanced_matplotlib_plot(
                pairs_df, color_by, similarity_threshold, activity_threshold
            )
            if fig:
                st.pyplot(fig)
                
                # Add download button for the matplotlib plot
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                st.download_button(
                    label="📥 Download Enhanced SAS Map (PNG)",
                    data=buf,
                    file_name=f"enhanced_sas_map_{fingerprint_type}.png",
                    mime="image/png"
                )
    
    # Statistics tab
    stats_tab_index = 1 if not (MATPLOTLIB_AVAILABLE and enhanced_plots) else 2
    with tabs[stats_tab_index]:
        st.subheader("Statistical Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # SALI distribution
            fig_sali = px.histogram(pairs_df, x="SALI", nbins=50, 
                                  title="SALI Distribution")
            st.plotly_chart(fig_sali, use_container_width=True)
            
            # Similarity distribution
            fig_sim = px.histogram(pairs_df, x="Similarity", nbins=50,
                                 title="Similarity Distribution")
            st.plotly_chart(fig_sim, use_container_width=True)
        
        with col2:
            # Activity difference distribution
            fig_act = px.histogram(pairs_df, x="Activity_Diff", nbins=50,
                                 title="Activity Difference Distribution")
            st.plotly_chart(fig_act, use_container_width=True)
            
            # Summary statistics table
            st.subheader("Summary Statistics")
            stats_df = pairs_df[['Similarity', 'Activity_Diff', 'SALI']].describe()
            st.dataframe(stats_df, use_container_width=True)

    # Pair Classification Section (if enabled)
    if show_classification and MATPLOTLIB_AVAILABLE:
        st.markdown("---")
        st.header("🔍 Compound Pair Classification")
        
        classifications = classify_compound_pairs(pairs_df, similarity_threshold, activity_threshold)
        
        # Display classification results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Activity Cliffs", 
                     f"{len(classifications['activity_cliffs']):,}",
                     help="High similarity, high activity difference")
        
        with col2:
            st.metric("Smooth SAR", 
                     f"{len(classifications['smooth_sar']):,}",
                     help="High similarity, low activity difference")
        
        with col3:
            st.metric("Scaffold Hopping", 
                     f"{len(classifications['scaffold_hopping']):,}",
                     help="Low similarity, low activity difference")
        
        with col4:
            st.metric("Activity Gaps", 
                     f"{len(classifications['activity_gaps']):,}",
                     help="Low similarity, high activity difference")

    # ---------- DOWNLOAD SECTION ----------
    st.markdown("---")
    st.header("📥 Download Results")
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Full pairs data as CSV
        csv_data = pairs_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Pairs (CSV)",
            data=csv_data,
            file_name=f"SAS_pairs_full_{fingerprint_type}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Top 100 cliffs only
        top_cliffs = pairs_df.nlargest(100, "SALI")
        top_cliffs_csv = top_cliffs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Top 100 Cliffs (CSV)",
            data=top_cliffs_csv,
            file_name=f"top_100_cliffs_{fingerprint_type}.csv",
            mime="text/csv"
        )

    st.success("🎉 Analysis complete! Use the download buttons above to save your results.")
