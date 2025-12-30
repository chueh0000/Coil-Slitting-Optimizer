import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="切割優化器", layout="wide")

st.title("✂️ 切割優化器")

# --- SIDEBAR: GLOBAL CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ 原料設定")
    MASTER_WIDTH = st.number_input("寬度 (mm)", value=1230.0, step=10.0)
    THICKNESS = st.number_input("厚度 (mm)", value=1.0, step=0.1)
    # User inputs kg/m^3, we convert to kg/mm^3 internally
    density_input = st.number_input("密度 (kg/m³)", value=7930.0, step=10.0)
    DENSITY_KG_MM3 = density_input * 1e-9 
    KERF = st.number_input("每次裁切損失寬度 (mm)", value=1.0, step=0.1)

# --- MAIN SECTION: ORDER INPUT ---
st.header("1. 訂單")
st.info("請編輯下方的訂單清單。")

# Default data matching your script
default_data = [
    {"編號": "A", "寬度 (mm)": 124.0, "重量 (kg)": 1500},
    {"編號": "B", "寬度 (mm)": 77.6,  "重量 (kg)": 2200},
    {"編號": "C", "寬度 (mm)": 68.8,  "重量 (kg)": 1800},
    {"編號": "D", "寬度 (mm)": 115.0, "重量 (kg)": 2500},
    {"編號": "E", "寬度 (mm)": 164.0, "重量 (kg)": 2800},
    {"編號": "F", "寬度 (mm)": 55.3,  "重量 (kg)": 4000},
]

df_input = pd.DataFrame(default_data)
edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)

# --- CALCULATION LOGIC ---
def solve_cutting_stock(orders_df):
    # Prepare Data
    widths = orders_df["寬度 (mm)"].values
    weights = orders_df["重量 (kg)"].values
    ids = orders_df["編號"].values
    
    # Validation
    if any(w + KERF > MASTER_WIDTH for w in widths):
        st.error("錯誤: 訂單寬度大於原料寬度!")
        return None

    # Calculate Demands (Length required for each ID)
    demands_length = weights / (widths * THICKNESS * DENSITY_KG_MM3)
    n_items = len(widths)

    # Initial Patterns (Identity Matrix approach)
    patterns = []
    for i in range(n_items):
        pat = [0] * n_items
        max_count = int((MASTER_WIDTH + KERF) / (widths[i] + KERF))
        pat[i] = max_count
        patterns.append(pat)
    
    patterns = np.array(patterns).T

    # --- COLUMN GENERATION LOOP ---
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    iter_count = 0
    max_iter = 100 # Safety break
    
    while iter_count < max_iter:
        iter_count += 1
        status_text.text(f"迭代次數: {iter_count}")
        progress_bar.progress(min(iter_count * 2, 100))
        
        n_patterns = patterns.shape[1]
        
        # 1. Restricted Master Problem
        c = np.ones(n_patterns)
        A_ub = -patterns
        b_ub = -demands_length
        
        # Solve Master LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        
        if not res.success:
            st.error("優化失敗，請檢查輸入資料。")
            return None
            
        duals = -res.ineqlin.marginals
        
        # 2. Sub-Problem (Knapsack)
        # Minimize reduced cost: 1 - sum(duals * count)
        # We use MILP to solve the Knapsack problem
        
        capacity_eff = MASTER_WIDTH + KERF
        weights_eff = [w + KERF for w in widths]
        
        sub_c = -duals # Maximize value
        sub_A = [weights_eff]
        sub_b = [capacity_eff]
        
        sub_res = linprog(sub_c, A_ub=sub_A, b_ub=sub_b, bounds=(0, None), integrality=1)
        
        best_pattern = np.round(sub_res.x).astype(int)
        reduced_cost = 1 + sub_res.fun
        
        if reduced_cost >= -1e-5:
            break # Optimal found
            
        # Add new pattern
        patterns = np.column_stack((patterns, best_pattern))

    progress_bar.empty()
    status_text.empty()
    
    return res.x, patterns, demands_length, widths, ids

# --- EXECUTION BUTTON ---
if st.button("🚀 開始優化"):
    if edited_df.empty:
        st.warning("請輸入至少一筆訂單資料。")
    else:
        with st.spinner("優化中，請稍候..."):
            result = solve_cutting_stock(edited_df)
            
        if result:
            final_run_lengths, final_patterns, demands, item_widths, item_ids = result
            
            st.divider()
            st.header("2. 優化結果")
            
            # --- SUMMARY METRICS ---
            total_master_length = sum(final_run_lengths)
            
            # Waste Calculation
            total_used_mass = total_master_length * MASTER_WIDTH * THICKNESS * DENSITY_KG_MM3
            total_order_mass = edited_df["重量 (kg)"].sum()
            waste_mass = total_used_mass - total_order_mass
            waste_pct = (waste_mass / total_used_mass) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("總原料長度需求", f"{total_master_length/1000:,.2f} m")
            col2.metric("總處理重量", f"{total_used_mass:,.0f} kg")
            col3.metric("預估廢料", f"{waste_pct:.2f}%", delta_color="inverse")
            
            # --- DETAILED PATTERN TABLE ---
            st.subheader("切割方案明細")
            
            results_list = []
            
            for i, length in enumerate(final_run_lengths):
                if length > 0.1:  # Filter out unused patterns
                    pat_col = final_patterns[:, i]
                    
                    # Create a readable string for the mix
                    mix_str = []
                    used_width = 0
                    
                    # For visualization logic
                    viz_widths = []
                    viz_labels = []
                    
                    for j, count in enumerate(pat_col):
                        if count > 0:
                            mix_str.append(f"{item_ids[j]}: {int(count)}個")
                            used_width += count * (item_widths[j] + KERF)
                            # Add to viz lists
                            for _ in range(int(count)):
                                viz_widths.append(item_widths[j])
                                viz_labels.append(item_ids[j])
                    
                    # Add Kerf adjustment for final usage calculation (remove last kerf)
                    used_width -= KERF 
                    
                    results_list.append({
                        # "Pattern ID": f"P{i+1}",
                        "原料長度 (m)": length / 1000,
                        "配置": ", ".join(mix_str),
                        "寬度利用率 (%)": (used_width / MASTER_WIDTH) * 100
                    })

            results_df = pd.DataFrame(results_list)
            st.dataframe(
                results_df.style.format({
                    "原料長度 (m)": "{:.2f}", 
                    "寬度利用率 (%)": "{:.3f}%"
                }), 
                use_container_width=True
            )
