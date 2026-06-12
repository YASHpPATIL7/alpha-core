import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent.parent
OUT_DIR = BASE_DIR / "paper_analysis" / "outputs"
OLD_DIR = BASE_DIR / "paper_analysis" / "outputs_old"

def safe_read(path, old=False):
    p = OLD_DIR / path.name if old else path
    if not p.exists():
        return None
    try:
        # Table 3 A, B, D etc usually have headers. Some might have indexes.
        return pd.read_csv(p)
    except Exception as e:
        print(f"Error reading {p}: {e}")
        return None

def main():
    diffs = []
    
    def add_diff(table, metric, old_v, new_v):
        diffs.append({
            "table": table,
            "metric": metric,
            "old_value": old_v,
            "new_value": new_v
        })

    # 1. Regime Day Counts (from table3_fii_descriptives.csv or regime_labels.csv)
    # Let's read regime_labels.csv directly
    rl_new = BASE_DIR / "data" / "regime_labels.csv"
    rl_old = BASE_DIR / "data" / "regime_labels_old.csv"
    if rl_new.exists() and rl_old.exists():
        df_new = pd.read_csv(rl_new)
        df_old = pd.read_csv(rl_old)
        
        counts_new = df_new["regime_name"].value_counts().to_dict()
        counts_old = df_old["regime_name"].value_counts().to_dict()
        
        for reg in ["Bull", "Sideways", "Bear"]:
            old_c = counts_old.get(reg, 0)
            new_c = counts_new.get(reg, 0)
            add_diff("Regime Counts", f"{reg} days", old_c, new_c)
            
        # Also compute % changed
        merged = df_old.merge(df_new, on="Date", suffixes=('_old', '_new'))
        changed = (merged["regime_name_old"] != merged["regime_name_new"]).sum()
        pct_changed = (changed / len(merged)) * 100
        add_diff("Regime Labels", "% Changed", "0.0", f"{pct_changed:.2f}%")
        
        # Check COVID Crash (2020-03-23)
        if "2020-03-23" in df_new["Date"].values:
            val = df_new.loc[df_new["Date"] == "2020-03-23", "regime_name"].iloc[0]
            add_diff("Sanity Check", "COVID Crash (2020-03-23)", "Bear", val)

    # 2. Table 2 (Regime Factor Matrix)
    t2_new = safe_read(OUT_DIR / "table2_regime_factor_matrix.csv", False)
    t2_old = safe_read(OUT_DIR / "table2_regime_factor_matrix.csv", True)
    if t2_new is not None and t2_old is not None:
        if "regime" in t2_new.columns and "factor" in t2_new.columns:
            for idx, row in t2_new.iterrows():
                reg = row["regime"]
                fac = row["factor"]
                old_row = t2_old[(t2_old["regime"] == reg) & (t2_old["factor"] == fac)]
                if not old_row.empty:
                    for col in ["ann_ret", "sr", "t_stat", "p_value"]:
                        if col in row and col in old_row.columns:
                            add_diff("Table 2", f"{reg}x{fac} {col}", old_row.iloc[0][col], row[col])

    # 3. Panel A F/p
    pa_new = safe_read(OUT_DIR / "table3_panel_a.csv", False)
    pa_old = safe_read(OUT_DIR / "table3_panel_a.csv", True)
    if pa_new is not None and pa_old is not None:
        for idx, row in pa_new.iterrows():
            reg = row["regime"]
            old_row = pa_old[pa_old["regime"] == reg]
            if not old_row.empty:
                for col in ["f_stat", "p_value"]:
                    add_diff("Panel A", f"{reg} {col}", old_row.iloc[0][col], row[col])

    # 4. Panel B all 15 cells raw + FDR
    pb_new = safe_read(OUT_DIR / "table3_panel_b.csv", False)
    pb_old = safe_read(OUT_DIR / "table3_panel_b.csv", True)
    if pb_new is not None and pb_old is not None:
        for idx, row in pb_new.iterrows():
            reg = row["regime"]
            fac = row["factor"]
            old_row = pb_old[(pb_old["regime"] == reg) & (pb_old["factor"] == fac)]
            if not old_row.empty:
                for col in ["f_stat", "p_value", "p_fdr", "fdr_reject"]:
                    if col in row and col in old_row.columns:
                        add_diff("Panel B", f"{reg}x{fac} {col}", old_row.iloc[0][col], row[col])
                        if col == "fdr_reject" and row[col] == True:
                            print(f"FLAG: {reg}x{fac} SURVIVED FDR!")

    # 5. Panel C AUC
    # Read table3_auc.txt directly
    auc_new = OUT_DIR / "table3_auc.txt"
    auc_old = OLD_DIR / "table3_auc.txt"
    if auc_new.exists() and auc_old.exists():
        add_diff("Panel C", "AUC", auc_old.read_text().strip(), auc_new.read_text().strip())

    # 6. Panel D Bear F/p
    pd_new = safe_read(OUT_DIR / "table3_panel_d.csv", False)
    pd_old = safe_read(OUT_DIR / "table3_panel_d.csv", True)
    if pd_new is not None and pd_old is not None:
        for idx, row in pd_new.iterrows():
            if row["regime"] == "Bear":
                pred = row["predictor"]
                old_row = pd_old[(pd_old["regime"] == "Bear") & (pd_old["predictor"] == pred)]
                if not old_row.empty:
                    add_diff("Panel D", f"Bear {pred} f_stat", old_row.iloc[0]["f_stat"], row["f_stat"])
                    add_diff("Panel D", f"Bear {pred} p_value", old_row.iloc[0]["p_value"], row["p_value"])

    # 7. DCC Bear/Bull means + t
    dcc_new = safe_read(OUT_DIR / "dcc_panel_b.csv", False)
    dcc_old = safe_read(OUT_DIR / "dcc_panel_b.csv", True)
    if dcc_new is not None and dcc_old is not None:
        # Columns might be different depending on dcc generator
        for idx, row in dcc_new.iterrows():
            if "regime" in row:
                reg = row["regime"]
                old_row = dcc_old[dcc_old["regime"] == reg]
                if not old_row.empty and "mean_corr" in row:
                    add_diff("DCC", f"{reg} mean_corr", old_row.iloc[0]["mean_corr"], row["mean_corr"])
        
        # t-stat is usually in dcc_panel_c.csv
        dc_new = safe_read(OUT_DIR / "dcc_panel_c.csv", False)
        dc_old = safe_read(OUT_DIR / "dcc_panel_c.csv", True)
        if dc_new is not None and dc_old is not None:
            for idx, row in dc_new.iterrows():
                pair = row.get("pair", str(idx))
                old_row = dc_old[dc_old.index == idx] if "pair" not in dc_old else dc_old[dc_old["pair"] == pair]
                if not old_row.empty:
                    for col in ["t_stat", "p_value"]:
                        if col in row:
                            add_diff("DCC", f"Test {pair} {col}", old_row.iloc[0][col], row[col])

    # 8. Table 4 Strategy Comparison
    t4_new = safe_read(OUT_DIR / "table4_strategy_comparison.csv", False)
    t4_old = safe_read(OUT_DIR / "table4_strategy_comparison.csv", True)
    if t4_new is not None and t4_old is not None:
        for idx, row in t4_new.iterrows():
            strat = row.get("Strategy", str(idx))
            old_row = t4_old[t4_old.index == idx] if "Strategy" not in t4_old else t4_old[t4_old["Strategy"] == strat]
            if not old_row.empty:
                for col in ["Ann Ret %", "Sharpe", "MaxDD %", "t-stat"]:
                    if col in row:
                        add_diff("Table 4", f"{strat} {col}", old_row.iloc[0][col], row[col])

    # 9. A4 ex-COVID values
    pbc_new = safe_read(OUT_DIR / "table3_panel_b_covid_robustness.csv", False)
    pbc_old = safe_read(OUT_DIR / "table3_panel_b_covid_robustness.csv", True)
    if pbc_new is not None and pbc_old is not None:
        for idx, row in pbc_new.iterrows():
            reg = row["regime"]
            fac = row["factor"]
            old_row = pbc_old[(pbc_old["regime"] == reg) & (pbc_old["factor"] == fac)]
            if not old_row.empty:
                for col in ["f_stat_excovid", "p_value_excovid", "fdr_reject_excovid"]:
                    if col in row:
                        add_diff("A4 Panel B", f"{reg}x{fac} {col}", old_row.iloc[0][col], row[col])
                        if col == "fdr_reject_excovid" and row[col] == True:
                            print(f"FLAG: {reg}x{fac} SURVIVED FDR EX-COVID!")

    diff_df = pd.DataFrame(diffs)
    out_path = OUT_DIR / "paper_number_diffs.csv"
    diff_df.to_csv(out_path, index=False)
    print(f"Diffs saved to {out_path}")

if __name__ == "__main__":
    main()
