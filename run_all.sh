#!/bin/bash
set -e

echo "========================================="
echo "Running FIX 1 & 2 Execution Sequence"
echo "========================================="

cd /Users/yashpatil/Local_Mark1/alpha-core
source venv/bin/activate

# 1. Back up old regime labels and outputs
echo "Backing up old outputs..."
cp data/regime_labels.csv data/regime_labels_old.csv
mkdir -p paper_analysis/outputs_old
cp paper_analysis/outputs/*.csv paper_analysis/outputs_old/ 2>/dev/null || true
cp paper_analysis/outputs/*.txt paper_analysis/outputs_old/ 2>/dev/null || true

# 2. Re-run HMM (Fix 1)
echo "Running HMM Regeneration (Fix 1)..."
export PYTHONPATH="/Users/yashpatil/Local_Mark1/alpha-core"
python alpha_core/hmm_regime.py

# 3. Re-run Paper Analysis (Fix 2)
echo "Running Table 1 & A1 generator (IIMA factors)..."
python paper_analysis/download_iima_factors.py
echo "Running Table 2 generator..."
python paper_analysis/table2_regime_factor_matrix.py

echo "Running Table 3 generator..."
python paper_analysis/table3_fii_factor_regimes.py

echo "Running Table DCC Stress generator..."
cd /Users/yashpatil/Local_Mark1/alpha-core
python paper_analysis/table_dcc_stress.py

echo "Running Table 4 Strategy Comparison generator..."
python paper_analysis/table4_strategy_comparison.py

# 4. Generate Diffs
echo "Generating Diffs..."
cd /Users/yashpatil/Local_Mark1/alpha-core
export PYTHONPATH="/Users/yashpatil/Local_Mark1/alpha-core"
python paper_analysis/generate_paper_diffs.py

echo "========================================="
echo "Done! Check paper_analysis/outputs/paper_number_diffs.csv"
echo "========================================="
