"""
Cross-Algorithm Comparison and Aggregator
=========================================
Reads the optimization Excel results (*_Results.xlsx) from results/excel/,
extracts key statistical metrics (Min, Mean, Median, Std Dev fitness),
generates a comprehensive summary leaderboard (console, CSV, styled Excel),
and plots an overlaid convergence curve comparing the best runs of all algorithms.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side


def load_algorithm_data(excel_path):
    """
    Extracts summary statistics and the best-run convergence curve from a single
    algorithm results workbook.
    """
    wb = openpyxl.load_workbook(excel_path, data_only=True)
    filename = os.path.basename(excel_path)
    default_algo = (filename.replace('_mtdm_Results.xlsx', '')
                            .replace('_tdm_Results.xlsx', '')
                            .replace('_Results.xlsx', '')
                            .replace('.xlsx', ''))

    algo_name = default_algo
    min_score = None
    mean_score = None
    median_score = None
    std_score = None
    best_run = 1

    # 1. Try extracting from Statistics sheet
    if 'Statistics' in wb.sheetnames:
        ws_stat = wb['Statistics']
        for row in ws_stat.iter_rows(values_only=True):
            if not row or len(row) < 2 or row[0] is None:
                continue
            k, v = str(row[0]).strip(), row[1]
            if k == 'Algorithm' and v:
                algo_name = str(v)
            elif 'Min (Best)' in k and v is not None:
                min_score = float(v)
            elif 'Mean' in k and 'Time' not in k and mean_score is None and v is not None:
                mean_score = float(v)
            elif 'Median' in k and 'Time' not in k and median_score is None and v is not None:
                median_score = float(v)
            elif 'Std Dev' in k and 'Time' not in k and std_score is None and v is not None:
                std_score = float(v)
            elif 'Best Run #' in k and v is not None:
                try:
                    best_run = int(v)
                except Exception:
                    pass

    # 2. Fall back to Summary sheet if any metric was missing
    if 'Summary' in wb.sheetnames:
        ws_sum = wb['Summary']
        rows = list(ws_sum.iter_rows(values_only=True))
        if len(rows) > 1:
            headers = [str(h).strip() if h else '' for h in rows[0]]
            data = rows[1:]
            try:
                # Find fitness column (either 'Best Fitness' or 'RMSE')
                fit_col = None
                for candidate in ['Best Fitness', 'RMSE', 'Fitness', 'Score']:
                    if candidate in headers:
                        fit_col = headers.index(candidate)
                        break

                if fit_col is not None:
                    scores = [float(r[fit_col]) for r in data if r[fit_col] is not None]
                    if scores:
                        if min_score is None: min_score = float(np.min(scores))
                        if mean_score is None: mean_score = float(np.mean(scores))
                        if median_score is None: median_score = float(np.median(scores))
                        if std_score is None: std_score = float(np.std(scores))
                        best_run = int(np.argmin(scores)) + 1
            except Exception:
                pass

    # 3. Extract best convergence curve
    best_curve = []
    if 'Convergence_All' in wb.sheetnames:
        ws_conv = wb['Convergence_All']
        best_col_idx = best_run + 1
        for row in ws_conv.iter_rows(min_row=2, min_col=best_col_idx, max_col=best_col_idx, values_only=True):
            if row and row[0] is not None:
                try:
                    best_curve.append(float(row[0]))
                except Exception:
                    pass

    return {
        'Algorithm': default_algo,
        'Full_Name': algo_name,
        'Min_Fitness': min_score,
        'Mean_Fitness': mean_score,
        'Median_Fitness': median_score,
        'Std_Fitness': std_score,
        'Best_Run': best_run,
        'Best_Curve': np.array(best_curve, dtype=float),
    }


def compare_algorithms(excel_dir="results/excel", figures_dir="results/figures"):
    excel_files = sorted(glob.glob(os.path.join(excel_dir, "*_Results.xlsx")))
    excel_files = [
        f for f in excel_files
        if not os.path.basename(f).startswith("Problem_Results")
        and not os.path.basename(f).startswith("Algorithm_Comparison_Summary")
    ]
    if not excel_files:
        excel_files = sorted(glob.glob(os.path.join(excel_dir, "*.xlsx")))
        excel_files = [
            f for f in excel_files
            if not os.path.basename(f).startswith("Problem_Results")
            and not os.path.basename(f).startswith("Algorithm_Comparison_Summary")
        ]

    if not excel_files:
        print(f"[!] No algorithm Excel results found in '{excel_dir}/'.")
        print("    Please run algorithms first (e.g., 'python run_all_algorithms.py --smoke').")
        return None

    print("=" * 80)
    print(f"  AGGREGATING RESULTS ACROSS {len(excel_files)} ALGORITHMS")
    print("=" * 80)

    records = []
    curves_dict = {}

    for path in excel_files:
        try:
            info = load_algorithm_data(path)
            records.append({
                'Algorithm': info['Algorithm'],
                'Display Name': info['Full_Name'],
                'Min Fitness': info['Min_Fitness'],
                'Mean Fitness': info['Mean_Fitness'],
                'Median Fitness': info['Median_Fitness'],
                'Std Dev': info['Std_Fitness'],
                'Best Run': info['Best_Run'],
            })
            if len(info['Best_Curve']) > 0:
                curves_dict[info['Algorithm']] = info['Best_Curve']
        except Exception as e:
            print(f"  [!] Error reading {path}: {e}")

    if not records:
        print("[!] No valid records could be extracted.")
        return None

    df = pd.DataFrame(records)
    # Sort by Min Fitness ascending (best to worst)
    if 'Min Fitness' in df.columns:
        df = df.sort_values(by='Min Fitness', ascending=True).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = 'Rank'

    print("\n" + "=" * 80)
    print("  OPTIMIZATION BENCHMARK — ALGORITHM COMPARISON LEADERBOARD")
    print("=" * 80)
    disp_cols = ['Algorithm', 'Min Fitness', 'Mean Fitness', 'Median Fitness', 'Std Dev']
    print(df[disp_cols].to_string())
    print("=" * 80 + "\n")

    # Save to CSV
    os.makedirs(excel_dir, exist_ok=True)
    csv_path = os.path.join(excel_dir, "Problem_Results.csv")
    df.to_csv(csv_path)
    try:
        df.to_csv(os.path.join(excel_dir, "Algorithm_Comparison_Summary.csv"))
    except Exception:
        pass
    print(f"  [✓] Summary CSV saved:   {csv_path}")

    # Save to styled Excel
    xlsx_path = os.path.join(excel_dir, "Problem_Results.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Comparison_Leaderboard"

    C_HEADER = "1F4E79"
    C_BEST   = "E2EFDA"
    C_ALT    = "F2F7FC"
    thin = Side(style='thin', color="AAAAAA")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    headers = ['Rank'] + list(df.columns)
    ws.append(headers)

    for rank, row in df.iterrows():
        ws.append([rank] + list(row))

    # Header styling
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = PatternFill("solid", fgColor=C_HEADER)
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border
    ws.row_dimensions[1].height = 22

    # Data row styling
    for r in range(2, len(df) + 2):
        fc = C_BEST if r == 2 else (C_ALT if r % 2 == 1 else "FFFFFF")
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=r, column=c)
            cell.fill = PatternFill("solid", fgColor=fc)
            cell.border = border
            val = cell.value
            if isinstance(val, float):
                if abs(val) < 1e-3 or abs(val) > 1e4:
                    cell.number_format = '0.000000E+00'
                else:
                    cell.number_format = '0.0000'
                cell.alignment = Alignment(horizontal="right")
            else:
                cell.alignment = Alignment(horizontal="center")

    # Autofit columns
    for col in ws.columns:
        ml = max((len(str(c.value or '')) for c in col), default=8)
        ws.column_dimensions[col[0].column_letter].width = max(min(ml + 4, 34), 12)
    ws.freeze_panes = 'A2'

    wb.save(xlsx_path)
    try:
        wb.save(os.path.join(excel_dir, "Algorithm_Comparison_Summary.xlsx"))
    except Exception:
        pass
    print(f"  [✓] Styled Excel saved: {xlsx_path}")

    # =========================================================
    # Overlaid Convergence Plot
    # =========================================================
    if curves_dict:
        os.makedirs(figures_dir, exist_ok=True)
        plt.figure(figsize=(15, 9))

        try:
            cmap = matplotlib.colormaps['tab20'].resampled(max(len(curves_dict), 20))
        except Exception:
            cmap = plt.cm.get_cmap('tab20', max(len(curves_dict), 20))

        linestyles = ['-', '--', '-.', ':']

        sorted_algos = sorted(curves_dict.keys(),
                              key=lambda k: curves_dict[k][-1] if len(curves_dict[k]) > 0 else np.inf)

        for idx, algo in enumerate(sorted_algos):
            curve = curves_dict[algo]
            ls = linestyles[(idx // 20) % len(linestyles)]
            c = cmap(idx % 20)
            iterations = np.arange(1, len(curve) + 1)
            final_val = curve[-1] if len(curve) > 0 else 0.0

            # Use semilogy if curve is strictly positive, otherwise regular plot
            if np.all(curve > 0):
                plt.semilogy(iterations, curve, label=f"{algo} ({final_val:.2e})",
                             color=c, linestyle=ls, linewidth=1.8, alpha=0.85)
            else:
                plt.plot(iterations, curve, label=f"{algo} ({final_val:.2e})",
                         color=c, linestyle=ls, linewidth=1.8, alpha=0.85)

        plt.xlabel('Iteration', fontsize=14, fontweight='bold')
        plt.ylabel('Best Objective Value (Log Scale)', fontsize=14, fontweight='bold')
        plt.title('Benchmark Convergence Comparison Across All Algorithms\n(Best Independent Run per Algorithm)',
                  fontsize=15, fontweight='bold')
        plt.grid(True, which="both", ls="-", alpha=0.25)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9.5, frameon=True)
        plt.tight_layout()

        plot_path = os.path.join(figures_dir, "Comparison_Convergence_All.png")
        plt.savefig(plot_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"  [✓] Overlaid convergence plot saved: {plot_path}")

    print("=" * 80 + "\n")
    return df


if __name__ == "__main__":
    compare_algorithms()
