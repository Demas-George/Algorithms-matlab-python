# =============================================================================
#  ALGORITHM REFERENCE
#  ---------------------------------------------------------------------------
#  Algorithm: Artificial Protozoa Optimizer (APO)
#  Author:    Wang Xiaopeng, et al.
#  Journal:   Knowledge-Based Systems (2024)
#  DOI:       https://doi.org/10.1016/j.knosys.2024.111737
# =============================================================================

import numpy as np
from copy import deepcopy
import time
import os
from opfunu.cec_based import cec2022   # opfunu==1.0.0

# =============================================================================
# 1. Global Settings
# =============================================================================
PopSize = 100               # Population size
Trials = 30                 # Number of independent runs
Dims = [10, 20]                 # List of dimensions to test
MaxFEs = 10000              # Total budget of Function Evaluations

APO_np = 1                  # APO parameter: Number of neighbor pairs
APO_pf_max = 0.1            # APO parameter: Maximum proportion fraction
LB_val = -100               # Lower bound for search space
UB_val = 100                # Upper bound for search space
SavePath = "./APOwr_Data/CEC2022/" 

# =============================================================================
# 2. APO Core Algorithm
# =============================================================================
def APO_optimizer(func, dim, pop_size, iter_max, lb, ub, np_pairs, pf_max):
    Xmin, Xmax = np.array(lb), np.array(ub)
    
    # Initialization
    protozoa = Xmin + np.random.rand(pop_size, dim) * (Xmax - Xmin)
    protozoa_fit = np.array([func.evaluate(ind) for ind in protozoa])
    
    best_id = np.argmin(protozoa_fit)
    best_fit = protozoa_fit[best_id]
    best_protozoa = deepcopy(protozoa[best_id])
    trace = [best_fit]
    
    for it in range(1, iter_max):
        # 1. Ranking/Sorting: protozoa population
        idx_sort = np.argsort(protozoa_fit)
        protozoa = protozoa[idx_sort]
        protozoa_fit = protozoa_fit[idx_sort]
        
        new_protozoa = np.zeros((pop_size, dim))
        pf = pf_max * np.random.rand()
        # Randomly select indices for dormancy or reproduction
        ri = np.random.choice(range(pop_size), int(np.ceil(pop_size * pf)), replace=False)
        
        for i in range(pop_size):
            if i in ri:
                # --- A. Dormancy and Reproduction Forms ---
                pdr = 0.5 * (1 + np.cos((1 - (i+1) / pop_size) * np.pi))
                if np.random.rand() < pdr:
                    # Dormancy: Random re-initialization
                    new_protozoa[i] = Xmin + np.random.rand(dim) * (Xmax - Xmin)
                else:
                    # Reproduction: Local perturbation
                    flag = 1 if np.random.rand() < 0.5 else -1
                    mr = np.zeros(dim)
                    # Mr is a mapping vector in reproduction
                    mr[np.random.choice(range(dim), int(np.ceil(np.random.rand()*dim)), replace=False)] = 1
                    new_protozoa[i] = protozoa[i] + flag * np.random.rand() * (Xmin + np.random.rand(dim) * (Xmax - Xmin)) * mr
            else:
                # --- B. Foraging Form ---
                f_factor = np.random.rand() * (1 + np.cos(it / iter_max * np.pi))
                mf = np.zeros(dim)
                # Mf is a mapping vector in foraging
                mf[np.random.choice(range(dim), int(np.ceil(dim * (i+1) / pop_size)), replace=False)] = 1
                pah = 0.5 * (1 + np.cos(it / iter_max * np.pi))
                epn = np.zeros((np_pairs, dim))
                
                if np.random.rand() < pah: 
                    # Autotroph 
                    j = np.random.randint(0, pop_size)
                    for k in range(1, np_pairs + 1):
                        if i == 0: km, kp = i, i + np.random.randint(1, pop_size - i)
                        elif i == pop_size - 1: km, kp = np.random.randint(0, pop_size - 1), i
                        else: km, kp = np.random.randint(0, i), i + np.random.randint(1, pop_size - i)
                        # Weight factor for autotroph form
                        # wa = np.exp(-abs(protozoa_fit[km] / (protozoa_fit[kp] + np.finfo(float).eps)))
                        # Replace fitness-based weighting with ranking. Rank starts from 1.
                        wa = np.exp(-(1 - (km + 1) / (kp + 1)))
                        epn[k-1] = wa * (protozoa[km] - protozoa[kp])
                    new_protozoa[i] = protozoa[i] + f_factor * (protozoa[j] - protozoa[i] + (1/np_pairs) * np.sum(epn, axis=0)) * mf
                else: 
                    # Heterotroph 
                    for k in range(1, np_pairs + 1):
                        imk, ipk = max(0, i - k), min(pop_size - 1, i + k)
                        # Weight factor for heterotroph form
                        # wh = np.exp(-abs(protozoa_fit[imk] / (protozoa_fit[ipk] + np.finfo(float).eps)))
                        # Replace fitness-based weighting with ranking. Rank starts from 1.
                        wh = np.exp(-(1 - (imk + 1) / (ipk + 1)))
                        epn[k-1] = wh * (protozoa[imk] - protozoa[ipk])
                    flag = 1 if np.random.rand() < 0.5 else -1
                    x_near = (1 + flag * np.random.rand(dim) * (1 - it/iter_max)) * protozoa[i]
                    new_protozoa[i] = protozoa[i] + f_factor * (x_near - protozoa[i] + (1/np_pairs) * np.sum(epn, axis=0)) * mf

        # Boundary control and Greedy selection
        new_protozoa = np.clip(new_protozoa, Xmin, Xmax)
        for i in range(pop_size):
            nf = func.evaluate(new_protozoa[i])
            if nf < protozoa_fit[i]:
                protozoa_fit[i], protozoa[i] = nf, new_protozoa[i]
        
        # Update current best
        c_best = np.argmin(protozoa_fit)
        best_fit = protozoa_fit[c_best]
        best_protozoa = deepcopy(protozoa[c_best])
        
        trace.append(best_fit)
        
    return trace, best_protozoa, best_fit

# =============================================================================
# 3. Main Experiment Execution
# =============================================================================
def main():
    if not os.path.exists(SavePath): os.makedirs(SavePath)

    for dim in Dims:
        dim_path = os.path.join(SavePath, f"{dim}D")
        if not os.path.exists(dim_path): os.makedirs(dim_path)
        
        lb, ub = [LB_val] * dim, [UB_val] * dim
        # Calculate maximum iterations based on MaxFEs and PopSize
        max_iter = int(MaxFEs / PopSize)
        
        # Load CEC2022 Benchmark Functions F1-F12
        test_funcs = [getattr(cec2022, f"F{i}2022")(dim) for i in range(1, 13)]
        
        print(f"\n>>> Running Experiments: {dim}D | MaxFEs: {MaxFEs} | MaxIter: {max_iter}")
        
        # This will accumulate ONLY pure algorithm execution time
        pure_algo_total_time = 0.0
        
        # Lists to store summary data for the current dimension
        summary_stats = []
        all_functions_best_f = [] # Combined results for all functions (Wilcoxon analysis)

        for i, func in enumerate(test_funcs):
            f_idx = i + 1
            all_trial_trace, all_trial_best_f, all_trial_best_x = [], [], []
            
            for t in range(Trials):
                np.random.seed(2022 + 7 * t)
                
                # --- PURE ALGORITHM TIMING START ---
                t_start = time.process_time()
                tr, bx, bf = APO_optimizer(func, dim, PopSize, max_iter, lb, ub, APO_np, APO_pf_max)
                t_end = time.process_time()
                # --- PURE ALGORITHM TIMING END ---
                
                pure_algo_total_time += (t_end - t_start)
                
                all_trial_trace.append(tr)
                all_trial_best_f.append(bf)
                all_trial_best_x.append(bx)
            
            # --- Data Archiving (Excluded from pure_algo_total_time) ---
            file_base = os.path.join(dim_path, f"F{f_idx}_{dim}D")
            best_f_arr = np.array(all_trial_best_f)
            best_run_idx = np.argmin(best_f_arr)
            
            min_val, max_val = np.min(best_f_arr), np.max(best_f_arr)
            mean_val, std_val = np.mean(best_f_arr), np.std(best_f_arr)
            best_run_num = best_run_idx + 1

            # 1. Save Convergence Curves for each function
            np.savetxt(f"{file_base}_Trace.csv", all_trial_trace, delimiter=",")
            
            # Add to Summary Lists: Function ID + Statistics
            # This replaces the need for individual _Stats.csv files
            summary_stats.append([f_idx, min_val, max_val, mean_val, std_val, best_run_num])
            
            # Prepend the Func_ID to the fitness array for identifying the row in the combined file
            all_functions_best_f.append(np.insert(best_f_arr.astype(float), 0, f_idx))

            # 2. Save Best Individual Vector (Vertical Arrangement) in all Trials
            np.savetxt(f"{file_base}_BestX_inTrials.csv", all_trial_best_x[best_run_idx], delimiter=",")
            print(f"[{dim}D] F{f_idx} Done:\n"
                  f"Min={min_val:.4e} | Max={max_val:.4e} | Mean={mean_val:.4e} | Std={std_val:.4e} | Best Run={best_run_num}")
            

        # --- Save Summary Statistics for All Functions ---
        # This file contains the data previously stored in individual Stats files
        summary_stats_file = os.path.join(dim_path, f"All_Functions_Stats_{dim}D.csv")
        np.savetxt(summary_stats_file, np.array(summary_stats), delimiter=",",
                   header="Func_ID,Min,Max,Mean,Std,Best_Run_Idx", comments='')

        # --- Save Combined Best Fitness Results (for Wilcoxon/Statistical Analysis) ---
        # Format: Each row corresponds to a function, columns are the Trials
        combined_fitness_file = os.path.join(dim_path, f"All_Functions_BestF_{dim}D.csv")
        header_best_f = "Func_ID," + ",".join([f"Trial_{t+1}" for t in range(Trials)])
        np.savetxt(combined_fitness_file, np.array(all_functions_best_f), delimiter=",",
                   header=header_best_f, comments='')

        # --- Record and Save Execution Time (Pure Algorithm Time Only) ---
        avg_time_per_trial = pure_algo_total_time / (len(test_funcs) * Trials)
        time_results = np.array([[pure_algo_total_time, avg_time_per_trial]])
        np.savetxt(os.path.join(dim_path, f"Execution_Time_{dim}D.csv"), time_results, 
                   delimiter=",", header="Total_Time_Seconds,Avg_Time_Per_Trial", comments='')
        
        print(f"\n>>> {dim}D Algorithm Time: Total {pure_algo_total_time:.4f}s | Avg/Trial {avg_time_per_trial:.4f}s")

if __name__ == "__main__":
    main()