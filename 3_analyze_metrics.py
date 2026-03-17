import os
import re
import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from collections import defaultdict
from radon.complexity import cc_visit
from radon.raw import analyze
import concurrent.futures

def get_code_metrics(code):
    try:
        blocks = cc_visit(code)
        cc_val = sum(obj.complexity for obj in blocks) if blocks else 1.0
        raw = analyze(code)
        loc = raw.sloc if hasattr(raw, 'sloc') else raw.loc 
        density = cc_val / loc if loc > 0 else 0
        return cc_val, loc, density
    except Exception:
        return None, None, None

def run_evalplus_evaluation():
    sample_files = glob.glob("eval_samples/samples_*.jsonl")
    print(f"找到 {len(sample_files)} 个样本文件，准备启动 EvalPlus 沙箱...")

    # 留出 1-2 个核保证系统稳定性
    max_workers = max(1, os.cpu_count() - 2)
    print(f"EvalPlus 将使用 {max_workers} 个并行 Worker 评测单文件。")

    for file_path in sample_files:
        result_file = file_path.replace(".jsonl", "_eval_results.json")
        if os.path.exists(result_file):
            print(f"跳过已评测文件: {file_path}")
            continue
        
        print(f"正在评测: {file_path}")
        # 【核心修正】：调用内置并行机制
        cmd = [
            "evalplus.evaluate", 
            "--dataset", "humaneval", 
            "--samples", file_path,
            "--parallel", str(max_workers)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

def analyze_and_plot():
    records = []
    
    if not os.path.exists("phase1_dataset.jsonl"):
        print("错误: 找不到 phase1_dataset.jsonl")
        return
        
    with open("phase1_dataset.jsonl", "r", encoding="utf-8") as f:
        context_data = {json.loads(line)["task_id"]: json.loads(line) for line in f}

    result_files = glob.glob("eval_samples/*_eval_results.json")
    if not result_files:
        print("未找到任何 eval_results.json。")
        return

    print("\n>>> 正在对齐评测结果并利用多进程计算 AST 复杂度...")
    ast_tasks = []
    
    for res_file in tqdm(result_files, desc="Processing files"):
        basename = os.path.basename(res_file)
        match = re.search(r'samples_(.+?)_(G[1-5])_eval_results\.json', basename)
        if not match: continue
        model_id, group = match.groups()
        
        sample_file = res_file.replace("_eval_results.json", ".jsonl")
        completions_by_task = defaultdict(list)
        with open(sample_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                completions_by_task[d["task_id"]].append(d["completion"])

        with open(res_file, "r") as f:
            eval_data = json.load(f).get("eval", {})
        
        for task_id, result_list in eval_data.items():
            ctx_key = {"G1": None, "G2": "metrics", "G3": "metrics", "G4": "metrics", "G5": "metrics"}[group]
            prompt_tokens = 0
            if ctx_key:
                if group in ["G2", "G4"]: prompt_tokens = context_data[task_id]["metrics"]["clean_tokens"]
                elif group in ["G3", "G5"]: prompt_tokens = context_data[task_id]["metrics"]["dirty_tokens"]

            codes = completions_by_task[task_id]
            if len(codes) != len(result_list): continue
            
            for i, res in enumerate(result_list):
                if res.get("plus_status") == "pass":
                    ast_tasks.append({
                        "Model": model_id, "Task": task_id, "Group": group,
                        "Context_Tokens": prompt_tokens, "Code": codes[i],
                        "Total_Samples": len(result_list)
                    })

    # 【核心突破】：ProcessPoolExecutor 解决 AST 纯本地算力瓶颈
    task_stats = defaultdict(lambda: {"Success": 0, "Total": 0, "CC": [], "LOC": [], "Density": [], "Tokens": 0})
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_task = {executor.submit(get_code_metrics, task["Code"]): task for task in ast_tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(ast_tasks), desc="AST 解析"):
            task = future_to_task[future]
            cc_val, loc_val, density_val = future.result()
            
            key = (task["Model"], task["Group"], task["Task"])
            stats = task_stats[key]
            stats["Success"] += 1
            stats["Total"] = task["Total_Samples"]
            stats["Tokens"] = task["Context_Tokens"]
            
            if cc_val is not None:
                stats["CC"].append(cc_val)
                stats["LOC"].append(loc_val)
                stats["Density"].append(density_val)

    for (model, group, task_id), stats in task_stats.items():
        pass_at_1 = stats["Success"] / stats["Total"] if stats["Total"] > 0 else 0
        records.append({
            "Model": model, "Task": task_id, "Group": group, 
            "Context_Tokens": stats["Tokens"],
            "Pass@1_Unbiased": pass_at_1, 
            "CC_Avg": sum(stats["CC"])/len(stats["CC"]) if stats["CC"] else None,
            "LOC_Avg": sum(stats["LOC"])/len(stats["LOC"]) if stats["LOC"] else None,
            "Density_Avg": sum(stats["Density"])/len(stats["Density"]) if stats["Density"] else None
        })

    df = pd.DataFrame(records)
    print("\n[表 1] 总体模型表现：Pass@1 与 复杂度密度 (Density) 均值")
    summary_table = df.groupby(["Model", "Group"]).agg({"Pass@1_Unbiased": "mean", "Density_Avg": "mean"}).unstack()
    print(summary_table.round(4))

    df_context = df[df['Group'].isin(['G2', 'G3', 'G4', 'G5'])].copy()
    if not df_context.empty:
        df_context['Token_Bin'] = pd.qcut(df_context['Context_Tokens'], q=3, labels=['Short', 'Medium', 'Long'], duplicates='drop')
        print("\n[表 2] 长度控制下的逻辑退化 (Density in Token Bins)")
        stratified_density = df_context.groupby(["Model", "Token_Bin", "Group"], observed=True)["Density_Avg"].mean().unstack()
        print(stratified_density.round(4))
      
        sns.set_theme(style="whitegrid")
        g = sns.catplot(data=df_context, x="Token_Bin", y="Density_Avg", hue="Group", col="Model", kind="bar", height=5, aspect=0.8, errorbar=None)
        g.set_axis_labels("Context Length (Binned)", "Complexity Density (CC/LOC)")
        g.set_titles("{col_name}")
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle("Code Complexity Density by Context Dirtiness (Only Passed Samples)")
        plt.savefig("density_stratified_analysis.png", dpi=300)
        print("\n[图表生成] density_stratified_analysis.png")

if __name__ == "__main__":
    try: from tqdm import tqdm
    except ImportError: tqdm = lambda x, **kwargs: x
    run_evalplus_evaluation()
    analyze_and_plot()
