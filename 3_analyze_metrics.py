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

    for file_path in sample_files:
        result_file = file_path.replace(".jsonl", "_eval_results.json")
        if os.path.exists(result_file):
            print(f"跳过已评测文件: {file_path}")
            continue
        print(f"正在评测: {file_path}")
        # 保持 --base，保证对比的基线公平
        cmd = ["evalplus.evaluate", "--dataset", "humaneval", "--samples", file_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)

def analyze_and_plot():
    records = []
    
    if not os.path.exists("phase1_dataset.jsonl"):
        print("错误: 找不到 phase1_dataset.jsonl")
        return
        
    with open("phase1_dataset.jsonl", "r", encoding="utf-8") as f:
        context_data = {json.loads(line)["task_id"]: json.loads(line) for line in f}

    result_files = glob.glob("eval_samples/*_eval_results.json")
    if not result_files:
        print("未找到任何 eval_results.json，请确保已成功运行 EvalPlus。")
        return

    print("\n>>> 正在对齐评测结果与原始代码，并计算 AST 复杂度 (仅限通过的样本)...")
    
    for res_file in tqdm(result_files, desc="Processing results"):
        basename = os.path.basename(res_file)
        
        # 【科研级修复】：强制正则提取，杜绝下划线导致的数组越界灾难
        match = re.search(r'samples_(.+?)_(G[1-5])_eval_results\.json', basename)
        if not match:
            print(f"  [跳过] 无法匹配文件格式: {basename}")
            continue
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
            
            # 严格对齐样本数量校验
            if len(codes) != len(result_list):
                print(f"  [对齐警告] {model_id} {group} {task_id}: 代码数({len(codes)})与评测数({len(result_list)})不匹配！")
                continue
            
            success_count = 0
            task_cc, task_loc, task_density = [], [], []
            
            for i, res in enumerate(result_list):
                # 【科研级修复】：适配最新版 EvalPlus JSON 格式
                is_success = (res.get("plus_status") == "pass")
                if is_success:
                    success_count += 1
                    cc_val, loc_val, density_val = get_code_metrics(codes[i])
                    if cc_val is not None:
                        task_cc.append(cc_val)
                        task_loc.append(loc_val)
                        task_density.append(density_val)
            
            total_samples = len(result_list)
            pass_at_1_expected = success_count / total_samples if total_samples > 0 else 0
            
            records.append({
                "Model": model_id, 
                "Task": task_id, 
                "Group": group, 
                "Context_Tokens": prompt_tokens,
                "Pass@1_Unbiased": pass_at_1_expected, 
                "CC_Avg": sum(task_cc)/len(task_cc) if task_cc else None,
                "LOC_Avg": sum(task_loc)/len(task_loc) if task_loc else None,
                "Density_Avg": sum(task_density)/len(task_density) if task_density else None
            })

    df = pd.DataFrame(records)
    
    print("\n[表 1] 总体模型表现：Pass@1 与 复杂度密度 (Density) 均值")
    summary_table = df.groupby(["Model", "Group"]).agg({
        "Pass@1_Unbiased": "mean",
        "Density_Avg": "mean"
    }).unstack()
    print(summary_table.round(4))

    df_context = df[df['Group'].isin(['G2', 'G3', 'G4', 'G5'])].copy()
    if not df_context.empty:
        df_context['Token_Bin'] = pd.qcut(df_context['Context_Tokens'], q=3, labels=['Short', 'Medium', 'Long'], duplicates='drop')
        
        print("\n[表 2] 长度控制下的逻辑退化铁证 (Density in Token Bins)")
        stratified_density = df_context.groupby(["Model", "Token_Bin", "Group"], observed=True)["Density_Avg"].mean().unstack()
        print(stratified_density.round(4))
      
        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=df_context, x="Token_Bin", y="Density_Avg", hue="Group", col="Model", 
            kind="bar", height=5, aspect=0.8, errorbar=None
        )
        g.set_axis_labels("Context Length (Binned)", "Complexity Density (CC/LOC)")
        g.set_titles("{col_name}")
        g.fig.subplots_adjust(top=0.85)
        g.fig.suptitle("Code Complexity Density by Context Dirtiness (Only Passed Samples)")
        plt.savefig("density_stratified_analysis.png", dpi=300)
        print("\n[图表生成] 控制长度后的复杂度密度对比图已保存至 density_stratified_analysis.png")

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs): return iterable

    print(">>> 启动 EvalPlus 沙箱并发评测...")
    run_evalplus_evaluation()
    print("\n>>> 开始解析数据、计算合法代码的AST复杂度并生成图表...")
    analyze_and_plot()
