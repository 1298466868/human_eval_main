import json
import re
import time
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

MODELS_CONFIG = {
    "gpt-4o-mini": {
        "client": OpenAI(api_key="sk-你购买的国内中转Key", base_url="https://中转商给你的具体地址/v1"),
        "model_name": "gpt-4o-mini"
    },
    "Qwen2.5-Coder-32B": {
        "client": OpenAI(api_key="sk-你的开源平台Key", base_url="https://api.siliconflow.cn/v1"),
        "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct" 
    },
    "Llama-3-8B-Instruct": {
        "client": OpenAI(api_key="sk-你的开源平台Key", base_url="https://api.siliconflow.cn/v1"),
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct"
    }
}

def clean_completion(code_str, prompt_str):
    """【科研级补丁】：彻底解决大模型复读函数签名导致的缩进/重定义错误"""
    code_str = code_str.strip('\r\n')
    sig_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', prompt_str)
    if sig_match:
        func_name = sig_match.group(1)
        pattern = rf"^def\s+{func_name}\s*\(.*?\):\s*\n"
        code_str = re.sub(pattern, "", code_str, count=1, flags=re.MULTILINE|re.DOTALL)
        
        # 如果模型连带文档字符串一起复读了，也需要清理
        doc_pattern = r'^(\s*"""(?:.*?)"""\s*\n|\s*\'\'\'(?:.*?)\'\'\'\s*\n)'
        code_str = re.sub(doc_pattern, "", code_str, count=1, flags=re.DOTALL)
    
    # 防止多余的前导空格导致 EvalPlus 缩进报错
    lines = code_str.split('\n')
    if lines and lines[0].startswith('    ') and not code_str.startswith('def '):
        code_str = '\n'.join([line[4:] if line.startswith('    ') else line for line in lines])
        
    return code_str.strip('\r\n')

def extract_code(text):
    match = re.search(r'```(?:[pP]ython)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip('\r\n')
    return text.strip('\r\n')

def fetch_single_sample(client, model, prompt_text, task_prompt, max_retries=5):
    """单次 API 调用任务，独立重试逻辑，供线程池调度"""
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer. Only output pure Python code wrapped in ```python ```."},
                    {"role": "user", "content": prompt_text}
                ],
                n=1,
                temperature=0.7, 
                max_tokens=1024,
                timeout=30
            )
            raw_code = extract_code(res.choices[0].message.content)
            final_code = clean_completion(raw_code, task_prompt)
            return final_code
        except Exception as e:
            time.sleep(2 ** attempt)
    return "" # 重试耗尽返回空字符串，EvalPlus将判为Fail，维持数据的客观性

def evaluate_tasks():
    if not os.path.exists("phase1_dataset.jsonl"):
        print("错误：请先运行 Phase 1 生成数据集。")
        return

    with open("phase1_dataset.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    os.makedirs("eval_samples", exist_ok=True)

    for model_id, config in MODELS_CONFIG.items():
        client = config["client"]
        target_model = config["model_name"]
        print(f"\n================ 开始采样模型: {model_id} ================")

        # 预先生成所有任务配置
        tasks_to_run = []
        for item in dataset:
            task_id = item["task_id"]
            task_prompt = item["task_prompt"]
            prompts = {
                "G1": f"Please complete the following code:\n{task_prompt}",
                "G2": f"Helper context (you can ignore):\n{item['g2_irrelevant_clean']}\n\nPlease complete:\n{task_prompt}",
                "G3": f"Helper context (you can ignore):\n{item['g3_irrelevant_dirty']}\n\nPlease complete:\n{task_prompt}",
                "G4": f"Helper context:\n{item['g4_clean']}\n\nPlease complete:\n{task_prompt}",
                "G5": f"Helper context:\n{item['g5_dirty']}\n\nPlease complete:\n{task_prompt}",
            }
            for group, p_text in prompts.items():
                for _ in range(10): # n=10
                    tasks_to_run.append((group, task_id, task_prompt, p_text))

        # 结果容器
        results_by_group = {g: [] for g in ["G1", "G2", "G3", "G4", "G5"]}

        # 【核心突破】：启动高并发线程池打满 I/O，最大允许 30 个并发连接
        with ThreadPoolExecutor(max_workers=30) as executor:
            future_to_meta = {
                executor.submit(fetch_single_sample, client, target_model, p_text, t_prompt): (group, t_id)
                for group, t_id, t_prompt, p_text in tasks_to_run
            }
            
            for future in tqdm(as_completed(future_to_meta), total=len(tasks_to_run), desc="API 并发采样"):
                group, t_id = future_to_meta[future]
                code = future.result()
                results_by_group[group].append({"task_id": t_id, "completion": code})

        # 统一写入文件
        for group, records in results_by_group.items():
            # 必须按照 task_id 排序保证 JSONL 对齐
            records.sort(key=lambda x: int(x["task_id"].split("/")[-1]))
            with open(f"eval_samples/samples_{model_id}_{group}.jsonl", "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

    print("\nPhase 2 结束，所有模型的采样文件已保存。")

if __name__ == "__main__":
    evaluate_tasks()
