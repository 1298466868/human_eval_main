import json
import re
import time
from openai import OpenAI
from tqdm import tqdm
import os

MODELS_CONFIG = {
    "gpt-4o-mini": {
        "client": OpenAI(api_key="sk-你购买的国内中转Key", base_url="https://中转商给你的具体地址/v1" ),
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

def extract_code(text):
    """【科研级修复】：支持大小写忽略、忽略前导空格的鲁棒提取"""
    match = re.search(r'```(?:[pP]ython)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip() # 保底：直接返回原文

def generate_with_retry(client, model, prompt, num_samples=10, max_retries=5):
    """
    【工程级修复】：弃用危险的 n=10 单次调用，改为稳健的 10 次串行 n=1。
    确保不会因为某个生成过程卡死导致 10 个样本全部作废。
    """
    samples = []
    for _ in range(num_samples):
        sample_code = ""
        for attempt in range(max_retries):
            try:
                res = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert Python programmer. Only output pure Python code wrapped in ```python ```."},
                        {"role": "user", "content": prompt}
                    ],
                    n=1,             # 强制 n=1
                    temperature=0.7, 
                    max_tokens=1024,
                    timeout=30       # 单次调用的超时设为 30s
                )
                sample_code = extract_code(res.choices.message.content)
                break # 成功拿到一个样本，跳出重试循环
            except Exception as e:
                wait_time = 2 ** attempt
                print(f"  API 异常 ({e})，休眠 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        samples.append(sample_code)
    return samples

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
        print(f"\n================ 开始采样模型: {model_id} (n=10) ================")

        file_handles = {
            group: open(f"eval_samples/samples_{model_id}_{group}.jsonl", "w", encoding="utf-8") 
            for group in ["G1", "G2", "G3", "G4", "G5"]
        }

        for item in tqdm(dataset, desc=f"Evaluating {model_id}"):
            task_id = item["task_id"]
            task_prompt = item["task_prompt"]
        
            prompts = {
                "G1": f"Please complete the following code:\n{task_prompt}",
                "G2": f"Helper context (you can ignore):\n{item['g2_irrelevant_clean']}\n\nPlease complete:\n{task_prompt}",
                "G3": f"Helper context (you can ignore):\n{item['g3_irrelevant_dirty']}\n\nPlease complete:\n{task_prompt}",
                "G4": f"Helper context:\n{item['g4_clean']}\n\nPlease complete:\n{task_prompt}",
                "G5": f"Helper context:\n{item['g5_dirty']}\n\nPlease complete:\n{task_prompt}",
            }

            for group, prompt_text in prompts.items():
                codes = generate_with_retry(client, target_model, prompt_text, num_samples=10)
                for code in codes:
                    record = {"task_id": task_id, "completion": code}
                    file_handles[group].write(json.dumps(record) + "\n")
                    file_handles[group].flush()

        for fh in file_handles.values(): fh.close()
    print("\nPhase 2 结束，所有模型的采样文件已保存。")

if __name__ == "__main__":
    evaluate_tasks()
