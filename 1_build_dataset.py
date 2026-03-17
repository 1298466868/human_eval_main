import json
import subprocess
import tiktoken
from openai import OpenAI
from evalplus.data import get_human_eval
from tqdm import tqdm

client = OpenAI(
    api_key="sk-你的国内中转Key", 
    base_url="https://中转商提供的地址/v1"  
)
MODEL_NAME = "gpt-4o"
tokenizer = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text): 
    return len(tokenizer.encode(text))

def run_sandbox(code):
    try:
        cmd = ["docker", "run", "-i", "--rm", "--memory=256m", "python:3.9-slim", "python", "-"]
        res = subprocess.run(
            cmd, 
            input=code,           
            text=True, 
            capture_output=True, 
            timeout=15            
        )
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception: 
        return False

def build_dataset(problems, max_items=164):
    dataset = []
    print(f"开始使用 {MODEL_NAME} 构造高难度对抗数据集...")
  
    # 【核心修改】：将其转变为结构清晰、指令强烈的专业英文 System Prompt
    sys_msg = """You are a senior software security testing expert and Python architect. You must strictly output in JSON format.
To evaluate code comprehension models, please design an independent [pre-requisite helper function] related to the provided target problem context. (Note: Do NOT solve the original problem itself; write a relevant standalone utility function).

You must provide exactly these three fields in your JSON output:
1. 'clean_code': A clean Python helper function with clear logic, optimal control flow, and strict PEP8 compliance.
2. 'dirty_code': Functionally strictly equivalent to `clean_code`, but intentionally severely obfuscated to maximize its Cyclomatic Complexity and minimize readability.
   - Required strategies: extremely deep and meaningless if/else nesting, dead code branches that are never triggered (e.g., complex variants of `if False`), redundant intermediate variable type conversions, completely unnecessary for/while loops, and other similar obfuscation techniques.
   - Hard Constraints: It must NEVER change the final output result. It must NEVER raise exceptions.
3. 'test_cases': At least 3 Python assertion statements (format: `assert helper_function(input) == output`). These tests MUST pass successfully for both `clean_code` and `dirty_code`."""

    for prob in tqdm(problems[:max_items]):
        for attempt in range(5):
            temp = 0.4 + (attempt * 0.2)
            # 【核心修改】：英文 User Prompt
            user_msg = f"Original Task Context:\n{prob['prompt']}\n\nBased on this scenario, please construct the pair of clean_code and dirty_code, and output the required JSON."
          
            try:
                res = client.chat.completions.create(
                    model=MODEL_NAME, 
                    messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                    temperature=temp, 
                    response_format={"type": "json_object"}
                )
                data = json.loads(res.choices[0].message.content)
                clean, dirty, tests = data['clean_code'], data['dirty_code'], data['test_cases']
          
                if run_sandbox(clean + "\n" + tests) and run_sandbox(dirty + "\n" + tests):
                    dataset.append({
                        "task_id": prob["task_id"], "task_prompt": prob["prompt"], "official_test": prob["test"],
                        "g4_clean": clean, "g5_dirty": dirty,
                        "metrics": {"clean_tokens": count_tokens(clean), "dirty_tokens": count_tokens(dirty)}
                    })
                    break 
            except Exception: 
                continue
      
    N = len(dataset)
    if N == 0:
        print("警告：未生成任何有效数据，请检查网络或Docker环境。")
        return
      
    for i in range(N):
        dataset[i]["g2_irrelevant_clean"] = dataset[(i+1)%N]["g4_clean"]
        dataset[i]["g3_irrelevant_dirty"] = dataset[(i+1)%N]["g5_dirty"]
  
    with open("phase1_dataset.jsonl", "w", encoding="utf-8") as f:
        for d in dataset: f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Phase 1 完成，共生成 {len(dataset)} 条高标准对抗数据。")

if __name__ == "__main__":
    evalplus_dataset = list(get_human_eval().values())
    evalplus_dataset = sorted(evalplus_dataset, key=lambda x: int(x["task_id"].split("/")[-1]))
    build_dataset(evalplus_dataset, max_items=164)
