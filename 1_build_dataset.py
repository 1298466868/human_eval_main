import json
import subprocess
import tiktoken
from radon.complexity import cc_visit
from openai import OpenAI
from evalplus.data import get_human_eval
from tqdm import tqdm

client = OpenAI(api_key="0000") # 记得替换
MODEL_NAME = "gpt-4o"
tokenizer = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text): 
    return len(tokenizer.encode(text))

def run_sandbox(code):
    """
    【工程级修复】：通过 stdin 管道将代码送入容器，完美避开 Windows/Linux 引号转义和路径挂载的 Bug。
    同时限制内存和执行时间，防止脏代码死循环拖垮系统。
    """
    try:
        cmd = ["docker", "run", "-i", "--rm", "--memory=256m", "python:3.9-slim", "python", "-"]
        res = subprocess.run(
            cmd, 
            input=code,           # 通过标准输入流传入完整代码
            text=True, 
            capture_output=True, 
            timeout=15            # 延长到 15 秒，避免误杀正常但跑得慢的脏代码
        )
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        # 超时被视为不合法，不抛出异常打断流程
        return False
    except Exception: 
        return False

def build_dataset(problems, max_items=164):
    dataset = []
    print(f"开始使用 {MODEL_NAME} 构造高难度对抗数据集...")
    
    sys_msg = """你是一位资深的软件安全测试专家与Python架构师。你需要输出严格的JSON格式。
为了测试代码理解模型，请为提供的目标函数设计一个【前置辅助函数】（注意：不是完成原题，而是写一个相关的独立工具函数）。
你必须提供以下三个字段：
1. 'clean_code': 逻辑清晰、控制流最优、遵循PEP8的干净Python辅助函数。
2. 'dirty_code': 与clean_code【功能完全等价】，但刻意极度恶化其可读性和圈复杂度（Cyclomatic Complexity）。
   - 必须引入的策略：极深的无用 if/else 嵌套、永不触发的死代码分支（如 if False 变体）、冗余的中间变量类型转换、毫无必要的 for/while 循环等。
   - 限制：绝对不能改变最终返回结果，绝不能抛出异常。
3. 'test_cases': 包含不少于 3 条 `assert 辅助函数(输入) == 输出` 的Python测试代码。必须能同时跑通 clean_code 和 dirty_code。"""

    for prob in tqdm(problems[:max_items]):
        for attempt in range(5):
            temp = 0.4 + (attempt * 0.2)
            user_msg = f"原题任务：{prob['prompt']}\n请基于该场景，构造一对 clean_code 和 dirty_code，并提供 JSON。"
            
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
    # 直接使用 EvalPlus 内部的清洗版 HumanEval 数据，确保与 Phase3 评测完全对齐！
    # get_human_eval() 返回的是字典 {task_id: task_dict}，转化为列表即可
    evalplus_dataset = list(get_human_eval().values())
    
    # 按照任务 ID 排序，保证每次运行顺序一致（科研基本操作）
    evalplus_dataset = sorted(evalplus_dataset, key=lambda x: int(x["task_id"].split("/")[-1]))
    
    build_dataset(evalplus_dataset, max_items=164)
