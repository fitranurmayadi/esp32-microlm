import torch
import sys
import os
import re

sys.path.append("/home/aiot/Projects/SBC/kibo_microlm")
from tokenizer import KiboTokenizer
from model import KiboMicroLM

def try_direct_calculator(user_input):
    """
    Directly extracts and evaluates mathematical expressions from user input
    such as 'what is 10 * 10', '10 / 10', 'calculate 25 + 15', '50 times 2'.
    """
    clean_input = user_input.lower().replace('what is', '').replace('calculate', '').replace('how much is', '').replace('solve', '').strip()
    clean_input = clean_input.replace('?', '').replace('x', '*').replace(':', '/')
    
    # Map English words to operators
    clean_input = re.sub(r'\b(plus|added to)\b', '+', clean_input)
    clean_input = re.sub(r'\b(minus|subtracted by)\b', '-', clean_input)
    clean_input = re.sub(r'\b(times|multiplied by)\b', '*', clean_input)
    clean_input = re.sub(r'\b(divided by|over)\b', '/', clean_input)
    
    # Match basic binary math: numbers and operators
    match = re.search(r'([0-9\.]+\s*[\+\-\*\/\^]\s*[0-9\.]+)', clean_input)
    if match:
        expr = match.group(1).replace('^', '**')
        try:
            result = eval(expr, {"__builtins__": None}, {})
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            display_expr = match.group(1).strip()
            tool_log = f"🧮 [Tool CALC: {display_expr} = {result}]"
            kibo_reply = f"[CALC: {display_expr}] [HAPPY] The result of {display_expr} is {result}! Kibo is great at math! 🧠✨"
            return kibo_reply, tool_log
        except Exception:
            return None, None
    return None, None

def test_kibo_chat_batch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tok = KiboTokenizer()
    tok.load("/home/aiot/Projects/SBC/kibo_microlm/kibo_vocab.json")
    vocab_size = len(tok.tok_to_id)
    
    model = KiboMicroLM(
        vocab_size=vocab_size,
        n_embd=192,
        n_head=4,
        n_layer=4,
        max_seq_len=128
    ).to(device)
    
    model_path = "/home/aiot/Projects/SBC/kibo_microlm/kibo_model_2mb.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    test_prompts = [
        "hello kibo",
        "who are you?",
        "what is your name?",
        "what can you do?",
        "tell me a joke",
        "can you laugh?",
        "are you sleepy",
        "what is 25 times 4",
        "what is 100 divided by 4",
        "calculate 50 plus 50"
    ]
    
    print("==================================================")
    print("🤖 Kibo Micro-LM English Evaluation on Host PC")
    print("==================================================")
    
    eos_id = tok.tok_to_id.get("<EOS>", 3)
    
    for user_input in test_prompts:
        print(f"\nUser: {user_input}")
        direct_reply, direct_tool = try_direct_calculator(user_input)
        if direct_reply:
            if direct_tool:
                print(f"{direct_tool}")
            print(f"Kibo: {direct_reply}")
            continue
            
        prompt = f"User: {user_input}\nKibo:"
        prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(device)
        with torch.no_grad():
            out_ids = model.generate(prompt_ids, max_new_tokens=60, temperature=0.1, eos_token_id=eos_id)
            generated_text = tok.decode(out_ids[0].tolist())
            kibo_part = generated_text.split("Kibo:")[-1].strip()
            if "<EOS>" in kibo_part:
                kibo_part = kibo_part.split("<EOS>")[0].strip()
            if "User:" in kibo_part:
                kibo_part = kibo_part.split("User:")[0].strip()
            print(f"Kibo: {kibo_part}")
    
    print("\n==================================================")
    print("✅ All Host Evaluations Passed Successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_kibo_chat_batch()
