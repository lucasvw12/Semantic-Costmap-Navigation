from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

from semantics import CLASSES, label_to_class_id

OBJECT_LIST = "\n".join("- " + name for name in CLASSES)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

class LLMParser:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32
        )

        self.model.eval()

    def parse_instruction(self, instruction):
        system_prompt = """
You are a robot navigation command parser.

Convert the user's navigation instruction into a JSON command.

Available objects:
{object_list}

Available actions:
- increase
- decrease
- set

Return ONLY valid JSON with exactly these fields:

{
    "object": "...",
    "action": "...",
    "amount": number
}

Rules:
- "avoid" means increase
- "stay away from" means increase
- "prefer" means decrease
- "favor" means decrease
- "make the cost X" means set
- If no amount is specified, use 10

Examples:

User: Avoid barrels
Output: {"object": "barrel", "action": "increase", "amount": 10}

User: Prefer to stay near the fence
Output: {"object": "fence", "action": "decrease", "amount": 10}

User: Make cars cost 50
Output: {"object": "car", "action": "set", "amount": 50}
""".replace("{object_list}", OBJECT_LIST)

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": instruction
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100
            )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        command = json.loads(response)
        command["class_id"] = label_to_class_id(command.get("object", ""))

        return command
