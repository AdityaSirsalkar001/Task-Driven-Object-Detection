"""Text-only task normalization. Object knowledge never enters this prompt."""
import json
import math
import re


def normalize_task(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def validate_response(raw, prompt, task_nodes):
    result = {'raw_prompt': prompt, 'task': 'unknown', 'task_confidence': None, 'constraints': {}, 'raw_response': raw, 'confidence_source': 'unavailable', 'validation_error': None}
    try:
        text = raw.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError('Response must be a JSON object')
        aliases = {normalize_task(name): name for name in task_nodes}
        for name, n in task_nodes.items():
            for alias in n.get('aliases', []):
                aliases[normalize_task(alias)] = name
        task = aliases.get(normalize_task(data.get('task', 'unknown')), 'unknown')
        confidence = data.get('task_confidence', data.get('confidence'))
        if confidence is not None and (type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError('Invalid task confidence')
        constraints = data.get('constraints', {})
        if not isinstance(constraints, dict) or constraints:
            raise ValueError('Constraints are not implemented; refusing to silently ignore them')
        result.update(task=task, task_confidence=confidence, confidence_source='slm_self_reported_uncalibrated' if confidence is not None else 'unavailable')
    except (ValueError, TypeError) as exc:
        result['validation_error'] = str(exc)
    return result


class TaskParser:
    def __init__(self, config, task_nodes):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config, self.tasks = config, task_nodes
        torch.set_num_threads(config['threads'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['slm_path'], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(config['slm_path'], dtype=torch.bfloat16, local_files_only=True).to('cpu').eval()

    def parse(self, prompt):
        import torch
        instruction = self.config['slm_prompt_template'].replace('{tasks}', json.dumps(sorted(self.tasks) + ['unknown']))
        messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.config['max_new_tokens'], do_sample=False)
        raw = self.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return validate_response(raw, prompt, self.tasks)
