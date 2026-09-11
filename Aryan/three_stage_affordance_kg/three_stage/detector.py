"""Adapter reusing the tested baseline's Florence grounding implementation."""


class FlorenceDetector:
    def __init__(self, config):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        from anshdeep_basic_pipeline_runner import florence_ground_objects
        torch.set_num_threads(config['threads'])
        self.processor = AutoProcessor.from_pretrained(config['od_path'], local_files_only=True)
        self.model = Florence2ForConditionalGeneration.from_pretrained(config['od_path'], dtype=torch.float32, local_files_only=True).to('cpu').eval()
        self.ground = florence_ground_objects

    def detect(self, image_path, labels):
        from PIL import Image
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        detections = self.ground(self.model, self.processor, image, labels)
        for item in detections:
            item['od_confidence'] = None
            item['confidence_source'] = 'not_provided_by_florence'
        return detections

    def detect_reference(self, image_path):
        """Run Florence's standard OD task once to label query-box clusters."""
        import torch
        from PIL import Image
        task = '<OD>'
        with Image.open(image_path) as source:
            image = source.convert('RGB')
        inputs = self.processor(text=task, images=image, return_tensors='pt').to('cpu')
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
        text = self.processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(text, task=task, image_size=image.size).get(task, {})
        if isinstance(parsed, str):
            return []
        labels = parsed.get('bboxes_labels', parsed.get('labels', []))
        return [
            {'label': str(label), 'bbox': [float(value) for value in bbox], 'source': 'florence_standard_od'}
            for label, bbox in zip(labels, parsed.get('bboxes', []))
        ]
