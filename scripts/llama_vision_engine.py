from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
import torch
from PIL import Image
SYSTEM_MESSAGE = (
    "You are a medical AI assistant with expertise in medical genetics, "
    "dysmorphology, and image-based phenotype recognition. Your task is to "
    "analyze facial and neck images to identify any observable phenotypes "
    "related to known human diseases. These phenotypes can include "
    "abnormalities in morphology, development, skin, face, hair, eyes, ears, "
    "nose, mouth, or neck. You must use the Human Phenotype Ontology (HPO) "
    "to normalize all detected phenotypes."
)
USER_QUERY = (
    "Analyze the attached facial image. "
    "Output only observable characteristics without implying any underlying "
    "cause, disease, or mechanism. All phenotypes should be separated by ||. "
    "Please do not provide any additional texts or formatting. "
    "The response should be like this: Phenotype A || Phenotype B || Phenotype C"
)
class LLaMA_Generator():
    def __init__(self, lora_ckpt, base_ckpt = 'meta-llama/Llama-3.2-11B-Vision-Instruct'):
        # 1️⃣ Load base model and processor
        self.base_model = AutoModelForVision2Seq.from_pretrained(
            base_ckpt,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(base_ckpt)
        #self.processor = MllamaProcessor.from_pretrained(base_ckpt)
        # 3️⃣ Fix tokenizer padding if needed
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.processor.tokenizer.padding_side = "left"
        self.base_model.resize_token_embeddings(len(self.processor.tokenizer))  # Now apply after PeftModel

        # 4️⃣ Set padding ID in config
        self.base_model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # 2️⃣ Attach the LoRA weights BEFORE resizing embeddings
        self.model = PeftModel.from_pretrained(self.base_model, lora_ckpt)
        self.model.eval()
    def base_preprocess(self, img: Image.Image, target_size: int = 112) -> Image.Image:
        img = img.convert("RGB")
        # Make the smallest edge = target_size while keeping aspect ratio
        img.thumbnail((target_size, target_size), Image.Resampling.BICUBIC)
        # Put it on a square canvas so rotations don’t clip
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        left = (target_size - img.width) // 2
        top  = (target_size - img.height) // 2
        new_img.paste(img, (left, top))
        return new_img
    def generate_descriptions(self, img_path: str, *, max_new_tokens: int = 1024,
                            top_p: float = 1.0, temperature: float = 0.8) -> str:
        """
        Parameters
        ----------
        img_path      :  <path/to/image>
        max_new_tokens, top_p, temperature : generation hyper‑params

        Returns
        -------
        str : model's JSON‑formatted phenotype list
        """

        # 1. Load image (keep RGB)
        img = Image.open(img_path)#.convert("RGB")
        img = self.base_preprocess(img)
        # 2. Build chat template (single turn, image first, then text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},                          # placeholder token for the processor
                    {"type": "text", "text": SYSTEM_MESSAGE + "\n" + USER_QUERY},
                ],
            }
        ]
        #text_prompt = self.processor.apply_chat_template(messages, tokenize=False)
        fallback_chat_template = """<|begin_of_text|>{% for message in messages %}
<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n
{% for item in message['content'] %}
{% if item['type'] == 'image' %}<|image|>{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}
{% endfor %}<|eot_id|>{% endfor %}"""

        ### Apply chat template (fallback if missing)
        try:
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False)
        except (ValueError, AttributeError):
            if not hasattr(self.processor, 'chat_template') or self.processor.chat_template is None:
                self.processor.chat_template = fallback_chat_template
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False)

        # 3. Tokenize / preprocess
        inputs = self.processor(
            text=[text_prompt],        # list because processor expects batch
            images=[[img]],            # nested list → one turn, one image
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, torch.float16 if self.model.config.torch_dtype == torch.float16 else torch.float32)

        # 4. Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
            )

        # 5. Trim the echo of the prompt so we keep only new tokens
        prompt_len = inputs.input_ids.shape[1]
        new_tokens = generated_ids[:, prompt_len:]

        # 6. Decode
        output = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        
        return output.replace('assistant','').strip()
    def process_output(self, outout):
        if len(output.strip()) == 0:
            return []
        if "||" in output:
            output_list = output.split("||")
            output_list = [x.replace("*", "").strip() for x in output_list]
            return output_list
        else:
            output_list = output.split("\n")
            output_list = [x.replace("*", "").strip() for x in output_list]
            return output_list
    def convert_hpo(self, output_list):
        pass