import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
disable_torch_init()
SYSTEM_MESSAGE = (
    "You are a medical AI assistant with expertise in medical genetics, dysmorphology, "
    "and image-based phenotype recognition. Your task is to analyze facial and neck images "
    "to identify any observable phenotypes related to known human diseases. "
    "These phenotypes can include abnormalities in morphology, development, skin, "
    "face, hair, eyes, ears, nose, mouth, or neck. You must use the Human "
    "Phenotype Ontology (HPO) to normalize all detected phenotypes. "
    "All phenotypes should be separated by ||. Please do not provide any additional texts or formatting."
)
class LLaVA_Generator():
    def __init__(self, model_path, model_base = 'microsoft/llava-med-v1.5-mistral-7b'):
        #MODEL = "/home/nguyenqm/projects/github/LLaVA/phenotype_vision/llava-med-v1.5-13b-task-lora-gmdb-80mix-clip"
        model_name = get_model_name_from_path(model_name)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                                                            model_path=model_path, 
                                                            model_base=model_base, 
                                                            model_name=model_name, 
                                                            load_4bit=False) #True )
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
        qs = SYSTEM_MESSAGE
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(img_path)#.convert('RGB')
        image = self.base_preprocess(img)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                # no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return(outputs)
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

