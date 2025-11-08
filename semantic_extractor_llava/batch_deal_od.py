import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json
from io import StringIO
import contextlib

anchor_seg_od = """
    The optic disc (optic nerve head) is the circular or oval region where retinal ganglion cell axons exit the eye; its margin corresponds to the inner edge of the scleral ring at the RPE termination.
    Please describe the size, shape, color and the position within the image of the optic disc in summary.
"""


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    conv_mode = "mistral_instruct"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] 自动推断的对话模式是 {}, 而`--conv-mode`是 {}, 使用 {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    with open(args.json_in, "r", encoding="utf-8") as f:
        data = json.load(f)

    # items = data

    for i, it in enumerate(data, start=1):
        conv = conv_templates[args.conv_mode].copy()
        image_path=it['image']
        image = load_image(image_path)
        image_tensor = process_images([image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        inp=anchor_seg_od

        if image is not None:
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        anchor = conv.get_anchor()

        input_ids = tokenizer_image_token(anchor, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        
        attention_mask = torch.ones(input_ids.shape, device=model.device)
        
        pad_token_id = model.config.eos_token_id

        stopping_criteria = []
        temperature = 0.7
        max_new_tokens = 1024

        streamer = TextStreamer(tokenizer, skip_anchor=True, skip_special_tokens=True)

        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            with torch.inference_mode():
                _ = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    pad_token_id=pad_token_id,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    streamer=streamer,
                )

        stream_text = buf.getvalue()
        stream_text = stream_text.strip()

        print(f"{image_path}:")
        print("[STREAM CAPTURED]", stream_text)
        print('-'*30)

        conv.messages[-1][-1] = stream_text

        it['problem']=stream_text

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"已写出处理结果到: {args.json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    parser.add_argument("--json_in", type=str, default="fundus_result_od.json")
    parser.add_argument("--json_out", type=str, default="fundus_result_od_after.json")

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    main(args)