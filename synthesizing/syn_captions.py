import os

import fire
import json
import math
import random

from llama import Llama
from tqdm import tqdm
from typing import Optional
from utils.llama_prompt_templates import TEMPLATE_new as TEMPLATE
from utils.classnames import CLASSNAMES


def check_lost(json_path, total_captions):
    if json_path is None or not (os.path.exists(json_path)):
        return None
    with open(json_path, 'r') as file:
        line_count = sum(1 for line in file)
    lost_num = total_captions - line_count
    return lost_num if lost_num > 0 else None


def get_utils(bg_path, attribute_path, category):

    with open('utils/lighting_conditions_viewpoints.json', 'r') as f:
        data = json.load(f)
    lighting_list = data['lighting_condition']
    image_quality = data["low_image_quality"]

    with open('utils/viewpoints.json', 'r') as v:
        viewpoints = json.load(v)[category]

    background_dict = None
    if os.path.exists(bg_path):
        with open(bg_path, 'r') as f:
            background_dict = json.load(f)

    with open(attribute_path, 'r') as f:
        attribute_dict = json.load(f)

    return attribute_dict, viewpoints, background_dict, lighting_list,  image_quality


def generate_format_args(generated_mode, examples,
                         chosen_idx, foreground, category, bg_path, attribute_path):
    format_args = []

    format_args.append('"' + foreground + '"')

    attribute_dict, viewpoints, background_dict, lighting_list,  image_quality = get_utils(bg_path, attribute_path, category)
    
    if "BG" in generated_mode:
        if foreground in background_dict.keys():
            background_list = background_dict[foreground]
        else:
            background_list = background_dict[category]

        def weighted_choice():
            n = len(background_list)
            high_weight = 4
            low_weight = 1.5

            weights = [high_weight] * int(n * 0.6) + [low_weight] * max(0, n - int(n * 0.6))
            return random.choices(background_list, weights=weights, k=1)[0]
        background_info = weighted_choice()

    mode_mapping = {
        "LC": "Lighting conditions",
        "CD": "Image quality",
        "BG": "Background"
    }

    for idx in chosen_idx:
        format_args.append(examples[idx]['Attribute'])
        format_args.append(examples[idx]['Viewpoint'])

        for key in mode_mapping:
            if key in generated_mode:
                format_args.append(examples[idx][mode_mapping[key]])

        format_args.append(examples[idx]['content'])

    format_args.append(random.choice(attribute_dict[foreground]))
    format_args.append(random.choice(viewpoints))

    if "LC" in generated_mode:
        format_args.append(random.choice(lighting_list))
    if "CD" in generated_mode:
        format_args.append(random.choice(image_quality))
    if "BG" in generated_mode:
        format_args.append(background_info)

    return format_args


def get_generate_mode(category):
    if category in ['dtd', 'eurosat']:
        generated_modes = ['Base']
        weights = [1]
    elif category == 'sun397':
        generated_modes = ['Base', 'LC', 'CD']
        weights = [0.7, 0.2, 0.1]
    else:
        generated_modes = ['Base', 'BG', 'LC', 'CD']
        weights = [0.4, 0.3, 0.2, 0.1]
    return random.choices(generated_modes, weights, k=1)[0]


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        category: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
        total_captions: int = 1000,
        seed: int = 42,
):
    output_filename = f"./captions/{category}_caption"
    os.makedirs(output_filename, exist_ok=True)

    bg_path = f"./backgrounds/{category}.json"
    attribute_path = f"./attributes/{category}.json"

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    generator=None
    # process saving dir
    new_prompt_filename = output_filename
    print('saving to ', new_prompt_filename)

    random.seed(seed)

    classes = CLASSNAMES[category]

    
    for i, foreground in enumerate(classes):
        foreground_ori = foreground
        foreground = foreground.replace("_","/")
        print(f"Generating NO.{i} {foreground}:")
        foreground = foreground.strip()
        prompt_filename = new_prompt_filename + f'/{foreground_ori}.txt'
        lost_num = check_lost(prompt_filename, total_captions)
        print(lost_num)
        if lost_num:
            num_batches = math.ceil(lost_num / max_batch_size)
        elif os.path.exists(prompt_filename):
            continue
        else:
            num_batches = math.ceil(total_captions / max_batch_size)

        new_prompts = []

        for batch_idx in tqdm(range(num_batches)):
            prompts = []
            for in_batch_idx in range(max_batch_size):
                generated_mode = get_generate_mode(category)

                example_path = f"./examples/{category}/{generated_mode}_examples.json"
                example_dict = json.load(open(example_path))

                examples = example_dict[foreground]

                num_example = len(examples)
                chosen_idx = random.sample(range(num_example), 3)
                prompt_template = TEMPLATE[generated_mode]

                format_args = generate_format_args(generated_mode, examples, chosen_idx, foreground, category, bg_path, attribute_path)
                current_prompt = prompt_template.format(*format_args)
                prompts.append(current_prompt)

            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            for res_idx, result in enumerate(results):
                generation = result['generation']
                new_prompt = generation.split('\n')[0]
                new_prompt = new_prompt.replace('\n', ' ')
                new_prompt = new_prompt.strip()
                new_prompts.append(new_prompt)

        if '/' in foreground:
            foreground = foreground.replace('/', '_')
        prompt_filename = new_prompt_filename + f'/{foreground}.txt'
        with open(prompt_filename, 'a') as f:
            f.writelines([p.strip().replace('\n', ' ') + '\n' for p in new_prompts])


if __name__ == "__main__":
    fire.Fire(main)