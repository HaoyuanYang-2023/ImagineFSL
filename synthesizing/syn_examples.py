
import openai
import os
import json
import random
import time
import argparse
from utils.example_gpt_templates import context_templates, base_context_templates
from utils.classnames import CLASSNAMES

parser = argparse.ArgumentParser(description='OpenAI ChatGPT API')
parser.add_argument('--model', type=str, default='gpt-4o', help='model')
parser.add_argument('--dataset', type=str, default='flowers', help='Dataset')
parser.add_argument('--api_key', type=str, default='', help='Your API Key')
args = parser.parse_args()

def load_gpt(api_key):
    os.environ["OPENAI_API_KEY"] = api_key
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")
    return client


def get_utils(bg_path, attribute_path):

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

    return attribute_dict, viewpoints, background_dict, lighting_list, image_quality



def fill_blanks(type, cls, bg_path, attribute_path, weighted=False):
    format_list = []
    save_dict = {"label": cls}
    attribute_dict, viewpoints, background_dict, lighting_list,  image_quality = get_utils(bg_path, attribute_path)

    if cls not in attribute_dict.keys():
        return "Not in attribute_dict", None
    attribute = random.choice(attribute_dict[cls])
    format_list.append(attribute)
    save_dict["Attribute"] = attribute

    viewpoint = random.choice(viewpoints)

    format_list.append(viewpoint)
    save_dict["Viewpoint"] = viewpoint

    if "BG" in type:
        def weighted_choice():
            if cls in background_dict.keys():
                backgrounds = background_dict[cls]
            else:
                backgrounds = background_dict[category]
            if not weighted:
                return random.choices(backgrounds, k=1)
            else:
                n = len(backgrounds)
                high_weight = 4
                low_weight = 1.5

                weights = [high_weight] * int(n * 0.6) + [low_weight] * max(0, n - int(n * 0.6))
                return random.choices(backgrounds, weights=weights, k=1)
        background = weighted_choice()
        format_list.append(background[0])
        save_dict["Background"] = background[0]
    elif "LC" in type:
        light = random.choice(lighting_list)
        format_list.append(light)
        save_dict["Lighting conditions"] = light
    elif "CD" in type:
        image_quality_info = random.choice(image_quality)
        format_list.append(image_quality_info)
        save_dict["Image quality"] = image_quality_info

    return format_list, save_dict


def get_prompts(generate_mode, cat_description, cls, path, bg_path, attribute_path, weighted=True):
    format_list, save_dict = fill_blanks(generate_mode, cls, bg_path, attribute_path, weighted)
    if format_list == "Not in attribute_dict":
        return "Not in attribute_dict", None
    if generate_mode == 'Base':
        template = base_context_templates
        prompt = template.format(cat_description + cls, cls, cls, cls, *format_list, cls)

    else:
        template = context_templates

        description = {
            "LC": f'lighting condition where the {cls} is photographed',
            "BG": f'background where the {cls} is photographed',
            "CD": "degradation causes resulting in deterioration of image quality"
        }
        # prompt = template.format(cls, cls, description[generate_mode], cls, attribute, viewpoint_info, specified_info, cls)
        prompt = template.format(cat_description + cls, cls, description[generate_mode], cls, *format_list,
                                 cls)

    return prompt, save_dict


def convert_unicode(unicode_str):
    return unicode_str.encode().decode('unicode_escape')

def main(category, description, prompt_type, path, bg_path, attribute_path, save_path,
         api_key):
    client = load_gpt(api_key)

    classes = CLASSNAMES[category]

    example_dict = {}
    if os.path.isfile(save_path):
        with open(save_path, 'r') as f:
            example_dict = json.load(f)

    for cls in classes:
        if cls in example_dict.keys() and len(example_dict[cls]) >= 20:
            continue
        if cls not in example_dict.keys():
            print(f"cls: {cls}")
            example_dict[cls] = []

            while len(example_dict[cls]) < 10:
                time.sleep(0.1)
                prompt, save_dict = get_prompts(prompt_type, description, cls, path, bg_path, attribute_path)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                print(prompt)
                try:
                    response = client.chat.completions.create(
                        model=llm_model,  # 模型名称
                        max_tokens=100,  # 最大输出token的数量
                        temperature=0.8,  # 温度
                        # Lower values for temperature result in more consistent outputs (e.g. 0.2), while higher values generate more diverse and creative results (e.g. 1.0). Select a temperature value based on the desired trade-off between coherence and creativity for your specific application. The temperature can range is from 0 to 2. Reference: https://platform.openai.com/docs/guides/text-generation
                        messages=messages,  # 输入的文本
                    )

                    # 提取并解析GPT返回的响应
                    content = response.choices[0].message.content
                    if "\"" in content:
                        content = content.replace("\"", "")
                    content = content.split("\n")[0]
                    if content.startswith("1."):
                        content = content[2:]
                    content = content.replace("**", "")
                    content = content.strip()
                    content = convert_unicode(content)
                    save_dict["content"] = content
                    example_dict[cls].append(save_dict)
                except Exception as e:
                    print(e)
                    print(f"Failed to generate examples for {cls}")
                    continue

            with open(save_path, 'w') as f:
                json.dump(example_dict, f, indent=4)



if __name__ == '__main__':

    datasets = {
        'flowers': 'a type of flower: ',
        "food101": 'a type of food: ',
        'cars': 'a car: ',
        'pets': 'a type of pet: ',
        'aircraft': 'an aircraft: ',
        'dtd': ['the texture: ',
                'an object with the texture: ',
                'an object with the pattern: '],
        'sun397': 'a scene: ',
        'caltech101': 'an object: ',
        'eurosat': 'a type of satellite land cover: ',
        'ucf101_frames': 'an action: ',
        'imagenet': 'an object: '
    }
    description = datasets[args.dataset]
    category = args.dataset


    if category in ['dtd', 'eurosat']:
        generated_modes = ['Base']
    elif category == 'sun397':
        generated_modes = ['Base', 'LC', 'CD']
    else:
        generated_modes = ['Base','BG', 'LC', 'CD']
    
    # generated_modes = ['LC','CD']
    if isinstance(description, list):
        description = random.choice(description)

    for generated_mode in generated_modes:

        # 读取json文件
        lighting_path = "lighting_conditions.json"
        bg_path = f"./backgrounds/{category}.json"
        attribute_path = f"./attributes/{category}.json"

        save_path = f"./examples/{category}/{generated_mode}_examples.json"
        os.makedirs(f"./examples/{category}", exist_ok=True)
        

        llm_model = args.model
        api_key = args.api_key

        main(category, description, generated_mode, lighting_path, bg_path, attribute_path, save_path, api_key)


