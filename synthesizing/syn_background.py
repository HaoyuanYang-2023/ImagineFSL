import openai
import os
import json
import time
import argparse


from utils.attribute_gpt_templates import *
from utils.classnames import CLASSNAMES as labels


parser = argparse.ArgumentParser(description='OpenAI ChatGPT API')
parser.add_argument('--api_key', type=str, default='', help='Your API Key')
parser.add_argument('--model', type=str, default='gpt-4o', help='model') 
parser.add_argument('--dataset', type=str, default='flowers', help='Dataset')
args = parser.parse_args()

# IF NEED PROXY
os.environ['HTTP_PROXY'] = "127.0.0.1:1080"
os.environ['HTTPS_PROXY'] = "127.0.0.1:1080"


def load_gpt(api_key):

    os.environ["OPENAI_API_KEY"] = api_key
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1")

    return client

def convert_unicode(unicode_str):
    return unicode_str.encode().decode('unicode_escape')

def get_prompt(category,):

    prompt = origion_background.format(category, category)

    print(prompt)

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    return messages


def get_attribute(dataset, classes, save_path):
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"{dataset}.json")
    for cls in classes:
        print(f"processing: {cls}")
        attribute_dict = get_attribute_from_json(save_path)
        if cls not in attribute_dict.keys():
            while True:
                time.sleep(0.01)
                cls_def = cls
                messages = get_prompt(cls_def)
                response = client.chat.completions.create(
                    model=llm_model,  
                    temperature=0.8,  
                    # Lower values for temperature result in more consistent outputs (e.g. 0.2), while higher values generate more diverse and creative results (e.g. 1.0). Select a temperature value based on the desired trade-off between coherence and creativity for your specific application. The temperature can range is from 0 to 2. Reference: https://platform.openai.com/docs/guides/text-generation
                    messages=messages, 
                )

                # If the response is not in python code block, continue to get the response
                response_text = response.choices[0].message.content
                if "```python" not in response_text:
                    continue
                
                print(response_text)

                if response_text.startswith("```python"):
                    response_text = response_text[len("```python")+1:]
                else:
                    response_split = response_text.split("python")
                    if len(response_split) != 1:
                        response_text = response_split[-1]
                if response_text.startswith("attributes = "):
                    response_text = response_text[len("attributes = "):]
                response_split = response_text.split("=")
                if len(response_split) != 1:
                    response_text = response_split[-1]
                if response_text.endswith("```"):
                    response_text = response_text[:-len("```")].strip()
                if not response_text.endswith(']'):
                    response_split = response_text.split("]")
                    response_text = response_split[0] + ']'
                    
                # Convert ASCII to Unicode, and then convert Unicode to Python string
                # Avoid "\uxxxx" in the string
                try:
                    response_text = convert_unicode(response_text)
                    attribute_dict[cls] = eval(response_text)
                except:
                    continue
                else:
                    break

        with open(save_path, 'w') as f:
            json.dump(attribute_dict, f, indent=4)


def get_attribute_from_json(json_file):
    attribute_dict = {}
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            attribute_dict = json.load(f)
    return attribute_dict

if __name__ == '__main__':
    
    print("====> Start to get backgrounds")
    client = load_gpt(api_key=args.api_key)
    llm_model = args.model

    dataset = args.dataset

    classes = labels[dataset]
    print(f'Number of classes: {len(classes)}')

    save_path = f"./backgrounds"

    get_attribute(dataset, classes, save_path)
