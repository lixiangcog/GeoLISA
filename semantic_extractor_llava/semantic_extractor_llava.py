import json
import requests
import os

headers = {"User-Agent": "LLaVA-Med Client"}

controller_endpoint = "http://localhost:10000"

def get_model_response(image_path):
    url = f"{controller_endpoint}/worker_generate_stream"
    anchor = "analysis the picture"
    
    payload = {
        "image": image_path,
        "anchor": anchor,
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json().get("text", "No response")
    else:
        print(f"Error: {response.status_code}")
        return None

def process_json_file(input_file, output_file):
    print(f"Processing file: {input_file}")
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    for entry in data:
        image_path = entry.get('image', '')
        if image_path:
            problem = get_model_response(image_path)
            if problem:
                entry['problem'] = problem

    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)

input_file = "origin_data/Fundus/fundus_result_oc.json"
output_file = "origin_data/Fundus/fundus_result_oc_after.json"

process_json_file(input_file, output_file)

print(f"Processed file saved to {output_file}")
