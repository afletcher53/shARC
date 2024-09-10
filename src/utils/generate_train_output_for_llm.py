'''
Generates a training output Images for the LLM model to describe the input and output grids of a challenge.
'''


import base64
import os
from classes.data_loader import DataLoader

from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key='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')

def generate_nld_of_example(sample:int = None):
    """
    Generates a natural language description of images using an LLM model to describe the input and output grids of a challenge.
    """


    
    files = [f for f in os.listdir('output/singles') if os.path.isfile(os.path.join('output/singles', f))]

    if sample is not None:
        files = files[:sample]
        
    
    batches = {}
    for f in files:
        batch = f.split('_')[0]
        if batch not in batches:
            batches[batch] = []
        batches[batch].append(f)
    
    
    responses = {}
    for batch in batches:
        images_encoded = []
        for f in batches[batch]:
            
            with open(f'output/singles/{f}', 'rb') as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                images_encoded.append(encoded_image)

        
        context = (
            "You are the world's best natural language describer of logic puzzles presented in grid format. "
            "These grids represent abstract logic problems where the challenge is to understand and describe "
            "the transformation that occurs between the input and output grids. Your task is to analyze the "
            "patterns, changes, and rules governing these transformations with exceptional detail and clarity."
        )
        prompt = (
            f"{context}\n\n"
            f"Here are some training examples for the challenge {batch}. Describe, in detail, the transformation "
            f"that occurs between the input and output grids. Focus on identifying patterns, rules, and logical "
            f"operations that explain the changes observed. Be as precise and comprehensive as possible in your description."
        )

        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        
        for image in images_encoded:
            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                }
            )

        
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=messages,
            max_tokens=300
        )

        reply = response.choices[0].message.content

        responses[batch] = reply

    return responses

        
        

