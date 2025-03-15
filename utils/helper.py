import json
from together import Together
import pandas as pd
from sklearn.model_selection import train_test_split
import json

def read_json(dataset: str) -> str:
    # Open and read the JSON file
    with open(dataset, 'r') as file:
        data = json.load(file)
    return json.dumps(data, indent=2, default=str)


def gen_preprocessing_tips(csv_json:str) -> str:
    ''' General function to call the LLM '''

    client = Together()

    message = [
        {"role": "system", "content": """
# Instructions:
You are a data scientist assistant. Given some csv information in json format, you will give me some preprocessing tips.
Your preprocessing tips will be specific to the data that I am about to give you.

# Output:
- You will give me some output on how clean/unclean the data is.
- You will generate the most useful 3-5 tips in bullet format, no excuses. Each tip will be maximum 1 sentence long.

## Note:
- You will not generate anything else, no meta-commentary, nothing. You must try your best.
- Please do not confuse this with feature engineering tips. I want as many preprocessing tips here.

"""},
        {"role": "user", "content": f"""
Here is the information of my csv:
-----
{csv_json}    
-----
"""}
    ]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=message,
        max_tokens=None,
        temperature=0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
    )
    answer = completion.choices[0].message.content
    print("Preprocessing Response:\n", answer)

    return answer 

def gen_feature_tips(csv_json:str, goal:str="", column:str="") -> str:
    ''' General function to call the LLM '''

    client = Together()

    # column of interest
    if column=="" or column=="None":
        column_prompt=""
    else:
        column_prompt=f"- The column of interest for the above objective is: **{column}**"

    # goal of interest
    if goal=="" or goal=="None":
        goal_prompt=""
    else: # classification, regression
        goal_prompt = f"""
# Data Goal:
- The user's objective is to utilize this data for: **{goal}**
{column_prompt}

"""

    message = [
        {"role": "system", "content": f"""
# Instructions:
You are a data scientist assistant. Given some csv information in json format, you will give me some feature engineering tips.
Your feature engineering tips will be specific to the data that I am about to give you.

# Output:
- You will generate the most useful 3-5 tips in bullet format, no excuse. Each tip will be maximum 1 sentence long.

## Note:
- You will not generate anything else, no meta-commentary, nothing. You must try your best.
- Please do not confuse this with preprocessing tips. I want as many feature engineering here.

{goal_prompt}
"""},
        {"role": "user", "content": f"""
Here is the information of my csv:
-----
{csv_json}    
-----
"""}
    ]
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=message,
        max_tokens=None,
        temperature=0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
    )
    answer = completion.choices[0].message.content
    print("Feature Response:\n", answer)

    return answer

def split_data(df, train_size, val_size, test_size, random_seed):
    if val_size > 0:
        train_data, temp_data = train_test_split(df, train_size=train_size, random_state=random_seed)
        val_data, test_data = train_test_split(temp_data, test_size=test_size / (val_size + test_size), random_state=random_seed)
    else:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_seed)
        val_data = pd.DataFrame()
    return train_data, val_data, test_data

def save_json_to_file(content, file_path="messages.json"):
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(content, json_file, indent=4, ensure_ascii=True)
        return True, f"Data successfully saved to {file_path}"
    except IOError as e:
        return False, f"IO Error: Failed to write to file {file_path}. {str(e)}"
    except TypeError as e:
        return False, f"Type Error: Content is not JSON serializable. {str(e)}"
    except Exception as e:
        return False, f"Unexpected error while saving JSON: {str(e)}"
