import json
from generator.hooks import start_system, tie

# List of unwanted characters to remove after "System: "

def clean_pred(pred):
    # Check if there's a missing space after "System:" and add it
    if pred.startswith("System:") and not pred.startswith("System: "):
        pred = "System: " + pred[7:]
    
    # Remove unwanted characters and leading whitespace after "System: "
    content = pred[8:].lstrip()  # Get content after "System: " and strip leading spaces
    
    # Remove characters from remove_list or digits at the start of content
    while content and (content[0] in tie or content[0].isdigit()):
        content = content[1:].lstrip()

    # Check if the content has at least 3 words
    if len(content.split()) >= 3:
        return "System: " + content
    # If the content starts with common greetings, keep it even if it has fewer than 3 words
    elif any(content.startswith(greet) for greet in start_system):
        return "System: " + content
    else:
        return "System:"
    

def rewrite4rec(dataset, prefix_data_type, data_type):
    filtered_data = []
    ## current_dir = ..src/
    with open(f'./save/{dataset}/{prefix_data_type}_{data_type}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['pred'] != "System:":  # Filter out exact matches of "System:"
                # Clean up the 'pred' string
                cleaned_pred = clean_pred(data['pred'])
                if cleaned_pred:  # Only append if the cleaned string meets conditions
                    data['pred'] = cleaned_pred
                    filtered_data.append(data)

    # Save the filtered and cleaned data to a new file
    with open(f'./save/{dataset}/{prefix_data_type}_{data_type}.jsonl', 'w') as f:
        for item in filtered_data:
            f.write(json.dumps(item) + '\n')
