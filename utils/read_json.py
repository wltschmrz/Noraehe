import re
import json
import fsspec

def read_json(json_path):
    config_dict = {}
    try:
        with fsspec.open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.decoder.JSONDecodeError:
        # backwards compat.
        data = read_json_with_comments(json_path)
    config_dict.update(data)
    return config_dict

def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data
