from collections import defaultdict
import re
import os
import inspect
import sys
import importlib


def deep_nested():
    return defaultdict(lambda: defaultdict(list))


def revert_defaultdict(obj):
    if isinstance(obj, defaultdict):
        return {k: revert_defaultdict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: revert_defaultdict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [revert_defaultdict(i) for i in obj]
    else:
        return obj


def parser_response_into_coordinates(text, meta=None):
    pattern = r"""
        (?:x\s*[:=]?\s*)?                 
        [\(\[\{]?              
        \s*([-+]?(?:\d+\.\d+|\.\d+|\d+))\s* 
        [,\s;]+                       
        (?:y\s*[:=]?\s*)?        
        \s*([-+]?(?:\d+\.\d+|\.\d+|\d+))\s* 
        [\)\]\}]?    
    """

    matches = re.findall(pattern, text, re.IGNORECASE | re.VERBOSE)
    if len(matches) == 0:
        return None
    else:
        return [(float(x), float(y)) for x, y in matches]


def parser_answers_into_option(text, meta=None):
    patterns = [
        r"\b([A-F])[\.:](?!\w)",  # parser: A.  B:
        r"\bOption\s+([A-F])\b",  # Option A / Option B
        r"\bAnswer\s*[:：]?\s*([A-F])\b",  # Answer: A
        r"^[ \t]*([A-F])[\.:]?",  # the first letter in a row: A. or A:
        r"[\'\"]([A-F])[\'\"]",  # 'A' or "B"
        r"\b([A-F])\b(?!\s+\w)",  # a single alphabet A-F, without following word
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def dynamic_import_function(path: str, bash_path: str = None):
    try:
        module_name = ".".join(path.split(".")[:-1])
        function_name = path.split(".")[-1]

        if bash_path is None:
            caller_file = inspect.stack()[1].filename
            bash_path = os.path.abspath(os.path.join(os.path.dirname(caller_file), ".."))
        
        if bash_path not in sys.path:
            sys.path.append(bash_path)
        
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
        return function
    except Exception as e:
        raise ImportError(f"Failed to import function {path} from {bash_path}: {e}")