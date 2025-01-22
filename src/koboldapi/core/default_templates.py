default_templates = {
    "alpaca": {
        "name": [
            "Alpaca",
            "open-orca-platypus2",
            "nous-hermes",
            "deepseek-coder",
            "wizardcoder",
            "wizard-math",
            "codeup",
            "codebooga"
        ],
        "system_start": "",
        "system_end": "\n\n",
        "user_start": "### Instruction:\n",
        "user_end": "\n\n",
        "assistant_start": "### Response:\n",
        "assistant_end": "</s>\n\n"
    },
    "athene-v2": {
        "name": [
            "athene-v2"
        ],
        "system_start": "<|im_start|>system\n",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "tool_instruction": "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": \n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n"
    },
    "chatglm_v3": {
        "name": [
            "ChatGLM-3",
            "ChatGLM-4"
        ],
        "system_start": "[gMASK]sop<|system|>\n",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "",
        "user_start": "<|user|>\n",
        "user_end": "",
        "assistant_start": "<|assistant|>",
        "assistant_end": ""
    },
    "chatml": {
        "name": [
            "ChatML",
            "obsidian",
            "Nous",
            "Hermes",
            "qwen",
            "MiniCPM-V-2.6",
            "QvQ",
            "nous-hermes2-mixtral",
            "smallthinker",
            "solar-pro",
            "marco-o1",
            "opencoder",
            "dolphin3",
            "sailor2",
            "mistral-openorca",
            "nous-hermes2",
            "openhermes",
            "tinydolphin",
            "samantha-mistral",
            "orca2",
            "dolphin-phi",
            "meditron",
            "megadolphin",
            "yi",
            "qwen2",
            "qwen2.5",
            "dolphin-mixtral",
            "dolphin-mistral",
            "dolphin-llama3",
            "smollm",
            "codeqwen",
            "stable-code",
            "stablelm2",
            "dolphincoder",
            "yi-coder",
            "internlm2",
            "reader-lm",
            "dbrx",
            "bespoke-minicheck"
        ],
        "system_start": "<|im_start|>system\n",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n"
    },
    "cmdr": {
        "name": [
            "Command-r",
            "aya",
            "cmdr",
            "command-r-plus",
            "aya-expanse",
            "c4ai",
            "command-r7b"
        ],
        "system_start": "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
        "system_end": "<|END_OF_TURN_TOKEN|>",
        "user_start": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        "user_end": "<|END_OF_TURN_TOKEN|>",
        "assistant_start": "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        "assistant_end": "<|END_OF_TURN_TOKEN|>",
        "tools_instruction": "# Safety Preamble\nThe instructions in this section override those in the task description and style guide sections.\n\n# System Preamble\n## Basic Rules\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.\n\n\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n```python\ndef test_function() -> List[Dict]:\n'''\n'''\npass\n```"
    },
    "deepseek": {
        "name": [
            "DeepSeek"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "",
        "user_start": "### Instruction:\n",
        "user_end": "\n",
        "assistant_start": "### Response:\n",
        "assistant_end": "\n<|EOT|>\n"
    },
	"DeepSeek-2": {
        "name": [
            "DeepSeek-2",
            "deepseek-coder-v2"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "\n\n",
        "user_start": "User: ",
        "user_end": "\n\n",
        "assistant_start": "Assistant: ",
        "assistant_end": "<\uff5cend\u2581of\u2581sentence\uff5c>"
    },
    "deepseek-2.5": {
        "name": [
            "DeepSeek-2.5"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "\n",
        "user_start": "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>",
        "user_end": "<\uff5cend\u2581of\u2581sentence\uff5c>",
        "assistant_start": "<\uff5cAssistant\uff5c>",
        "assistant_end": "<\uff5cend\u2581of\u2581sentence\uff5c>"
    },
    "deepseek-3": {
        "name": [
            "DeepSeek-3"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "\n\n",
        "user_start": "<\uff5cUser\uff5c>",
        "user_end": "",
        "assistant_start": "<\uff5cAssistant\uff5c>",
        "assistant_end": "<\uff5cend\u2581of\u2581sentence\uff5c>"
    },
    "default": {
        "name": [
            "default"
        ],
        "system_start": "System: ",
        "system_end": "\n\n",
        "user_start": "User: ",
        "user_end": "\n",
        "assistant_start": "Assistant: ",
        "assistant_end": "\n"
    },
    "exaone_v3": {
        "name": [
            "ExaOne-3"
        ],
        "system_start": "[|system|]",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "[|endofturn|]\n",
        "user_start": "[|user|]",
        "user_end": "\n",
        "assistant_start": "[|assistant|]",
        "assistant_end": "[|endofturn|]\n"
    },
	"Gemma": {
        "name": [
            "Gemma, codegemma"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "",
        "user_start": "<start_of_turn>user\n",
        "user_end": "<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n",
        "assistant_end": "<end_of_turn>\n"
    },
    "gemma-2": {
        "name": [
            "gemma2"
        ],
        "system_start": "<start_of_turn>system\n",
        "system_end": "<end_of_turn>\n",
        "user_start": "<start_of_turn>user\n",
        "user_end": "<end_of_turn>\n",
        "assistant_start": "<start_of_turn>model\n",
        "assistant_end": "<end_of_turn>\n"
    },
    "gigachat": {
        "name": [
            "GigaChat"
        ],
        "system_start": "<s>",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "<|message_sep|>",
        "user_start": "user<|role_sep|>",
        "user_end": "<|message_sep|>available functions<|role_sep|>[]<|message_sep|>",
        "assistant_start": "assistant<|role_sep|>",
        "assistant_end": "<|message_sep|>"
    },
    "granite": {
        "name": [
            "Granite"
        ],
        "system_start": "<|start_of_role|>system<|end_of_role|>",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "<|end_of_text|>\n",
        "user_start": "<|start_of_role|>user<|end_of_role|>",
        "user_end": "<|end_of_text|>\n",
        "assistant_start": "<|start_of_role|>assistant<|end_of_role|>",
        "assistant_end": "<|end_of_text|>\n"
    },
    "hermes3": {
        "name": [
            "hermes3"
        ],
        "system_start": "<|im_start|>system\n",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "tool_instruction": "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools: <tools>\n{{- range .Tools }}\n{\"type\": \"function\", \"function\": {{ .Function }}}\n{{- end }}  </tools> Use the following pydantic model json schema for each tool call you will make: {\"properties\": {\"arguments\": {\"title\": \"Arguments\", \"type\": \"object\"}, \"name\": {\"title\": \"Name\", \"type\": \"string\"}}, \"required\": [\"arguments\", \"name\"], \"title\": \"FunctionCall\", \"type\": \"object\"} For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{\"arguments\": <args-dict>, \"name\": <function-name>}\n</tool_call>\n"
    },
    "llama2": {
        "name": [
            "Llama-2"
        ],
        "system_start": "",
        "system_end": "",
        "user_start": "<s>[INST] ",
        "user_end": " [/INST]",
        "assistant_start": " ",
        "assistant_end": " </s>"
    },
    "llama3": {
        "name": [
            "Llama-3",
            "MiniCPM-V-2.5",
            "llama3.3",
            "llama3.1",
            "llama3.2",
            "llama3.1-vision",
            "llama3-grandient",
            "reflection",
            "llama3-groq-tool-use",
            "firefunction-v2"
        ],
        "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end": "<|eot_id|>",
        "tools_instruction": "Cutting Knowledge Date: December 2023\n\nWhen you receive a tool call response, use the output to format an answer to the orginal user question.\n\nYou are a helpful assistant with tool calling capabilities."
    },
    "megrez": {
        "name": [
            "Megrez"
        ],
        "system_start": "<|role_start|>system<|role_end|>",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "<|turn_end|>",
        "user_start": "<|role_start|>user<|role_end|>",
        "user_end": "<|turn_end|>",
        "assistant_start": "<|role_start|>assistant<|role_end|>",
        "assistant_end": "<|turn_end|>"
    },
    "metharme": {
        "name": [
            "Metharme"
        ],
        "system_start": "<|system|>",
        "system_end": "",
        "user_start": "<|user|>",
        "user_end": "",
        "assistant_start": "<|model>",
        "assistant_end": ""
    },
    "minicpm": {
        "name": [
            "MiniCPM"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "",
        "user_start": "<\u7528\u6237>",
        "user_end": "<AI>",
        "assistant_start": "",
        "assistant_end": ""
    },
    "mistral": {
        "name": [
            "Mistral",
            "Miqu",
            "Mixtral",
            "mathstral",
            "codestral",
            "Mistral-V1"
        ],
        "system_start": "",
        "system_end": "",
        "user_start": " [INST] ",
        "user_end": "",
        "assistant_start": " [/INST]",
        "assistant_end": "</s>"
    },
    "mistral_large": {
        "name": [
            "Mistral Large",
            "Mistral Small",
            "Mistral 2409",
            "Mistral-V3",
            "Mistral-V2"
        ],
        "system_start": "",
        "system_end": "",
        "user_start": "[INST] ",
        "user_end": "",
        "assistant_start": "[/INST]",
        "assistant_end": "</s>"
    },
    "mistral_nemo": {
        "name": [
            "Mistral Nemo",
            "Mistral-V3-Tekken"
        ],
        "system_start": "",
        "system_end": "",
        "user_start": "[INST]",
        "user_end": "",
        "assistant_start": "[/INST]",
        "assistant_end": "</s>"
    },
    "nemotron": {
        "name": [
            "nemotron"
        ],
        "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "assistant_end": "<|eot_id|>",
        "tools_instruction": "You have access to the following functions. To call a function, please respond with JSON for a function call. Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables.\n\nmap[Function:map[Arguments:{\"test\": \"value\"} Name:test_function]]\n\n"
    },
    "nemotron-mini": {
        "name": [
            "nemotron-mini"
        ],
        "system_start": "<extra_id_0>System",
        "system_end": "",
        "user_start": "<extra_id_1>User",
        "user_end": "",
        "assistant_start": "<extra_id_1>Assistant",
        "assistant_end": "",
        "tools_instruction": "<extra_id_0>System\n<tool> map[Function:map[Arguments:{\"test\": \"value\"} Name:test_function]] </tool>"
    },
    "neural": {
        "name": [
            "neural-chat"
        ],
        "system_start": "### System:\n",
        "system_end": "\n\n",
        "user_start": "### User:\n",
        "user_end": "\n\n",
        "assistant_start": "### Assistant:\n",
        "assistant_end": "\n\n"
    },
	"OpenChat": {
        "name": [
            "OpenChat"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "<|end_of_turn|>",
        "user_start": "GPT4 Correct User: ",
        "user_end": "<|end_of_turn|>",
        "assistant_start": "GPT4 Correct Assistant: ",
        "assistant_end": "<|end_of_turn|>"
    },
    "phi": {
        "name": [
            "Phi-3"
        ],
        "system_start": "<|system|>\n",
        "system_end": "<|end|>\n",
        "user_start": "<|user|>\n",
        "user_end": "<|end|>\n",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "<|end|>\n"
    },
    "qwq": {
        "name": [
            "qwq"
        ],
        "system_start": "<|im_start|>system",
        "system_end": "<|im_end|>",
        "system_instruction": "",
        "user_start": "<|im_start|>user",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "tools_instruction": "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": map[Arguments:{\"test\": \"value\"} Name:test_function]}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    },
    "rwkv_world": {
        "name": [
            "RWKV-World"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "",
        "user_start": "User: ",
        "user_end": "\n\nAssistant:",
        "assistant_start": "",
        "assistant_end": "\n\n"
    },
    "smollm2": {
        "name": [
            "smollm2"
        ],
        "system_start": "<|im_start|>system\n",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "tool_instruction": "You are an expert in composing functions. You are given a question and a set of possible functions.\nBased on the question, you will need to make one or more function/tool calls to achieve the purpose.\nIf none of the functions can be used, point it out and refuse to answer.\nIf the given question lacks the parameters required by the function, also point it out.\n\nYou have access to the following tools:\n<tools></tools>\n\nThe output MUST strictly adhere to the following format, and NO other text MUST be included.\nThe example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.\n<tool_call>[\n{\"name\": \"func_name1\", \"arguments\": {\"argument1\": \"value1\", \"argument2\": \"value2\"}},\n(more tool calls as required)\n]</tool_call>"
    },
    "vicuna": {
        "name": [
            "Vicuna"
        ],
        "system_start": "",
        "system_instruction": "",
        "system_end": "\n\n",
        "user_start": "USER: ",
        "user_end": "\n",
        "assistant_start": "ASSISTANT: ",
        "assistant_end": "</s>\n"
    },
    "vicuna_orca": {
        "name": [
            "Vicuna-Orca"
        ],
        "system_start": "SYSTEM: ",
        "system_instruction": "You are a helpful assistant.",
        "system_end": "\n",
        "user_start": "USER: ",
        "user_end": "\n",
        "assistant_start": "ASSISTANT: ",
        "assistant_end": "</s>\n"
    }
}