"""
Standalone LoCoMo QA evaluation script using the OpenAI API.

No HuggingFace models, no shell script, no env.sh needed. Just edit the
CONFIG block below (or export OPENAI_API_KEY) and run:

    python3 run_locomo_eval_openai.py

Optionally override any config value from the CLI, e.g.:

    python3 run_locomo_eval_openai.py --model gpt-4o-mini --data-file ./data/locomo10.json

Requires:
    pip install openai tiktoken tqdm
"""

import os
import json
import time
import random
import argparse

from tqdm import tqdm

try:
    import tiktoken
except ImportError:
    tiktoken = None

from openai import OpenAI


# ============================================================
# 1. CONFIG -- edit these directly, no env.sh required
# ============================================================

CONFIG = {
    "model": "gpt-4o-mini",        # any Chat Completions model name
    "data_file": "./data/locomo10.json",
    "out_file": "./outputs/locomo10_qa_openai.json",

    "batch_size": 1,                # only batch_size == 1 is implemented
    "overwrite": False,             # re-predict questions that already have an answer
    "temperature": 0.4,
    "top_p": 0.9,

    # put your key directly here if you don't want to export it
    "api_key": os.environ.get("OPENAI_API_KEY", ""),

    # optional: point to a compatible endpoint (Azure OpenAI, proxies, etc.)
    "base_url": os.environ.get("OPENAI_BASE_URL", None),

    "max_retries": 5,
    "retry_backoff_sec": 5,
}

# context window per model (used to decide how much conversation history
# fits before the question); trimmed conservatively below max in practice
MAX_LENGTH = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 1000000,
    "gpt-4.1-mini": 1000000,
    "o3-mini": 200000,
}

ANS_TOKENS_PER_QUES = 50


# ============================================================
# 2. Prompts
# ============================================================

QA_PROMPT = """
Based on the above conversations, write a short answer for the following question in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

Question: {}
"""

CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. The conversation "
    "takes place over multiple days and the date of each conversation is "
    "written at the beginning of the conversation.\n\n"
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant whose job is to "
    "understand the following conversation and answer questions based on "
    "the conversation. If you don't know the answer to a question, please "
    "don't share false information."
)


# ============================================================
# 3. Tokenizer helper (tiktoken, falls back to a rough char-based estimate)
# ============================================================

def get_encoding(model_name):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text, encoding):
    if encoding is not None:
        return len(encoding.encode(text))
    # rough fallback: ~4 chars per token
    return max(1, len(text) // 4)


# ============================================================
# 4. Context building (same truncation logic as the HF version)
# ============================================================

def get_input_context(data, question_prompt, encoding, max_len, batch_size):
    question_tokens = count_tokens(question_prompt, encoding)

    speakers_names = list(set([d["speaker"] for d in data["session_1"]]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = count_tokens(start_prompt, encoding)

    query_conv = ""
    total_tokens = 0
    stop = False
    session_nums = [
        int(k.split("_")[-1])
        for k in data.keys()
        if "session" in k and "date_time" not in k
    ]

    for i in range(min(session_nums), max(session_nums) + 1):
        if "session_%s" % i in data:
            for dialog in data["session_%s" % i][::-1]:
                turn = dialog["speaker"] + ' said, "' + dialog["text"] + '"' + "\n"
                if "blip_caption" in dialog:
                    turn += " and shared %s." % dialog["blip_caption"]
                turn += "\n"

                new_tokens = count_tokens(
                    "DATE: " + data["session_%s_date_time" % i] + "\n" + "CONVERSATION:\n" + turn,
                    encoding,
                )
                if (start_tokens + new_tokens + total_tokens + question_tokens) < (
                    max_len - (ANS_TOKENS_PER_QUES * batch_size)
                ):
                    query_conv = turn + query_conv
                    total_tokens += count_tokens(turn, encoding)
                else:
                    stop = True
                    break

            query_conv = (
                "\nDATE: " + data["session_%s_date_time" % i] + "\n" + "CONVERSATION:\n" + query_conv
            )

        if stop:
            break

    return start_prompt + query_conv


# ============================================================
# 5. OpenAI call with retry
# ============================================================

def call_openai(client, model, messages, temperature, top_p, max_tokens, max_retries, backoff_sec):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # covers RateLimitError, APIError, APIConnectionError, etc.
            last_err = e
            print(f"[warn] OpenAI call failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


def run_openai(client, question, data, encoding, model, max_len, cfg):
    question_prompt = QA_PROMPT.format(question)
    query_conv = get_input_context(data["conversation"], question_prompt, encoding, max_len, cfg["batch_size"])

    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {"role": "user", "content": query_conv + "\n\n" + question_prompt},
    ]

    return call_openai(
        client,
        model=model,
        messages=messages,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=cfg["batch_size"] * ANS_TOKENS_PER_QUES,
        max_retries=cfg["max_retries"],
        backoff_sec=cfg["retry_backoff_sec"],
    )


# ============================================================
# 6. QA loop for a single conversation sample
# ============================================================

def get_openai_answers(in_data, out_data, client, encoding, max_len, cfg):
    batch_size = cfg["batch_size"]

    for batch_start_idx in range(0, len(in_data["qa"]) + batch_size, batch_size):
        questions = []
        cat_5_idxs = []
        cat_5_answers = []

        for i in range(batch_start_idx, batch_start_idx + batch_size):
            if i >= len(in_data["qa"]):
                break
            qa = in_data["qa"][i]

            pred_key = f"{cfg['model']}_prediction"
            if pred_key in qa and not cfg["overwrite"]:
                print("Skipping -->", qa["question"])
                continue

            if qa["category"] == 2:
                questions.append(qa["question"] + " Use DATE of CONVERSATION to answer with an approximate date.")
            elif qa["category"] == 5:
                question = qa["question"] + " (a) {} (b) {}. Select the correct answer by writing (a) or (b)."
                if random.random() < 0.5:
                    question = question.format("No information available", qa["answer"])
                    answer = {"a": "No information available", "b": qa["answer"]}
                else:
                    question = question.format(qa["answer"], "No information available")
                    answer = {"b": "No information available", "a": qa["answer"]}
                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
            else:
                questions.append(qa["question"])

        if not questions:
            continue

        if batch_size != 1:
            raise NotImplementedError("Only batch_size == 1 is implemented.")

        answer = run_openai(client, questions[0], in_data, encoding, cfg["model"], MAX_LENGTH.get(cfg["model"], 8192), cfg)
        print(questions[0], "->", answer)

        answer = answer.replace('\\"', "'").strip()
        non_empty_lines = [w.strip() for w in answer.split("\n") if w.strip()]
        answer = non_empty_lines[0] if non_empty_lines else ""

        if len(cat_5_idxs) > 0:
            answer_lower = answer.lower().strip()
            if "(a)" in answer_lower:
                answer = cat_5_answers[0]["a"]
            else:
                answer = cat_5_answers[0]["b"]
        else:
            answer = (
                answer.lower()
                .replace("(a)", "")
                .replace("(b)", "")
                .replace("a)", "")
                .replace("b)", "")
                .replace("answer:", "")
                .strip()
            )

        out_data["qa"][batch_start_idx][f"{cfg['model']}_prediction"] = answer

    return out_data


# ============================================================
# 7. Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run LoCoMo QA eval with the OpenAI API (no shell script needed).")
    parser.add_argument("--model", default=CONFIG["model"])
    parser.add_argument("--data-file", default=CONFIG["data_file"])
    parser.add_argument("--out-file", default=CONFIG["out_file"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--overwrite", action="store_true", default=CONFIG["overwrite"])
    parser.add_argument("--api-key", default=CONFIG["api_key"])
    parser.add_argument("--base-url", default=CONFIG["base_url"])
    parser.add_argument("--temperature", type=float, default=CONFIG["temperature"])
    parser.add_argument("--top-p", type=float, default=CONFIG["top_p"])
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG.update(
        {
            "model": args.model,
            "data_file": args.data_file,
            "out_file": args.out_file,
            "batch_size": args.batch_size,
            "overwrite": args.overwrite,
            "api_key": args.api_key,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
    )

    if not CONFIG["api_key"]:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY env var, pass --api-key, "
            "or fill in CONFIG['api_key'] directly."
        )

    os.makedirs(os.path.dirname(CONFIG["out_file"]) or ".", exist_ok=True)

    client_kwargs = {"api_key": CONFIG["api_key"]}
    if CONFIG["base_url"]:
        client_kwargs["base_url"] = CONFIG["base_url"]
    client = OpenAI(**client_kwargs)

    encoding = get_encoding(CONFIG["model"])
    max_len = MAX_LENGTH.get(CONFIG["model"], 8192)

    with open(CONFIG["data_file"]) as f:
        samples = json.load(f)

    if os.path.exists(CONFIG["out_file"]) and not CONFIG["overwrite"]:
        with open(CONFIG["out_file"]) as f:
            out_samples = json.load(f)
    else:
        out_samples = json.loads(json.dumps(samples))  # deep copy

    for idx, sample in enumerate(tqdm(samples, desc="Samples")):
        print(f"\n=== Sample {idx + 1}/{len(samples)} (id={sample.get('sample_id', idx)}) ===")

        out_data = {"qa": out_samples[idx]["qa"]}
        out_data = get_openai_answers(
            in_data=sample,
            out_data=out_data,
            client=client,
            encoding=encoding,
            max_len=max_len,
            cfg=CONFIG,
        )
        out_samples[idx]["qa"] = out_data["qa"]

        # save progressively after every conversation sample
        with open(CONFIG["out_file"], "w") as f:
            json.dump(out_samples, f, indent=2)

    print(f"\nDone. Predictions written to {CONFIG['out_file']}")


if __name__ == "__main__":
    main()