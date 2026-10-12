"""Count the tokens the model reads for each benchmark request, with the model's own tokenizer.

    python3 scripts/count_tokens.py [--model models/original] [--out results/tokens.json]

The count includes the prompt the server prepends ("title: none | text: " for a Document,
"task: search result | query: " for a SearchQuery) and the tokenizer's special tokens, because
the model reads all of them. cost.py reads the result to price a million input tokens.
"""

import argparse
import json
import os
import sys

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import DOC, QUERY  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="models/original")
    p.add_argument("--out", default="results/tokens.json")
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = json.load(open(os.path.join(a.model, "config_sentence_transformers.json")))["prompts"]
    out = {
        "query_tokens": len(tok(prompts["SearchQuery"] + QUERY)["input_ids"]),
        "query_text_tokens": len(tok(QUERY, add_special_tokens=False)["input_ids"]),
        "doc_tokens": len(tok(prompts["Document"] + DOC)["input_ids"]),
        "doc_text_tokens": len(tok(DOC, add_special_tokens=False)["input_ids"]),
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
