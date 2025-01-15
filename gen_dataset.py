import random

import ollama
import pysbd
from datasets import load_dataset

TP = """Given """

# every thing said is supported by...
# Examples:
# Original: Gelfand was a famous Soviet mathematician. He worked on many branches of mathematics, including
# group theory and other areas related to differential equations.
#
# supported by
# Gelfand was a famous mathematician.
# Gelfand was a Soviet mathematician.
# Gelfand was Soviet.
# Gelfand was a mathematician.
#
# *NOT* supported by
# Gelfand was a famous physicist.
# Gelfand was a Soviet physicist.
# Gelfand was American.
# Gelfand was an American mathematician.

if __name__ == "__main__":
    model = "llama3.3:70b"
    client = ollama.Client("http://atlas1api.eurecom.fr:8019")

    max_chunk_size = 5000
    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.simple")
    contents = wikipedia['train']
    page = random.choice(contents)
    page_text: str = page["text"]
    start_index = random.randint(0, max(0, len(page_text) - max_chunk_size))
    random_chunk = page_text[start_index:start_index + max_chunk_size]

    response = client.chat(model,
                           messages=[{"role": "system", "content": "Copy, identically, from the following text chunk "
                                                                   "the narrative content, written in paragraph or "
                                                                   "essay-like form. Ignore structured or list-based "
                                                                   "contents \"References,\" or \"Other websites.\""},
                                     {"role": "user", "content": random_chunk}],
                           options={"num_ctx": 32_000, "temperature": 0.})

    cleaned: str = response["message"]["content"]
    cleaned = cleaned.strip()

    seg = pysbd.Segmenter(language="en", clean=False)

    extract = random.choice(seg.segment(cleaned)).strip()
    response = client.chat(model,
                           messages=[{"role": "system", "content": "Rewrite the following sentence so to make it look "
                                                                   "like original. The new original content must "
                                                                   "remain factually correct. Output only the new "
                                                                   "content (no closing or opening explanations).\n"
                                                                   "Example:\n\n"
                                                                   "```\n"
                                                                   "Israel Moiseevich Gelfand was a famous Soviet mathematician. He worked on many branches of mathematics, including group theory and other areas related to differential equations. "
                                                                   "He received many awards, including the Order of Lenin and the first Wolf Prize, he was also a Foreign Fellow of the Royal Society and professor at Moscow State University, and worked at Rutgers University after immigrating to the United States shortly before his 76th birthday."
                                                                   "```\n\n"
                                                                   "Sentence: He worked on many branches of mathematics, including group theory and other areas related to differential equations.\n"
                                                                   "Output: He contributed to many areas of mathematics, such as differential equations and group theory."},
                                     {"role": "user",
                                      "content": f"```\n{extract}\n```\n\nSentence: {extract}\nOutput: "}],
                           options={"num_ctx": 32_000, "temperature": 0.})

    tp: str = response["message"]["content"]

    extract = random.choice(seg.segment(cleaned)).strip()
    response = client.chat(model,
                           messages=[{"role": "system", "content": "Rewrite the following sentence so to make it "
                                                                   "missing a factual information. The new original "
                                                                   "content must remain factually correct. Output only "
                                                                   "the new content (no closing or opening explanations).\n"
                                                                   "Example:\n\n"
                                                                   "```\n"
                                                                   "Israel Moiseevich Gelfand was a famous Soviet mathematician. He worked on many branches of mathematics, including group theory and other areas related to differential equations. "
                                                                   "He received many awards, including the Order of Lenin and the first Wolf Prize, he was also a Foreign Fellow of the Royal Society and professor at Moscow State University, and worked at Rutgers University after immigrating to the United States shortly before his 76th birthday."
                                                                   "```\n\n"
                                                                   "Sentence: He worked on many branches of mathematics, including group theory and other areas related to differential equations.\n"
                                                                   "Output: He contributed to many areas of mathematics, such as group theory."},
                                     {"role": "user",
                                      "content": f"```\n{extract}\n```\n\nSentence: {tp}\nOutput: "}],
                           options={"num_ctx": 32_000, "temperature": 0.})

    tp2: str = response["message"]["content"]

    extract = random.choice(seg.segment(cleaned)).strip()
    response = client.chat(model,
                           messages=[{"role": "system",
                                      "content": "Rewrite the following sentence so to introduce one subtle factual error. "
                                                 "Output only the new content (no closing or opening explanations).\n"
                                                 "Example:\n\n"
                                                 "```\n"
                                                 "Israel Moiseevich Gelfand was a famous Soviet mathematician. He worked on many branches of mathematics, including group theory and other areas related to differential equations. "
                                                 "He received many awards, including the Order of Lenin and the first Wolf Prize, he was also a Foreign Fellow of the Royal Society and professor at Moscow State University, and worked at Rutgers University after immigrating to the United States shortly before his 76th birthday."
                                                 "```\n\n"
                                                 "Sentence: He worked on many branches of mathematics, including group theory and other areas related to differential equations.\n"
                                                 "Output: He contributed to many areas of physics, such as group theory and differential equations."},
                                     {"role": "user",
                                      "content": f"```\n{extract}\n```\n\nSentence: {tp}\nOutput: "}],
                           options={"num_ctx": 32_000, "temperature": 0.})

    fn: str = response["message"]["content"]

    # TODO: should I allow the system to "pick" elements from the context rather than selecting a sentence for rewriting? Here is an alternative pipeline:
    #       For example: re-write sentence while preserving factual correctness -> then rewrite so to miss out one factual information -> rewrite so to introduce a factual error.
    #                                                                           ------- (this step may or may not be skipped) ---------->
