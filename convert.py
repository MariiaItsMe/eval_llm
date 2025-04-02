import json

if __name__ == "__main__":
    with open("evaluation_dataset.json", "r") as f:
        data = json.loads(f.read())

    output = []
    for q in data["questions"]:
        answers = {k: v["human"] for k, v in q["answers"].items()}
        item = {"question": q["question"], "ground_truth": q["ground_truth"], "answers": answers}
        output.append(item)

    with open("dataset.json",  "w") as f:
        f.write(json.dumps(output, indent=4))
