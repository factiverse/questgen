"""Test Seq2Seq Language Model."""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore

if __name__ == "__main__":
    model_checkpoint = (
        "src/QuestionGeneration/T5/wandb/"
        "run-20230709_044116-c1924a9g/files/model/checkpoint-23500"
    )
    print(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, use_auth_token=True
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    print("type something")
    while True:
        inp = input()
        if inp == "q":
            break
        input_ids = tokenizer(inp, return_tensors="pt")
        print(input_ids["input_ids"])
        outputs = model.generate(input_ids["input_ids"])
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))