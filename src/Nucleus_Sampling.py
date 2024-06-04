from transformers import (  # type: ignore
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from pathlib import Path
import typing
import numpy as np
import torch
import time


def top_k_top_p_filtering(logits, top_k=50, top_p=0.95):
    # Sort logits and keep top k tokens only
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = -float("Inf")

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Sort the probabilities to identify the cumulative sum up to top_p
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, descending=True
    )
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # Remove tokens with a cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
        ..., :-1
    ].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    probabilities[indices_to_remove] = 0

    return probabilities


def generate_sequences(
    input_text, model, tokenizer, num_sequences=3, max_length=500
):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated_sequences = []

    for _ in range(num_sequences):
        output_sequence = input_ids
        for _ in range(max_length):
            outputs = model(output_sequence, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            filtered_probabilities = top_k_top_p_filtering(next_token_logits)
            next_token = torch.multinomial(filtered_probabilities, 1)
            next_token = next_token.to(device)
            output_sequence = torch.cat([output_sequence, next_token], dim=-1)

            # Stop generating if the model produces the end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break

        decoded_sequence = tokenizer.decode(
            output_sequence, skip_special_tokens=True
        )
        generated_sequences.append(decoded_sequence)

    return generated_sequences


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model_path = "models/bart-large_claim_decomp_fact_checking_briefs_faviq_a_set_faviq_r_set_gpt_generated_4/checkpoint-52090"
    # model_path = "Factiverse/bart-large-claimdecomp"
    model_path = "models/bart-base_all_0/checkpoint-20836"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_auth_token=True, device=device
    )

    while True:
        inp = input("Enter prompt: ")
        if inp == "q":
            break

        # print(generate_sequences(inp, model, tokenizer))
        model_inputs = tokenizer(inp, return_tensors="pt").to(device)
        start_time = time.time()
        torch.manual_seed(int(input("Enter seed: ")))
        sample_outputs = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=3,
            repetition_penalty=1.2,
            temperature=1.5,
            epsilon_cutoff=3e-4,
            diversity_penalty=0.5,
            num_beam_groups=3,
            num_beams=9,
        )
        end_time = time.time()
        time_taken = end_time - start_time
        for i, sample_output in enumerate(sample_outputs):
            print(
                "{}: {}".format(
                    i, tokenizer.decode(sample_output, skip_special_tokens=True)
                )
            )
        print(f"Time taken: {time_taken} seconds")
        print("Output:\n" + 100 * "-")
