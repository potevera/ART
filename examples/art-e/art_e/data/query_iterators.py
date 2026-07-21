from typing import List, Literal, Optional

from art_e.data.types_enron import SyntheticQuery
from datasets import load_dataset

# Define the Hugging Face repository ID
HF_REPO_ID = "corbt/enron_emails_sample_questions"


def load_synthetic_queries(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[SyntheticQuery]:
    dataset = load_dataset(HF_REPO_ID, split=split)  # type: ignore

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    # Passing a seed implies shuffling (deterministically), even if shuffle is False.
    if shuffle or seed is not None:
        dataset = dataset.shuffle(seed=seed)

    queries = [
        SyntheticQuery(**row, split=split)  # type: ignore
        for row in dataset  # type: ignore
    ]

    if limit is not None:
        return queries[:limit]
    return queries
