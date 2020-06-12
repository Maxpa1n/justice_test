import torch
import glob
import json
import os
import tqdm
from torch.utils.data.dataset import Dataset
from filelock import FileLock
from dataclasses import dataclass
from typing import List, Optional
import logging
from enum import Enum

from model.tokenization_bert import BertTokenizer
from model.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    endings: List[str]
    label: List[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    data_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: List[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


def _create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (_, data_raw) in enumerate(lines):
        race_id = "%s-%s" % (set_type, data_raw["race_id"])
        article = data_raw["article"]
        for i in range(len(data_raw["answers"])):
            truth = str(ord(data_raw["answers"][i]) - ord("A"))
            question = data_raw["questions"][i]
            options = data_raw["options"][i]

            examples.append(
                InputExample(
                    example_id=race_id,
                    question=question,
                    contexts=[article, article, article, article],  # this is not efficient but convenient
                    endings=[options[0], options[1], options[2], options[3]],
                    label=truth,
                )
            )
    return examples


class JusticeRaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train.json")
        high = self._read_json(high)
        return self._create_examples(high, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev.json")
        high = self._read_json(high)
        return self._create_examples(high, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test.json")
        high = self._read_json(high)
        return self._create_examples(high, "test")

    def get_labels(self):
        """See base class."""
        return ["A", "B", "C", "D"]

    def _read_json(self, input_dir):
        lines = []
        with open(input_dir, "r", encoding="utf-8") as fin:
            for i in tqdm.tqdm(fin, desc="read files"):
                data_raw = json.loads(i)
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["id"])
            question = data_raw['statement']
            truth = data_raw['answer']
            options = data_raw["option_list"]

            examples.append(
                InputExample(
                    example_id=race_id,
                    question=question,
                    endings=[options['A'], options['B'], options['C'], options['D']],
                    label=truth,
                )
            )
        return examples


processors = {"justice_race": JusticeRaceProcessor, }
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"justice_race", 4}


class MultipleChoiceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length), task, ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                logger.info("Training examples: %s", len(examples))
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                    examples,
                    label_list,
                    max_seq_length,
                    tokenizer,
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        data_id = example.example_id
        for ending_idx, ending in enumerate(example.endings):
            text_a = ending
            text_b = example.question

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = 0
        if "A" in example.label:
            label += 1
        if "B" in example.label:
            label += 2
        if "C" in example.label:
            label += 4
        if "D" in example.label:
            label += 8


        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                data_id=data_id,
                example_id=example.example_id,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


if __name__ == '__main__':

    pretrained_token = './albert_model_pretrain/'
    tokenizer = BertTokenizer.from_pretrained(pretrained_token)
    dataset = MultipleChoiceDataset(data_dir='data/',
                                    tokenizer=tokenizer,
                                    task='justice_race',
                                    max_seq_length=512,
                                    overwrite_cache=False,
                                    mode=Split.train, )
