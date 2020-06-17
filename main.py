# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from model.hf_argparser import HfArgumentParser
from model.tokenization_bert import BertTokenizer
from model.configuration_albert import AlbertConfig
from model.justice_model import AlbertForJustice

#
# from transformers import (
#     AutoConfig,
#     AutoModelForMultipleChoice,
#     AutoTokenizer,
#     EvalPrediction,
#     HfArgumentParser,
#     Trainer,
#     TrainingArguments,
#     set_seed,
# )
from justice_examples import MultipleChoiceDataset, Split, processors
from model.trainer import set_seed, Trainer
from model.trainer_utils import EvalPrediction
from model.training_args import TrainingArguments

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="eval_model_dir/",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(
        default="justice_race",
        metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())}
    )

    data_dir: str = field(
        default="/input/",
        metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AlbertForJustice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        cache_dir=model_args.cache_dir,
    )

    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = p.predictions
        ac = (preds == p.label_ids).mean()
        # preds = np.argmax(p.predictions, axis=1)
        return {"acc": np.array(ac).mean()}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluation
    output_map = {
        0: [],
        1: ['A'],
        2: ['B'],
        4: ['C'],
        8: ['D'],
        5: ['A', 'B'],
        6: ['B', 'C'],
        7: ['A', 'B', 'C'],
        9: ['A', 'D'],
        10: ['B', 'D'],
        11: ['A', 'B', 'D'],
        12: ['C', 'D'],
        13: ['A', 'C', 'D'],
        14: ['B', 'C', 'D'],
        15: ['A', 'B', 'C', 'D']
    }

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # result = trainer.evaluate()
        result = trainer.predict(eval_dataset)
        result_out = {}
        for i, j in zip(result.predictions, result.id):
            result_out[j] = output_map[i]

        output_eval_file = os.path.join(training_args.output_dir, "result.txt")

        json.dump(result_out, open(output_eval_file, "w", encoding="utf8"), indent=2, ensure_ascii=False,
                  sort_keys=True)


if __name__ == "__main__":
    main()
