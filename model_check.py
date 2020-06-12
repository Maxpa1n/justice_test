import transformers
from model.tokenization_bert import BertTokenizer
from model.configuration_albert import AlbertConfig
from model.justice_model import AlbertForJustice
import torch

if __name__ == '__main__':
    pretrained_token = './albert_model_pretrain/'
    tokenizer = BertTokenizer.from_pretrained(pretrained_token)
    # config = AlbertConfig.from_pretrained('./albert_model_pretrain/')
    # config.output_hidden_states = True
    model = AlbertForJustice.from_pretrained('./albert_model_pretrain/')
    choices = ["今天心情很好", "今天心情很好"]
    input_ids = torch.tensor(
        [tokenizer.encode(s, add_special_tokens=True, max_length=10, pad_to_max_length=True) for s in
         choices]).unsqueeze(0)
    labels = torch.tensor([1, 0]).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=labels)

    loss, classification_scores = outputs[:2]
    print(classification_scores)
    print(loss)
