import warnings
warnings.filterwarnings("ignore")
import torch
from pathlib import Path

from src.data.tokenizer import EnTokenizer, ViTokenizer, BaseTokenizer
from src.data.parallel_dataset import nopeak_mask 
from src.utils import load_config, is_file_exist
from src.model.transformer import Transformer, get_model

def choose_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    return torch.device(device)
    
def load_latest_checkpoint(config, model, device):
    if is_file_exist(config['checkpoint_last']):
        if device == torch.device('cpu'):
            checkpoint = torch.load(config['checkpoint_last'], map_location='cpu')
        else:
            checkpoint = torch.load(config['checkpoint_last'])
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f'No checkpoint found at {config["checkpoint_last"]}')
    return model
    
def get_tokenizers(config):
    src_tokenizer = EnTokenizer(config['vocab_en'])
    trg_tokenizer = ViTokenizer(config['vocab_vi'])
    return src_tokenizer, trg_tokenizer
  
def set_up_necessary_objects():
    device = choose_device()
    
    current_file_path = Path(__file__).resolve() 
    current_dir = current_file_path.parent 
    config_path = current_dir / 'config.yaml'
    config = load_config(config_path)
    src_tokenizer, trg_tokenizer = get_tokenizers(config) 
    model = get_model(config=config, src_tokenizer=src_tokenizer, trg_tokenizer=trg_tokenizer).to(device)
    print(f'Loading model weights...')
    model = load_latest_checkpoint(config, model, device)
    model.eval()
    return model, src_tokenizer, trg_tokenizer, device, config

def translate(input_sentence: str, config, model: Transformer, src_tokenizer: BaseTokenizer, trg_tokenizer: BaseTokenizer, device):
    input_sentence = input_sentence.lower()
    sos_id = trg_tokenizer.vocab.sos_id
    eos_id = trg_tokenizer.vocab.eos_id
    with torch.no_grad():
        tokenized_input = src_tokenizer.tokenize(input_sentence)
        if len(tokenized_input) <= config['max_seq_len'] - 2:
            input_tensor = src_tokenizer.vocab.sentence_to_tensor(tokenized_input)
            input_num_padding_tokens = config['max_seq_len'] - len(input_tensor) - 2 
            enc_input = torch.cat(
                [
                    torch.tensor([src_tokenizer.vocab.sos_id], dtype=torch.int64),
                    input_tensor,
                    torch.tensor([src_tokenizer.vocab.eos_id], dtype=torch.int64),
                    torch.tensor([src_tokenizer.vocab.pad_id] * input_num_padding_tokens, dtype=torch.int64),
                ],
                dim=0,
            ).unsqueeze(0).to(device)
            enc_mask = (enc_input != src_tokenizer.vocab.pad_id).unsqueeze(0).unsqueeze(0).int().to(device)
            enc_output = model.encoder(enc_input, enc_mask)
            dec_input = torch.full((1, 1), sos_id, dtype=enc_input.dtype, device=device)
            next_token = ''
            while dec_input.size(1) != config['max_seq_len']+1 and next_token != eos_id:
                dec_mask = nopeak_mask(dec_input.size(1)).type_as(enc_mask).to(device)
                dec_output = model.decoder(dec_input, enc_output, enc_mask, dec_mask)
                prob = model.linear(dec_output[:, -1]) # [:, -1] for the last token
                _, next_token = torch.max(prob, dim=1)
                dec_input = torch.cat([
                    dec_input, torch.full((1, 1), next_token.item(), dtype=enc_input.dtype, device=device)
                ], dim=1)
            pred_sent = trg_tokenizer.tensor_to_sentence(dec_input[0, 1:-1]) # remove sos and eos tokens
        else: 
            pred_sent = 'ERROR: Input sentence is too long'
    return pred_sent
  
if __name__ == '__main__':
    model, src_tokenizer, trg_tokenizer, device, config = set_up_necessary_objects()
    while input_sentence := input('Enter an English sentence: '):
        if input_sentence == 'exit':
            break
        pred_sent = translate(input_sentence, config, model, src_tokenizer, trg_tokenizer, device)
        print(f'Vietnamese translation: {pred_sent}')
        print('---------------------------------------------------------------------')