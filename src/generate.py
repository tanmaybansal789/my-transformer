import torch
from model import *


class Generator:
    def __init__(self, model, encoder, ):
        self.model = model
        self.encoder = encoder

    def generate_next(self, x, temperature=1.0):
        # x is a list of token ids
        x_cut = x[:, -self.model.rot.max_len:]  # cut to max context length
        logits = self.model(x_cut)
        next_token_logits = logits[:, -1, :] / temperature  # get logits for the last token and apply temperature

        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)  # sample the next token id
        return next_token_id

    @torch.inference_mode()
    def generate_multiple(self, prompt, max_new_tokens=100, temperature=1.0):
        # unsqueeze(0) to add batch dimension, and to(device) to move to the same device as the model
        x = torch.tensor(self.encoder.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_new_tokens):
            next_token_id = self.generate_next(x, temperature)

            # next_token_id already has shape (batch, 1)
            x = torch.cat([x, next_token_id], dim=1)  # append the new token id to the sequence

        return self.encoder.decode(x.squeeze().tolist())


if __name__ == '__main__':
    checkpoint = torch.load('2026-05-30-03-59---5000ep-model.pt', map_location=device)

    model = Transformer(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    encoder = CharEncoder(checkpoint['text']) if checkpoint['encoder_type'] == 'char' else tiktoken.get_encoding(checkpoint['encoder_type'])
    generator = Generator(model, encoder)
    prompt = "Once upon a time"
    generated_text = generator.generate_multiple(prompt, max_new_tokens=500)

    print(generated_text)