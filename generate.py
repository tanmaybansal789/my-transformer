import torch
from model import *


class Generator:
    def __init__(self, model, encoder, ):
        self.model = model
        self.encoder = encoder

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0):
        # unsqueeze(0) to add batch dimension, and to(device) to move to the same device as the model
        x = torch.tensor(self.encoder.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        for _ in range(max_new_tokens):
            # B, T = shape
            x_cut = x[:, -self.model.rot.max_len:]
            logits = self.model(x_cut)
            next_token_logits = logits[:, -1, :] / temperature

            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token_id], dim=1)

        return self.encoder.decode(x.squeeze().tolist())


if __name__ == '__main__':
    checkpoint = torch.load('2026-05-30-03-59---5000ep-model.pt', map_location=device)

    model = Transformer(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    encoder = CharEncoder(checkpoint['text']) if checkpoint['encoder_type'] == 'char' else tiktoken.get_encoding(checkpoint['encoder_type'])
    generator = Generator(model, encoder)
    prompt = "Once upon a time"
    generated_text = generator.generate(prompt, max_new_tokens=500)

    print(generated_text)