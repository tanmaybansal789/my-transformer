from model import *
import datetime

class Trainer:
    @staticmethod
    def from_checkpoint(checkpoint):
        model = Transformer(**checkpoint['model_config'])
        return Trainer(
            encoder_type=checkpoint['encoder_type'],
            text=checkpoint['text'],
            model_config=checkpoint['model_config'],
            training_config=checkpoint['training_config'],
            model=model
        )

    def __init__(self, encoder_type, text, model_config, training_config, model=None):
        self.text = text

        self.encoder_type = encoder_type
        self.encoder = CharEncoder(text) if encoder_type == 'char' else tiktoken.get_encoding(encoder_type)

        self.model_config = model_config
        self.model_config['n_vocab'] = self.encoder.n_vocab

        self.training_config = training_config

        self.model = model if model is not None else Transformer(**model_config)
        self.optimiser = torch.optim.Adam(self.model.parameters())

        self.load_data(training_config['train_split'])

    def load_data(self, train_split):
        self.data = torch.tensor(self.encoder.encode(self.text), dtype=torch.long)
        n = int(train_split * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def get_batch(self, split, batch_size, block_size):
        data = self.train_data if split == 'train' else self.val_data
        # generate random indices (< len(data) - block_size) to make sure we have enough room to get a full sequence of block_size characters
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        # each character predicts the next character, so we offset by 1
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y
    
    def train(self, save_checkpoint=True):
        print('Starting training...')
        self.model.train()
    
        try:
            for epoch in range(self.training_config['epochs']):
                x, y = self.get_batch('train', self.training_config['batch_size'], self.training_config['block_size'])

                logits = self.model(x)

                loss = F.cross_entropy(einops.rearrange(logits, 'b t c -> (b t) c'), 
                                    einops.rearrange(y, 'b t -> (b t)'))

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f'epoch {epoch+1}/{self.training_config["epochs"]}, loss={loss.item()}')
        except KeyboardInterrupt:
            print('Training interrupted. Saving checkpoint...')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'encoder_type': self.encoder_type,
            'text': self.text
        }
        if save_checkpoint:
            torch.save(checkpoint, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}---{self.encoder_type}-{self.training_config["epochs"]}ep-model.pt')
        else:
            return checkpoint

if __name__ == '__main__':
    text = open('input.txt', 'r').read()

    model_config = dict(
        n_vocab=None,
        max_len=256,      # Expanded from 32! Model can now read/remember 256 characters at once
        d_embed=384,      # Increased from 128. Gives each token a much richer vector space
        n_heads=6,        # 384 / 6 = 64 head dimension (d_k stays 64, which is the sweet spot)
        d_hidden=1536,
        n_blocks=6,
        base=10_000
    )

    training_config = dict(
        batch_size=64,
        block_size=256,
        epochs=100,
        train_split=0.9
    )

    trainer = Trainer(
        encoder_type='char',
        text=text,
        model_config=model_config,
        training_config=training_config
    )
    trainer.train()