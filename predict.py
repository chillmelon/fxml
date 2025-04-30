import torch
from models.gru_module import ForexGRU


CHECKPOINT_PATH = r'lightning_logs\forex_gru\version_1\checkpoints\epoch=41-step=188328.ckpt'

def main():
    model = ForexGRU.load_from_checkpoint(CHECKPOINT_PATH)
    model.to('cpu')
    model.eval()
    x = torch.randn(1, 30).unsqueeze(-1)
    with torch.no_grad():
        prediction = model(x)
        print(prediction)

if __name__ == '__main__':
    main()
