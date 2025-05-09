import torch

from models.gru_model import GRUModule


CHECKPOINT_PATH = r'lightning_logs\prob_gru\version_6\checkpoints\best_checkpoint.ckpt'

def main():
    model = GRUModule.load_from_checkpoint(CHECKPOINT_PATH)
    model.to('cpu')
    model.eval()

    with torch.no_grad():
        for i in range(10):
            x = torch.randn(1, 30, 1)
            prediction = model(x)
            print(prediction)

if __name__ == '__main__':
    main()
