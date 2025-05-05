import torch

from models.gru_prob_regr import ProbabilisticGRURegressor


CHECKPOINT_PATH = r'lightning_logs\prob_gru\version_10\checkpoints\best_checkpoint.ckpt'

def main():
    model = ProbabilisticGRURegressor.load_from_checkpoint(CHECKPOINT_PATH)
    model.to('cpu')
    model.eval()

    with torch.no_grad():
        for i in range(10):
            x = torch.randn(1, 30).unsqueeze(-1)
            prediction = model(x)
            print(prediction)

if __name__ == '__main__':
    main()
