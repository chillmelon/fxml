from datetime import datetime, timezone, timedelta
import socket
import threading
import time
import queue
import numpy as np
import pandas as pd
from models.gru_module import ForexGRU
import torch
import joblib

CHECKPOINT_PATH = r'lightning_logs\forex_gru\version_1\checkpoints\epoch=41-step=188328.ckpt'
SCALER_PATH = './scaler.pkl'

# Load *model* and *scaler* from file
scaler = joblib.load(SCALER_PATH)
model = ForexGRU.load_from_checkpoint(CHECKPOINT_PATH)
model.to('cpu')
model.eval()

resolution = 30
# tick_queue = queue.Queue()
rate_queue = queue.Queue()
bar_queue = queue.Queue()
tensor_queue = queue.Queue()

# def read_tick(conn, addr):
#     try:
#         while True:
#             msg = conn.recv(8192).decode()
#             if msg != "":
#                 print("[INFO]\tMessage:", msg)
#                 tick_queue.put(msg)
#     except ConnectionError:
#         print("[INFO]\tread lost connection to ", addr)

def read_rate(conn, addr):
    try:
        while True:
            msg = conn.recv(8192).decode()
            if msg != "":
                print("[INFO]\tMessage:", msg)
                rate_queue.put(msg)
    except ConnectionError:
        print("[INFO]\tread lost connection to ", addr)

def write(conn, addr):
    try:
        while True:
            conn.send("hi\n".encode())
            time.sleep(3)
    except ConnectionError:
        print("[INFO]\twrite lost connection to ", addr)

def to_tensor():
    while True:
        rates = rate_queue.get()
        print(rates)
        tensor_queue.put(msg_to_tensor(rates))


def floor_time(dt, seconds=60):
    """Rounds down a datetime to nearest interval."""
    return dt - timedelta(seconds=dt.second % seconds,
                          microseconds=dt.microsecond)

# def compressor(resolution=60):
#     bars = {}
#     bar = {}
#     current_bar_time = None

#     while True:
#         tick = tick_queue.get()
#         price = list(tick.values())[0]  # Assuming tick is like {'price': ...}

#         tick_time = datetime.now(timezone.utc)
#         bar_time = floor_time(tick_time, resolution)

#         # New bar started
#         if current_bar_time is None or bar_time > current_bar_time:
#             if bar:
#                 bar_queue.put(bar.copy())
#                 bars[current_bar_time] = bar.copy()
#                 print(f"[BAR @ {current_bar_time}] {bar}")

#             # start new bar
#             bar = {
#                 'open': price,
#                 'high': price,
#                 'low': price,
#                 'close': price,
#             }
#             current_bar_time = bar_time
#         else:
#             # Update current bar
#             try:
#                 bar['high'] = max(bar['high'], price)
#                 bar['low'] = min(bar['low'], price)
#                 bar['close'] = price
#             except:
#                 print('first bar forming')


def gru_strategy():

    while True:
        with torch.no_grad():
            x = tensor_queue.get()
            prediction = model(x)
            print(x, prediction)


def msg_to_tick(msg: str):
    tick = {}
    values = msg.rstrip("\n").split(',')
    ts = datetime.fromtimestamp(float(values[1]) / 1000.0)
    tick[ts] = values[2]
    return tick


def msg_to_tensor(msg: str):
    # msg: symbol,timestamp,close1,close2,...,closeN
    data = msg.strip().split(',')[2:]
    closes = np.array(data, dtype=np.float32)
    delta = np.diff(closes)
    pct_delta = delta / closes[:-1]

    pct_delta = pct_delta.reshape(-1, 1)  # (seq_len, 1)

    pct_delta_scaled = scaler.transform(pct_delta)

    seq = pct_delta_scaled.reshape(1, -1, 1)  # (batch_size=1, seq_len, feature_dim=1)
    tensor = torch.FloatTensor(seq)
    return tensor


def main():

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 8888))
    server.listen(10)
    try:
        connection, address = server.accept()
        print("[INFO]\tConnection established with: ", address)

        thread_read = threading.Thread(target=read_rate, args=(connection, address))
        thread_read.start()

        thread_write = threading.Thread(target=write, args=(connection, address))
        thread_write.start()

        thread_tensor = threading.Thread(target=to_tensor)
        thread_tensor.start()

        # thread_compressor = threading.Thread(target=compressor, args=(resolution,))
        # thread_compressor.start()

        thread_strategy = threading.Thread(target=gru_strategy)
        thread_strategy.start()

    except KeyboardInterrupt:
        print("Shutting down server.")
        server.close()


if __name__ == '__main__':
    main()
