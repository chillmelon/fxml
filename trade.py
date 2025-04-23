from datetime import datetime, timezone, timedelta
import socket
import threading
import time
import queue
import numpy as np
import pandas as pd
from models.gru import GRUModel
import torch
import joblib


resolution = 30
tick_queue = queue.Queue()
rate_queue = queue.Queue()
bar_queue = queue.Queue()
tensor_queue = queue.Queue()
scaler = joblib.load('models/saved/scaler.pkl')

def read_tick(conn, addr):
    try:
        while True:
            msg = conn.recv(8192).decode()
            if msg != "":
                print("[INFO]\tMessage:", msg)
                tick_queue.put(msg)
    except ConnectionError:
        print("[INFO]\tread lost connection to ", addr)

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
        rates = scaler.transform(rates)
        print(rates)
        tensor_queue.put(msg_to_tensor(rates))


def floor_time(dt, seconds=60):
    """Rounds down a datetime to nearest interval."""
    return dt - timedelta(seconds=dt.second % seconds,
                          microseconds=dt.microsecond)

def compressor(resolution=60):
    bars = {}
    bar = {}
    current_bar_time = None

    while True:
        tick = tick_queue.get()
        price = list(tick.values())[0]  # Assuming tick is like {'price': ...}

        tick_time = datetime.now(timezone.utc)
        bar_time = floor_time(tick_time, resolution)

        # New bar started
        if current_bar_time is None or bar_time > current_bar_time:
            if bar:
                bar_queue.put(bar.copy())
                bars[current_bar_time] = bar.copy()
                print(f"[BAR @ {current_bar_time}] {bar}")

            # start new bar
            bar = {
                'open': price,
                'high': price,
                'low': price,
                'close': price,
            }
            current_bar_time = bar_time
        else:
            # Update current bar
            try:
                bar['high'] = max(bar['high'], price)
                bar['low'] = min(bar['low'], price)
                bar['close'] = price
            except:
                print('first bar forming')

# def macd_strategy(conn):
#     short_ema_span = 12
#     long_ema_span = 26
#     signal_span = 9

#     while True:
#         bar = bar_queue.get()
#         print(bar)
#         sw.add(bar['close'])  # feed closing price to sliding window

#         if len(sw.data) >= long_ema_span + signal_span:
#             # get close price window
#             closes = np.array(sw.data, dtype=np.float64)

#             # calculate EMAs
#             ema_short = pd.Series(closes).ewm(span=short_ema_span, adjust=False).mean()
#             ema_long = pd.Series(closes).ewm(span=long_ema_span, adjust=False).mean()
#             macd_line = ema_short - ema_long
#             signal_line = macd_line.ewm(span=signal_span, adjust=False).mean()

#             # detect crossover
#             if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
#                 print("[STRATEGY] MACD Crossover detected: BUY")
#                 conn.send(b"buy\n")
#             elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
#                 print("[STRATEGY] MACD Crossunder detected: SELL")
#                 conn.send(b"sell\n")
#             else:
#                 conn.send(b"hold\n")


def gru_strategy():
    gru_model = GRUModel(
        input_size=1,
        hidden_size=64,
        num_layers=1,
        output_size=1,
        dropout=0.0,
        bidirectional=False
    )

    while True:
        tensor = tensor_queue.get()
        prediction = gru_model(tensor).item()
        print(prediction)


def msg_to_tick(msg: str):
    tick = {}
    values = msg.rstrip("\n").split(',')
    ts = datetime.fromtimestamp(float(values[1]) / 1000.0)
    tick[ts] = values[2]
    return tick


def msg_to_tensor(msg: str):
    data = msg.strip().split(',')[1:]

    seq = np.array(data, dtype=np.float32).reshape(1, -1, 1)  # (1, seq_len, 1)
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

        # thread_write = threading.Thread(target=write, args=(connection, address))
        # thread_write.start()

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
