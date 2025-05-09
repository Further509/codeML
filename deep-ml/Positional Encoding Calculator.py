import numpy as np

def pos_encoding(position: int, d_model: int):
    # Your code here
    if position == 0 or d_model <= 0:
        return -1
    pos_enc = np.zeros((position, d_model))
    pos = np.arange(position)[:, np.newaxis]
    _2i = np.arange(0, d_model, 2)[np.newaxis, :]

    pos_enc[:, 0::2] = np.sin(pos / (10000 ** (_2i / d_model)))
    pos_enc[:, 1::2] = np.cos(pos / (10000 ** (_2i / d_model)))
    pos_enc = np.float16(pos_enc)
    return pos_enc

if __name__ == "__main__":
    print(pos_encoding(2, 8))
    print(pos_encoding(8, 2))