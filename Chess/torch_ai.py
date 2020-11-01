import chess
import numpy as np
import pandas as pd
import os
import random

indice = 500
data = pd.read_csv('E:/Chess/games.csv')
df = data['moves'].tolist()[:indice]

chess_dict = {
    'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
    'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
    'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
    'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
    'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
    'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
    'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
    'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
    'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
    'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
    'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
    'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
    '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

alpha_dict = {
    'a' : [0,0,0,0,0,0,0],
    'b' : [1,0,0,0,0,0,0],
    'c' : [0,1,0,0,0,0,0],
    'd' : [0,0,1,0,0,0,0],
    'e' : [0,0,0,1,0,0,0],
    'f' : [0,0,0,0,1,0,0],
    'g' : [0,0,0,0,0,1,0],
    'h' : [0,0,0,0,0,0,1],
}

number_dict = {
    1 : [0,0,0,0,0,0,0],
    2 : [1,0,0,0,0,0,0],
    3 : [0,1,0,0,0,0,0],
    4 : [0,0,1,0,0,0,0],
    5 : [0,0,0,1,0,0,0],
    6 : [0,0,0,0,1,0,0],
    7 : [0,0,0,0,0,1,0],
    8 : [0,0,0,0,0,0,1],
}

board = chess.Board()

def make_matrix(board:chess.Board) :
    """

    :param board:
    :return:
    """
    pgn = board.epd()
    pieces = pgn.split(" ",1)[0]
    rows = pieces.split('/')
    matrix = []
    for row in rows :
        row_filler = []
        for item in row :
            if item.isdigit() :
                for i in range(int(item)) : row_filler.append('.')
            else :
                row_filler.append(item)
        matrix.append(row_filler)
    return matrix

def translate(matrix,chess_dict) :
    rows = []
    for row in matrix :
        terms = []
        for term in row :
            terms.append(chess_dict[term])
        rows.append(terms)
    return rows

# create data
def create_data(indice=indice) :
    split_data = []
    for point in df[:indice]:
        point = point.split()
        split_data.append(point)

    data = []
    for game in split_data:
        board = chess.Board()
        for move in game:
            data.append(board.copy())
            board.push_san(move)
    trans_data = []
    for board in data:
        matrix = make_matrix(board)
        trans = translate(matrix, chess_dict)
        trans_data.append(trans)
    return trans_data

trans_data = create_data(indice=indice)
pieces = []
alphas = []
numbers = []


def seed_everything(seed=0) :
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    try :
        tf.random.set_seed(seed)
    except : pass