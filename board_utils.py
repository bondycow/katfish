import os
from pathlib import Path
import chess.polyglot
from chess.polyglot import zobrist_hash, open_reader

BOOK_PATH = Path(os.path.expanduser("~/PycharmProjects/Katfish/book/Cerebellum3Merge.bin"))
try:
    _book_file = open(BOOK_PATH, "rb")
    BOOK_READER = open_reader(BOOK_PATH)
except (OSError, FileNotFoundError):
    BOOK_READER = None
    print("Opening book not found; engine will search from move 1.")

def tkey(board: chess.Board) -> int:
    return board.transposition_key() if hasattr(board, "transposition_key") else zobrist_hash(board)

def book_move(board):
    if BOOK_READER is None:
        return None
    try:
        entries = list(BOOK_READER.find_all(board))
    except IndexError:
        return None
    if not entries:
        return None
    best = max(entries, key=lambda e: e.weight)
    print(f"Book: {board.san(best.move)} (w={best.weight}, learn={best.learn})")
    return best.move
