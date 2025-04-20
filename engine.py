import math, time
import chess
from collections import namedtuple
from board_utils import tkey


MATE_VAL = 100_000                 # bigger than any eval magnitude

TTEntry = namedtuple("TTEntry", "depth score flag best_move")
EXACT, LOWER, UPPER = 0, 1, 2

# Stockfish‑style material (centipawns)
MATERIAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Stockfish 14 middlegame PSQT (white’s view; black mirrored)
# Values adapted verbatim from Stockfish 14 evaluate.cpp. :contentReference[oaicite:0]{index=0}
PSQT = {
    chess.PAWN: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10, -20, -20,  10,  10,   5,
         5,  -5, -10,   0,   0, -10,  -5,   5,
         0,   0,   0,  20,  20,   0,   0,   0,
         5,   5,  10,  25,  25,  10,   5,   5,
        10,  10,  20,  30,  30,  20,  10,  10,
        50,  50,  50,  50,  50,  50,  50,  50,
         0,   0,   0,   0,   0,   0,   0,   0],
    chess.KNIGHT: [
       -50, -40, -30, -30, -30, -30, -40, -50,
       -40, -20,   0,   0,   0,   0, -20, -40,
       -30,   0,  10,  15,  15,  10,   0, -30,
       -30,   5,  15,  20,  20,  15,   5, -30,
       -30,   0,  15,  20,  20,  15,   0, -30,
       -30,   5,  10,  15,  15,  10,   5, -30,
       -40, -20,   0,   5,   5,   0, -20, -40,
       -50, -40, -30, -30, -30, -30, -40, -50],
    chess.BISHOP: [
       -20, -10, -10, -10, -10, -10, -10, -20,
       -10,   0,   0,   0,   0,   0,   0, -10,
       -10,   0,   5,  10,  10,   5,   0, -10,
       -10,   5,   5,  10,  10,   5,   5, -10,
       -10,   0,  10,  10,  10,  10,   0, -10,
       -10,  10,  10,  10,  10,  10,  10, -10,
       -10,   5,   0,   0,   0,   0,   5, -10,
       -20, -10, -10, -10, -10, -10, -10, -20],
    chess.ROOK: [
         0,   0,   0,   0,   0,   0,   0,   0,
         5,  10,  10,  10,  10,  10,  10,   5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         0,   0,   0,   5,   5,   0,   0,   0],
    chess.QUEEN: [
       -20, -10, -10,  -5,  -5, -10, -10, -20,
       -10,   0,   0,   0,   0,   0,   0, -10,
       -10,   0,   5,   5,   5,   5,   0, -10,
        -5,   0,   5,   5,   5,   5,   0,  -5,
         0,   0,   5,   5,   5,   5,   0,  -5,
       -10,   5,   5,   5,   5,   5,   0, -10,
       -10,   0,   5,   0,   0,   0,   0, -10,
       -20, -10, -10,  -5,  -5, -10, -10, -20],
    chess.KING: [
       -30, -40, -40, -50, -50, -40, -40, -30,
       -30, -40, -40, -50, -50, -40, -40, -30,
       -30, -40, -40, -50, -50, -40, -40, -30,
       -30, -40, -40, -50, -50, -40, -40, -30,
       -20, -30, -30, -40, -40, -30, -30, -20,
       -10, -20, -20, -20, -20, -20, -20, -10,
        20,  20,   0,   0,   0,   0,  20,  20,
        20,  30,  10,   0,   0,  10,  30,  20]
}

class Engine:
    def __init__(self):
        self.tt = {}
        #self.nn = NNUE()

    # ------------------------------------------------------------------
    def evaluate(self, board: chess.Board, ply_from_root: int) -> int:
        """Static evaluation with mate‑distance scaling."""
        if board.is_checkmate():
            # side‑to‑move is mated
            return -(MATE_VAL - ply_from_root)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        sc = 0
        for p, v in MATERIAL.items():
            for sq in board.pieces(p, chess.WHITE):
                sc += v + PSQT[p][sq]
            for sq in board.pieces(p, chess.BLACK):
                sc -= v + PSQT[p][chess.square_mirror(sq)]
        return sc if board.turn else -sc

    # --------------  Quiescence  --------------------------------------
    def _qsearch(self, board, alpha, beta, ply):
        stand = self.evaluate(board, ply)
        if stand >= beta:
            return beta
        alpha = max(alpha, stand)

        for mv in board.legal_moves:
            if board.is_capture(mv) or mv.promotion or board.gives_check(mv):
                board.push(mv)
                score = -self._qsearch(board, -beta, -alpha, ply + 1)
                board.pop()
                if score >= beta:
                    return beta
                alpha = max(alpha, score)
        return alpha

    # --------------  PVS / negamax with check extension  --------------
    def _search(self, board, depth, alpha, beta, ply):
        if depth == 0:
            return self._qsearch(board, alpha, beta, ply)

        key   = tkey(board)
        entry = self.tt.get(key)
        if entry and entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.score
            if entry.flag == LOWER:
                alpha = max(alpha, entry.score)
            elif entry.flag == UPPER:
                beta  = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        alpha_orig = alpha
        best_val, best_move = -math.inf, None
        moves = sorted(board.legal_moves, key=board.is_capture, reverse=True)

        for mv in moves:
            board.push(mv)
            extra   = 1 if board.is_check() else 0    # check extension
            score   = -self._search(board,
                                     depth - 1 + extra,
                                     -beta, -alpha, ply + 1)
            board.pop()

            if score > best_val:
                best_val, best_move = score, mv
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        flag = EXACT
        if best_val <= alpha_orig:
            flag = UPPER
        elif best_val >= beta:
            flag = LOWER
        self.tt[key] = TTEntry(depth, best_val, flag, best_move)
        return best_val

    # --------------  Iterative deepening  -----------------------------
    def best_move(self, board, time_limit=2.0,
                  min_depth=2, max_depth=64):
        start      = time.time()
        best_move  = None

        for depth in range(1, max_depth + 1):
            self._search(board, depth, -math.inf, math.inf, 0)

            entry = self.tt.get(tkey(board))
            if entry and entry.best_move:
                best_move = entry.best_move

            # don’t look at the clock until min_depth done
            if depth >= min_depth and time.time() - start >= time_limit:
                break
        if best_move is None and not board.is_game_over():
            # fallback: pick *any* legal move to delay mate
            for mv in board.legal_moves:
                return mv
        return best_move
