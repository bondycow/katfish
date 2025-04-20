"""
Katfish – a compact CPU chess engine with a PyQt5 GUI.

• GUI uses PyQt5 only ⇒ no extra GUI dependencies.
• Engine = iterative‑deepening α‑β/negamax + transposition table.
• Evaluation: Stockfish‑inspired material + PSQT (centipawns).
"""

import sys, math, time, threading
from pathlib import Path
from collections import namedtuple

from PyQt5.QtCore import Qt, QTimer, QSize, QRect
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtGui import QFont, QPainter, QColor, QPixmap, QPainter
from PyQt5.QtSvg import QSvgRenderer

from katfish_nnue import NNUE

import os
import chess
import chess.polyglot
from chess.polyglot import zobrist_hash, open_reader

BOOK_PATH = Path(os.path.expanduser(
        "~/PycharmProjects/Katfish/book/Cerebellum3Merge.bin"))
try:
    _book_file  = open(BOOK_PATH, "rb")
    BOOK_READER = open_reader(BOOK_PATH)   # returns a Reader‑like object
except (OSError, FileNotFoundError):
    BOOK_READER = None
    print("Opening book not found; engine will search from move 1.")

def book_move(board):
    """Return a Polyglot book move or None."""
    if BOOK_READER is None:
        return None
    try:
        entries = list(BOOK_READER.find_all(board))
    except IndexError:                       # position not in book
        return None
    if not entries:
        return None

    # pick highest‑weight entry; use weighted random if you prefer
    best = max(entries, key=lambda e: e.weight)
    print(f"Book: {board.san(best.move)} "
          f"(w={best.weight}, learn={best.learn})")
    return best.move

def tkey(board: chess.Board) -> int:
    """Return a 64‑bit Zobrist key – works on every python‑chess release."""
    return board.transposition_key() if hasattr(board, "transposition_key") else zobrist_hash(board)

# ───────────────────  ENGINE  ─────────────────────────────────────────

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
            if (board.is_capture(mv) or mv.promotion or board.gives_check(mv)):
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


# ───────────────────────  PyQt GUI  ───────────────────────────────────
class ChessWindow(QWidget):
    SQ     = 80
    LIGHT  = QColor("#f0d9b5")
    DARK   = QColor("#b58863")
    HL     = QColor("#f6f669")

    def __init__(self, human_is_white: bool):
        super().__init__()
        self.coord_font = QFont("Arial", int(self.SQ * 0.18))
        self.human_is_white = human_is_white
        title_side = "White" if human_is_white else "Black"
        self.setWindowTitle(f"Katfish – you are {title_side}")
        self.setFixedSize(self.SQ * 8, self.SQ * 8)

        # one‑time load of SVG pixmaps
        self.pix = {}
        svg_dir  = Path(__file__).with_name("pieces")
        for c, clr in (("w", chess.WHITE), ("b", chess.BLACK)):
            for sym, pt in zip("KQRBNP",
                               (chess.KING, chess.QUEEN, chess.ROOK,
                                chess.BISHOP, chess.KNIGHT, chess.PAWN)):
                fn   = svg_dir / f"{c}{sym}.svg"
                ren  = QSvgRenderer(str(fn))
                pm   = QPixmap(self.SQ, self.SQ)
                pm.fill(Qt.transparent)
                p    = QPainter(pm); ren.render(p); p.end()
                self.pix[(clr, pt)] = pm

        self.board    = chess.Board()
        self.redo_stack = []
        self.engine   = Engine()
        self.selected = None

        self.reply_timer = QTimer(singleShot=True)
        self.reply_timer.timeout.connect(self.engine_reply)
        self.setFocusPolicy(Qt.StrongFocus)

        if not human_is_white:  # engine (white) moves first
            QTimer.singleShot(200, self.engine_reply)

    # --- board ➜ screen --------------------------------------------------
    def to_screen(self, file_i: int, rank_i: int):
        """
        Convert board file/rank (0‑7) to window x,y pixels.
        Handles both White‑ and Black‑at‑bottom orientations.
        """
        col = file_i if self.human_is_white else 7 - file_i
        row = 7 - rank_i if self.human_is_white else rank_i
        return col * self.SQ, row * self.SQ

    # --- screen ➜ board --------------------------------------------------
    def to_board(self, x_px: int, y_px: int):
        """
        Convert window pixel coordinates to board file/rank (0‑7).
        """
        col = x_px // self.SQ
        row = y_px // self.SQ
        file_i = col if self.human_is_white else 7 - col
        rank_i = 7 - row if self.human_is_white else row
        return file_i, rank_i

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Left:  # UNDO one ply
            if self.board.move_stack:
                mv = self.board.pop()
                self.redo_stack.append(mv)
                # self.engine.nn.refresh(self.board)  # resync NNUE
                self.selected = None
                self.update()
        elif ev.key() == Qt.Key_Right:  # REDO one ply
            if self.redo_stack:
                mv = self.redo_stack.pop()
                self.board.push(mv)
                # self.engine.nn.refresh(self.board)
                self.selected = None
                self.update()
        else:
            super().keyPressEvent(ev)

    def disp_rank(self, rank_0_top: int) -> int:
        """Convert board rank to display row depending on orientation."""
        return (7 - rank_0_top) if self.human_is_white else rank_0_top

    # ------------- drawing -------------------------------------------
    # ───────────────────────── helpers (unchanged) ─────────────────────
    # to_screen(file, rank)  – board → window pixel
    # to_board(x_px, y_px)  – window pixel → board file,rank

    # ───────────────────────── paintEvent ───────────────────────────────
    def paintEvent(self, _):
        qp = QPainter(self)

        # 1. Draw squares (a1 dark).
        for br in range(8):  # board rank 0..7
            for bf in range(8):  # board file 0..7
                x, y = self.to_screen(bf, br)
                colour = self.DARK if (bf + br) & 1 == 0 else self.LIGHT
                qp.fillRect(x, y, self.SQ, self.SQ, colour)

        # 2. Coordinate labels: files on bottom, ranks on left.
        qp.setFont(self.coord_font)

        for board_rank in range(8):  # 0 = rank 1 (White side)
            for board_file in range(8):  # 0 = file a
                x, y = self.to_screen(board_file, board_rank)

                # pick label colour for contrast with square
                qp.setPen(QColor("#000000") if (board_file + board_rank) & 1 else QColor("#ffffff"))

                # file letters on *board* rank 1
                if board_rank == 0:
                    qp.drawText(x + 2, y + self.SQ - 4, chr(ord('a') + board_file))

                # rank numbers on *board* file a
                if board_file == 0:
                    qp.drawText(x + 2, y + 12, str(board_rank + 1))

        # 3. Draw pieces.
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if not piece:
                continue
            bf, br = chess.square_file(sq), chess.square_rank(sq)
            x, y = self.to_screen(bf, br)
            qp.drawPixmap(x, y, self.pix[(piece.color, piece.piece_type)])

        # 4. Highlight selected square.
        if self.selected is not None:
            bf, br = chess.square_file(self.selected), chess.square_rank(self.selected)
            x, y = self.to_screen(bf, br)
            qp.setPen(self.HL)
            qp.setBrush(Qt.NoBrush)
            qp.drawRect(x, y, self.SQ, self.SQ)

    # ───────────────────────── mousePressEvent ──────────────────────────
    def mousePressEvent(self, ev):
        file, rank = self.to_board(ev.x(), ev.y())
        sq = chess.square(file, rank)

        if self.selected is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.human_is_white:
                self.selected = sq
        else:
            mv = chess.Move(self.selected, sq)
            if mv in self.board.legal_moves and self.board.turn == self.human_is_white:
                self.make_move(mv)
                self.reply_timer.start(200)
            self.selected = None
        self.update()

    def make_move(self, mv):
        self.redo_stack.clear()
        self.board.push(mv)
        if self.board.is_game_over():
            QMessageBox.information(self, "Game over", self.board.result())

    def engine_reply(self):
        if self.board.is_game_over() or self.board.turn == self.human_is_white:
            return
        # ---- NEW: try book first ---------------------------------------
        mv = book_move(self.board)
        if mv:
            self.make_move(mv)
            self.update()
            return
        # ----------------------------------------------------------------
        mv = self.engine.best_move(self.board, 2.0)
        if mv:
            self.make_move(mv)
            self.update()
        if mv is None:
            # should happen only in game‑over positions
            print("No legal moves – game must be over.")

# ----------------------------- main -----------------------------------
def main():
    play_white = True  # default
    if len(sys.argv) > 1 and sys.argv[1].lower().startswith("b"):
        play_white = False  # “python main.py black”
    app = QApplication(sys.argv)
    win = ChessWindow(human_is_white=play_white)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()