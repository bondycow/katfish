from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QPainter, QPixmap, QFont
from PyQt5.QtSvg import QSvgRenderer

import chess
from pathlib import Path

from engine import Engine
from board_utils import book_move
# from .katfish_nnue import NNUE  # if using NNUE later

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
        # ---- NEW: try books first ---------------------------------------
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
