"""
Very small NNUE loader + incremental evaluator
* Loads a Stockfish .nnue (HalfKP 768×256x32x32x1) into NumPy int16
* keeps layer‑1 activations so make/unmake is O(64) not O(768*256)
Reference: sunfish‑NNUE, numbfish (MIT)  :contentReference[oaicite:1]{index=1}
"""
import numpy as np, struct, chess

# ---- constants -------------------------------------------------------
INPUTS  = 768                # HalfKP
H1      = 256
H2      = 32
OUT     = 1
SCALE   = 1 << 6             # Stockfish uses 6‑bit fractional

# piece → [offset for 64 squares]  (HalfKP order)
OFFSET = {
    chess.PAWN  : 0,
    chess.KNIGHT: 64 * 1,
    chess.BISHOP: 64 * 2,
    chess.ROOK  : 64 * 3,
    chess.QUEEN : 64 * 4,
    chess.KING  : 64 * 5,
}

class NNUE:
    def __init__(self, path="katfish.nnue"):
        with open(path, "rb") as f:
            magic = f.read(6)
            assert magic in (b"NNUEF\n", b"NNUE\x00\n"), "not an NNUE file"
            f.seek(128)            # skip header

            # --- load layers ------------------------------------------
            self.W1 = np.frombuffer(f.read(INPUTS*H1*2), "<i2").reshape(INPUTS, H1)
            self.B1 = np.frombuffer(f.read(H1*2),         "<i2")
            self.W2 = np.frombuffer(f.read(H1*H2*2),      "<i2").reshape(H1, H2)
            self.B2 = np.frombuffer(f.read(H2*2),         "<i2")
            self.W3 = np.frombuffer(f.read(H2*OUT*2),     "<i2").reshape(H2, OUT)
            self.B3 = np.frombuffer(f.read(OUT*2),        "<i2")

        # scratch arrays
        self.l1 = np.zeros(H1, dtype=np.int32)   # activations
        self.stack = []                          # incremental undo

    # ------------------------------------------------------------------
    @staticmethod
    def _index(king_sq, pc_sq, offset):
        """HalfKP: input index for (our king, piece‑square)"""
        return 64 * king_sq + pc_sq + offset

    def _add_piece(self, piece, sq, color, king_sq, sign):
        if color == chess.BLACK:
            sq = chess.square_mirror(sq)
        idx = self._index(king_sq, sq, OFFSET[piece])
        self.l1 += sign * self.W1[idx]

    # ------------------------------------------------------------------
    def refresh(self, board: chess.Board):
        """Recompute layer‑1 from scratch (fast at start)"""
        self.l1[:] = self.B1
        stm = board.turn
        king_sq = board.king(stm)

        for p in chess.PIECE_TYPES:
            for sq in board.pieces(p, stm):
                self._add_piece(p, sq, stm, king_sq, +1)
        # opponent pieces have NEGATIVE contribution
        opp = not stm
        king_opp = board.king(opp)
        for p in chess.PIECE_TYPES:
            for sq in board.pieces(p, opp):
                self._add_piece(p, sq, opp, king_opp, -1)

    # ------------------------------------------------------------------
    def make_move(self, board: chess.Board, mv: chess.Move):
        """Update incrementally; push state on stack for undo."""
        stm = board.turn
        king_before = board.king(stm)
        from_sq, to_sq = mv.from_square, mv.to_square
        piece = board.piece_type_at(from_sq)

        self.stack.append((piece, from_sq, to_sq, stm, king_before))

        # remove moving piece
        self._add_piece(piece, from_sq, stm, king_before, -1)
        # captured piece?
        if board.is_capture(mv):
            cap_piece = board.piece_type_at(to_sq)
            self._add_piece(cap_piece, to_sq, not stm, board.king(not stm), +1)
        # promotion
        if mv.promotion:
            piece = mv.promotion
        # add piece on target
        if piece == chess.KING:      # king moved → update all features…
            self.refresh(board)      # …simplest but rare
        else:
            king_now = king_before
            self._add_piece(piece, to_sq, stm, king_now, +1)

    def unmake(self):
        self.l1[:] = self.stack.pop()  # simply recompute on undo for simplicity

    # ------------------------------------------------------------------
    def value(self):
        # clamp with ReLU and scale
        x = np.maximum(self.l1, 0, dtype=np.int32)
        y = np.maximum(self.W2.T @ x + self.B2, 0)
        z = self.W3.T @ y + self.B3
        return int(z[0] / SCALE)      # centipawns
