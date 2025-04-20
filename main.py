"""
Katfish – a compact CPU chess engine with a PyQt5 GUI.

• GUI uses PyQt5 only ⇒ no extra GUI dependencies.
• Engine = iterative‑deepening α‑β/negamax + transposition table.
• Evaluation: Stockfish‑inspired material + PSQT (centipawns).
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui import ChessWindow

def main():
    play_white = True
    if len(sys.argv) > 1 and sys.argv[1].lower().startswith("b"):
        play_white = False
    app = QApplication(sys.argv)
    win = ChessWindow(human_is_white=play_white)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()