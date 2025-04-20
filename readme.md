# Katfish

**Katfish** is a lightweight, CPU-based chess engine with a PyQt5 GUI and optional NNUE integration.  
It supports move search, an opening book (Polyglot format), user orientation (play as White or Black), etc.  
The engine uses iterative deepening with alpha-beta pruning, quiescence search, transposition tables, and optionally Stockfish-style NNUE evaluation.

---

## Features

- Play as **White or Black**
- **Stockfish-style** positional evaluation with material & PSQT
- Optional **NNUE evaluation** for stronger play (still **in development**)
- Supports **Polyglot opening books** (e.g. Cerebellum)
- **Undo / redo** via keyboard arrows (←/→)
- Responsive **PyQt5 interface**

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bondycow/katfish.git
cd katfish
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
or manually
```bash
pip install python-chess PyQt5 numpy
```

### 3. Run the GUI
To play as **white**:
```bash
python main.py
```
To play as **black**:
```bash
python main.py -b
```

### 4. Keyboard Shortcuts
← Return to the previous board state

→ Go to the next board state (if applicable)

Click a piece, then click a square to move it