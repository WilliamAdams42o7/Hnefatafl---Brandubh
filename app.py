from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

ROWS, COLUMNS = 7, 7
THRONE = (3, 3)
CORNERS = [(0, 0), (0, 6), (6, 0), (6, 6)]
SPECIAL_SQUARES = CORNERS + [THRONE]

# ===================== Pieces & Board =====================
class Piece:
    def __init__(self, team, is_king=False):
        self.team = team
        self.is_king = is_king

def initialize_board():
    return [
        [None, None, None, Piece('attacker'), None, None, None],
        [None, None, None, Piece('attacker'), None, None, None],
        [None, None, None, Piece('defender'), None, None, None],
        [Piece('attacker'), Piece('attacker'), Piece('defender'),
         Piece('defender', is_king=True), Piece('defender'), Piece('attacker'), Piece('attacker')],
        [None, None, None, Piece('defender'), None, None, None],
        [None, None, None, Piece('attacker'), None, None, None],
        [None, None, None, Piece('attacker'), None, None, None],
    ]

# ===================== Game state =====================
game_state = {
    "board": initialize_board(),
    "current_turn": 'attacker',
    "game_mode": None,
    "selected_piece": None,
    "valid_moves": [],
    "game_over": False,
    "winner": None
}

# ===================== Game logic =====================
def get_valid_moves(board, row, column):
    valid_moves = []
    piece = board[row][column]
    if not piece:
        return valid_moves
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dr, dc in directions:
        r, c = row + dr, column + dc
        while 0 <= r < ROWS and 0 <= c < COLUMNS:
            if board[r][c] is not None:
                break
            if (r, c) == THRONE:
                break
            if (r, c) in CORNERS and not piece.is_king:
                break
            valid_moves.append((r, c))
            r += dr
            c += dc
    return valid_moves

def move_piece(start, end, state):
    sr, sc = start
    er, ec = end
    piece = state["board"][sr][sc]
    state["board"][er][ec] = piece
    state["board"][sr][sc] = None
    check_captures((er, ec), state)
    check_victory_conditions(state)

def check_captures(end_pos, state):
    er, ec = end_pos
    moved_piece = state["board"][er][ec]
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    for dr, dc in directions:
        adj_r, adj_c = er+dr, ec+dc
        opp_r, opp_c = er+2*dr, ec+2*dc
        if not (0 <= adj_r < ROWS and 0 <= adj_c < COLUMNS):
            continue
        target_piece = state["board"][adj_r][adj_c]
        if not target_piece or target_piece.team == moved_piece.team:
            continue
        if target_piece.is_king:
            if king_captured(adj_r, adj_c, state):
                state["board"][adj_r][adj_c] = None
            continue
        if not (0 <= opp_r < ROWS and 0 <= opp_c < COLUMNS):
            continue
        opposite = state["board"][opp_r][opp_c]
        opp_pos = (opp_r, opp_c)
        if opposite and opposite.team == moved_piece.team:
            state["board"][adj_r][adj_c] = None
            continue
        if opp_pos in CORNERS and opposite is None:
            state["board"][adj_r][adj_c] = None
            continue
        if opp_pos == THRONE and opposite is None:
            state["board"][adj_r][adj_c] = None
            continue

def king_captured(r, c, state):
    king = state["board"][r][c]
    if not king or not king.is_king:
        return False
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    def is_attacker(pos):
        nr, nc = pos
        if 0 <= nr < ROWS and 0 <= nc < COLUMNS:
            neighbor = state["board"][nr][nc]
            return neighbor is not None and neighbor.team == 'attacker'
        return False
    def is_special_square(pos):
        return pos in SPECIAL_SQUARES
    if (r, c) == THRONE:
        return all(is_attacker((r+dr, c+dc)) for dr, dc in directions)
    for dr, dc in directions:
        nr, nc = r+dr, c+dc
        opp_r, opp_c = r-dr, c-dc
        if not (0 <= nr < ROWS and 0 <= nc < COLUMNS):
            continue
        neighbor = state["board"][nr][nc]
        if neighbor is None or neighbor.team != 'attacker':
            continue
        if 0 <= opp_r < ROWS and 0 <= opp_c < COLUMNS:
            opposite = state["board"][opp_r][opp_c]
            if (opposite and opposite.team == 'attacker') or is_special_square((opp_r, opp_c)):
                return True
    return False

def hostile_square(r, c, attacking_team, state):
    if not (0 <= r < ROWS and 0 <= c < COLUMNS):
        return False
    piece = state["board"][r][c]
    if (r, c) == THRONE:
        return piece is None
    if (r, c) in CORNERS:
        return piece is None
    return piece is not None and piece.team == attacking_team

def check_victory_conditions(state):
    if state["game_over"]:
        return
    for (r, c) in CORNERS:
        piece = state["board"][r][c]
        if piece and piece.is_king:
            state["winner"] = 'defender'
            state["game_over"] = True
            return
    king_exists = any(piece for row in state["board"] for piece in row if piece and piece.is_king)
    if not king_exists:
        state["winner"] = 'attacker'
        state["game_over"] = True

def game_over_check(board):
    """
    Returns True if either the king is captured or has reached an escape square.
    Replace this with your actual victory condition if you have one.
    """
    king_alive = False
    for r in range(ROWS):
        for c in range(COLUMNS):
            piece = board[r][c]
            if piece and piece.is_king:
                king_alive = True
                # Optional: king escape condition (Brandubh style)
                if (r, c) in [(0, 0), (0, ROWS - 1), (ROWS - 1, 0), (ROWS - 1, COLUMNS - 1)]:
                    return True  # King has escaped
    return not king_alive  # King dead = game over

# ===================== AI =====================
def opponent_move(state):
    movable = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            piece = state["board"][r][c]
            if piece and piece.team == state["current_turn"]:
                moves = get_valid_moves(state["board"], r, c)
                if moves:
                    movable.append(((r,c), moves))
    if not movable:
        return
    start, moves = random.choice(movable)
    end = random.choice(moves)
    move_piece(start, end, state)
    state["current_turn"] = 'defender' if state["current_turn"] == 'attacker' else 'attacker'

# ===================== Flask Routes =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_game", methods=["POST"])
def start_game():
    mode = request.json.get("mode", "PvP")
    game_state["board"] = initialize_board()
    game_state["current_turn"] = 'attacker'
    game_state["game_mode"] = mode
    game_state["selected_piece"] = None
    game_state["valid_moves"] = []
    game_state["game_over"] = False
    game_state["winner"] = None
    return jsonify({"board": board_to_dict(game_state)})

@app.route("/make_move", methods=["POST"])
def make_move():
    if game_state["game_over"]:
        return jsonify({"error": "game over"}), 400

    data = request.json
    try:
        start = tuple(map(int, data["start"]))
        end = tuple(map(int, data["end"]))
    except Exception:
        return jsonify({"error": "bad coordinates"}), 400

    sr, sc = start
    er, ec = end

    # Basic bounds check
    if not (0 <= sr < ROWS and 0 <= sc < COLUMNS and 0 <= er < ROWS and 0 <= ec < COLUMNS):
        return jsonify({"error": "out of bounds"}), 400

    piece = game_state["board"][sr][sc]
    if piece is None:
        return jsonify({"error": "no piece at start"}), 400

    # Ensure player may only move their own piece
    if piece.team != game_state["current_turn"]:
        return jsonify({"error": "not your piece"}), 400

    # Get valid moves and check the destination
    valid = get_valid_moves(game_state["board"], sr, sc)
    if (er, ec) not in valid:
        return jsonify({"error": "illegal move", "valid_moves": valid}), 400

    # Apply the move
    move_piece((sr, sc), (er, ec), game_state)

    # Switch turn (only if game not ended by the move)
    if not game_state["game_over"]:
        game_state["current_turn"] = 'defender' if game_state["current_turn"] == 'attacker' else 'attacker'

    # Optional: if mode requires immediate AI response, handle it
    if ai_turn(game_state) and not game_state["game_over"]:
        opponent_move(game_state)

    return jsonify({
        "board": board_to_dict(game_state),
        "current_turn": game_state["current_turn"],
        "game_over": game_state["game_over"],
        "winner": game_state["winner"]
            })


def ai_turn(state):
    mode = state["game_mode"]
    turn = state["current_turn"]
    if mode == 'PvAI' and turn == 'attacker':
        return True
    if mode == 'AIvP' and turn == 'defender':
        return True
    if mode == 'AIvAI':
        return True
    return False

def board_to_dict(state):
    result = []
    for row in state["board"]:
        r = []
        for piece in row:
            if not piece:
                r.append(None)
            else:
                r.append({"team": piece.team, "is_king": piece.is_king})
        result.append(r)
    return result

if __name__ == "__main__":
    app.run(debug=True)
