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
    "winner": None,
    "last_moves": []  # for repetition rule (ignored for now)
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

            # BLOCKED square
            if board[r][c] is not None:
                break

            # Throne cannot be stopped on, but sliding over is allowed
            if (r, c) == THRONE:
                r += dr
                c += dc
                continue

            # Corners only for the king
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
        adj_r, adj_c = er + dr, ec + dc
        opp_r, opp_c = er + 2*dr, ec + 2*dc

        if not (0 <= adj_r < ROWS and 0 <= adj_c < COLUMNS):
            continue

        target_piece = state["board"][adj_r][adj_c]
        if not target_piece or target_piece.team == moved_piece.team:
            continue

        # King capture check
        if target_piece.is_king:
            if king_captured(adj_r, adj_c, state):
                state["board"][adj_r][adj_c] = None
            continue

        # Normal piece capture
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

    # Strong capture zone near throne
    if (r, c) == THRONE or any((r+dr, c+dc) == THRONE for dr, dc in directions):
        return all(
            is_attacker((r+dr, c+dc)) or is_special_square((r+dr, c+dc))
            for dr, dc in directions
            if 0 <= r+dr < ROWS and 0 <= c+dc < COLUMNS
        )

    # Normal capture
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


def check_victory_conditions(state):
    if state["game_over"]:
        return

    # King escape
    for (r, c) in CORNERS:
        piece = state["board"][r][c]
        if piece and piece.is_king:
            state["winner"] = 'defender'
            state["game_over"] = True
            return

    # King dead
    king_exists = any(
        piece for row in state["board"] for piece in row
        if piece and piece.is_king
    )
    if not king_exists:
        state["winner"] = 'attacker'
        state["game_over"] = True


def is_ai_controlled(state, team):
    """
    Return True if given team ('attacker' or 'defender') is controlled by AI
    for the current game_mode.
    """
    mode = state.get("game_mode")
    if mode == "PvP" or mode is None:
        return False
    if mode == "AI_Attacker_vs_Player":
        return team == "attacker"
    if mode == "Player_vs_AI_Defender":
        return team == "defender"
    if mode == "AI_vs_AI":
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


# ===================== Flask Routes =====================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_game", methods=["POST"])
def start_game():
    data = request.get_json() or {}
    mode = data.get("mode", "PvP")

    # Initialize game
    game_state["board"] = initialize_board()
    game_state["current_turn"] = 'attacker'
    game_state["game_mode"] = mode
    game_state["selected_piece"] = None
    game_state["valid_moves"] = []
    game_state["game_over"] = False
    game_state["winner"] = None
    game_state["last_moves"] = []   # reset repetition tracking

    # If the starting side is AI, perform exactly one AI move and return.
    # (For AI_vs_AI we do one AI move per request to avoid blocking the server.)
    if is_ai_controlled(game_state, game_state["current_turn"]) and not game_state["game_over"]:
        opponent_move(game_state)

    return jsonify({
        "board": board_to_dict(game_state),
        "current_turn": game_state["current_turn"],
        "game_over": game_state["game_over"],
        "winner": game_state["winner"],
        "game_mode": game_state["game_mode"]
    })


@app.route("/make_move", methods=["POST"])
def make_move():
    if game_state["game_over"]:
        return jsonify({"error": "game over"}), 400

    data = request.json
    try:
        start = tuple(map(int, data["start"]))
        end = tuple(map(int, data["end"]))
    except:
        return jsonify({"error": "bad coordinates"}), 400

    sr, sc = start
    er, ec = end

    if not (0 <= sr < ROWS and 0 <= sc < COLUMNS and 0 <= er < ROWS and 0 <= ec < COLUMNS):
        return jsonify({"error": "out of bounds"}), 400

    piece = game_state["board"][sr][sc]
    if piece is None:
        return jsonify({"error": "no piece at start"}), 400

    if piece.team != game_state["current_turn"]:
        return jsonify({"error": "not your piece"}), 400

    valid = get_valid_moves(game_state["board"], sr, sc)
    if (er, ec) not in valid:
        return jsonify({"error": "illegal move", "valid_moves": valid}), 400

    # Apply player's move
    move_piece((sr, sc), (er, ec), game_state)

    # Toggle turn if game isn't over
    if not game_state["game_over"]:
        game_state["current_turn"] = 'defender' if game_state["current_turn"] == 'attacker' else 'attacker'

    # After the human move, if the next side is AI, perform AI moves.
    # Safety: For AI_vs_AI we only do one AI move per request to avoid blocking.
    # For single-side AI modes the AI will move until control returns to human (usually one move).
    mode = game_state.get("game_mode")
    # We'll perform AI moves while:
    # - current side is AI-controlled
    # - game not over
    # - and (if AI_vs_AI) we only do one AI move in this request
    ai_moves_done = 0
    while not game_state["game_over"] and is_ai_controlled(game_state, game_state["current_turn"]):
        opponent_move(game_state)
        ai_moves_done += 1
        # If it's AI_vs_AI, stop after a single AI move and return so frontend can poll again.
        if mode == "AI_vs_AI":
            break
        # In other AI modes (single side AI), after one AI move control should return to human so loop will stop naturally.

    return jsonify({
        "board": board_to_dict(game_state),
        "current_turn": game_state["current_turn"],
        "game_over": game_state["game_over"],
        "winner": game_state["winner"],
        "game_mode": game_state["game_mode"]
    })

def ai_turn(state):
    """Return True if the current_turn is controlled by AI."""
    return is_ai_controlled(state, state["current_turn"])
@app.route("/tick", methods=["POST"])

def tick():
    if game_state["game_over"]:
        return jsonify({
            "board": board_to_dict(game_state),
            "current_turn": game_state["current_turn"],
            "game_over": True,
            "winner": game_state["winner"]
        })

    # If it's an AI turn, make exactly one AI move
    if ai_turn(game_state):
        opponent_move(game_state)

        return jsonify({
            "board": board_to_dict(game_state),
            "current_turn": game_state["current_turn"],
            "game_over": game_state["game_over"],
            "winner": game_state["winner"]
        })

    return jsonify({"status": "no_ai_turn"})

# ===================== AI =====================
def opponent_move(state):
    """
    Make one heuristic-based move for the side `state['current_turn']`.
    Returns True if a move was made, False if none available.
    """
    team = state["current_turn"]
    board = state["board"]

    movable = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            piece = board[r][c]
            if piece and piece.team == team:
                moves = get_valid_moves(board, r, c)
                if moves:
                    movable.append(((r, c), moves))

    if not movable:
        # No legal moves -> loss for this team
        state["game_over"] = True
        state["winner"] = 'defender' if team == 'attacker' else 'attacker'
        return False

    best_moves = []
    best_score = -999999

    # Evaluate each move
    for start, moves in movable:
        sr, sc = start

        for end in moves:
            er, ec = end

            # Copy the board for evaluation
            temp_state = {
                "board": [row[:] for row in state["board"]],
                "current_turn": team,
                "game_over": False,
                "winner": None
            }

            # Count pieces before move
            before = sum(
                1 for row in temp_state["board"] for p in row
                if p is not None
            )

            # Apply the move to temp_state
            move_piece(start, end, temp_state)

            # Count pieces after move
            after = sum(
                1 for row in temp_state["board"] for p in row
                if p is not None
            )

            captured = before - after

            # Score calculation
            if temp_state["game_over"]:
                # If the AI itself won → big bonus
                if temp_state["winner"] == team:
                    score = 1000
                else:
                    score = -1000
            else:
                score = captured * 10

            # Track best moves
            if score > best_score:
                best_score = score
                best_moves = [(start, end)]
            elif score == best_score:
                best_moves.append((start, end))

    # Pick among equal-best moves at random
    start, end = random.choice(best_moves)

    # Apply to real game
    move_piece(start, end, state)

    if not state["game_over"]:
        state["current_turn"] = 'defender' if team == 'attacker' else 'attacker'

    return True

# =====================
if __name__ == "__main__":
    app.run(debug=True)

