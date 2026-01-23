from flask import Flask, render_template, request, jsonify
import random
import copy
import math

app = Flask(__name__)

ROWS, COLUMNS = 7, 7
THRONE = (3, 3)
CORNERS = [(0, 0), (0, 6), (6, 0), (6, 6)]
SPECIAL_SQUARES = CORNERS + [THRONE]

# Search depth for minimax 
SEARCH_DEPTH = 3

eval_cache = {}       # Cache for evaluate_state results
minimax_cache = {}    # Cache for minimax results

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
    "last_moves": [],
    "ai_thinking": False  
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
                r += dr
                c += dc
                continue
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
    if (r, c) == THRONE or any((r+dr, c+dc) == THRONE for dr, dc in directions):
        return all(
            ( (0 <= r+dr < ROWS and 0 <= c+dc < COLUMNS) and
              (is_attacker((r+dr, c+dc)) or is_special_square((r+dr, c+dc))) )
            for dr, dc in directions
        )
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
def is_ai_controlled(state, team):
    mode = state.get("game_mode")
    if mode == "PvP" or mode is None:
        return False
    if mode in ("AI_Attacker_vs_Player", "AIvP", "AIvA"):
        return team == "attacker"
    if mode in ("Player_vs_AI_Defender", "PvAI"):
        return team == "defender"
    if mode in ("AI_vs_AI", "AIvAI"):
        return True
    if mode == "PvAI":
        return state["current_turn"] == 'attacker'
    if mode == "AIvP":
        return state["current_turn"] == 'defender'
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
def board_hash(board):
    return tuple(
        tuple(
            None if p is None else (p.team, p.is_king)
            for p in row
        )
        for row in board
    )
def king_has_clear_escape(board):
    for r in range(ROWS):
        for c in range(COLUMNS):
            p = board[r][c]
            if p and p.is_king:
                kr, kc = r, c
                break
        else:
            continue
        break
    else:
        return False
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    for dr, dc in directions:
        r, c = kr + dr, kc + dc
        clear = True
        while 0 <= r < ROWS and 0 <= c < COLUMNS:
            if board[r][c] is not None:
                clear = False
                break
            r += dr
            c += dc
        if clear:
            return True
    return False

def king_can_reach_edge_next(board):
    for r in range(ROWS):
        for c in range(COLUMNS):
            p = board[r][c]
            if p and p.is_king:
                kr, kc = r, c
                break
        else:
            continue
        break
    else:
        return False

    moves = get_valid_moves(board, kr, kc)
    for mr, mc in moves:
        if mr == 0 or mr == ROWS-1 or mc == 0 or mc == COLUMNS-1:
            return True
    return False

def is_potential_capture(state, move):
    (_, _), (er, ec) = move
    board = state["board"]
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    for dr, dc in directions:
        r, c = er + dr, ec + dc
        if 0 <= r < ROWS and 0 <= c < COLUMNS:
            p = board[r][c]
            if p and p.team != state["current_turn"]:
                return True
    return False

def get_forced_win_moves(state, team):
    forced = []
    for start, end in get_all_moves_for_state(state, team):
        temp = copy.deepcopy(state)
        move_piece(start, end, temp)
        if team == 'defender' and king_has_clear_escape(temp["board"]):
            forced.append((start, end))
        elif team == 'attacker' and not any(p.is_king for row in temp["board"] for p in row if p):
            forced.append((start, end))
    return forced

def evaluate_state_cached(state, ai_team):
    h = board_hash(state["board"])
    key = (h, ai_team, state["current_turn"])
    if key in eval_cache:
        return eval_cache[key]
    val = evaluate_state(state, ai_team)
    eval_cache[key] = val
    return val

def blocks_escape(prev_board, new_board):
    return king_has_clear_escape(prev_board) and not king_has_clear_escape(new_board)

def simulate_move(state, move):
    temp = copy.deepcopy(state)
    start, end = move
    move_piece(start, end, temp)
    if not temp["game_over"]:
        temp["current_turn"] = (
            'defender' if state["current_turn"] == 'attacker' else 'attacker'
        )
    return temp
# ===================== Flask Routes =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_game", methods=["POST"])
def start_game():
    data = request.get_json() or {}
    mode = data.get("mode", "PvP")
    game_state["board"] = initialize_board()
    game_state["current_turn"] = 'attacker'
    game_state["game_mode"] = mode
    game_state["selected_piece"] = None
    game_state["valid_moves"] = []
    game_state["game_over"] = False
    game_state["winner"] = None
    game_state["last_moves"] = []
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
    move_piece((sr, sc), (er, ec), game_state)
    if not game_state["game_over"]:
        game_state["current_turn"] = 'defender' if game_state["current_turn"] == 'attacker' else 'attacker'
    mode = game_state.get("game_mode")
    if not game_state["game_over"] and is_ai_controlled(game_state, game_state["current_turn"]):
        opponent_move(game_state)
    return jsonify({
        "board": board_to_dict(game_state),
        "current_turn": game_state["current_turn"],
        "game_over": game_state["game_over"],
        "winner": game_state["winner"],
        "game_mode": game_state["game_mode"]
    })

@app.route("/tick", methods=["POST"])
def tick():
    if (
        game_state["game_mode"] in ("AI_vs_AI", "AIvAI")
        and not game_state["game_over"]
        and not game_state.get("ai_thinking")
        and is_ai_controlled(game_state, game_state["current_turn"])
    ):
        opponent_move(game_state)

    return jsonify({
        "board": board_to_dict(game_state),
        "current_turn": game_state["current_turn"],
        "game_over": game_state["game_over"],
        "winner": game_state["winner"]
    })

# ===================== AI: minimax with alpha-beta =====================
def evaluate_state(state, ai_team):
    board = state["board"]
    material = 0
    king_pos = None
    for r in range(ROWS):
        for c in range(COLUMNS):
            p = board[r][c]
            if not p:
                continue
            value = 5
            if p.is_king:
                value = 40
                king_pos = (r, c)
            material += value if p.team == ai_team else -value
    king_score = 0
    if king_pos:
        kr, kc = king_pos
        min_corner_dist = min(abs(kr - cr) + abs(kc - cc) for cr, cc in CORNERS)
        if ai_team == 'defender':
            king_score += (10 - min_corner_dist) * 3
        else:
            king_score -= (10 - min_corner_dist) * 3
        nearest_attacker = math.inf
        for r in range(ROWS):
            for c in range(COLUMNS):
                p = board[r][c]
                if p and p.team == 'attacker' and not p.is_king:
                    d = abs(r - kr) + abs(c - kc)
                    if d < nearest_attacker:
                        nearest_attacker = d
        if nearest_attacker != math.inf:
            if ai_team == 'defender':
                king_score -= max(0, 6 - nearest_attacker) * 2
            else:
                king_score += max(0, 6 - nearest_attacker) * 2
        danger = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = kr + dr, kc + dc
            orr, occ = kr - dr, kc - dc
            if 0 <= nr < ROWS and 0 <= nc < COLUMNS:
                p = board[nr][nc]
                if p and p.team == 'attacker':
                    danger += 3
            if (
                0 <= nr < ROWS and 0 <= nc < COLUMNS and
                0 <= orr < ROWS and 0 <= occ < COLUMNS
            ):
                p1 = board[nr][nc]
                p2 = board[orr][occ]
                if (
                    p1 and p1.team == 'attacker' and
                    (p2 and p2.team == 'attacker' or (orr, occ) in SPECIAL_SQUARES)
                ):
                    danger += 6
        if ai_team == 'defender':
            king_score -= danger * 4
        else:
            king_score += danger * 4
    ai_moves = 0
    opp_moves = 0
    for r in range(ROWS):
        for c in range(COLUMNS):
            p = board[r][c]
            if not p: continue
            moves = get_valid_moves(board, r, c)
            if p.team == ai_team:
                ai_moves += len(moves)
            else:
                opp_moves += len(moves)
    mobility = ai_moves - opp_moves
    if ai_team == 'defender' and king_has_clear_escape(board):
        return 10_000
    if ai_team == 'attacker' and king_has_clear_escape(board):
        return -10_000
    if king_has_clear_escape(board):
        if ai_team == 'defender':
            king_score += 2500
        else:
            king_score -= 2500
    if king_can_reach_edge_next(board):
        if ai_team == 'defender':
            king_score += 1500
        else:
            king_score -= 1500
    if ai_team == 'attacker':
        if king_has_clear_escape(board):
            return -12_000
    if ai_team == 'attacker':
        if not king_has_clear_escape(board):
            king_score += 80
    if ai_team == 'defender' and king_has_clear_escape(board):
        return 12_000
    return material + king_score + mobility * 1.5

def get_all_moves_for_state(state, team):
    moves = []
    for r in range(ROWS):
        for c in range(COLUMNS):
            p = state["board"][r][c]
            if p and p.team == team:
                valid = get_valid_moves(state["board"], r, c)
                for mv in valid:
                    moves.append(((r, c), mv))
    return moves

def minimax(state, depth, alpha, beta, ai_team):
    if state["game_over"]:
        if state["winner"] == ai_team:
            return 100_000 - (SEARCH_DEPTH - depth), None
        else:
            return -100_000 + (SEARCH_DEPTH - depth), None
    current = state["current_turn"]
    moves = get_all_moves_for_state(state, current)
    for start, end in moves:
        temp = copy.deepcopy(state)
        move_piece(start, end, temp)
        if temp["game_over"] and temp["winner"] == ai_team:
            return 100_000, (start, end)
    maximizing = (state["current_turn"] == ai_team)
    current = state["current_turn"]
    if depth == 0 or state["game_over"]:
        score = evaluate_state_cached(state, ai_team)
        return score, None
    moves = get_all_moves_for_state(state, current)
    if not moves:
        score = -10_000 if current == ai_team else 10_000
        return score, None
    #random.shuffle(moves)
    moves.sort(key=lambda m: (blocks_escape(state["board"], simulate_move(state, m)["board"]),is_potential_capture(state, m)),reverse=True)
    best_value = -float('inf') if maximizing else float('inf')
    best_moves = []
    for start, end in moves:
        temp = copy.deepcopy(state)
        move_piece(start, end, temp)
        if not temp["game_over"]:
            temp["current_turn"] = (
                'defender' if current == 'attacker' else 'attacker'
            )
        score, _ = minimax(temp, depth - 1, alpha, beta, ai_team)
        if maximizing:
            if score > best_value:
                best_value = score
                best_moves = [(start, end)]
            elif score == best_value:
                best_moves.append((start, end))
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        else:
            if score < best_value:
                best_value = score
                best_moves = [(start, end)]
            elif score == best_value:
                best_moves.append((start, end))
            beta = min(beta, best_value)
            if alpha >= beta:
                break
    result = (best_value, random.choice(best_moves) if best_moves else None)
    return result

def opponent_move(state):
    eval_cache.clear()
    minimax_cache.clear()
    if state.get("game_over"):
        return False
    if state.get("ai_thinking"):
        return False
    team = state["current_turn"]
    if not is_ai_controlled(state, team):
        return False
    state["ai_thinking"] = True
    try:
        moves = get_all_moves_for_state(state, team)
        if not moves:
            state["game_over"] = True
            state["winner"] = 'defender' if team == 'attacker' else 'attacker'
            return False
        score, best = minimax(
            copy.deepcopy(state),
            SEARCH_DEPTH,
            -float('inf'),
            float('inf'),
            team
        )
        if not best:
            best = random.choice(moves)
        move_piece(best[0], best[1], state)
        state_hash = board_hash(state["board"])
        state.setdefault("position_history", []).append(state_hash)
        if len(state["position_history"]) > 20:
            state["position_history"].pop(0)

        if not state["game_over"]:
            state["current_turn"] = (
                'defender' if team == 'attacker' else 'attacker'
            )
        return True
    finally:
        state["ai_thinking"] = False

# =====================
if __name__ == "__main__":
    app.run(debug=True)
