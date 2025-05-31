# multiple_divisor.py

def can_take(a, b):
    return a % b == 0 if a >= b else b % a == 0

# Memoization cache
memo = {}

def get_valid_moves(stones, last):
    return [x for x in stones if can_take(x, last)]

def is_winning(stones, last):
    key = (tuple(sorted(stones)), last)
    if key in memo:
        return memo[key]

    valid_moves = get_valid_moves(stones, last)
    if not valid_moves:
        memo[key] = False  # No moves => losing position
        return False

    # Try all possible moves
    for move in sorted(valid_moves, key=lambda m: len(get_valid_moves([x for x in stones if x != m], m))):
        next_stones = [x for x in stones if x != move]
        if not is_winning(next_stones, move):  # If opponent cannot win, then this move wins
            memo[key] = True
            return True

    # No winning move
    memo[key] = False
    return False

def player(stones, last):
    for move in sorted(stones, key=lambda m: len(get_valid_moves([x for x in stones if x != m], m))):
        if can_take(move, last):
            next_stones = [x for x in stones if x != move]
            if not is_winning(next_stones, move):
                return move  # This move wins
    return False  # No winning move
