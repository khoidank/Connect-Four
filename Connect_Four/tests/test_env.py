import pytest
from config import PLAYER_RED, PLAYER_YELLOW, ROWS, COLS
from env.game_env import ConnectFourEnv


def test_initial_state():
    env = ConnectFourEnv()
    state = env.reset()
    assert state.shape == (ROWS, COLS)
    assert env.current_player == PLAYER_RED
    assert not env.is_terminal()
    assert env.legal_actions() == [0, 1, 2, 3, 4, 5, 6]


def test_gravity_and_full_column():
    env = ConnectFourEnv()
    for i in range(ROWS):
        expected_player = PLAYER_RED if i % 2 == 0 else PLAYER_YELLOW
        assert env.current_player == expected_player
        env.step(3)

    assert 3 not in env.legal_actions()

    with pytest.raises(ValueError):
        env.step(3)


def test_horizontal_win():
    env = ConnectFourEnv()
    moves = [0, 0, 1, 1, 2, 2, 3]
    for m in moves:
        _, reward, done, _ = env.step(m)

    assert done
    assert env.winner == PLAYER_RED
    assert env.result(PLAYER_RED) == 1.0
    assert env.result(PLAYER_YELLOW) == -1.0


def test_vertical_win():
    env = ConnectFourEnv()
    moves = [2, 5, 2, 5, 2, 5, 2]
    for m in moves:
        env.step(m)

    assert env.is_terminal()
    assert env.winner == PLAYER_RED


def test_diagonal_win():
    env = ConnectFourEnv()
    moves = [
        0,  # Red (5,0)
        1,  # Yellow (5,1)
        1,  # Red (4,1)
        2,  # Yellow (5,2)
        2,  # Red (4,2)
        3,  # Yellow (5,3)
        2,  # Red (3,2)
        3,  # Yellow (4,3)
        3,  # Red (3,3)
        6,  # Yellow (5,6)
        3,  # Red (2,3)
    ]
    for m in moves:
        env.step(m)

    assert env.is_terminal()
    assert env.winner == PLAYER_RED


def test_draw_game():
    env = ConnectFourEnv()
    pattern_cols = [0, 1, 0, 1, 0, 1,
                    1, 0, 1, 0, 1, 0,
                    2, 3, 2, 3, 2, 3,
                    3, 2, 3, 2, 3, 2,
                    4, 5, 4, 5, 4, 5,
                    5, 4, 5, 4, 5, 4,
                    6, 6, 6, 6, 6, 6]
    for c in pattern_cols:
        assert not env.is_terminal()
        env.step(c)

    assert env.is_terminal()
    assert env.winner == 0
    assert env.result(PLAYER_RED) == 0.0
    assert env.result(PLAYER_YELLOW) == 0.0


def test_clone_independence():
    env = ConnectFourEnv()
    env.step(3)
    cloned_env = env.clone()

    cloned_env.step(3)
    cloned_env.step(4)

    assert env.move_count == 1
    assert env.current_player == PLAYER_YELLOW
    assert env.board.grid[4, 3] == 0