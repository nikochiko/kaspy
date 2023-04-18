from __future__ import annotations

from contextlib import contextmanager
from abc import ABC
from typing import Iterator

import chess


class Engine(ABC):
    def __init__(self, board: chess.Board):
        raise NotImplementedError

    def suggest(self) -> chess.Move:
        raise NotImplementedError


class EvaluationTree:
    board: chess.Board
    eval: float
    is_game_over: bool
    best_move: chess.Move | None
    children: dict[chess.Move, EvaluationTree]


class Kaspy(Engine):
    def __init__(self, board: chess.Board, breadth=10, depth=2):
        self.board = board
        self.depth = depth  # max-depth
        self.breadth = breadth

        self.piece_types = [
            chess.PAWN,
            chess.KING,
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        ]
        self.piece_value = {
            chess.PAWN: 1.0,
            chess.KING: 2.5,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.5,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0
        }

    def suggest(self, depth=5, breadth=10) -> chess.Move | None:
        if self.board.is_variant_end():
            print("game is over")
            return
        elif self.board.is_stalemate():
            print("stalemate")
            return

        evals = []
        for move in self.generate_candidates(board=self.board, k=breadth):
            with make_move(self.board, move) as tmp_board:
                evals.append(
                    (move, self.eval(tmp_board, depth=depth, breadth=breadth))
                )
        if self.board.turn == chess.WHITE:
            return max(evals, key=lambda x: x[1])
        else:
            return min(evals, key=lambda x: x[1])

    def eval(self, board: chess.Board, depth: int = 0, breadth: int = 0) -> float:
        # check if game has ended
        outcome = board.outcome()
        if outcome and outcome.winner == chess.WHITE:
            return 1000.0
        elif outcome and outcome.winner == chess.BLACK:
            return -1000.0
        elif outcome and outcome is None:
            return 0.0

        if depth == 0:
            return sum([
                self.get_material_score(board, color=chess.WHITE),
                -self.get_material_score(board, color=chess.BLACK),
                self.get_attacked_squares_score(board, color=chess.WHITE),
                -self.get_attacked_squares_score(board, color=chess.BLACK),
                self.get_king_safety_score(board, color=chess.WHITE),
                -self.get_king_safety_score(board, color=chess.BLACK),
            ])
        else:
            evals = []
            for move in self.generate_candidates(board, k=breadth or 1):
                with make_move(board, move) as new_board:
                    evals.append(
                        self.eval(new_board, depth=depth-1 if breadth > 0 else 0, breadth=breadth//2)
                    )
            if not evals:
                raise Exception(f"evals is empty. board: {board}")
            if board.turn == chess.WHITE:
                return max(evals or [0.0])
            else:
                return min(evals or [0.0])

    def generate_candidates(
        self,
        board: chess.Board,
        k: int
    ) -> Iterator[chess.Move]:
        # return list(board.legal_moves)[:k]
        evals = []
        for move in board.legal_moves:
            with make_move(board, move) as new_board:
                evals.append(
                    (move, self.eval(new_board))
                )
        return (
            move for move, _ in
            sorted(
                evals,
                key=lambda x: x[1], reverse=board.turn == chess.WHITE)
        )

    def get_material_score(
        self, board: chess.Board, color: chess.Color,
    ) -> float:
        score = 0
        for piece in self.piece_types:
            num_pieces = len(board.pieces(piece_type=piece, color=color))
            score += num_pieces * self.piece_value[piece]
        return score

    def get_attacked_squares_score(
        self, board: chess.Board, color: chess.Color,
    ) -> float:
        score = 0
        for piece in self.piece_types:
            for square in board.pieces(piece_type=piece, color=color):
                score += len(board.attacks(square))
        return score * 0.1

    def get_king_safety_score(
        self, board: chess.Board, color: chess.Color,
    ) -> float:
        score = 0
        king_square = board.king(color=color)
        for square in board.attacks(king_square):
            score += board.color_at(square) == color
        return score * 0.2

    def generate_moves(self) -> list[chess.Move]:
        return self.board.legal_moves


@contextmanager
def make_move(board: chess.Board, move: chess.Move):
    new_board = board.copy()

    new_board.push(move)
    yield new_board
