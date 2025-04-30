# chess-engin
import numpy as np
from typing import List, Tuple, Dict, Optional

# Board representation using bitboards (64-bit integers)
class ChessBoard:
    def __init__(self):
        # Initialize piece bitboards
        self.white_pawns = 0xFF00
        self.white_knights = 0x42
        self.white_bishops = 0x24
        self.white_rooks = 0x81
        self.white_queens = 0x08
        self.white_king = 0x10
        
        self.black_pawns = 0xFF000000000000
        self.black_knights = 0x4200000000000000
        self.black_bishops = 0x2400000000000000
        self.black_rooks = 0x8100000000000000
        self.black_queens = 0x0800000000000000
        self.black_king = 0x1000000000000000
        
        self.white_pieces = (self.white_pawns | self.white_knights | 
                            self.white_bishops | self.white_rooks | 
                            self.white_queens | self.white_king)
        
        self.black_pieces = (self.black_pawns | self.black_knights | 
                            self.black_bishops | self.black_rooks | 
                            self.black_queens | self.black_king)
        
        self.all_pieces = self.white_pieces | self.black_pieces
        self.side_to_move = 'white'
        self.castling_rights = {'K': True, 'Q': True, 'k': True, 'q': True}
        self.en_passant_target = None
        self.halfmove_clock = 0
        self.fullmove_number = 1

# Move generation using magic bitboards for sliding pieces
class MoveGenerator:
    def __init__(self, board: ChessBoard):
        self.board = board
        self.move_lookup = self._initialize_move_tables()
        
    def _initialize_move_tables(self):
        # Pre-computed move tables for all pieces
        tables = {}
        # Fill with magic bitboard lookups for sliding pieces
        # and jump patterns for knights and kings
        return tables
    
    def generate_legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        # Generate pseudo-legal moves
        if self.board.side_to_move == 'white':
            moves.extend(self._generate_pawn_moves(self.board.white_pawns, 'white'))
            moves.extend(self._generate_knight_moves(self.board.white_knights, 'white'))
            moves.extend(self._generate_bishop_moves(self.board.white_bishops, 'white'))
            moves.extend(self._generate_rook_moves(self.board.white_rooks, 'white'))
            moves.extend(self._generate_queen_moves(self.board.white_queens, 'white'))
            moves.extend(self._generate_king_moves(self.board.white_king, 'white'))
        else:
            # Generate moves for black pieces
            # Similar to white but with different bitboards
            pass
        
        # Filter out illegal moves (moves that leave king in check)
        legal_moves = [move for move in moves if not self._leaves_king_in_check(move)]
        return legal_moves

# Neural network evaluation component
class NeuralNetEvaluator:
    def __init__(self):
        self.input_layer_size = 768  # 12 piece types × 64 squares
        self.hidden_layer_size = 256
        self.output_size = 1
        
        # Initialize weights (would be loaded from trained model)
        self.weights1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.weights2 = np.random.randn(self.hidden_layer_size, self.output_size)
        
    def board_to_features(self, board: ChessBoard) -> np.ndarray:
        # Convert board to input features
        features = np.zeros(self.input_layer_size)
        
        # Fill features based on piece positions
        # Each piece type on each square becomes a binary feature
        
        return features
    
    def evaluate(self, board: ChessBoard) -> float:
        features = self.board_to_features(board)
        hidden = np.tanh(np.dot(features, self.weights1))
        output = np.tanh(np.dot(hidden, self.weights2))
        
        # Convert to centipawn score (100 = 1 pawn advantage)
        return float(output[0] * 100)

# Search algorithm with alpha-beta pruning and principal variation
class SearchEngine:
    def __init__(self, board: ChessBoard, evaluator: NeuralNetEvaluator):
        self.board = board
        self.evaluator = evaluator
        self.move_generator = MoveGenerator(board)
        self.transposition_table = {}
        self.killer_moves = [[] for _ in range(64)]  # Store good moves at each depth
        self.history_table = {}  # Store historically good moves
        
    def search(self, depth: int) -> Tuple[float, List[Tuple[int, int]]]:
        alpha = float('-inf')
        beta = float('inf')
        best_score, best_line = self._negamax(depth, 0, alpha, beta, True)
        return best_score, best_line
    
    def _negamax(self, depth: int, ply: int, alpha: float, beta: float, 
                is_pv_node: bool) -> Tuple[float, List[Tuple[int, int]]]:
        if depth == 0:
            return self._quiescence_search(alpha, beta), []
        
        # Check transposition table
        board_hash = hash(str(self.board.all_pieces))
        if board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            entry = self.transposition_table[board_hash]
            if entry['type'] == 'exact':
                return entry['score'], entry['pv']
            elif entry['type'] == 'lowerbound' and entry['score'] >= beta:
                return entry['score'], entry['pv']
            elif entry['type'] == 'upperbound' and entry['score'] <= alpha:
                return entry['score'], entry['pv']
        
        moves = self.move_generator.generate_legal_moves()
        
        # Move ordering for better pruning
        moves = self._order_moves(moves, ply)
        
        if not moves:
            # Check if checkmate or stalemate
            if self._is_in_check():
                return -100000 + ply, []  # Checkmate, prefer later checkmates
            else:
                return 0, []  # Stalemate
        
        best_score = float('-inf')
        best_move = None
        best_line = []
        
        for move in moves:
            self._make_move(move)
            score, line = self._negamax(depth - 1, ply + 1, -beta, -alpha, is_pv_node)
            score = -score
            self._unmake_move(move)
            
            if score > best_score:
                best_score = score
                best_move = move
                best_line = [move] + line
                
            alpha = max(alpha, best_score)
            if alpha >= beta:
                # Store killer move
                if not self._is_capture(move):
                    self.killer_moves[ply].append(move)
                break
        
        # Store in transposition table
        entry_type = 'exact'
        if best_score <= alpha:
            entry_type = 'upperbound'
        elif best_score >= beta:
            entry_type = 'lowerbound'
            
        self.transposition_table[board_hash] = {
            'score': best_score,
            'depth': depth,
            'type': entry_type,
            'pv': best_line
        }
        
        return best_score, best_line
    
    def _quiescence_search(self, alpha: float, beta: float) -> float:
        # Stand-pat score
        stand_pat = self.evaluator.evaluate(self.board)
        
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
            
        # Only consider captures
        capture_moves = [move for move in self.move_generator.generate_legal_moves() 
                        if self._is_capture(move)]
        
        for move in capture_moves:
            self._make_move(move)
            score = -self._quiescence_search(-beta, -alpha)
            self._unmake_move(move)
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

# Main engine class that ties everything together
class ChessEngine5000:
    def __init__(self):
        self.board = ChessBoard()
        self.evaluator = NeuralNetEvaluator()
        self.search_engine = SearchEngine(self.board, self.evaluator)
        
    def get_best_move(self, position_fen: str, time_limit_ms: int = 1000) -> str:
        self._set_position(position_fen)
        
        # Iterative deepening
        best_move = None
        for depth in range(1, 20):  # Start shallow, go deeper
            try:
                # Time management would be added here
                score, line = self.search_engine.search(depth)
                if line:
                    best_move = line[0]
                print(f"Depth {depth}: score {score/100:.2f}, best line: {self._format_line(line)}")
            except TimeoutError:
                break
                
        return self._format_move(best_move)
Python practice project
