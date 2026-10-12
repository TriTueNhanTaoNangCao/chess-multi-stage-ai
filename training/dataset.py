"""Thai: PGN -> (state, legal policy target, outcome); split by game to avoid leakage."""

def build_dataset(pgn_path, output_path):
    raise NotImplementedError("Parse valid PGN, use ChessEncoder, split train/validation/test by game")
