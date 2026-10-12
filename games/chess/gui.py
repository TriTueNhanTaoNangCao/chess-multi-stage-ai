"""Pygame dashboard. Workers receive snapshots; stale results never alter a new game."""
import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
import queue
import threading
from time import perf_counter
import chess
import pygame
from agents import AGENT_REGISTRY, build_agent
from agents.stage1_heuristics.pst import evaluate_board
from games.chess.board import ChessBoard

BG = "#0d1521"
CARD = "#192638"
TEXT = "#edf2f7"
MUTED = "#93a4b7"
ACCENT = "#b6d987"
LIGHT = "#e9edcc"
DARK = "#779954"

class ChessGUI:
    def __init__(self, agent_key="alphabeta", depth=3, human_color=chess.WHITE, seed=42, board=None):
        pygame.init()
        pygame.display.set_caption("Chess Multi-Stage AI")
        desktop = pygame.display.get_desktop_sizes()[0]
        self.window = pygame.display.set_mode(self.fit_window_size(desktop), pygame.RESIZABLE)
        # Draw at a stable logical resolution; scale the entire dashboard and
        # invert the same transform for mouse events.
        self.screen = pygame.Surface((1000, 920))
        self.update_viewport()
        self.clock = pygame.time.Clock()
        self.fonts = {size: pygame.font.SysFont("segoeui,dejavusans,arial", size) for size in (14, 16, 18, 22, 30, 36)}
        self.piece_font = pygame.font.SysFont("segoeuisymbol,dejavusans,freeserif", 65)
        self.board_rect = pygame.Rect(180, 190, 640, 640)
        self.square_size = 80
        self.agent_key, self.depth, self.seed = agent_key, depth, seed
        self.agent = build_agent(agent_key, depth, seed)
        self.human_color = human_color
        self.board = board.clone() if board is not None else ChessBoard()
        self.selected = None
        self.promotion_moves = []
        self.last_seconds = 0.0
        self.error = ""
        self.generation = 0
        self.working = False
        self.results = queue.Queue()
        self.running = True
        self._eval_fen = None
        self._eval = 0
        self.piece_images = {}
        for color in chess.COLORS:
            for kind in chess.PIECE_TYPES:
                glyph = chess.Piece(kind, chess.BLACK).unicode_symbol()
                fill = "#fffef0" if color else "#303738"
                outline = "#303738" if color else "#d6dfba"
                face = self.piece_font.render(glyph, True, fill)
                edge = self.piece_font.render(glyph, True, outline)
                image = pygame.Surface((face.get_width()+4, face.get_height()+4), pygame.SRCALPHA)
                for dx, dy in ((0,2),(4,2),(2,0),(2,4)):
                    image.blit(edge, (dx,dy))
                image.blit(face,(2,2))
                self.piece_images[(kind,color)] = image

    @staticmethod
    def fit_window_size(desktop):
        # Reserve space for OS title bar, taskbar and window borders.
        scale = min(0.8, max(1, desktop[0]-96)/1000, max(1, desktop[1]-140)/920)
        return max(1, int(1000*scale)), max(1, int(920*scale))

    def update_viewport(self):
        width, height = self.window.get_size()
        scale = min(width/1000, height/920)
        size = max(1, int(1000*scale)), max(1, int(920*scale))
        self.viewport = pygame.Rect((width-size[0])//2, (height-size[1])//2, *size)

    def window_to_logical(self, pos):
        if not self.viewport.collidepoint(pos):
            return None
        return ((pos[0]-self.viewport.x)*1000/self.viewport.width,
                (pos[1]-self.viewport.y)*920/self.viewport.height)

    def text(self, value, size, color, pos, center=False):
        image = self.fonts[size].render(str(value), True, color)
        rect = image.get_rect(center=pos) if center else image.get_rect(topleft=pos)
        self.screen.blit(image, rect)

    def square_to_screen(self, square):
        file, rank = chess.square_file(square), chess.square_rank(square)
        col, row = (file, 7-rank) if self.human_color else (7-file, rank)
        return self.board_rect.x + col*80, self.board_rect.y + row*80

    def screen_to_square(self, x, y):
        if not self.board_rect.collidepoint(x, y):
            return None
        col, row = int((x-self.board_rect.x)//80), int((y-self.board_rect.y)//80)
        return chess.square(col,7-row) if self.human_color else chess.square(7-col,row)

    def invalidate(self):
        self.generation += 1
        self.selected = None
        self.promotion_moves = []
        self.error = ""
        self.last_seconds = 0
        self._eval_fen = None

    def restart(self):
        self.invalidate()
        self.board.reset()
        # A fresh instance prevents overlapping state with a stale worker.
        self.agent = build_agent(self.agent_key, self.depth, self.seed)

    def undo(self):
        self.invalidate()
        if self.board.board.move_stack:
            self.board.pop()
            if self.board.turn != self.human_color and self.board.board.move_stack:
                self.board.pop()
        self.agent = build_agent(self.agent_key, self.depth, self.seed)

    def switch_bot(self):
        keys = list(AGENT_REGISTRY)
        self.agent_key = keys[(keys.index(self.agent_key)+1) % len(keys)]
        self.restart()

    def flip(self):
        self.human_color = not self.human_color
        self.restart()

    def apply_move(self, move):
        self.board.step(move)
        self.selected = None
        self.promotion_moves = []

    def handle_click(self, pos):
        if self.board.is_terminal() or self.board.turn != self.human_color or self.error:
            return
        if self.promotion_moves:
            for i, kind in enumerate((chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)):
                if pygame.Rect(300+i*100, 445, 100, 90).collidepoint(pos):
                    self.apply_move(next(m for m in self.promotion_moves if m.promotion == kind))
                    return
            return
        square = self.screen_to_square(*pos)
        if square is None:
            self.selected = None
            return
        if self.selected is not None:
            moves = [m for m in self.board.get_legal_actions()
                     if m.from_square == self.selected and m.to_square == square]
            if moves:
                if len(moves) > 1:
                    self.promotion_moves = moves
                else:
                    self.apply_move(moves[0])
                return
        piece = self.board.piece_at(square)
        self.selected = square if piece and piece.color == self.human_color else None

    def start_worker(self):
        if self.working or self.error or self.board.is_terminal() or self.board.turn == self.human_color:
            return
        self.working = True
        generation, snapshot, agent = self.generation, self.board.clone(), self.agent
        def work():
            started = perf_counter()
            try:
                move, error = agent.act(snapshot), ""
            except Exception as exc:
                move, error = None, f"{type(exc).__name__}: {exc}"
            self.results.put((generation, move, perf_counter()-started, error))
        threading.Thread(target=work, daemon=True).start()

    def poll_worker(self):
        try:
            generation, move, seconds, error = self.results.get_nowait()
        except queue.Empty:
            return
        self.working = False
        if generation != self.generation:
            return
        self.last_seconds = seconds
        if error:
            self.error = error
        else:
            try:
                if not self.board.is_legal(move):
                    raise ValueError("Bot returned an illegal move")
                self.apply_move(move)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

    def status(self):
        if self.error:
            return "Lỗi AI — nhấn R để chơi lại"
        outcome = self.board.outcome()
        if outcome:
            if outcome.winner is None:
                return "Hòa cờ!"
            return "Chiếu Hết! Bạn Thắng!" if outcome.winner == self.human_color else "Chiếu Hết! Bạn Thua!"
        if self.board.is_check():
            return "Chiếu Vua! — Đến lượt bạn" if self.board.turn == self.human_color else "Chiếu Vua! — AI đang nghĩ"
        return "Đến Lượt Bạn!" if self.board.turn == self.human_color else "AI Đang Suy Nghĩ..."

    def render(self):
        self.screen.fill(BG)
        self.text("CHESS / MULTI-STAGE AI", 16, ACCENT, (24, 13))
        self.text("CLASSICAL ENGINE  /  STAGE 01–02", 14, MUTED, (694, 15))
        fen = self.board.fen()
        if fen != self._eval_fen:
            self._eval_fen = fen
            self._eval = evaluate_board(self.board.board, claim_draw=self.board.claim_draw)
        turn = ("Bạn" if self.board.turn == self.human_color else "AI") + (" (Trắng)" if self.board.turn else " (Đen)")
        cards = [("AI MODEL", self.agent.name), ("LƯỢT ĐI", turn),
                 ("NƯỚC ĐI", str(len(self.board.board.move_stack)) + " ply"),
                 ("T.GIAN AI", f"{self.last_seconds:.3f}s"), ("EVAL SCORE", f"{self._eval/100:+.2f}")]
        for i, (label, value) in enumerate(cards):
            rect = pygame.Rect(24+i*193, 45, 183, 65)
            pygame.draw.rect(self.screen, CARD, rect, border_radius=12)
            self.text(label, 14, MUTED, (rect.x+12, rect.y+9))
            self.text(value, 16 if i == 0 else 22, TEXT, (rect.x+12, rect.y+31))
        self.text(self.status(), 30, "#ff648e" if self.board.is_check() else ACCENT, (500, 143), True)
        self.text("Chọn quân • Chọn ô đến", 14, MUTED, (500, 171), True)
        pygame.draw.rect(self.screen, CARD, self.board_rect.inflate(18,18), border_radius=8)
        position = self.board.board
        last = position.peek() if position.move_stack else None
        hints = {m.to_square for m in position.legal_moves if m.from_square == self.selected}
        for square in chess.SQUARES:
            x,y = self.square_to_screen(square)
            rect = pygame.Rect(x,y,80,80)
            color = LIGHT if (chess.square_rank(square)+chess.square_file(square)) % 2 else DARK
            pygame.draw.rect(self.screen, color, rect)
            overlay = None
            if last and square in (last.from_square, last.to_square):
                overlay = (248,224,90,105)
            if square == self.selected:
                overlay = (255,233,95,150)
            if position.is_check() and square == position.king(position.turn):
                overlay = (245,65,79,185)
            if overlay:
                surface = pygame.Surface((80,80), pygame.SRCALPHA)
                surface.fill(overlay)
                self.screen.blit(surface, (x,y))
            piece = position.piece_at(square)
            if piece:
                image = self.piece_images[(piece.piece_type,piece.color)]
                self.screen.blit(image, image.get_rect(center=rect.center))
            if square in hints:
                pygame.draw.circle(self.screen, "#425344", rect.center, 33 if piece else 9, 4 if piece else 0)
        for i in range(8):
            self.text(chess.FILE_NAMES[i if self.human_color else 7-i], 14, MUTED, (220+i*80, 845), True)
            self.text(str(8-i if self.human_color else i+1), 14, MUTED, (161,230+i*80), True)
        self.text("TRẮNG" if self.human_color else "ĐEN", 14, MUTED, (65, 220), True)
        self.text("BẠN", 18, ACCENT, (65, 246), True)
        self.text("01 / 02", 18, ACCENT, (907, 220), True)
        self.text("HEURISTIC", 14, MUTED, (907, 247), True)
        self.text("& SEARCH", 14, MUTED, (907, 268), True)
        self.text("[R] Chơi lại    |    [U] Lùi nước    |    [M] Đổi đối thủ AI    |    [F] Đổi bên", 18, TEXT, (500, 881), True)
        if self.promotion_moves:
            shade = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            shade.fill((0,0,0,150))
            self.screen.blit(shade,(0,0))
            pygame.draw.rect(self.screen, CARD, (280,390,440,165), border_radius=18)
            self.text("Phong cấp — chọn quân", 22, TEXT, (500,418),True)
            for i, kind in enumerate((chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT)):
                image = self.piece_images[(kind,self.human_color)]
                self.screen.blit(image,image.get_rect(center=(350+i*100,490)))
        self.window.fill(BG)
        self.window.blit(pygame.transform.smoothscale(self.screen, self.viewport.size), self.viewport)
        pygame.display.flip()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.window = pygame.display.set_mode((max(1, event.w), max(1, event.h)), pygame.RESIZABLE)
            self.update_viewport()
        elif event.type == pygame.KEYDOWN:
            commands = {pygame.K_r:self.restart, pygame.K_u:self.undo,
                        pygame.K_m:self.switch_bot, pygame.K_f:self.flip}
            if event.key == pygame.K_ESCAPE:
                if self.promotion_moves:
                    self.promotion_moves = []
                else:
                    self.running = False
            elif event.key in commands:
                commands[event.key]()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = self.window_to_logical(event.pos)
            if pos is not None:
                self.handle_click(pos)

    def run(self, max_frames=None):
        frames = 0
        try:
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                self.poll_worker()
                self.start_worker()
                self.render()
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
                self.clock.tick(60)
        finally:
            self.close()

    def close(self):
        self.generation += 1
        pygame.quit()
