"""Offline interactive chessboard. SVG pieces supplied by python-chess."""
import chess
import chess.svg
import streamlit as st
from functools import lru_cache


_BOARD = st.components.v2.component(
    "chess_arena_board",
    html='<div class="wrap"><div class="frame"><div class="ranks"></div><div class="board" aria-label="Bàn cờ"></div><div></div><div class="files"></div></div><div class="promotion"></div><p class="hint"></p></div>',
    css="""
    .wrap {width:100%;margin:0 auto;font-family:var(--st-font,sans-serif);}
    .frame {display:grid;grid-template-columns:18px 1fr;grid-template-rows:1fr 20px;
      padding:12px 16px 2px 7px;border-radius:12px;border:1px solid #665b42;
      background:linear-gradient(145deg,#272a30,#12151b);box-shadow:0 18px 42px #0006,inset 0 1px #e4d0a225;}
    .board {display:grid;grid-template-columns:repeat(8,minmax(0,1fr));aspect-ratio:1;
      border:1px solid #8c7c5d;border-radius:3px;overflow:hidden;}
    .ranks {display:grid;grid-template-rows:repeat(8,1fr);}
    .files {display:grid;grid-template-columns:repeat(8,1fr);}
    .ranks span,.files span {display:grid;place-items:center;color:#b3a382;font:10px ui-monospace,monospace;}
    .sq {position:relative;aspect-ratio:1;min-width:0;border:0;border-radius:0;padding:5%;
      display:flex;align-items:center;justify-content:center;cursor:default;transition:box-shadow .15s;}
    .sq[aria-disabled=false] {cursor:pointer;}
    .sq.light {background:linear-gradient(145deg,#c7c3b7,#b7b3a7);}
    .sq.dark {background:linear-gradient(145deg,#616671,#505660);}
    .sq.last.light {background:linear-gradient(145deg,#d1be8f,#c0aa76);}
    .sq.last.dark {background:linear-gradient(145deg,#91805c,#7b6d4e);}
    .sq.selected {box-shadow:inset 0 0 0 3px #ebc982;z-index:1;}
    .sq.check {background:radial-gradient(ellipse,#c55e55,#713b3b);}
    .sq:focus-visible {outline:3px solid #f3d99b;outline-offset:-3px;z-index:2;}
    .sq svg {width:100%;height:100%;filter:drop-shadow(0 3px 2px #0005);pointer-events:none;transition:transform .14s;}
    .sq[aria-disabled=false].occupied:hover svg {transform:translateY(-2px);}
    .sq.legal:after {content:'';position:absolute;width:20%;height:20%;border-radius:50%;background:#2a261e80;border:1px solid #dcc38a90;pointer-events:none;}
    .sq.legal.occupied:after {width:85%;height:85%;background:none;border:3px solid #e7c889;box-sizing:border-box;}
    .promotion {display:flex;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap;}
    .promotion:empty {display:none;} .promotion button {padding:10px 16px;border:1px solid #8d784f;
      border-radius:8px;background:#242219;color:#f0d8a5;font-size:14px;cursor:pointer;}
    .promotion button:hover {background:#3c3423;}
    .hint {text-align:center;font-size:11px;color:#8d95a3;margin:9px 0 0;}
    @media(max-width:480px) {.frame {padding:7px 9px 0 3px;grid-template-columns:14px 1fr;grid-template-rows:1fr 16px;}
      .ranks span,.files span {font-size:9px;} .sq {padding:4%;}}
    @media(prefers-reduced-motion:reduce) {.sq,.sq svg {transition:none;}}
    """,
    js="""
    export default function({data, parentElement, setTriggerValue}) {
      const board = parentElement.querySelector('.board');
      const promo = parentElement.querySelector('.promotion');
      const hint = parentElement.querySelector('.hint');
      const rankLabels = parentElement.querySelector('.ranks');
      const fileLabels = parentElement.querySelector('.files');
      let state = parentElement._chess;
      if (!state || state.token !== data.token) {
        state = {token:data.token, selected:null, choices:[], sent:false};
        parentElement._chess = state;
      }
      function send(uci) {
        if (state.sent) return;
        state.sent = true; state.selected = null; state.choices = [];
        setTriggerValue('move', {uci, token:data.token}); render();
      }
      function click(name) {
        if (!data.interactive || state.sent) return;
        const choices = data.legal.filter(m => m.slice(0,2) === state.selected && m.slice(2,4) === name);
        if (choices.length === 1) {send(choices[0]); return;}
        if (choices.length > 1) {state.choices = choices; render(); return;}
        state.choices = [];
        state.selected = state.selected === name ? null : name;
        render();
      }
      function render() {
        board.replaceChildren(); promo.replaceChildren();
        rankLabels.replaceChildren(); fileLabels.replaceChildren();
        const files = data.flipped ? 'hgfedcba' : 'abcdefgh';
        const ranks = data.flipped ? '12345678' : '87654321';
        for (const rank of ranks) {const e=document.createElement('span');e.textContent=rank;rankLabels.appendChild(e);}
        for (const file of files) {const e=document.createElement('span');e.textContent=file;fileLabels.appendChild(e);}
        for (const [ri, rank] of [...ranks].entries()) {
          for (const [fi, file] of [...files].entries()) {
            const name = file+rank;
            const square = data.squares[name];
            const el = document.createElement('button'); el.type = 'button';
            const dark = (file.charCodeAt(0)-97+Number(rank)-1)%2 === 0;
            el.className = 'sq '+(dark?'dark':'light');
            if(square) {el.innerHTML = square; el.classList.add('occupied');}
            if(data.last.includes(name)) el.classList.add('last');
            if(data.check === name) el.classList.add('check');
            if(state.selected === name) el.classList.add('selected');
            if(data.interactive && data.legal.some(m => m.slice(0,2)===state.selected && m.slice(2,4)===name)) el.classList.add('legal');
            el.setAttribute('aria-label', name); el.title = name;
            el.setAttribute('aria-disabled', String(!data.interactive));
            el.onclick = () => click(name);
            el.draggable = data.interactive && data.legal.some(m=>m.slice(0,2)===name);
            el.ondragstart = e => {
              state.selected=name;state.choices=[];
              e.dataTransfer.setData('text/plain',name);e.dataTransfer.effectAllowed='move';
              for(const target of board.querySelectorAll('.sq')) {
                target.classList.toggle('legal',data.legal.some(m=>m.slice(0,2)===name && m.slice(2,4)===target.title));
                target.classList.toggle('selected',target.title===name);
              }
            };
            el.ondragover = e => {if(data.interactive) {e.preventDefault();e.dataTransfer.dropEffect='move';}};
            el.ondrop = e => {e.preventDefault();if(data.interactive) click(name);};
            board.appendChild(el);
          }
        }
        if(state.choices.length) {
          for(const uci of state.choices) {
            const button=document.createElement('button'); button.type='button';
            button.textContent=({q:'Hậu',r:'Xe',b:'Tượng',n:'Mã'})[uci[4]];
            button.onclick=()=>send(uci); promo.appendChild(button);
          }
          const cancel=document.createElement('button'); cancel.textContent='Hủy';
          cancel.onclick=()=>{state.choices=[];state.selected=null;render();}; promo.appendChild(cancel);
        }
        hint.textContent = state.choices.length ? 'Chọn quân phong cấp' :
          data.interactive ? 'Kéo quân hoặc chọn quân rồi chọn ô đích' : data.caption;
      }
      board.onkeydown=e=>{if(e.key==='Escape'){state.selected=null;state.choices=[];render();}};
      render();
    }
    """,
)


@lru_cache(maxsize=12)
def piece_svg(symbol):
    svg = chess.svg.piece(chess.Piece.from_symbol(symbol))
    # Keep the original vector detail; recolor ivory and graphite surfaces.
    return (svg.replace('#ffffff', '#F4ECD9').replace('#fff', '#F4ECD9')
            .replace('#ececec', '#C9C4B9').replace('#000000', '#161A22').replace('#000', '#161A22'))


def chessboard(board, *, key, token="replay", interactive=False, flipped=False, caption=""):
    last = board.peek() if board.move_stack else None
    data = dict(squares={chess.square_name(s): piece_svg(p.symbol()) for s, p in board.piece_map().items()},
                legal=[m.uci() for m in board.legal_moves] if interactive else [],
                last=[chess.square_name(last.from_square), chess.square_name(last.to_square)] if last else [],
                check=chess.square_name(board.king(board.turn)) if board.is_check() else "",
                token=token, interactive=interactive, flipped=flipped, caption=caption)
    result = _BOARD(data=data, key=key, on_move_change=lambda: None)
    return result.move
