"""Small presentation helpers for the dark chess workspace."""
from html import escape

import streamlit as st


def apply_design():
    # Only our own classes/keys are styled; native controls use the TOML theme.
    st.html("""<style>
    /* Compact single-screen layout (no vertical scroll needed) */
    .stMainBlockContainer, .block-container {
      padding-top: 1rem !important;
      padding-bottom: 0.5rem !important;
      padding-left: 1.5rem !important;
      padding-right: 1.5rem !important;
      max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
      height: 2.2rem !important;
    }
    .st-key-workspace {max-width:none; margin:0 auto;}
    @media(min-width:768px) {
      .st-key-workspace {width:calc(100vw - 36px);margin-left:calc(50% - 50vw + 18px);}
    }
    .st-key-board_stage {width:100%;max-width:490px;margin:0 auto;}
    .st-key-setup_bar {background:linear-gradient(110deg,#16191f,#11141a);
      border-color:#34352f; box-shadow:0 4px 16px #00000018; padding:4px 8px !important;}
    .lab-eyebrow {font-size:10px;letter-spacing:.22em;color:#c9a96e;font-weight:600;margin:0 0 3px;}
    .lab-intro h1 {font-size:22px;font-weight:600;line-height:1.2;margin:2px 0 4px;padding:0;}
    .lab-intro p {font-size:11px;color:#9399a4;margin:0;}
    .player-strip {display:flex;align-items:center;gap:8px;padding:6px 12px;
      border:1px solid #2b3039;border-radius:8px;background:#15181e;min-height:44px;box-sizing:border-box;}
    .player-strip.active {border-color:#8c764d;background:linear-gradient(110deg,#242117,#15181e);}
    .player-avatar {width:28px;height:28px;flex-shrink:0;border-radius:7px;display:grid;place-items:center;
      font-family:Georgia,serif;font-size:20px;border:1px solid #635a47;}
    .player-avatar.white {background:#e6dfd1;color:#16191f;}
    .player-avatar.black {background:#252a34;color:#dbc59b;}
    .player-info {min-width:0;flex:1;}
    .player-name {font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#eee8dc;}
    .player-sub {font-size:9px;letter-spacing:.04em;color:#9da3af;margin-top:1px;}
    .player-clock {font:500 17px ui-monospace,Consolas,monospace;color:#d9c49c;text-align:right;}
    .clock-label {font:8px sans-serif;color:#8c929f;letter-spacing:.08em;margin-top:1px;}
    .match-status {display:flex;justify-content:space-between;align-items:center;gap:8px;
      color:#aeb4bf;font-size:11px;letter-spacing:.02em;margin:0 0 2px;}
    .status-dot {display:inline-block;width:6px;height:6px;border-radius:100%;background:#c9a96e;margin-right:6px;}
    .status-detail {color:#747c8b;white-space:nowrap;}
    .side-note {border-left:2px solid #645437;padding-left:9px;color:#969eac;font-size:11px;line-height:1.5;margin-top:6px;}
    .empty-moves {text-align:center;padding:24px 10px;color:#8b93a1;font-size:11px;line-height:1.6;}
    .empty-moves .glyph {font-size:24px;color:#b99b64;margin-bottom:8px;}
    .st-key-left_panel, .st-key-right_panel {padding:8px 12px !important;}
    @media(max-width:700px) {.st-key-left_panel {display:none;} .player-strip {padding:6px 8px;min-height:40px;}
      .player-clock {font-size:15px;} .status-detail {display:none;}}
    </style>""")


def page_intro(title, subtitle):
    st.html(f'<header class="lab-intro"><div class="lab-eyebrow">CHESS LAB / MULTI-STAGE AI</div>'
            f'<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></header>')


def player_strip(name, *, white, seconds=0.0, active=False, ready=True):
    side = "Trắng" if white else "Đen"
    status = "Đến lượt" if active else "Sẵn sàng" if not ready else "Trong ván"
    st.html(f"""<div class="player-strip {'active' if active else ''}">
      <div class="player-avatar {'white' if white else 'black'}">{'♔' if white else '♚'}</div>
      <div class="player-info"><div class="player-name">{escape(name)}</div>
      <div class="player-sub">{side} &nbsp;·&nbsp; {status}</div></div>
      <div class="player-clock">{seconds:.2f}<span style="font-size:12px"> s</span>
      <div class="clock-label">THỜI GIAN AI</div></div></div>""")


def match_status(text, detail):
    st.html(f'<div class="match-status"><span><i class="status-dot"></i>{escape(text)}</span>'
            f'<span class="status-detail">{escape(detail)}</span></div>')
