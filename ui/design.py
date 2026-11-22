"""Small presentation helpers for the dark chess workspace."""
from html import escape

import streamlit as st


def apply_design():
    # Only our own classes/keys are styled; native controls use the TOML theme.
    st.html("""<style>
    /* Compact single-screen layout (zero vertical scrolling required) */
    .stMainBlockContainer, .block-container {
      padding-top: 2.4rem !important;
      padding-bottom: 0.2rem !important;
      padding-left: 1rem !important;
      padding-right: 1rem !important;
      max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
      height: 2.2rem !important;
      min-height: 2.2rem !important;
      padding: 0 !important;
    }
    footer { display: none !important; }
    hr { margin: 0.35rem 0 !important; }
    .st-key-workspace { max-width: none; margin: 0 auto; }
    @media(min-width: 768px) {
      .st-key-workspace { width: calc(100vw - 32px); margin-left: calc(50% - 50vw + 16px); }
    }
    .st-key-board_stage {
      width: 100%;
      max-width: min(calc(100vh - 160px), 730px);
      margin: 0 auto;
    }
    .st-key-setup_bar {
      background: linear-gradient(110deg,#16191f,#11141a);
      border-color: #34352f;
      box-shadow: 0 4px 16px #00000018;
      padding: 3px 6px !important;
      margin-bottom: 2px !important;
    }
    .lab-eyebrow { font-size: 9px; letter-spacing: .2em; color: #c9a96e; font-weight: 600; margin: 0 0 2px; }
    .lab-intro h1 { font-size: 20px; font-weight: 600; line-height: 1.2; margin: 2px 0 3px; padding: 0; }
    .lab-intro p { font-size: 10px; color: #9399a4; margin: 0; }
    .player-strip {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border: 1px solid #2b3039;
      border-radius: 6px;
      background: #15181e;
      min-height: 34px;
      height: 34px;
      box-sizing: border-box;
      margin: 2px 0;
    }
    .player-strip.active { border-color: #8c764d; background: linear-gradient(110deg,#242117,#15181e); }
    .player-avatar {
      width: 22px;
      height: 22px;
      flex-shrink: 0;
      border-radius: 5px;
      display: grid;
      place-items: center;
      font-family: Georgia, serif;
      font-size: 16px;
      border: 1px solid #635a47;
    }
    .player-avatar.white { background: #e6dfd1; color: #16191f; }
    .player-avatar.black { background: #252a34; color: #dbc59b; }
    .player-info { min-width: 0; flex: 1; }
    .player-name { font-size: 12px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #eee8dc; }
    .player-sub { font-size: 8px; letter-spacing: .04em; color: #9da3af; margin-top: 0; }
    .player-clock { font: 500 15px ui-monospace, Consolas, monospace; color: #d9c49c; text-align: right; }
    .clock-label { display: none; }
    .match-status {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      color: #aeb4bf;
      font-size: 11px;
      letter-spacing: .02em;
      margin: 0 0 2px;
    }
    .status-dot { display: inline-block; width: 5px; height: 5px; border-radius: 100%; background: #c9a96e; margin-right: 5px; }
    .status-detail { color: #747c8b; white-space: nowrap; }
    .side-note { border-left: 2px solid #645437; padding-left: 8px; color: #969eac; font-size: 10px; line-height: 1.4; margin-top: 4px; }
    .empty-moves { text-align: center; padding: 16px 8px; color: #8b93a1; font-size: 10px; line-height: 1.5; }
    .empty-moves .glyph { font-size: 20px; color: #b99b64; margin-bottom: 4px; }
    .st-key-left_panel, .st-key-right_panel {
      padding: 6px 10px !important;
      max-height: calc(100vh - 65px);
      overflow-y: auto;
    }
    [data-testid="stMetricValue"] { font-size: 18px !important; }
    [data-testid="stMetricLabel"] { font-size: 11px !important; }
    [data-testid="stMetric"] { padding: 1px 0 !important; }
    .stButton button { padding: 2px 8px !important; min-height: 32px !important; }
    @media(max-width: 700px) {
      .st-key-left_panel { display: none; }
      .player-strip { padding: 3px 6px; min-height: 30px; }
      .player-clock { font-size: 13px; }
      .status-detail { display: none; }
    }
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
