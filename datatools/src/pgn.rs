// PGN reader for OpenBench/fastchess datagen output.
//
// Each game has tags ([Result], optionally [FEN]) and movetext of SAN moves,
// each followed by a comment such as {+0.52/12 0.004s, n=8123, sd=17}. The
// score is in pawns from the mover's point of view; `+M3`/`-M2` are mates.
use crate::position::*;
use crate::viri::{self, Game};

pub struct PgnGame {
    pub tags: Vec<(String, String)>,
    pub movetext: String,
}

impl PgnGame {
    pub fn tag(&self, name: &str) -> Option<&str> {
        self.tags.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

/// Splits PGN text into games. A game starts at a tag line following movetext
/// (or at the first tag line).
pub fn split_games(text: &str) -> Vec<PgnGame> {
    let mut games = Vec::new();
    let mut cur: Option<PgnGame> = None;
    let mut in_movetext = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') && !(in_movetext && t.contains('{')) {
            if in_movetext || cur.is_none() {
                if let Some(g) = cur.take() {
                    games.push(g);
                }
                cur = Some(PgnGame { tags: Vec::new(), movetext: String::new() });
                in_movetext = false;
            }
            if let Some((k, v)) = parse_tag(t) {
                cur.as_mut().unwrap().tags.push((k, v));
            }
        } else if !t.is_empty() {
            if let Some(g) = cur.as_mut() {
                in_movetext = true;
                g.movetext.push_str(t);
                g.movetext.push(' ');
            }
        }
    }
    games.extend(cur);
    games
}

fn parse_tag(t: &str) -> Option<(String, String)> {
    let inner = &t[1..t.len() - 1];
    let sp = inner.find(' ')?;
    let val = inner[sp + 1..].trim();
    let val = val.strip_prefix('"')?.strip_suffix('"')?;
    Some((inner[..sp].to_string(), val.replace("\\\"", "\"")))
}

/// Parses the score at the start of a fastchess comment: Some(cp) from the
/// mover's point of view, or None for a mate score or no parseable score.
/// Mates aren't trained on, so they don't need a value.
pub fn parse_score(comment: &str) -> Option<i16> {
    let s = comment.trim().split('/').next()?.trim();
    let pawns: f64 = s.parse().ok()?;
    Some((pawns * 100.0).round().clamp(-32000.0, 32000.0) as i16)
}

fn legal_moves(pos: &Position) -> Vec<Move> {
    let mut list = MoveList::new();
    pos.gen_moves(&mut list, false);
    let mut v = Vec::with_capacity(list.len);
    for i in 0..list.len {
        let mut c = *pos;
        if c.make_move(list.moves[i]) {
            v.push(list.moves[i]);
        }
    }
    v
}

fn sq_name(sq: usize) -> String {
    format!("{}{}", (b'a' + (sq % 8) as u8) as char, (b'1' + (sq / 8) as u8) as char)
}

/// SAN for `m` without check/mate suffixes. `legal` is the legal move list.
pub fn san(pos: &Position, m: Move, legal: &[Move]) -> String {
    match mflag(m) {
        F_KCASTLE => return "O-O".to_string(),
        F_QCASTLE => return "O-O-O".to_string(),
        _ => {}
    }
    let (from, to) = (mfrom(m), mto(m));
    let pt = pc_type(pos.board[from]);
    let capture = is_capture(m);
    let mut s = String::new();
    if pt == PAWN {
        if capture {
            s.push((b'a' + (from % 8) as u8) as char);
            s.push('x');
        }
        s.push_str(&sq_name(to));
        if is_promo(m) {
            s.push('=');
            s.push(b"PNBRQK"[promo_pt(m)] as char);
        }
        return s;
    }
    s.push(b"PNBRQK"[pt] as char);
    // Castling never needs telling apart from a normal move: in Chess960 the
    // king's castling destination can also be an ordinary king move.
    let rivals: Vec<usize> = legal
        .iter()
        .filter(|&&o| o != m && !is_castle(o) && mto(o) == to && pc_type(pos.board[mfrom(o)]) == pt)
        .map(|&o| mfrom(o))
        .collect();
    if !rivals.is_empty() {
        if rivals.iter().all(|&r| r % 8 != from % 8) {
            s.push((b'a' + (from % 8) as u8) as char);
        } else if rivals.iter().all(|&r| r / 8 != from / 8) {
            s.push((b'1' + (from / 8) as u8) as char);
        } else {
            s.push_str(&sq_name(from));
        }
    }
    if capture {
        s.push('x');
    }
    s.push_str(&sq_name(to));
    s
}

/// Finds the legal move matching a SAN token (annotations like +, #, !, ? are ignored).
pub fn find_san(pos: &Position, token: &str) -> Option<Move> {
    let want = token.trim_end_matches(|c| matches!(c, '+' | '#' | '!' | '?')).replace("0-0-0", "O-O-O").replace("0-0", "O-O");
    let legal = legal_moves(pos);
    legal.iter().copied().find(|&m| san(pos, m, &legal) == want)
}

/// Converts one PGN game. Returns Err with a reason if it can't be used.
pub fn to_game(g: &PgnGame) -> Result<Game, String> {
    let wdl = match g.tag("Result") {
        Some("1-0") => viri::WDL_WHITE_WIN,
        Some("0-1") => viri::WDL_BLACK_WIN,
        Some("1/2-1/2") => viri::WDL_DRAW,
        other => return Err(format!("unusable result {:?}", other)),
    };
    let start = match g.tag("FEN") {
        Some(fen) => Position::from_fen(fen).ok_or_else(|| format!("bad FEN {}", fen))?,
        None => Position::from_fen(START_FEN).unwrap(),
    };
    let mut pos = start;
    // (move, mover-relative cp); None = mate or unscored.
    let mut moves: Vec<(Move, Option<i16>)> = Vec::new();
    let mut scored: Vec<bool> = Vec::new(); // whether a comment was seen for each move
    let text = &g.movetext;
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'{' {
            let end = text[i..].find('}').map(|e| i + e).ok_or("unterminated comment")?;
            // The first comment after a move scores it (from the mover's side).
            if let Some(seen) = scored.last_mut() {
                if !*seen {
                    *seen = true;
                    moves.last_mut().unwrap().1 = parse_score(&text[i + 1..end]);
                }
            }
            i = end + 1;
        } else if c == b'(' || c == b';' {
            return Err("variations or line comments are not supported".into());
        } else {
            let end = text[i..].find(|ch: char| ch.is_whitespace() || ch == '{').map_or(text.len(), |e| i + e);
            let tok = &text[i..end];
            i = end;
            if tok.ends_with('.') || tok.starts_with('$') || matches!(tok, "1-0" | "0-1" | "1/2-1/2" | "*") {
                continue;
            }
            let tok = tok.rsplit('.').next().unwrap();
            let m = find_san(&pos, tok).ok_or_else(|| format!("illegal or unknown move {} in {}", tok, pos.to_fen()))?;
            moves.push((m, None));
            scored.push(false);
            pos.make_move(m);
        }
    }
    // Make scores white-relative; mates and unscored moves become NO_SCORE.
    let mut p = start;
    let mut out = Vec::with_capacity(moves.len());
    for (m, score) in moves {
        let white = match score {
            Some(cp) if p.stm == WHITE => cp,
            Some(cp) => -cp,
            None => viri::NO_SCORE,
        };
        out.push((m, white));
        p.make_move(m);
    }
    Ok(Game { start, wdl, moves: out })
}
