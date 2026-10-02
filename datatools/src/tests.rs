// Checks our viriformat output against the reference reader (the viriformat
// crate, a test-only dependency): it must decode the same positions, moves and
// scores that we wrote.
use crate::pgn;
use crate::position::*;
use crate::viri::{self, Game};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn legal(pos: &Position) -> Vec<Move> {
    let mut list = MoveList::new();
    pos.gen_moves(&mut list, false);
    let mut v = Vec::new();
    for i in 0..list.len {
        let mut c = *pos;
        if c.make_move(list.moves[i]) {
            v.push(list.moves[i]);
        }
    }
    v
}

/// Decodes `game` with the reference reader and checks every position it
/// visits packs to the same bytes as ours, with the same score.
fn check_roundtrip(game: &Game) {
    let mut bytes = Vec::new();
    game.write(&mut bytes).unwrap();
    let decoded = viriformat::dataformat::Game::deserialise_from(&mut &bytes[..], Vec::new()).unwrap();
    assert_eq!(decoded.len(), game.moves.len());
    let mut pos = game.start;
    let mut i = 0;
    decoded.visit_positions(|board, eval| {
        let theirs = board.to_marlinformat(0, game.wdl, 0).as_bytes();
        let ours = viri::pack_board(&pos, 0, game.wdl);
        assert_eq!(theirs, ours, "position {} differs: ours {}", i, pos.to_fen());
        assert_eq!(eval, game.moves[i].1 as i32);
        pos.make_move(game.moves[i].0);
        i += 1;
    });
    assert_eq!(i, game.moves.len());
    // Re-serialising the decoded game gives back our exact bytes.
    let mut again = Vec::new();
    decoded.serialise_into(&mut again).unwrap();
    assert_eq!(again, bytes);
}

#[test]
fn random_games_roundtrip() {
    crate::attacks::init();
    let starts = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 5 30",
        "8/2P5/8/8/8/8/5p2/K6k w - - 0 60",
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
        // Black double pushes beside White's fifth-rank pawns invite en passant.
        "4k3/pppppppp/8/1P1P1P1P/8/8/8/4K3 b - - 0 1",
        "4k3/8/8/8/p1p1p1p1/8/PPPPPPPP/4K3 w - - 0 1",
    ];
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (mut castles, mut eps, mut promos) = (0, 0, 0);
    for g in 0..600 {
        let start = Position::from_fen(starts[g % starts.len()]).unwrap();
        let mut pos = start;
        let mut moves = Vec::new();
        for _ in 0..200 {
            let l = legal(&pos);
            if l.is_empty() || pos.halfmove >= 100 {
                break;
            }
            let m = l[(rng.next() % l.len() as u64) as usize];
            match mflag(m) {
                F_KCASTLE | F_QCASTLE => castles += 1,
                F_EP => eps += 1,
                _ if is_promo(m) => promos += 1,
                _ => {}
            }
            moves.push((m, (rng.next() % 2001) as i16 - 1000));
            pos.make_move(m);
        }
        check_roundtrip(&Game { start, wdl: (g % 3) as u8, moves });
    }
    // Make sure the special moves were actually exercised.
    assert!(castles > 50 && eps > 50 && promos > 50, "castles {} eps {} promos {}", castles, eps, promos);
}

#[test]
fn san_roundtrips_for_every_legal_move() {
    crate::attacks::init();
    let mut rng = Rng(12345);
    let mut pos = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    for _ in 0..3000 {
        let l = legal(&pos);
        if l.is_empty() {
            pos = Position::from_fen(START_FEN).unwrap();
            continue;
        }
        for &m in &l {
            let s = pgn::san(&pos, m, &l);
            assert_eq!(pgn::find_san(&pos, &format!("{}+", s)), Some(m), "{} in {}", s, pos.to_fen());
        }
        pos.make_move(l[(rng.next() % l.len() as u64) as usize]);
    }
}

#[test]
fn fastchess_pgn() {
    crate::attacks::init();
    let text = r#"[Event "Fastchess Tournament"]
[White "Incipit-dev"]
[Black "Incipit-base"]
[Result "1-0"]
[FEN "r3k2r/ppp2ppp/8/3pP3/8/8/PPP2PPP/R3K2R w KQkq d6 0 12"]
[SetUp "1"]

13. exd6 {+1.50/10 0.004s, n=9000, sd=14} O-O-O {-1.20/9 0.004s, n=8000, sd=12}
14. O-O {+1.10/10 0.004s, n=9100, sd=15} Rxd6 {-0.90/10 0.004s, n=9000, sd=13}
15. Rad1 {0.00/9 0.004s, n=100, sd=3} Rhd8 {-M5/20 0.004s, n=9000, sd=13}
16. Rxd6 {+M4/9 0.004s, n=100, sd=3} 1-0

[Event "Fastchess Tournament"]
[Result "1/2-1/2"]

1. e4 {+0.30/10 0.004s} e5 {-0.25/10 0.004s} 1/2-1/2
"#;
    let games = pgn::split_games(text);
    assert_eq!(games.len(), 2);
    let g = pgn::to_game(&games[0]).unwrap();
    assert_eq!(g.wdl, viri::WDL_WHITE_WIN);
    let scores: Vec<i16> = g.moves.iter().map(|m| m.1).collect();
    // White-relative: black's -1.20 is +120 for white; mates are unscored.
    assert_eq!(scores, vec![150, 120, 110, 90, 0, viri::NO_SCORE, viri::NO_SCORE]);
    check_roundtrip(&g);
    let g2 = pgn::to_game(&games[1]).unwrap();
    assert_eq!(g2.wdl, viri::WDL_DRAW);
    assert_eq!(g2.moves.iter().map(|m| m.1).collect::<Vec<_>>(), vec![30, 25]);
    check_roundtrip(&g2);
}

/// A random Chess960 back rank: bishops on opposite colours, king between the
/// rooks.
fn chess960_rank(rng: &mut Rng) -> [u8; 8] {
    loop {
        let mut r = *b"RNBQKBNR";
        for i in (1..8).rev() {
            r.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let b: Vec<usize> = (0..8).filter(|&i| r[i] == b'B').collect();
        let rk: Vec<usize> = (0..8).filter(|&i| r[i] == b'R').collect();
        let k = r.iter().position(|&c| c == b'K').unwrap();
        if (b[0] + b[1]) % 2 == 1 && rk[0] < k && k < rk[1] {
            return r;
        }
    }
}

/// A DFRC start position (independent white and black back ranks) with
/// Shredder-FEN castling rights.
fn dfrc_start(rng: &mut Rng) -> String {
    let w = chess960_rank(rng);
    let b = chess960_rank(rng);
    let files = |r: &[u8; 8], upper: bool| -> String {
        (0..8)
            .filter(|&i| r[i] == b'R')
            .map(|i| {
                let f = (b'a' + i as u8) as char;
                if upper { f.to_ascii_uppercase() } else { f }
            })
            .collect()
    };
    format!(
        "{}/pppppppp/8/8/8/8/PPPPPPPP/{} w {}{} - 0 1",
        String::from_utf8(b.iter().map(|c| c.to_ascii_lowercase()).collect()).unwrap(),
        String::from_utf8(w.to_vec()).unwrap(),
        files(&w, true),
        files(&b, false)
    )
}

#[test]
fn dfrc_games_roundtrip() {
    crate::attacks::init();
    let mut rng = Rng(0xD1F7C0DE);
    let mut castles = 0;
    for g in 0..400 {
        let start = Position::from_fen(&dfrc_start(&mut rng)).unwrap();
        assert_eq!(start.castling, 15, "{}", start.to_fen());
        let mut pos = start;
        let mut moves = Vec::new();
        for _ in 0..160 {
            let l = legal(&pos);
            if l.is_empty() || pos.halfmove >= 100 {
                break;
            }
            // Prefer castling when it's legal, so plenty of it gets written.
            let castle = l.iter().copied().find(|&m| matches!(mflag(m), F_KCASTLE | F_QCASTLE));
            let m = match castle {
                Some(c) if rng.next() % 2 == 0 => c,
                _ => l[(rng.next() % l.len() as u64) as usize],
            };
            if matches!(mflag(m), F_KCASTLE | F_QCASTLE) {
                castles += 1;
            }
            // Every legal move's SAN finds that move again (Chess960 castling
            // can share its squares with a normal king move).
            for &o in &l {
                let s = pgn::san(&pos, o, &l);
                assert_eq!(pgn::find_san(&pos, &s), Some(o), "{} in {}", s, pos.to_fen());
            }
            // Packing then unpacking any position gives it back.
            let (back, _, _) = viri::unpack_board(&viri::pack_board(&pos, 0, 1)).unwrap();
            assert_eq!(back.to_fen(), pos.to_fen());
            moves.push((m, (rng.next() % 2001) as i16 - 1000));
            pos.make_move(m);
        }
        check_roundtrip(&Game { start, wdl: (g % 3) as u8, moves });
    }
    assert!(castles > 200, "castles {}", castles);
}

#[test]
fn dfrc_king_move_vs_castling_san() {
    crate::attacks::init();
    // King d8 with a queen-side right (rook a8): O-O-O puts the king on c8,
    // which is also an ordinary king move. Seen in real OB DFRC games.
    let pos = Position::from_fen("r2k1rb1/1pqpp3/6R1/n1P1Pn2/pQ6/P7/2P2P2/B1KR1B1N b fa - 2 19").unwrap();
    let k = pgn::find_san(&pos, "Kc8").unwrap();
    let c = pgn::find_san(&pos, "O-O-O").unwrap();
    assert_eq!((mfrom(k), mto(k), mflag(k)), (59, 58, F_QUIET));
    assert_eq!((mfrom(c), mto(c), mflag(c)), (59, 58, F_QCASTLE));
}

#[test]
fn dfrc_pgn() {
    crate::attacks::init();
    // Shredder-FEN start; king b1 / b8, rooks a and h. O-O-O: king c1, rook d1.
    let text = r#"[Event "Fastchess Tournament"]
[Variant "fischerandom"]
[Result "1/2-1/2"]
[FEN "rkbqnbnr/pppppppp/8/8/8/8/PPPPPPPP/RKBQNBNR w HAha - 0 1"]
[SetUp "1"]

1. d3 {+0.20/10 0.004s} d6 {-0.10/10 0.004s} 2. Be3 {+0.25/10 0.004s} Be6 {-0.15/10 0.004s} 3. Qd2 {+0.30/10 0.004s} Qd7 {-0.20/10 0.004s} 4. O-O-O {+0.35/10 0.004s} O-O-O {-0.25/10 0.004s} 1/2-1/2
"#;
    let games = pgn::split_games(text);
    let g = pgn::to_game(&games[0]).unwrap();
    assert_eq!(g.moves.len(), 8);
    check_roundtrip(&g);
    let mut pos = g.start;
    for &(m, _) in &g.moves {
        pos.make_move(m);
    }
    assert_eq!(pos.to_fen(), "2krnbnr/pppqpppp/3pb3/8/8/3PB3/PPPQPPPP/2KRNBNR w - - 6 5");
}

/// Opt-in: decode every game in a real file with the reference reader and
/// replay every move. `DATATOOLS_CHECK=file.vf cargo test --release -- --ignored`
#[test]
#[ignore]
fn check_file_with_reference_reader() {
    let path = std::env::var("DATATOOLS_CHECK").expect("set DATATOOLS_CHECK to a .vf file");
    let data = std::fs::read(&path).unwrap();
    let mut reader = &data[..];
    let (mut games, mut positions) = (0u64, 0u64);
    while !reader.is_empty() {
        let game = viriformat::dataformat::Game::deserialise_from(&mut reader, Vec::new()).unwrap();
        game.visit_positions(|_, _| positions += 1);
        games += 1;
    }
    println!("{}: {} games, {} positions decoded", path, games, positions);
}

/// Opt-in: check a converted old-datagen file against its source. Replaying
/// the .vf with the reference reader, the scored positions must be the .bin
/// records, in order, with the same boards, side to move and scores (only
/// records at the end of a chain may be missing).
/// `DATATOOLS_BIN=x.bin DATATOOLS_VF=x.vf cargo test --release -- --ignored`
#[test]
#[ignore]
fn check_oldbin_conversion() {
    let (Ok(bin), Ok(vf)) = (std::env::var("DATATOOLS_BIN"), std::env::var("DATATOOLS_VF")) else {
        return;
    };
    let records = std::fs::read(&bin).unwrap();
    let data = std::fs::read(&vf).unwrap();
    let mut reader = &data[..];
    let mut next = 0usize; // next .bin record to match
    let (mut matched, mut skipped) = (0u64, 0u64);
    while !reader.is_empty() {
        let game = viriformat::dataformat::Game::deserialise_from(&mut reader, Vec::new()).unwrap();
        game.visit_positions(|board, eval| {
            if eval == viri::NO_SCORE as i32 {
                return;
            }
            let packed = board.to_marlinformat(0, 0, 0).as_bytes();
            let occ = u64::from_le_bytes(packed[0..8].try_into().unwrap());
            let mut want = [NONE_PC; 64];
            let (mut bits, mut i) = (occ, 0);
            while bits != 0 {
                let sq = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let code = (packed[8 + i / 2] >> ((i & 1) * 4)) & 15;
                let pt = if code & 7 == 6 { ROOK } else { (code & 7) as usize };
                want[sq] = make_pc(pt, (code >> 3) as usize);
                i += 1;
            }
            let stm = (packed[24] >> 7) as usize;
            loop {
                assert!(next * 32 < records.len(), "ran out of records");
                let rec = crate::oldbin::decode(&records[next * 32..next * 32 + 32]);
                next += 1;
                if rec.board == want && rec.stm == stm && rec.score as i32 == eval {
                    matched += 1;
                    break;
                }
                skipped += 1;
            }
        });
    }
    println!("{} records matched in order, {} records not in the output", matched, skipped);
}

#[test]
fn score_parsing() {
    assert_eq!(pgn::parse_score("+0.35/12 0.010s"), Some(35));
    assert_eq!(pgn::parse_score("-12.07/9 0.009s, n=8836"), Some(-1207));
    assert_eq!(pgn::parse_score("+M3/10 0.009s"), None);
    // Tablebase scores (cp +-(20000 - plies)) are not evals.
    assert_eq!(pgn::parse_score("+199.99/1 0.000s"), None);
    assert_eq!(pgn::parse_score("-199.80/14 0.010s"), None);
}
