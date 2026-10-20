use std::ptr::{addr_of, addr_of_mut};

// Tunable search parameters. Built with `--features tune` (`make openbench`), they are
// advertised as UCI options for SPSA and can be set (also as bench ablations);
// `tune-spec` prints them in OpenBench's SPSA format. Other builds use the
// defaults as compile-time constants.
pub struct Param {
    pub name: &'static str,
    pub val: i32,
    pub min: i32,
    pub max: i32,
    pub step: i32,
}

macro_rules! params {
    ($($id:ident = $name:literal, $v:expr, $lo:expr, $hi:expr, $step:expr;)*) => {
        #[allow(non_camel_case_types, dead_code)]
        #[repr(usize)]
        pub enum P { $($id,)* COUNT }
        pub static mut PARAMS: [Param; P::COUNT as usize] = [$(Param { name: $name, val: $v, min: $lo, max: $hi, step: $step },)*];
        #[cfg(not(feature = "tune"))]
        const DEFAULTS: [i32; P::COUNT as usize] = [$($v,)*];

        /// A parameter's value: from PARAMS (settable) with `tune`, else its
        /// default as a constant, which folds into the code using it.
        #[inline(always)]
        pub fn tp(p: P) -> i32 {
            #[cfg(feature = "tune")]
            {
                unsafe { PARAMS[p as usize].val }
            }
            #[cfg(not(feature = "tune"))]
            {
                DEFAULTS[p as usize]
            }
        }
    };
}

// name = UCI name, default, min, max, SPSA step (c_end). Step = (max - min) / 20, at least 1.
params! {
    RfpMargin = "RfpMargin", 70, 30, 150, 6;
    // Razoring at any depth: eval + RazorBase + RazorMul * depth^2 <= alpha. Starting values
    // fit the old tuned linear margin (242 + 312 * depth, depth <= 3) at depths 1 and 3.
    RazorBase = "RazorBase", 400, 100, 800, 35;
    RazorMul = "RazorMul", 81, 20, 200, 9;
    NmpEvalDiv = "NmpEvalDiv", 165, 100, 400, 15;
    ProbcutMargin = "ProbcutMargin", 217, 100, 350, 13;
    FutBase = "FutBase", 73, 30, 250, 11;
    FutMul = "FutMul", 75, 40, 200, 8;
    // Futility margin by quiet history: a quiet's history sum (the ordering
    // score, four tables of +-16384) over this is added to the margin, so a
    // well-scored quiet survives a margin a badly scored one does not. Start:
    // 128, so a move at one table's ceiling adds 128 cp, about one depth of
    // FutMul.
    FutHistDiv = "FutHistDiv", 118, 32, 512, 12;
    HistPrune = "HistPrune", 3106, 500, 5000, 225;
    SeeQuiet = "SeeQuiet", 10, 0, 80, 4;
    // Quiet SEE bar by history: the history sum over this is subtracted from
    // the bar. Start 128, the same scale as FutHistDiv (a table ceiling of
    // 16384 moves the bar by 128 cp).
    SeeHistDiv = "SeeHistDiv", 137, 32, 512, 12;
    SeeNoisy = "SeeNoisy", 108, 10, 200, 10;
    LmrBaseX100 = "LmrBaseX100", 44, 30, 130, 5;
    LmrDivX100 = "LmrDivX100", 298, 150, 350, 10;
    LmrHistDiv = "LmrHistDiv", 9359, 3000, 16000, 650;
    AspDelta = "AspDelta", 15, 3, 40, 2;
    HistMul = "HistMul", 338, 100, 500, 20;
    HistOff = "HistOff", 146, 0, 500, 25;
    HistMax = "HistMax", 2796, 1000, 4000, 150;
    // Bonus for the opponent's previous quiet move when this node fails low (it refuted
    // every reply): PriorBonusPct% of the cutoff bonus at this depth. Half, as weaker
    // evidence than a cutoff; 0 disables it.
    PriorBonusPct = "PriorBonusPct", 54, 0, 150, 8;
    SeDouble = "SeDouble", 13, 5, 50, 2;
    // A reduced search that beats the best score so far by more than LmrDeeper is
    // re-searched one ply deeper than the full depth (half a pawn, round value).
    LmrDeeper = "LmrDeeper", 49, 10, 200, 10;
    RfpDepth = "RfpDepth", 7, 3, 12, 1;
    // Null move needs static_eval >= beta - NmpDepthMul * depth + NmpMarginBase.
    NmpDepthMul = "NmpDepthMul", 24, 0, 60, 3;
    NmpMarginBase = "NmpMarginBase", 142, 0, 400, 20;
    // Largest |TT score| still used for a cutoff at halfmove >= 90 (half a pawn).
    HmGuard = "HmGuard", 43, 0, 300, 15;
    NmpBase = "NmpBase", 5, 2, 7, 1;
    LmpBase = "LmpBase", 6, 1, 8, 1;
    SeMul = "SeMul", 30, 8, 40, 2;
    CapLmrDiv = "CapLmrDiv", 3530, 2000, 16000, 700;
    TmSoftDiv = "TmSoftDiv", 28, 12, 40, 1;
    TmIncPct = "TmIncPct", 79, 40, 100, 3;
    TmHardMul = "TmHardMul", 5, 2, 6, 1;
    // Exponent (x100) on the soft target's combined scale (node share x
    // stability x fail-lows): 100 = linear. SPSA #4304 (32k games, STC): 106.
    TmScalePow = "TmScalePow", 106, 50, 250, 15;
    // On a ponder hit, move at once if we pondered at least PonderHitPct% of
    // the time we'd want for this move; otherwise keep thinking until pondering
    // + thinking reaches it (PonderCredit% of the pondering time counts).
    PonderHitPct = "PonderHitPct", 80, 30, 1000, 8;
    PonderCredit = "PonderCredit", 100, 30, 100, 10;
    // Least thinking after a hit, % of the soft limit. With 0, a ponder that
    // covered the target ended in an instant move (24-39% of moves in local
    // ponder gauntlets, 38-49% on lichess) and the clock went unused; 30 keeps
    // instant moves at 2-4%, like other engines that ponder.
    PonderMinPct = "PonderMinPct", 30, 0, 100, 5;
    TmFinishPct = "TmFinishPct", 100, 50, 150, 10;
    TmFailLow = "TmFailLow", 28, 0, 100, 8;
    // Node-share scale (TmNodeBase/100 - best move's share of root nodes):
    // a larger base flattens it. Stability scale: the table's distance from 1
    // times TmStabAmp/100. Hand-set shapes, exposed so tune rounds can move them.
    TmNodeBase = "TmNodeBase", 150, 110, 250, 10;
    TmStabAmp = "TmStabAmp", 100, 0, 200, 15;
    TmFeedMax = "TmFeedMax", 250, 100, 400, 20;
    // Soft-target extensions stop at |score| >= TmDecided (internal cp; round
    // value at the start of the bucket where extension gained nothing).
    TmDecided = "TmDecided", 400, 200, 2000, 50;
    // Sudden death: keep this many moves' worth of the measured lag in reserve
    // (round value; lag-free matches measure ~0, so OB can't tune it).
    TmLagMoves = "TmLagMoves", 40, 0, 200, 10;
    // Depth gates (formerly constants).
    NmpDepth = "NmpDepth", 2, 2, 6, 1;
    // Verify a null-move cutoff with a reduced real search from this depth on
    // (null move stays off at that ply). Start: 12, where a null-move cut
    // prunes a subtree deep enough for a zugzwang to matter.
    NmpVerifyDepth = "NmpVerifyDepth", 12, 6, 30, 2;
    ProbcutDepth = "ProbcutDepth", 3, 3, 8, 1;
    ProbcutRed = "ProbcutRed", 5, 2, 6, 1;
    IirDepth = "IirDepth", 2, 2, 8, 1;
    FutDepth = "FutDepth", 10, 4, 12, 1;
    HistPruneDepth = "HistPruneDepth", 5, 2, 8, 1;
    SeeNoisyDepth = "SeeNoisyDepth", 6, 3, 10, 1;
    // A capture whose static exchange gain is at least this (cp, half a pawn
    // below zero) is ordered as a good capture, before the killers and quiets;
    // the rest follow the quiets, and with staged generation the quiets are
    // generated as soon as only those remain.
    GoodCaptureSee = "GoodCaptureSee", -56, -150, 0, 8;
    // SEE piece values (pawn = 100), textbook starting values: used by SEE pruning,
    // ProbCut, qsearch SEE and capture ordering.
    SeeKnight = "SeeKnight", 300, 150, 650, 25;
    SeeBishop = "SeeBishop", 342, 150, 650, 25;
    SeeRook = "SeeRook", 499, 250, 1000, 37;
    SeeQueen = "SeeQueen", 928, 500, 2000, 75;
    SeDepth = "SeDepth", 5, 4, 10, 1;
    // A TT entry qualifies its move for the singular test when its depth is
    // at least the node's depth minus this. Start: 3, the value used so far.
    SeTtDepth = "SeTtDepth", 4, 0, 6, 1;
    // TT replacement: a was-PV entry counts this many plies deeper when a
    // victim is chosen. Start 2, a small round value (one age step is 8).
    TtPvKeep = "TtPvKeep", 1, 0, 8, 1;
    // Correction history. The eval correction is (CorrPawnWeight * pawn entry +
    // CorrNonPawnWeight * (white + black non-pawn entries)) / 128; the defaults are the
    // original 2:1:1 weighting, as separate weights so SPSA can move each one.
    CorrPawnWeight = "CorrPawnWeight", 95, 0, 256, 13;
    CorrNonPawnWeight = "CorrNonPawnWeight", 124, 0, 256, 13;
    // Update: an entry moves toward the search's eval error (clamped to
    // +-CorrDiffClamp cp) by min(depth + 1, CorrUpdateCap) / 256; entries are
    // kept within +-CorrLimit cp.
    CorrUpdateCap = "CorrUpdateCap", 13, 4, 64, 3;
    CorrDiffClamp = "CorrDiffClamp", 416, 100, 1000, 45;
    CorrLimit = "CorrLimit", 48, 16, 256, 12;
    // Weight of the continuation correction (keyed by the previous move's
    // piece and square): half the pawn table's original weight of 128, as
    // one non-pawn table had, until SPSA says otherwise.
    CorrContWeight = "CorrContWeight", 64, 0, 256, 13;
    // Ply-2 continuation correction weight: the round start the ply-1 table had (64).
    CorrCont2Weight = "CorrCont2Weight", 69, 0, 256, 13;
    // Network output multiplier in percent, applied to every search eval. Lets a
    // net trained on differently scaled labels be run at Incipit's eval scale
    // (which the pruning margins and other cp thresholds are tuned to) without
    // retuning them. Step 0 keeps it out of tune-spec; set it per test.
    EvalScale = "EvalScale", 100, 25, 400, 0;
    // Lazy SMP vote: a thread's weight is rel * VoteBase + min(score - lowest
    // score, VoteCap), with rel its depth relative to the iteration (1 for
    // d - 1, 2 for d, 3 for d + 1). VoteBase is one unit per ply of relative
    // depth (a round number; only ratios matter). VoteCap 50: optimism
    // beyond half a pawn adds nothing, and the score term is worth at most
    // half a ply, so depth decides and the score only breaks near-ties.
    VoteBase = "VoteBase", 100, 20, 300, 10;
    VoteCap = "VoteCap", 50, 0, 200, 10;
    // Feature switches for ablation tests (1 = on, 0 = off). Step 0 keeps them out of
    // tune-spec; OpenBench builds advertise them, so a test can set e.g. UseProbcut=0.
    UseTtCut = "UseTtCut", 1, 0, 1, 0;
    UseRfp = "UseRfp", 1, 0, 1, 0;
    UseRazor = "UseRazor", 1, 0, 1, 0;
    UseNmp = "UseNmp", 1, 0, 1, 0;
    UseProbcut = "UseProbcut", 1, 0, 1, 0;
    UseIir = "UseIir", 1, 0, 1, 0;
    UseLmp = "UseLmp", 1, 0, 1, 0;
    UseFut = "UseFut", 1, 0, 1, 0;
    UseFutHist = "UseFutHist", 1, 0, 1, 0;
    UseHistPrune = "UseHistPrune", 1, 0, 1, 0;
    UseSeeQuiet = "UseSeeQuiet", 1, 0, 1, 0;
    UseSeeHist = "UseSeeHist", 1, 0, 1, 0;
    UseSeeNoisy = "UseSeeNoisy", 1, 0, 1, 0;
    UseSe = "UseSe", 1, 0, 1, 0;
    UseSeDoubleExt = "UseSeDoubleExt", 1, 0, 1, 0;
    UseMulticut = "UseMulticut", 1, 0, 1, 0;
    UseSeNegExt = "UseSeNegExt", 1, 0, 1, 0;
    UseSeNeg2 = "UseSeNeg2", 1, 0, 1, 0;
    UseLmr = "UseLmr", 1, 0, 1, 0;
    UseLmrCutNoTt = "UseLmrCutNoTt", 1, 0, 1, 0;
    UseLmrDeeper = "UseLmrDeeper", 1, 0, 1, 0;
    UseTtPv = "UseTtPv", 1, 0, 1, 0;
    UseTtPvKeep = "UseTtPvKeep", 1, 0, 1, 0;
    UseQsSee = "UseQsSee", 1, 0, 1, 0;
    UseQsEvasionLimit = "UseQsEvasionLimit", 1, 0, 1, 0;
    UseCorrHist = "UseCorrHist", 1, 0, 1, 0;
    UseAsp = "UseAsp", 1, 0, 1, 0;
    // Soft time limit scaled by best-move node share and stability (0: plain soft limit).
    UseTm = "UseTm", 1, 0, 1, 0;
    // Don't start a depth predicted (elapsed x last branching factor) to end past
    // TmFinishPct% of the hard limit: it would be cut off anyway.
    UseTmFinish = "UseTmFinish", 0, 0, 1, 0;
    // Extend the soft target by TmFailLow% per root aspiration fail-low in the
    // last depth (up to 3): the best line just got worse, so look harder.
    UseTmFailLow = "UseTmFailLow", 1, 0, 1, 0;
    // Extend the soft target when the score drops between depths.
    UseTmScore = "UseTmScore", 0, 0, 1, 0;
    // 1: the fail-low and score-drop extensions count once (the larger of the
    // two) instead of multiplying: they mostly fire on the same moves.
    UseTmExtMax = "UseTmExtMax", 0, 0, 1, 0;
    // Budget feedback: scale the soft limit by 1 / (running average of our own
    // clock spend / base soft limit), between 1 and TmFeedMax/100. Spends the
    // time ponder hits would otherwise leave unused. Only in games where the
    // GUI ponders (without pondering spend already meets the budget).
    UseTmFeed = "UseTmFeed", 1, 0, 1, 0;
    UseKillers = "UseKillers", 1, 0, 1, 0;
    // A failed null move's refuting capture names a hanging piece of ours;
    // its moves are ordered with the killers.
    UseNmpThreat = "UseNmpThreat", 1, 0, 1, 0;
    // Continuation history keyed by our move three plies back (prev6), at
    // half weight in move scoring like the ply-4 table.
    UseCont6 = "UseCont6", 1, 0, 1, 0;
    // Correction history: incremental updates toward zero residual (1) or the
    // old average toward the residual (0).
    UseCorrIncr = "UseCorrIncr", 1, 0, 1, 0;
    // Continuation correction history (the previous move's piece-square).
    UseCorrCont = "UseCorrCont", 1, 0, 1, 0;
    UseCorrCont2 = "UseCorrCont2", 1, 0, 1, 0;
}

/// A feature switch (a step-0 parameter) is on.
#[inline(always)]
pub fn on(p: P) -> bool {
    tp(p) != 0
}

/// The parameter's UCI name, if `name` (any case) is one.
pub fn find(name: &str) -> Option<&'static str> {
    unsafe { (*addr_of!(PARAMS)).iter().find(|p| p.name.eq_ignore_ascii_case(name)).map(|p| p.name) }
}

/// Sets a parameter by UCI name; false if there is none or (without `tune`)
/// parameters are constants.
pub fn set(name: &str, v: i32) -> bool {
    if !cfg!(feature = "tune") {
        return false;
    }
    unsafe {
        for p in (*addr_of_mut!(PARAMS)).iter_mut() {
            if p.name.eq_ignore_ascii_case(name) {
                p.val = v.clamp(p.min, p.max);
                return true;
            }
        }
    }
    false
}

pub fn print_options() {
    unsafe {
        for p in (*addr_of!(PARAMS)).iter() {
            println!("option name {} type spin default {} min {} max {}", p.name, p.val, p.min, p.max);
        }
    }
}

/// SPSA parameter list in OpenBench's input format: name, int, default, min, max, c_end, r_end.
pub fn print_spec() {
    unsafe {
        for p in (*addr_of!(PARAMS)).iter().filter(|p| p.step > 0) {
            println!("{}, int, {}, {}, {}, {}, 0.002", p.name, p.val, p.min, p.max, p.step);
        }
    }
}
