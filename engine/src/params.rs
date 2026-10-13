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
    RfpMargin = "RfpMargin", 75, 30, 150, 6;
    // Razoring at any depth: eval + RazorBase + RazorMul * depth^2 <= alpha. Starting values
    // fit the old tuned linear margin (242 + 312 * depth, depth <= 3) at depths 1 and 3.
    RazorBase = "RazorBase", 509, 100, 800, 35;
    RazorMul = "RazorMul", 79, 20, 200, 9;
    NmpEvalDiv = "NmpEvalDiv", 165, 100, 400, 15;
    ProbcutMargin = "ProbcutMargin", 221, 100, 350, 13;
    FutBase = "FutBase", 82, 30, 250, 11;
    FutMul = "FutMul", 93, 40, 200, 8;
    HistPrune = "HistPrune", 3124, 500, 5000, 225;
    SeeQuiet = "SeeQuiet", 8, 0, 80, 4;
    SeeNoisy = "SeeNoisy", 91, 10, 200, 10;
    LmrBaseX100 = "LmrBaseX100", 44, 30, 130, 5;
    LmrDivX100 = "LmrDivX100", 311, 150, 350, 10;
    LmrHistDiv = "LmrHistDiv", 8418, 3000, 16000, 650;
    AspDelta = "AspDelta", 15, 3, 40, 2;
    HistMul = "HistMul", 302, 100, 500, 20;
    HistOff = "HistOff", 187, 0, 500, 25;
    HistMax = "HistMax", 2734, 1000, 4000, 150;
    // Bonus for the opponent's previous quiet move when this node fails low (it refuted
    // every reply): PriorBonusPct% of the cutoff bonus at this depth. Half, as weaker
    // evidence than a cutoff; 0 disables it.
    PriorBonusPct = "PriorBonusPct", 50, 0, 150, 8;
    SeDouble = "SeDouble", 12, 5, 50, 2;
    RfpDepth = "RfpDepth", 8, 3, 12, 1;
    // Null move needs static_eval >= beta - NmpDepthMul * depth + NmpMarginBase.
    NmpDepthMul = "NmpDepthMul", 23, 0, 60, 3;
    NmpMarginBase = "NmpMarginBase", 145, 0, 400, 20;
    // Largest |TT score| still used for a cutoff at halfmove >= 90 (half a pawn).
    HmGuard = "HmGuard", 66, 0, 300, 15;
    NmpBase = "NmpBase", 5, 2, 7, 1;
    LmpBase = "LmpBase", 5, 1, 8, 1;
    SeMul = "SeMul", 25, 8, 40, 2;
    CapLmrDiv = "CapLmrDiv", 3416, 2000, 16000, 700;
    TmSoftDiv = "TmSoftDiv", 28, 12, 40, 1;
    TmIncPct = "TmIncPct", 79, 40, 100, 3;
    TmHardMul = "TmHardMul", 5, 2, 6, 1;
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
    TmFailLow = "TmFailLow", 30, 0, 100, 8;
    TmFeedMax = "TmFeedMax", 250, 100, 400, 20;
    // Depth gates (formerly constants).
    NmpDepth = "NmpDepth", 2, 2, 6, 1;
    // Verify a null-move cutoff with a reduced real search from this depth on
    // (null move stays off at that ply). Start: 12, where a null-move cut
    // prunes a subtree deep enough for a zugzwang to matter.
    NmpVerifyDepth = "NmpVerifyDepth", 12, 6, 30, 2;
    ProbcutDepth = "ProbcutDepth", 4, 3, 8, 1;
    ProbcutRed = "ProbcutRed", 6, 2, 6, 1;
    IirDepth = "IirDepth", 2, 2, 8, 1;
    FutDepth = "FutDepth", 9, 4, 12, 1;
    HistPruneDepth = "HistPruneDepth", 5, 2, 8, 1;
    SeeNoisyDepth = "SeeNoisyDepth", 7, 3, 10, 1;
    // SEE piece values (pawn = 100), textbook starting values: used by SEE pruning,
    // ProbCut, qsearch SEE and capture ordering.
    SeeKnight = "SeeKnight", 313, 150, 650, 25;
    SeeBishop = "SeeBishop", 328, 150, 650, 25;
    SeeRook = "SeeRook", 514, 250, 1000, 37;
    SeeQueen = "SeeQueen", 951, 500, 2000, 75;
    SeDepth = "SeDepth", 6, 4, 10, 1;
    // Correction history. The eval correction is (CorrPawnWeight * pawn entry +
    // CorrNonPawnWeight * (white + black non-pawn entries)) / 128; the defaults are the
    // original 2:1:1 weighting, as separate weights so SPSA can move each one.
    CorrPawnWeight = "CorrPawnWeight", 123, 0, 256, 13;
    CorrNonPawnWeight = "CorrNonPawnWeight", 104, 0, 256, 13;
    // Update: an entry moves toward the search's eval error (clamped to
    // +-CorrDiffClamp cp) by min(depth + 1, CorrUpdateCap) / 256; entries are
    // kept within +-CorrLimit cp.
    CorrUpdateCap = "CorrUpdateCap", 18, 4, 64, 3;
    CorrDiffClamp = "CorrDiffClamp", 404, 100, 1000, 45;
    CorrLimit = "CorrLimit", 57, 16, 256, 12;
    // Weight of the continuation correction (keyed by the previous move's
    // piece and square): half the pawn table's original weight of 128, as
    // one non-pawn table had, until SPSA says otherwise.
    CorrContWeight = "CorrContWeight", 64, 0, 256, 13;
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
    UseHistPrune = "UseHistPrune", 1, 0, 1, 0;
    UseSeeQuiet = "UseSeeQuiet", 1, 0, 1, 0;
    UseSeeNoisy = "UseSeeNoisy", 1, 0, 1, 0;
    UseSe = "UseSe", 1, 0, 1, 0;
    UseSeDoubleExt = "UseSeDoubleExt", 1, 0, 1, 0;
    UseMulticut = "UseMulticut", 1, 0, 1, 0;
    UseSeNegExt = "UseSeNegExt", 1, 0, 1, 0;
    UseLmr = "UseLmr", 1, 0, 1, 0;
    UseLmrCutNoTt = "UseLmrCutNoTt", 1, 0, 1, 0;
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
    // Correction history: incremental updates toward zero residual (1) or the
    // old average toward the residual (0).
    UseCorrIncr = "UseCorrIncr", 1, 0, 1, 0;
    // Continuation correction history (the previous move's piece-square).
    UseCorrCont = "UseCorrCont", 1, 0, 1, 0;
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
