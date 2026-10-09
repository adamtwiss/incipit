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
    RfpMargin = "RfpMargin", 71, 30, 150, 6;
    // Razoring at any depth: eval + RazorBase + RazorMul * depth^2 <= alpha. Starting values
    // fit the old tuned linear margin (242 + 312 * depth, depth <= 3) at depths 1 and 3.
    RazorBase = "RazorBase", 483, 100, 800, 35;
    RazorMul = "RazorMul", 86, 20, 200, 9;
    NmpEvalDiv = "NmpEvalDiv", 173, 100, 400, 15;
    ProbcutMargin = "ProbcutMargin", 219, 100, 350, 13;
    FutBase = "FutBase", 92, 30, 250, 11;
    FutMul = "FutMul", 96, 40, 200, 8;
    HistPrune = "HistPrune", 3028, 500, 5000, 225;
    SeeQuiet = "SeeQuiet", 9, 0, 80, 4;
    SeeNoisy = "SeeNoisy", 91, 10, 200, 10;
    LmrBaseX100 = "LmrBaseX100", 47, 30, 130, 5;
    LmrDivX100 = "LmrDivX100", 303, 150, 350, 10;
    LmrHistDiv = "LmrHistDiv", 8293, 3000, 16000, 650;
    AspDelta = "AspDelta", 11, 3, 40, 2;
    HistMul = "HistMul", 299, 100, 500, 20;
    HistOff = "HistOff", 198, 0, 500, 25;
    HistMax = "HistMax", 2649, 1000, 4000, 150;
    SeDouble = "SeDouble", 11, 5, 50, 2;
    RfpDepth = "RfpDepth", 7, 3, 12, 1;
    // Null move needs static_eval >= beta - NmpDepthMul * depth + NmpMarginBase.
    NmpDepthMul = "NmpDepthMul", 25, 0, 60, 3;
    NmpMarginBase = "NmpMarginBase", 137, 0, 400, 20;
    // Largest |TT score| still used for a cutoff at halfmove >= 90 (half a pawn).
    HmGuard = "HmGuard", 54, 0, 300, 15;
    NmpBase = "NmpBase", 4, 2, 7, 1;
    LmpBase = "LmpBase", 5, 1, 8, 1;
    SeMul = "SeMul", 23, 8, 40, 2;
    CapLmrDiv = "CapLmrDiv", 3260, 2000, 16000, 700;
    TmSoftDiv = "TmSoftDiv", 28, 12, 40, 1;
    TmIncPct = "TmIncPct", 79, 40, 100, 3;
    TmHardMul = "TmHardMul", 5, 2, 6, 1;
    // On a ponder hit, move at once if we pondered at least PonderHitPct% of
    // the time we'd want for this move; otherwise keep thinking until pondering
    // + thinking reaches it (PonderCredit% of the pondering time counts).
    PonderHitPct = "PonderHitPct", 80, 30, 1000, 8;
    PonderCredit = "PonderCredit", 100, 30, 100, 10;
    // Least thinking after a hit, % of the soft limit (0: none; the
    // PonderHitPct rule already prevents instant replies to short ponders).
    PonderMinPct = "PonderMinPct", 0, 0, 100, 5;
    TmFinishPct = "TmFinishPct", 100, 50, 150, 10;
    TmFailLow = "TmFailLow", 30, 0, 100, 8;
    TmFeedMax = "TmFeedMax", 250, 100, 400, 20;
    // Clock surplus (ponder games with an increment): reserve in increments,
    // and the number of moves to spend the surplus over.
    TmSurplusInc = "TmSurplusInc", 10, 2, 40, 2;
    TmSurplusDiv = "TmSurplusDiv", 10, 3, 30, 2;
    // Depth gates (formerly constants).
    NmpDepth = "NmpDepth", 3, 2, 6, 1;
    ProbcutDepth = "ProbcutDepth", 5, 3, 8, 1;
    ProbcutRed = "ProbcutRed", 5, 2, 6, 1;
    IirDepth = "IirDepth", 3, 2, 8, 1;
    FutDepth = "FutDepth", 9, 4, 12, 1;
    HistPruneDepth = "HistPruneDepth", 4, 2, 8, 1;
    SeeNoisyDepth = "SeeNoisyDepth", 7, 3, 10, 1;
    SeDepth = "SeDepth", 6, 4, 10, 1;
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
    // Spend the clock surplus above TmSurplusInc increments in ponder games.
    UseTmSurplus = "UseTmSurplus", 1, 0, 1, 0;
    UseKillers = "UseKillers", 1, 0, 1, 0;
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
