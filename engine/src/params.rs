use std::ptr::{addr_of, addr_of_mut};

// Tunable search parameters. Built with `--features tune` (`make openbench`), they are
// advertised as UCI options for SPSA; `tune-spec` prints them in OpenBench's SPSA format.
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
    };
}

// name = UCI name, default, min, max, SPSA step (c_end). Step = (max - min) / 20, at least 1.
params! {
    RfpMargin = "RfpMargin", 70, 30, 150, 6;
    // Razoring at any depth: eval + RazorBase + RazorMul * depth^2 <= alpha. Starting values
    // fit the old tuned linear margin (242 + 312 * depth, depth <= 3) at depths 1 and 3.
    RazorBase = "RazorBase", 487, 100, 800, 35;
    RazorMul = "RazorMul", 80, 20, 200, 9;
    NmpEvalDiv = "NmpEvalDiv", 169, 100, 400, 15;
    ProbcutMargin = "ProbcutMargin", 217, 100, 350, 13;
    FutBase = "FutBase", 92, 30, 250, 11;
    FutMul = "FutMul", 102, 40, 200, 8;
    HistPrune = "HistPrune", 3038, 500, 5000, 225;
    SeeQuiet = "SeeQuiet", 11, 0, 80, 4;
    SeeNoisy = "SeeNoisy", 87, 10, 200, 10;
    LmrBaseX100 = "LmrBaseX100", 50, 30, 130, 5;
    LmrDivX100 = "LmrDivX100", 297, 150, 350, 10;
    LmrHistDiv = "LmrHistDiv", 8293, 3000, 16000, 650;
    AspDelta = "AspDelta", 11, 3, 40, 2;
    HistMul = "HistMul", 299, 100, 500, 20;
    HistOff = "HistOff", 198, 0, 500, 25;
    HistMax = "HistMax", 2649, 1000, 4000, 150;
    SeDouble = "SeDouble", 10, 5, 50, 2;
    RfpDepth = "RfpDepth", 7, 3, 12, 1;
    // Null move needs static_eval >= beta - NmpDepthMul * depth + NmpMarginBase.
    NmpDepthMul = "NmpDepthMul", 23, 0, 60, 3;
    NmpMarginBase = "NmpMarginBase", 143, 0, 400, 20;
    // Largest |TT score| still used for a cutoff at halfmove >= 90 (half a pawn).
    HmGuard = "HmGuard", 54, 0, 300, 15;
    NmpBase = "NmpBase", 4, 2, 7, 1;
    LmpBase = "LmpBase", 5, 1, 8, 1;
    SeMul = "SeMul", 22, 8, 40, 2;
    CapLmrDiv = "CapLmrDiv", 3260, 2000, 16000, 700;
    TmSoftDiv = "TmSoftDiv", 28, 12, 40, 1;
    TmIncPct = "TmIncPct", 79, 40, 100, 3;
    TmHardMul = "TmHardMul", 5, 2, 6, 1;
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
    UseKillers = "UseKillers", 1, 0, 1, 0;
}

#[inline(always)]
pub fn tp(p: P) -> i32 {
    unsafe { PARAMS[p as usize].val }
}

/// A feature switch (a step-0 parameter) is on.
#[inline(always)]
pub fn on(p: P) -> bool {
    tp(p) != 0
}

pub fn set(name: &str, v: i32) -> bool {
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
