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
    RfpMargin = "RfpMargin", 68, 30, 150, 6;
    // Razoring at any depth: eval + RazorBase + RazorMul * depth^2 <= alpha. Starting values
    // fit the old tuned linear margin (242 + 312 * depth, depth <= 3) at depths 1 and 3.
    RazorBase = "RazorBase", 480, 100, 800, 35;
    RazorMul = "RazorMul", 80, 20, 200, 9;
    NmpEvalDiv = "NmpEvalDiv", 182, 100, 400, 15;
    ProbcutMargin = "ProbcutMargin", 226, 100, 350, 13;
    FutBase = "FutBase", 87, 30, 250, 11;
    FutMul = "FutMul", 100, 40, 200, 8;
    HistPrune = "HistPrune", 2720, 500, 5000, 225;
    SeeQuiet = "SeeQuiet", 11, 0, 80, 4;
    SeeNoisy = "SeeNoisy", 87, 10, 200, 10;
    LmrBaseX100 = "LmrBaseX100", 56, 30, 130, 5;
    LmrDivX100 = "LmrDivX100", 288, 150, 350, 10;
    LmrHistDiv = "LmrHistDiv", 8139, 3000, 16000, 650;
    AspDelta = "AspDelta", 10, 3, 40, 2;
    QsFut = "QsFut", 454, 50, 500, 23;
    HistMul = "HistMul", 301, 100, 500, 20;
    HistOff = "HistOff", 212, 0, 500, 25;
    HistMax = "HistMax", 2739, 1000, 4000, 150;
    SeDouble = "SeDouble", 13, 5, 50, 2;
    RfpDepth = "RfpDepth", 8, 3, 12, 1;
    // Null move needs static_eval >= beta - NmpDepthMul * depth + NmpMarginBase.
    NmpDepthMul = "NmpDepthMul", 20, 0, 60, 3;
    NmpMarginBase = "NmpMarginBase", 150, 0, 400, 20;
    // Largest |TT score| still used for a cutoff at halfmove >= 90 (half a pawn).
    HmGuard = "HmGuard", 50, 0, 300, 15;
    NmpBase = "NmpBase", 4, 2, 7, 1;
    LmpBase = "LmpBase", 6, 1, 8, 1;
    SeMul = "SeMul", 21, 8, 40, 2;
    CapLmrDiv = "CapLmrDiv", 3337, 2000, 16000, 700;
    TmSoftDiv = "TmSoftDiv", 27, 12, 40, 1;
    TmIncPct = "TmIncPct", 75, 40, 100, 3;
    TmHardMul = "TmHardMul", 5, 2, 6, 1;
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
    UseQsFut = "UseQsFut", 1, 0, 1, 0;
    UseQsSee = "UseQsSee", 1, 0, 1, 0;
    UseQsEvasionLimit = "UseQsEvasionLimit", 1, 0, 1, 0;
    UseCorrHist = "UseCorrHist", 1, 0, 1, 0;
    UseAsp = "UseAsp", 1, 0, 1, 0;
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
