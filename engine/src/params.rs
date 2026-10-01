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
    RfpMargin = "RfpMargin", 78, 30, 150, 6;
    RazorBase = "RazorBase", 233, 50, 400, 18;
    RazorMul = "RazorMul", 311, 100, 400, 15;
    NmpEvalDiv = "NmpEvalDiv", 190, 100, 400, 15;
    ProbcutMargin = "ProbcutMargin", 218, 100, 350, 13;
    FutBase = "FutBase", 89, 30, 250, 11;
    FutMul = "FutMul", 104, 40, 200, 8;
    HistPrune = "HistPrune", 2513, 500, 5000, 225;
    SeeQuiet = "SeeQuiet", 10, 10, 80, 4;
    SeeNoisy = "SeeNoisy", 81, 10, 200, 10;
    LmrBaseX100 = "LmrBaseX100", 60, 30, 130, 5;
    LmrDivX100 = "LmrDivX100", 286, 150, 350, 10;
    LmrHistDiv = "LmrHistDiv", 7931, 3000, 16000, 650;
    AspDelta = "AspDelta", 10, 3, 40, 2;
    QsFut = "QsFut", 449, 50, 500, 23;
    HistMul = "HistMul", 256, 100, 500, 20;
    HistOff = "HistOff", 221, 0, 500, 25;
    HistMax = "HistMax", 2686, 1000, 4000, 150;
    SeDouble = "SeDouble", 15, 5, 50, 2;
    RfpDepth = "RfpDepth", 7, 3, 12, 1;
    NmpBase = "NmpBase", 4, 2, 7, 1;
    LmpBase = "LmpBase", 6, 1, 8, 1;
    SeMul = "SeMul", 20, 8, 40, 2;
    CapLmrDiv = "CapLmrDiv", 3735, 2000, 16000, 700;
    TmSoftDiv = "TmSoftDiv", 28, 12, 40, 1;
    TmIncPct = "TmIncPct", 71, 40, 100, 3;
    TmHardMul = "TmHardMul", 6, 2, 6, 1;
}

#[inline(always)]
pub fn tp(p: P) -> i32 {
    unsafe { PARAMS[p as usize].val }
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
        for p in (*addr_of!(PARAMS)).iter() {
            println!("{}, int, {}, {}, {}, {}, 0.002", p.name, p.val, p.min, p.max, p.step);
        }
    }
}
