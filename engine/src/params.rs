// Tunable search parameters (exposed as UCI options for SPSA tuning).
pub struct Param {
    pub name: &'static str,
    pub val: i32,
    pub min: i32,
    pub max: i32,
}

macro_rules! params {
    ($($id:ident = $name:literal, $v:expr, $lo:expr, $hi:expr;)*) => {
        #[allow(non_camel_case_types, dead_code)]
        #[repr(usize)]
        pub enum P { $($id,)* COUNT }
        pub static mut PARAMS: [Param; P::COUNT as usize] = [$(Param { name: $name, val: $v, min: $lo, max: $hi },)*];
    };
}

params! {
    RfpMargin = "RfpMargin", 95, 30, 150;
    RazorBase = "RazorBase", 243, 50, 400;
    RazorMul = "RazorMul", 284, 100, 400;
    NmpEvalDiv = "NmpEvalDiv", 195, 100, 400;
    ProbcutMargin = "ProbcutMargin", 209, 100, 350;
    FutBase = "FutBase", 96, 30, 250;
    FutMul = "FutMul", 130, 40, 200;
    HistPrune = "HistPrune", 2269, 500, 5000;
    SeeQuiet = "SeeQuiet", 25, 10, 80;
    SeeNoisy = "SeeNoisy", 42, 40, 200;
    LmrBase = "LmrBase", 60, 30, 130;
    LmrDiv = "LmrDiv", 246, 150, 350;
    LmrHistDiv = "LmrHistDiv", 7888, 3000, 16000;
    AspDelta = "AspDelta", 8, 6, 40;
    QsFut = "QsFut", 294, 50, 300;
    HistMul = "HistMul", 226, 100, 500;
    HistOff = "HistOff", 241, 0, 500;
    HistMax = "HistMax", 2648, 1000, 4000;
    SeDouble = "SeDouble", 18, 5, 50;
    RfpDepth = "RfpDepth", 5, 5, 12;
    NmpBase = "NmpBase", 5, 2, 5;
    LmpBase = "LmpBase", 4, 1, 8;
    SeMul = "SeMul", 23, 8, 40;
    CapLmrDiv = "CapLmrDiv", 6036, 2000, 16000;
    TmSoftDiv = "TmSoftDiv", 25, 12, 40;
    TmIncPct = "TmIncPct", 75, 40, 100;
    TmHardMul = "TmHardMul", 4, 2, 6;
}

#[inline(always)]
pub fn tp(p: P) -> i32 {
    unsafe { PARAMS[p as usize].val }
}

pub fn set(name: &str, v: i32) -> bool {
    unsafe {
        for p in PARAMS.iter_mut() {
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
        for p in PARAMS.iter() {
            println!("option name {} type spin default {} min {} max {}", p.name, p.val, p.min, p.max);
        }
    }
}
