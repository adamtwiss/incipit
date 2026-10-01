// Trains an Incipit network with Bullet from viriformat data.
//
//   incipit-train <data_dir> <net_id> <out_dir> [superbatches] [hidden] [wdl] [mirror|plain|kb8|kb4] [screlu|pairwise] [key=value...]
//   Options (defaults = the recipe before they existed): lr=0.001 (peak LR),
//   floor=0.00000243 (final LR, 0.001 * 0.3^5), warmup=0 (fraction of the run
//   spent ramping the LR linearly up from lr/10), skip=0 (probability of
//   randomly skipping each position), min_ply=16 (drop earlier positions),
//   l1=0 (neurons in one hidden layer after the FT; kb4 SCReLU nets only).
//   incipit-train eval <checkpoint_dir> <hidden> <mirror|plain|kb8|kb4> <screlu|pairwise> <fen>...
//
// Reads every .vf file in <data_dir> (symlinks are fine), interleaving them so
// each batch mixes datasets. Architecture: 768 piece-square inputs, horizontally
// mirrored (one king bucket), plain, or mirrored with 8 king buckets (kb8, with a
// factoriser merged in at save time), SCReLU, 8 material output buckets,
// quantised 255/64 with eval scale 400. Convert the result with:
//   datatools net bullet <out_dir>/<net_id>-<N>/quantised.bin <dir> --hidden <H> [--mirror]
//     [--king-buckets <KB8, comma-separated>]
//
// `eval` prints Bullet's own evaluation (centipawns, side to move) of each FEN
// from a checkpoint, to compare against the engine with the converted net.
use bullet_lib::{
    game::{
        inputs::{Chess768, ChessBucketsMirrored},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::viribinpack::{Filter, ViriBinpackLoader, ViriFilter},
    },
};

const OUTPUT_BUCKETS: usize = 8;

/// King buckets for `kb8`, indexed by the perspective's king square after
/// horizontal mirroring (rank * 4 + file a-d): finer on the back two ranks.
/// This is the layout Incipit's engine used before the network format.
#[rustfmt::skip]
const KB8: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 3, 3,
    4, 4, 4, 4,
    5, 5, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    7, 7, 7, 7,
    7, 7, 7, 7,
];
const KB8_BUCKETS: usize = 8;

/// Coarser king buckets for `kb4`, by the king's rank only (no file split, so
/// castling doesn't change bucket): rank 1, rank 2, ranks 3-4, ranks 5-8.
#[rustfmt::skip]
const KB4: [usize; 32] = [
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];
const KB4_BUCKETS: usize = 4;

/// The network: inputs -> hidden (SCReLU, or pairwise CReLU giving hidden/2 per
/// perspective) x 2 -> 8 material output buckets.
/// (Macros because the trainer's concrete type is awkward to name.)
macro_rules! build {
    ($inputs:expr, $hidden:expr, $pairwise:expr) => {{
        let hidden: usize = $hidden;
        let pairwise: bool = $pairwise;
        ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs($inputs)
            .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
            .save_format(&[
                SavedFormat::id("l0w").round().quantise::<i16>(255),
                SavedFormat::id("l0b").round().quantise::<i16>(255),
                SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
                SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                let l0 = builder.new_affine("l0", 768, hidden);
                let l1 = builder.new_affine("l1", if pairwise { hidden } else { 2 * hidden }, OUTPUT_BUCKETS);
                let (stm, ntm) = if pairwise {
                    // Pairwise CReLU: each perspective's first half times its second half.
                    let ft = |input, a, b| l0.slice(a, b).forward(input).crelu();
                    let half = hidden / 2;
                    (ft(stm_inputs, 0, half) * ft(stm_inputs, half, hidden), ft(ntm_inputs, 0, half) * ft(ntm_inputs, half, hidden))
                } else {
                    (l0.forward(stm_inputs).screlu(), l0.forward(ntm_inputs).screlu())
                };
                l1.forward(stm.concat(ntm)).select(output_buckets)
            })
    }};
}

/// King-bucketed variant: 768 x buckets inputs plus a shared 768-input
/// factoriser, merged into every bucket's weights when saving.
macro_rules! build_kb {
    ($layout:expr, $nb:expr, $hidden:expr, $pairwise:expr) => {{
        let hidden: usize = $hidden;
        let pairwise: bool = $pairwise;
        const NB: usize = $nb;
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs(ChessBucketsMirrored::new($layout))
            .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
            .save_format(&[
                SavedFormat::id("l0w")
                    .transform(|store, weights| {
                        let factoriser = store.get("l0f").values.f32().repeat(NB);
                        weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                    })
                    .round()
                    .quantise::<i16>(255),
                SavedFormat::id("l0b").round().quantise::<i16>(255),
                SavedFormat::id("l1w").round().quantise::<i16>(64).transpose(),
                SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), InitSettings::Zeroed);
                let mut l0 = builder.new_affine("l0", 768 * NB, hidden);
                l0.weights = l0.weights + l0f.repeat(NB);
                let l1 = builder.new_affine("l1", if pairwise { hidden } else { 2 * hidden }, OUTPUT_BUCKETS);
                let (stm, ntm) = if pairwise {
                    let ft = |input, a, b| l0.slice(a, b).forward(input).crelu();
                    let half = hidden / 2;
                    (ft(stm_inputs, 0, half) * ft(stm_inputs, half, hidden), ft(ntm_inputs, 0, half) * ft(ntm_inputs, half, hidden))
                } else {
                    (l0.forward(stm_inputs).screlu(), l0.forward(ntm_inputs).screlu())
                };
                l1.forward(stm.concat(ntm)).select(output_buckets)
            });
        // The factoriser adds to each bucket's weights, so clip both tighter
        // to keep their sum within the quantised range.
        let clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
        trainer.optimiser.set_params_for_weight("l0w", clip);
        trainer.optimiser.set_params_for_weight("l0f", clip);
        trainer
    }};
}

/// King-bucketed FT with one hidden layer: FT (SCReLU) x 2 -> L1 neurons
/// (per output bucket, SCReLU) -> 8 material output buckets. Saved for the
/// engine's integer path: FT as now (i16, x255), L1 weights i8 at x64 (the
/// default +-1.98 clip keeps them within i8), L1 biases and the final layer f32.
macro_rules! build_kb_l1 {
    ($layout:expr, $nb:expr, $hidden:expr, $l1:expr) => {{
        let hidden: usize = $hidden;
        let l1n: usize = $l1;
        const NB: usize = $nb;
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs(ChessBucketsMirrored::new($layout))
            .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
            .save_format(&[
                SavedFormat::id("l0w")
                    .transform(|store, weights| {
                        let factoriser = store.get("l0f").values.f32().repeat(NB);
                        weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                    })
                    .round()
                    .quantise::<i16>(255),
                SavedFormat::id("l0b").round().quantise::<i16>(255),
                SavedFormat::id("l1w").round().quantise::<i8>(64).transpose(),
                SavedFormat::id("l1b"),
                SavedFormat::id("l2w").transpose(),
                SavedFormat::id("l2b"),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), InitSettings::Zeroed);
                let mut l0 = builder.new_affine("l0", 768 * NB, hidden);
                l0.weights = l0.weights + l0f.repeat(NB);
                let l1 = builder.new_affine("l1", 2 * hidden, OUTPUT_BUCKETS * l1n);
                let l2 = builder.new_affine("l2", l1n, OUTPUT_BUCKETS);
                let stm = l0.forward(stm_inputs).screlu();
                let ntm = l0.forward(ntm_inputs).screlu();
                let h = l1.forward(stm.concat(ntm)).select(output_buckets).screlu();
                l2.forward(h).select(output_buckets)
            });
        let clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
        trainer.optimiser.set_params_for_weight("l0w", clip);
        trainer.optimiser.set_params_for_weight("l0f", clip);
        trainer
    }};
}

#[derive(Clone, Copy, PartialEq)]
enum Inputs {
    Plain,
    Mirror,
    Kb8,
    Kb4,
}

/// Runs `$body` with `$t` bound to a trainer for the given input type.
macro_rules! with_trainer {
    ($inputs:expr, $hidden:expr, $pairwise:expr, |$t:ident| $body:block) => {
        match $inputs {
            Inputs::Mirror => {
                let mut $t = build!(ChessBucketsMirrored::new([0; 32]), $hidden, $pairwise);
                $body
            }
            Inputs::Plain => {
                let mut $t = build!(Chess768, $hidden, $pairwise);
                $body
            }
            Inputs::Kb8 => {
                let mut $t = build_kb!(KB8, KB8_BUCKETS, $hidden, $pairwise);
                $body
            }
            Inputs::Kb4 => {
                let mut $t = build_kb!(KB4, KB4_BUCKETS, $hidden, $pairwise);
                $body
            }
        }
    };
}

fn parse_activation(s: &str) -> bool {
    match s {
        "screlu" => false,
        "pairwise" => true,
        _ => panic!("expected screlu or pairwise, got {}", s),
    }
}

fn parse_inputs(s: &str) -> Inputs {
    match s {
        "mirror" => Inputs::Mirror,
        "plain" => Inputs::Plain,
        "kb8" => Inputs::Kb8,
        "kb4" => Inputs::Kb4,
        _ => panic!("expected mirror, plain, kb8 or kb4, got {}", s),
    }
}

/// Linear warm-up from peak/10 over the first `warmup` fraction of the run,
/// then cosine decay from `peak` to `floor` at the last superbatch.
#[derive(Clone, Debug)]
struct WarmupCosine {
    peak: f32,
    floor: f32,
    warmup: f32,
    superbatches: usize,
    batches_per_superbatch: usize,
}

impl lr::LrScheduler for WarmupCosine {
    fn lr(&self, batch: usize, superbatch: usize) -> f32 {
        let t = ((superbatch - 1) as f32 + batch as f32 / self.batches_per_superbatch as f32) / self.superbatches as f32;
        if t < self.warmup {
            return self.peak * (0.1 + 0.9 * t / self.warmup);
        }
        let u = if self.warmup < 1.0 { (t - self.warmup) / (1.0 - self.warmup) } else { 1.0 };
        self.floor + 0.5 * (self.peak - self.floor) * (1.0 + (std::f32::consts::PI * u.min(1.0)).cos())
    }

    fn colourful(&self) -> String {
        format!("warmup {:.0}% then cosine {} -> {}", self.warmup * 100.0, self.peak, self.floor)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: incipit-train <data_dir> <net_id> <out_dir> [superbatches=60] [hidden=512] [wdl=0.15] [mirror|plain|kb8|kb4] [screlu|pairwise]");
        eprintln!("       incipit-train eval <checkpoint_dir> <hidden> <mirror|plain|kb8|kb4> <screlu|pairwise> <fen>...");
        std::process::exit(1);
    }
    if args[1] == "eval" {
        let hidden: usize = args[3].parse().unwrap();
        let l1n: usize = args.iter().find_map(|a| a.strip_prefix("l1=").map(|v| v.parse().unwrap())).unwrap_or(0);
        let args: Vec<String> = args.iter().filter(|a| !a.starts_with("l1=")).cloned().collect();
        if l1n > 0 {
            assert!(parse_inputs(&args[4]) == Inputs::Kb4, "l1= needs kb4 inputs");
            let mut trainer = build_kb_l1!(KB4, KB4_BUCKETS, hidden, l1n);
            trainer.load_from_checkpoint(&args[2]);
            for fen in &args[6..] {
                println!("{:.1}\t{}", trainer.eval(fen) * 400.0, fen);
            }
            return;
        }
        with_trainer!(parse_inputs(&args[4]), hidden, parse_activation(&args[5]), |trainer| {
            trainer.load_from_checkpoint(&args[2]);
            for fen in &args[6..] {
                println!("{:.1}\t{}", trainer.eval(fen) * 400.0, fen);
            }
        });
        return;
    }
    // key=value options may follow the positional arguments.
    let opt = |k: &str, d: f32| -> f32 {
        args.iter()
            .find_map(|a| a.strip_prefix(k).and_then(|r| r.strip_prefix('=')).map(|v| v.parse().unwrap()))
            .unwrap_or(d)
    };
    let (peak_lr, floor_lr, warmup, skip) = (opt("lr", 0.001), opt("floor", 0.001 * 0.3f32.powi(5)), opt("warmup", 0.0), opt("skip", 0.0));
    let min_ply = opt("min_ply", 16.0) as u32;
    // One hidden layer of this many neurons after the FT (0 = none; kb4 only).
    let l1n = opt("l1", 0.0) as usize;
    let args: Vec<String> = args.iter().filter(|a| !a.contains('=')).cloned().collect();
    let (data_dir, net_id, out_dir) = (&args[1], &args[2], &args[3]);
    let superbatches: usize = args.get(4).map_or(60, |s| s.parse().unwrap());
    let hidden: usize = args.get(5).map_or(512, |s| s.parse().unwrap());
    // Weight on the game result (1 - lambda). The CPU-trainer generations found
    // lambda 0.85 (85% eval, 15% result) best; lambda 1.0 lost 27 Elo.
    let wdl_proportion: f32 = args.get(6).map_or(0.15, |s| s.parse().unwrap());
    let inputs = args.get(7).map_or(Inputs::Mirror, |s| parse_inputs(s));
    let pairwise = args.get(8).is_some_and(|s| parse_activation(s));

    let mut files: Vec<String> = std::fs::read_dir(data_dir)
        .expect("reading data dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "vf"))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .vf files in {}", data_dir);
    println!(
        "training {} on {} files from {}: hidden {}, {}, {}, {} superbatches, wdl {}",
        net_id,
        files.len(),
        data_dir,
        hidden,
        match inputs {
            Inputs::Mirror => "mirrored",
            Inputs::Plain => "plain",
            Inputs::Kb8 => "mirrored, 8 king buckets",
            Inputs::Kb4 => "mirrored, 4 king buckets (by rank)",
        },
        if pairwise { "pairwise CReLU" } else { "SCReLU" },
        superbatches,
        wdl_proportion
    );
    println!("lr {} -> {}, warmup {}, skip {}, min_ply {}, hidden layer {}", peak_lr, floor_lr, warmup, skip, min_ply, l1n);

    let schedule = TrainingSchedule {
        net_id: net_id.clone(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_proportion },
        lr_scheduler: WarmupCosine { peak: peak_lr, floor: floor_lr, warmup, superbatches, batches_per_superbatch: 6104 },
        save_rate: 10,
    };
    let settings = LocalSettings { threads: 8, test_set: None, output_directory: out_dir, batch_queue_size: 64 };
    let paths: Vec<&str> = files.iter().map(String::as_str).collect();
    let filter = Filter { min_ply, random_fen_skipping: skip > 0.0, random_fen_skip_probability: skip as f64, ..Filter::default() };
    let loader = ViriBinpackLoader::new_interleave_multiple(&paths, 1024, 8, ViriFilter::Builtin(filter));
    if l1n > 0 {
        assert!(inputs == Inputs::Kb4 && !pairwise, "l1= is only implemented for kb4 SCReLU nets");
        let mut trainer = build_kb_l1!(KB4, KB4_BUCKETS, hidden, l1n);
        trainer.run(&schedule, &settings, &loader);
        return;
    }
    with_trainer!(inputs, hidden, pairwise, |trainer| {
        trainer.run(&schedule, &settings, &loader);
    });
}
