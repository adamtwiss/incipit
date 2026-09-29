// Trains an Incipit network with Bullet from viriformat data.
//
//   incipit-train <data_dir> <net_id> <out_dir> [superbatches] [hidden] [wdl] [mirror|plain]
//   incipit-train eval <checkpoint_dir> <hidden> <mirror|plain> <fen>...
//
// Reads every .vf file in <data_dir> (symlinks are fine), interleaving them so
// each batch mixes datasets. Architecture: 768 piece-square inputs, horizontally
// mirrored (one king bucket) or plain, SCReLU, 8 material output buckets,
// quantised 255/64 with eval scale 400. Convert the result with:
//   datatools net bullet <out_dir>/<net_id>-<N>/quantised.bin <dir> --hidden <H> [--mirror]
//
// `eval` prints Bullet's own evaluation (centipawns, side to move) of each FEN
// from a checkpoint, to compare against the engine with the converted net.
use bullet_lib::{
    game::{
        inputs::{Chess768, ChessBucketsMirrored},
        outputs::MaterialCount,
    },
    nn::optimiser::AdamW,
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

/// The network: inputs -> hidden (SCReLU) x 2 -> 8 material output buckets.
/// (Macros because the trainer's concrete type is awkward to name.)
macro_rules! build {
    ($inputs:expr, $hidden:expr) => {{
        let hidden: usize = $hidden;
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
                let l1 = builder.new_affine("l1", 2 * hidden, OUTPUT_BUCKETS);
                let stm = l0.forward(stm_inputs).screlu();
                let ntm = l0.forward(ntm_inputs).screlu();
                l1.forward(stm.concat(ntm)).select(output_buckets)
            })
    }};
}

/// Runs `$body` with `$t` bound to a trainer for the given input type.
macro_rules! with_trainer {
    ($mirror:expr, $hidden:expr, |$t:ident| $body:block) => {
        if $mirror {
            let mut $t = build!(ChessBucketsMirrored::new([0; 32]), $hidden);
            $body
        } else {
            let mut $t = build!(Chess768, $hidden);
            $body
        }
    };
}

fn parse_mirror(s: &str) -> bool {
    match s {
        "mirror" => true,
        "plain" => false,
        _ => panic!("expected mirror or plain, got {}", s),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: incipit-train <data_dir> <net_id> <out_dir> [superbatches=60] [hidden=512] [wdl=0.15] [mirror|plain]");
        eprintln!("       incipit-train eval <checkpoint_dir> <hidden> <mirror|plain> <fen>...");
        std::process::exit(1);
    }
    if args[1] == "eval" {
        let hidden: usize = args[3].parse().unwrap();
        with_trainer!(parse_mirror(&args[4]), hidden, |trainer| {
            trainer.load_from_checkpoint(&args[2]);
            for fen in &args[5..] {
                println!("{:.1}\t{}", trainer.eval(fen) * 400.0, fen);
            }
        });
        return;
    }
    let (data_dir, net_id, out_dir) = (&args[1], &args[2], &args[3]);
    let superbatches: usize = args.get(4).map_or(60, |s| s.parse().unwrap());
    let hidden: usize = args.get(5).map_or(512, |s| s.parse().unwrap());
    // Weight on the game result (1 - lambda). The CPU-trainer generations found
    // lambda 0.85 (85% eval, 15% result) best; lambda 1.0 lost 27 Elo.
    let wdl_proportion: f32 = args.get(6).map_or(0.15, |s| s.parse().unwrap());
    let mirror = args.get(7).map_or(true, |s| parse_mirror(s));

    let mut files: Vec<String> = std::fs::read_dir(data_dir)
        .expect("reading data dir")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "vf"))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no .vf files in {}", data_dir);
    println!(
        "training {} on {} files from {}: hidden {}, {}, {} superbatches, wdl {}",
        net_id,
        files.len(),
        data_dir,
        hidden,
        if mirror { "mirrored" } else { "plain" },
        superbatches,
        wdl_proportion
    );

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
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.001 * 0.3f32.powi(5),
            final_superbatch: superbatches,
        },
        save_rate: 10,
    };
    let settings = LocalSettings { threads: 8, test_set: None, output_directory: out_dir, batch_queue_size: 64 };
    let paths: Vec<&str> = files.iter().map(String::as_str).collect();
    let loader = ViriBinpackLoader::new_interleave_multiple(&paths, 1024, 8, ViriFilter::Builtin(Filter::default()));
    with_trainer!(mirror, hidden, |trainer| {
        trainer.run(&schedule, &settings, &loader);
    });
}
