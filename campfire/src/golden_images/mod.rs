// Intentionally overrides the global Clippy settings because Campfire is not
// part of Ambient engine.
#![allow(clippy::disallowed_types)]

use anyhow::{bail, Context};
use clap::Parser;
use itertools::Itertools;
use serde::Deserialize;
use std::{path::PathBuf, process::Command, time::Instant};

mod failure;
use failure::*;

mod progress;
use progress::*;

const TEST_BASE_PATH: &str = "guest/rust/examples";
const TEST_MANIFEST: &str = "golden-image-manifest.toml";

#[derive(Parser, Clone)]
pub struct GoldenImages {
    /// Only run tests which start with the specified prefix
    #[arg(long)]
    prefix: Option<String>,

    /// Selects testing mode
    #[command(subcommand)]
    mode: Mode,
}

#[derive(Parser, Clone)]
enum Mode {
    /// For each test, updates the golden image
    Update,
    /// For each test, check the current image against the committed image
    Check,
}

const TEST_NAME_PLACEHOLDER: &str = "{test}";
const QUIC_INTERFACE_PORT_PLACEHOLDER: &str = "{quic-port}";
const HTTP_INTERFACE_PORT_PLACEHOLDER: &str = "{http-port}";

pub(crate) fn main(gi: &GoldenImages) -> anyhow::Result<()> {
    let start_time = Instant::now();

    // Get tests.
    let tests = parse_tests_from_manifest()?;

    // Filter tests.
    let tests = if let Some(prefix) = &gi.prefix {
        let total_test_count = tests.len();
        let filtered_tests = tests
            .into_iter()
            .filter(|test| test.starts_with(prefix))
            .collect_vec();
        log::info!(
            "--prefix {prefix} resulted in {} out of {total_test_count} tests",
            filtered_tests.len(),
        );
        filtered_tests
    } else {
        tests
    };
    if tests.is_empty() {
        bail!("Nothing to do!");
    }

    // Build tests.
    run(
        "Building",
        &[
            "run",
            "--release",
            "--",
            "build",
            "--release",
            TEST_NAME_PLACEHOLDER,
        ],
        &tests,
        true,
    )?;

    match gi.mode {
        Mode::Update => {
            run(
                "Updating",
                &[
                    "run",
                    "--release",
                    "--",
                    "run",
                    "--release",
                    TEST_NAME_PLACEHOLDER,
                    "--headless",
                    "--no-proxy",
                    "--quic-interface-port",
                    QUIC_INTERFACE_PORT_PLACEHOLDER,
                    "--http-interface-port",
                    HTTP_INTERFACE_PORT_PLACEHOLDER,
                    "golden-image-update",
                    // Todo: Ideally this waiting should be unnecessary, because
                    // we only care about rendering the first frame of the test,
                    // no matter how long it takes to start the test. Being able
                    // to stall the renderer before everything has been loaded
                    // eliminates the need for timeouts and reduces test
                    // flakiness.
                    "--wait-seconds",
                    "5.0",
                ],
                &tests,
                true,
            )?;
        }
        Mode::Check => {
            run(
                "Checking",
                &[
                    "run",
                    "--release",
                    "--",
                    "run",
                    "--release",
                    TEST_NAME_PLACEHOLDER,
                    "--headless",
                    "--no-proxy",
                    "--quic-interface-port",
                    QUIC_INTERFACE_PORT_PLACEHOLDER,
                    "--http-interface-port",
                    HTTP_INTERFACE_PORT_PLACEHOLDER,
                    "golden-image-check",
                    // Todo: See notes on --wait-seconds from above.
                    "--timeout-seconds",
                    "30.0",
                ],
                &tests,
                true,
            )
            .context(
                "Checking failed, possible causes:\n \
                - Missing golden image: consider running `cargo cf golden-images update` first.\n \
                - Golden image differs: investigate if the difference was intentional.\n",
            )?;
        }
    }

    log::info!(
        "Running {} golden image tests took {:.03} seconds",
        tests.len(),
        start_time.elapsed().as_secs_f64()
    );

    Ok(())
}

#[derive(Deserialize)]
struct Manifest {
    tests: Vec<String>,
}

fn parse_tests_from_manifest() -> anyhow::Result<Vec<String>> {
    let manifest_path = PathBuf::from(TEST_BASE_PATH).join(TEST_MANIFEST);
    let manifest = std::fs::read_to_string(&manifest_path)?;
    let manifest: Manifest = toml::from_str(&manifest)?;
    log::info!(
        "Read manifest from '{}', parsed {} tests",
        manifest_path.display(),
        manifest.tests.len()
    );
    Ok(manifest.tests)
}

fn run(command: &str, args: &[&str], tests: &[String], parallel: bool) -> anyhow::Result<()> {
    use rayon::prelude::*;

    // Run.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(if parallel { 0 } else { 1 })
        .build()?;
    let outputs = pool.install(|| {
        let pb = Progress::new(tests.len());
        pb.println(format!(
            "{command} {} tests across {} CPUs",
            tests.len(),
            if parallel { num_cpus::get() } else { 1 }
        ));
        let mut outputs = vec![];
        tests
            .into_par_iter()
            .enumerate()
            .map(|(test_index, test)| {
                let start_time = Instant::now();
                let args = args
                    .iter()
                    .map(|&arg| {
                        let arg = match arg {
                            TEST_NAME_PLACEHOLDER => format!("{TEST_BASE_PATH}/{test}"),
                            QUIC_INTERFACE_PORT_PLACEHOLDER => format!("{}", 9000 + test_index),
                            HTTP_INTERFACE_PORT_PLACEHOLDER => format!("{}", 10000 + test_index),
                            _ => arg.to_string(),
                        };
                        if arg == TEST_NAME_PLACEHOLDER {
                            format!("{TEST_BASE_PATH}/{test}")
                        } else {
                            arg.to_string()
                        }
                    })
                    .collect_vec();
                let output = Command::new("cargo").args(&args).output().unwrap();
                pb.println_and_inc(format!(
                    "{} | {:.03}s | cargo {}",
                    status_emoji(output.status.success()),
                    start_time.elapsed().as_secs_f64(),
                    args.join(" "),
                ));
                (test.to_string(), output)
            })
            .collect_into_vec(&mut outputs);
        pb.finish();
        outputs
    });

    // Collect failures.
    let failures = outputs
        .into_iter()
        .filter_map(|(test, output)| {
            if output.status.success() {
                None
            } else {
                Some(Failure::from_output(test, &output))
            }
        })
        .collect_vec();
    if !failures.is_empty() {
        for failure in &failures {
            failure.log();
        }
        bail!("{} tests failed", failures.len());
    }

    Ok(())
}

fn status_emoji(status: bool) -> char {
    if status {
        '✅'
    } else {
        '❌'
    }
}
