use std::process::Stdio;

use ambient_std::asset_url::AssetType;
use ambient_world_audio::AudioNode;
use anyhow::Context;
use futures::FutureExt;
use tokio::io::{AsyncRead, AsyncReadExt};
use tracing::{info_span, Instrument};

use super::{
    context::PipelineCtx,
    out_asset::{asset_id_from_url, OutAsset, OutAssetContent, OutAssetPreview},
};

pub const SOUND_GRAPH_EXTENSION: &str = "sgr";

pub async fn pipeline(ctx: &PipelineCtx) -> Vec<OutAsset> {
    ctx.process_files(
        |file| matches!(file.extension().as_deref(), Some("ogg") | Some("wav") | Some("mp3")),
        |ctx, file| async move {
            let contents = file.download_bytes(ctx.assets()).await?;

            let filename = file.path().file_name().unwrap().to_string();

            let rel_path = ctx.in_root().relative_path(file.path());

            let content_url = match file.extension().as_deref() {
                Some("ogg") => ctx.write_file(&rel_path, contents).await,
                Some(ext @ "wav") | Some(ext @ "mp3") => {
                    tracing::info!("Processing {ext:?} file");
                    // Make sure to take the contents, to avoid having both the input and output in
                    // memory at once
                    let contents = symphonia_convert(ext, std::io::Cursor::new(contents)).await?;
                    ctx.write_file(rel_path.with_extension("ogg"), contents).await
                }
                other => anyhow::bail!("Audio filetype {:?} is not yet supported", other.unwrap_or_default()),
            };

            let root_node = AudioNode::Vorbis { url: content_url.to_string() };
            let graph_url = ctx.write_file(&rel_path.with_extension(SOUND_GRAPH_EXTENSION), save_audio_graph(root_node).unwrap()).await;

            Ok(vec![
                OutAsset {
                    id: asset_id_from_url(&file),
                    type_: AssetType::VorbisTrack,
                    hidden: false,
                    name: filename.clone(),
                    tags: Vec::new(),
                    categories: Default::default(),
                    preview: OutAssetPreview::None,
                    content: OutAssetContent::Content(content_url),
                    source: Some(file.clone()),
                },
                OutAsset {
                    id: asset_id_from_url(&file.push("graph").unwrap()),
                    type_: AssetType::SoundGraph,
                    hidden: false,
                    name: filename,
                    tags: Vec::new(),
                    categories: Default::default(),
                    preview: OutAssetPreview::None,
                    content: OutAssetContent::Content(graph_url),
                    source: None,
                },
            ])
        },
    )
    .instrument(info_span!("audio_pipeline"))
    .await
}

fn save_audio_graph(root: AudioNode) -> anyhow::Result<Vec<u8>> {
    Ok(serde_json::to_string_pretty(&root).context("Invalid sound graph")?.into_bytes())
}

#[tracing::instrument(level = "info", skip(input))]
async fn ffmpeg_convert<A>(input: A) -> anyhow::Result<Vec<u8>>
where
    A: 'static + Send + AsyncRead,
{
    let mut child = tokio::process::Command::new("ffmpeg")
        .args(["-i", "pipe:", "-f", "ogg", "pipe:1"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .context("Failed to execute ffmpeg")?;

    tracing::info!("Writing to stdin");

    let mut stdin = child.stdin.take().expect("no stdin");
    let mut stdout = child.stdout.take().expect("no stdout");

    let input = tokio::task::spawn(async move {
        tokio::pin!(input);
        tokio::io::copy(&mut input, &mut stdin).await.context("Failed to write to stdin")
    })
    .map(|v| v.unwrap());

    let output = async move {
        let mut output = Vec::new();
        stdout.read_to_end(&mut output).await.unwrap();
        Ok(output)
    };

    let status = async { child.wait().await.context("Failed to wait for ffmpeg") };

    tracing::info!("Waiting for ffmpeg to complete");
    let (_, output, status) = tokio::try_join!(input, output, status)?;

    if !status.success() {
        anyhow::bail!("FFMPEG conversion failed")
    }

    tracing::info!("Converted to vorbis of {} kb", output.len() as f32 / 1000.0);

    Ok(output)
}

#[tracing::instrument(level = "info", skip(input))]
async fn symphonia_convert<A>(ext: &str, input: A) -> anyhow::Result<Vec<u8>>
where
    A: 'static + Send + AsyncRead,
{
    use std::num::{NonZeroU32, NonZeroU8};

    use symphonia::core::{
        codecs::{DecoderOptions, CODEC_TYPE_NULL},
        errors::Error,
        formats::FormatOptions,
        io::MediaSourceStream,
        meta::MetadataOptions,
        probe::Hint,
    };

    use vorbis_rs::{VorbisBitrateManagementStrategy, VorbisEncoder};

    let mut hint = Hint::new();
    hint.with_extension(ext);

    let mut buf = Vec::new();
    let mut input = Box::pin(input);
    input.read_to_end(&mut buf).await.unwrap();
    let mss = MediaSourceStream::new(Box::new(std::io::Cursor::new(buf)), Default::default());
    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts).context("Failed to probe audio format")?;
    let mut format = probed.format;
    let track = format.tracks().iter().find(|t| t.codec_params.codec != CODEC_TYPE_NULL).context("Failed to select default audio track")?;
    let dec_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts).context("Failed to create audio decoder")?;

    let stream_serial: i32 = rand::random();

    let sampling_rate: NonZeroU32 = decoder
        .codec_params()
        .sample_rate
        .context("Expected audio to have sample rate")?
        .try_into()
        .context("Audio must have >0 sampling rate")?;

    let channels = decoder.codec_params().channels.context("Audio does not have any channels")?.bits();
    let channels: NonZeroU8 = (channels as u8).try_into().context("Audio must have >0 channels")?;

    let bitrate = VorbisBitrateManagementStrategy::QualityVbr { target_quality: 0.9 };

    let mut encoder = VorbisEncoder::new(
        stream_serial,
        [("", ""); 0], // no tags
        sampling_rate,
        channels,
        bitrate,
        None,
        Vec::new(),
    )?;

    let result = loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(err) => break err,
        };

        let decoded = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(err) => break err,
        };

        let mut block = decoded.make_equivalent::<f32>();
        decoded.convert(&mut block);
        let planes = block.planes();
        let plane_samples = planes.planes();
        encoder.encode_audio_block(plane_samples)?;
    };

    match result {
        // "end of stream" is non-fatal, ignore it
        Error::IoError(err) if err.kind() == std::io::ErrorKind::UnexpectedEof && err.to_string() == "end of stream" => {}
        // return every other kind of error
        err => return Err(err.into()),
    }

    let output = encoder.finish()?;
    tracing::info!("Decoded {} samples", output.len());
    Ok(output)
}
