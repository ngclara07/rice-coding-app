from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rice_codec import (
    PREDICTION_FIRST_ORDER,
    PREDICTION_RAW,
    PREDICTION_SECOND_ORDER,

    STRATEGY_HYBRID_V1,
    STRATEGY_HYBRID_V2,
    STRATEGY_OFFICIAL,
    STRATEGY_RAW,
    STRATEGY_SECOND_ORDER,

    WavInfo,

    build_coursework_table,
    build_lossless_table,
    compress_wav,
    decode_ex2_to_wav,
    estimate_ex2_size_bytes,
    human_bytes,
    inspect_ex2,
    prediction_encode,
    prediction_name,
    read_wav,
    residual_statistics,
    resolve_prediction_mode,
    results_dataframe,
    rice_code_details,
    save_run_configuration,
)


# ============================================================
# PATHS
# ============================================================


BASE_DIR = Path(
    __file__
).resolve().parent

AUDIO_DIR = (
    BASE_DIR
    / "audio"
)

OUTPUTS_DIR = (
    BASE_DIR
    / "outputs"
)

TEMP_DIR = (
    BASE_DIR
    / "temp"
)

ASSETS_DIR = (
    BASE_DIR
    / "assets"
)

for directory in [
    AUDIO_DIR,
    OUTPUTS_DIR,
    TEMP_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PAGE CONFIG
# ============================================================


st.set_page_config(
    page_title=(
        "Signal Coding Lab | "
        "Exercise 2"
    ),
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================


css_path = (
    ASSETS_DIR
    / "styles.css"
)

if css_path.is_file():

    st.markdown(
        "<style>"
        + css_path.read_text(
            encoding="utf-8"
        )
        + "</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# CONSTANTS / LABELS
# ============================================================


STRATEGY_LABELS = {
    (
        "Official Coursework "
        "— First-order Delta"
    ):
        STRATEGY_OFFICIAL,

    (
        "Appendix "
        "— Second-order Delta"
    ):
        STRATEGY_SECOND_ORDER,

    (
        "Appendix "
        "— Raw Samples"
    ):
        STRATEGY_RAW,

    (
        "Appendix "
        "— Hybrid V1"
    ):
        STRATEGY_HYBRID_V1,

    (
        "Appendix "
        "— Hybrid V2"
    ):
        STRATEGY_HYBRID_V2,
}


PREDICTION_LABELS = {
    (
        "First-order Delta"
    ):
        PREDICTION_FIRST_ORDER,

    (
        "Second-order Delta"
    ):
        PREDICTION_SECOND_ORDER,

    "Raw Samples":
        PREDICTION_RAW,
}


# ============================================================
# HELPERS
# ============================================================


def discover_audio_files() -> list[
    Path
]:

    return sorted(
        [
            path
            for path
            in AUDIO_DIR.glob(
                "*.wav"
            )
            if path.is_file()
        ]
    )


def save_uploaded_wavs(
    uploaded_files,
) -> list[
    Path
]:

    paths = []

    for uploaded in (
        uploaded_files
        or []
    ):

        safe_name = (
            Path(
                uploaded.name
            ).name
        )

        path = (
            TEMP_DIR
            / safe_name
        )

        path.write_bytes(
            uploaded.getvalue()
        )

        paths.append(
            path
        )

    return paths


def save_uploaded_file(
    uploaded_file,
    prefix: str = "",
) -> Path:

    safe_name = (
        Path(
            uploaded_file.name
        ).name
    )

    path = (
        TEMP_DIR
        / (
            prefix
            + safe_name
        )
    )

    path.write_bytes(
        uploaded_file.getvalue()
    )

    return path


def format_seconds(
    seconds: float,
) -> str:

    if seconds < 60:

        return (
            f"{seconds:.2f} s"
        )

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds % 60
    )

    return (
        f"{minutes}m "
        f"{remaining:.1f}s"
    )


def copy_inputs_to_run(
    input_paths: list[
        Path
    ],
    input_directory: Path,
) -> list[
    Path
]:

    input_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = []

    used_names = set()

    for index, source in enumerate(
        input_paths,
        start=1,
    ):

        filename = (
            source.name
        )

        if filename in used_names:

            filename = (
                f"{source.stem}"
                f"_{index}"
                f"{source.suffix}"
            )

        used_names.add(
            filename
        )

        destination = (
            input_directory
            / filename
        )

        shutil.copy2(
            source,
            destination,
        )

        copied.append(
            destination
        )

    return copied


def preflight_dataframe(
    paths: list[
        Path
    ],
    k_values: list[
        int
    ],
    strategy: str,
) -> pd.DataFrame:

    rows = []

    for path in paths:

        try:

            samples, info = read_wav(
                path
            )

            prediction_mode = (
                resolve_prediction_mode(
                    strategy,
                    path.name,
                )
            )

            encoded_values = (
                prediction_encode(
                    samples,
                    prediction_mode,
                )
            )

            for k in k_values:

                bit_count, ex2_bytes = (
                    estimate_ex2_size_bytes(
                        encoded_values,
                        k,
                    )
                )

                original_size = (
                    path.stat().st_size
                )

                percentage = (
                    (
                        original_size
                        - ex2_bytes
                    )
                    / original_size
                    * 100
                )

                rows.append(
                    {
                        "File":
                            path.name,

                        "K":
                            k,

                        "Prediction":
                            prediction_name(
                                prediction_mode
                            ),

                        "Samples":
                            len(
                                samples
                            ),

                        "Duration":
                            (
                                f"{info.duration_seconds:.2f}s"
                            ),

                        "Original":
                            human_bytes(
                                original_size
                            ),

                        "Estimated .ex2":
                            human_bytes(
                                ex2_bytes
                            ),

                        "Estimated Compression":
                            f"{percentage:.2f}%",

                        "Estimated Bits":
                            f"{bit_count:,}",
                    }
                )

        except Exception as error:

            rows.append(
                {
                    "File":
                        path.name,

                    "K":
                        "—",

                    "Prediction":
                        "Invalid WAV",

                    "Samples":
                        "—",

                    "Duration":
                        "—",

                    "Original":
                        "—",

                    "Estimated .ex2":
                        "—",

                    "Estimated Compression":
                        "—",

                    "Estimated Bits":
                        str(
                            error
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def create_waveform_dataframe(
    path: Path,
    prediction_mode: str,
    sample_limit: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    samples, info = read_wav(
        path
    )

    values = prediction_encode(
        samples,
        prediction_mode,
    )

    count = min(
        sample_limit,
        len(samples),
    )

    time_axis = (
        np.arange(
            count
        )
        / info.frame_rate
    )

    waveform_df = pd.DataFrame(
        {
            "Time (s)":
                time_axis,

            "Amplitude":
                samples[
                    :count
                ],
        }
    ).set_index(
        "Time (s)"
    )

    residual_df = pd.DataFrame(
        {
            "Time (s)":
                time_axis,

            "Residual":
                values[
                    :count
                ],
        }
    ).set_index(
        "Time (s)"
    )

    return (
        waveform_df,
        residual_df,
    )


def safe_download(
    label: str,
    path: Path,
    mime: str,
    key: str,
    maximum_inline_mb: float = 100.0,
):

    if not path.is_file():
        return

    size_mb = (
        path.stat().st_size
        / (
            1024
            * 1024
        )
    )

    if size_mb > (
        maximum_inline_mb
    ):

        st.warning(
            (
                f"{path.name} is "
                f"{size_mb:.1f} MB. "
                "It has been saved locally at:"
            )
        )

        st.code(
            str(
                path.resolve()
            )
        )

        return

    with open(
        path,
        "rb",
    ) as file:

        st.download_button(
            label=label,
            data=file.read(),
            file_name=(
                path.name
            ),
            mime=mime,
            key=key,
            use_container_width=True,
        )


def show_run_summary(
    results_df: pd.DataFrame,
):

    if results_df.empty:
        return

    successful = int(
        results_df[
            "lossless_verified"
        ].sum()
    )

    best_index = (
        results_df[
            "percent_compression"
        ].idxmax()
    )

    best = results_df.loc[
        best_index
    ]

    average_time = float(
        results_df[
            "processing_time_seconds"
        ].mean()
    )

    total_encoded = int(
        results_df[
            "compressed_size_bytes"
        ].sum()
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Lossless Tests",
        (
            f"{successful}"
            f" / "
            f"{len(results_df)}"
        ),
    )

    c2.metric(
        "Best Compression",
        (
            f"{best['percent_compression']:.2f}%"
        ),
        (
            f"{best['file']} "
            f"K={int(best['K'])}"
        ),
    )

    c3.metric(
        "Average Runtime",
        format_seconds(
            average_time
        ),
    )

    c4.metric(
        "Total .ex2 Output",
        human_bytes(
            total_encoded
        ),
    )


# ============================================================
# HEADER
# ============================================================


header_left, header_right = (
    st.columns(
        [
            5,
            1.2,
        ]
    )
)

with header_left:

    st.markdown(
        """
        <div class="eyebrow">
            CM3065 · INTELLIGENT SIGNAL PROCESSING
        </div>

        <div class="main-title">
            Signal Coding Lab
        </div>

        <div class="main-subtitle">
            Exercise 2 · Lossless Audio Compression
            using Predictive Rice Coding
        </div>
        """,
        unsafe_allow_html=True,
    )


with header_right:

    st.markdown(
        """
        <div class="system-status">
            <span class="status-dot"></span>
            CODEC READY
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PIPELINE STRIP
# ============================================================


st.markdown(
    """
    <div class="pipeline-strip">

        <span>16-BIT WAV</span>
        <b>→</b>

        <span>Δ PREDICTOR</span>
        <b>→</b>

        <span>SIGNED MAP</span>
        <b>→</b>

        <span>RICE q + r</span>
        <b>→</b>

        <span>.EX2</span>
        <b>→</b>

        <span>RICE DECODE</span>
        <b>→</b>

        <span>Δ⁻¹</span>
        <b>→</b>

        <span>LOSSLESS WAV</span>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.markdown(
        "## Codec Control"
    )

    st.caption(
        (
            "Exercise 2 · "
            "Rice Encoder / Decoder"
        )
    )

    # --------------------------------------------------------
    # AUDIO SOURCE
    # --------------------------------------------------------

    st.markdown(
        "### Audio Source"
    )

    bundled_files = (
        discover_audio_files()
    )

    source_mode = (
        st.radio(
            "Input source",
            [
                "Coursework audio folder",
                "Upload WAV file(s)",
            ],
        )
    )

    selected_paths = []

    if (
        source_mode
        == "Coursework audio folder"
    ):

        if bundled_files:

            selected_names = (
                st.multiselect(
                    "Select WAV files",
                    [
                        path.name
                        for path
                        in bundled_files
                    ],
                    default=[
                        path.name
                        for path
                        in bundled_files
                    ],
                )
            )

            selected_paths = [
                path
                for path
                in bundled_files
                if path.name
                in selected_names
            ]

        else:

            st.warning(
                (
                    "Place Sound1.wav and "
                    "Sound2.wav inside audio/."
                )
            )

    else:

        uploaded_wavs = (
            st.file_uploader(
                "Upload 16-bit mono WAV",
                type=[
                    "wav"
                ],
                accept_multiple_files=True,
            )
        )

        selected_paths = (
            save_uploaded_wavs(
                uploaded_wavs
            )
        )

    st.divider()

    # --------------------------------------------------------
    # RICE PARAMETER
    # --------------------------------------------------------

    st.markdown(
        "### Rice Parameter"
    )

    k_values = (
        st.multiselect(
            "K values",
            options=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
            ],
            default=[
                2,
                4,
            ],
        )
    )

    st.caption(
        (
            "Coursework comparison: "
            "K = 2 and K = 4."
        )
    )

    # --------------------------------------------------------
    # PREDICTOR
    # --------------------------------------------------------

    st.markdown(
        "### Prediction Strategy"
    )

    strategy_label = (
        st.selectbox(
            "Method",
            list(
                STRATEGY_LABELS.keys()
            ),
            index=0,
        )
    )

    strategy = (
        STRATEGY_LABELS[
            strategy_label
        ]
    )

    if strategy == (
        STRATEGY_OFFICIAL
    ):

        st.success(
            (
                "Official Exercise 2 mode: "
                "first-order delta + Rice."
            )
        )

    else:

        st.warning(
            (
                "Appendix experiment mode. "
                "Do not use this result as the "
                "official coursework comparison table."
            )
        )

    # --------------------------------------------------------
    # VISUAL SETTINGS
    # --------------------------------------------------------

    with st.expander(
        "Visualisation"
    ):

        waveform_samples = (
            st.slider(
                "Waveform preview samples",
                min_value=250,
                max_value=20000,
                value=4000,
                step=250,
            )
        )

        inspector_rows = (
            st.slider(
                "Bit inspector rows",
                min_value=5,
                max_value=50,
                value=20,
            )
        )

    st.divider()

    run_button = (
        st.button(
            "▶  RUN RICE ENCODE + DECODE",
            type="primary",
            use_container_width=True,
        )
    )

    if st.button(
        "Clear previous results",
        use_container_width=True,
    ):

        st.session_state.pop(
            "exercise_2_last_run",
            None,
        )


# ============================================================
# TABS
# ============================================================


(
    codec_tab,
    signal_tab,
    results_tab,
    inspector_tab,
    decode_tab,
    config_tab,
) = st.tabs(
    [
        "🎵 Codec Lab",
        "〰 Signal & Residuals",
        "📊 Compression Analytics",
        "🧮 Rice Bit Inspector",
        "📦 Decode .EX2",
        "⚙ Configuration",
    ]
)


# ============================================================
# CODEC LAB
# ============================================================


with codec_tab:

    st.markdown(
        "### Encode → Decode → Verify"
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    files_metric = (
        m1.empty()
    )

    tasks_metric = (
        m2.empty()
    )

    status_metric = (
        m3.empty()
    )

    output_metric = (
        m4.empty()
    )

    files_metric.metric(
        "Selected WAV Files",
        len(
            selected_paths
        ),
    )

    tasks_metric.metric(
        "Rice Tasks",
        (
            len(
                selected_paths
            )
            * len(
                k_values
            )
        ),
    )

    status_metric.metric(
        "Codec Status",
        "Ready",
    )

    output_metric.metric(
        "Format",
        ".EX2",
    )

    st.markdown(
        "#### Preflight size analysis"
    )

    if (
        selected_paths
        and k_values
    ):

        preflight_df = (
            preflight_dataframe(
                selected_paths,
                k_values,
                strategy,
            )
        )

        st.dataframe(
            preflight_df,
            hide_index=True,
            use_container_width=True,
        )

        large_rows = (
            preflight_df[
                preflight_df[
                    "Estimated .ex2"
                ].str.contains(
                    "MB|GB",
                    regex=True,
                    na=False,
                )
            ]
        )

        if not large_rows.empty:

            st.warning(
                (
                    "Some Rice configurations can generate "
                    "very large files. This is expected when "
                    "the residual distribution is poorly "
                    "matched to the selected K."
                )
            )

    else:

        st.info(
            (
                "Select at least one WAV file "
                "and one Rice K value."
            )
        )

    status_placeholder = (
        st.empty()
    )

    progress_placeholder = (
        st.empty()
    )

    live_task_placeholder = (
        st.empty()
    )


# ============================================================
# SIGNAL TAB
# ============================================================


with signal_tab:

    st.markdown(
        "### Predictive Signal Analysis"
    )

    if selected_paths:

        preview_path = (
            st.selectbox(
                "Preview audio",
                selected_paths,
                format_func=lambda path:
                    Path(path).name,
                key="signal_preview_file",
            )
        )

        preview_mode = (
            resolve_prediction_mode(
                strategy,
                Path(
                    preview_path
                ).name,
            )
        )

        try:

            samples, info = read_wav(
                preview_path
            )

            residuals = prediction_encode(
                samples,
                preview_mode,
            )

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "Sample Rate",
                (
                    f"{info.frame_rate:,} Hz"
                ),
            )

            c2.metric(
                "Samples",
                f"{len(samples):,}",
            )

            c3.metric(
                "Duration",
                (
                    f"{info.duration_seconds:.3f}s"
                ),
            )

            c4.metric(
                "Prediction",
                prediction_name(
                    preview_mode
                ),
            )

            st.audio(
                str(
                    preview_path
                )
            )

            waveform_df, residual_preview_df = (
                create_waveform_dataframe(
                    Path(
                        preview_path
                    ),
                    preview_mode,
                    waveform_samples,
                )
            )

            left_plot, right_plot = (
                st.columns(2)
            )

            with left_plot:

                st.markdown(
                    "#### Original waveform"
                )

                st.line_chart(
                    waveform_df,
                    use_container_width=True,
                )

            with right_plot:

                st.markdown(
                    "#### Prediction residuals"
                )

                st.line_chart(
                    residual_preview_df,
                    use_container_width=True,
                )

            stats = (
                residual_statistics(
                    preview_path,
                    preview_mode,
                )
            )

            st.markdown(
                "#### Residual statistics"
            )

            s1, s2, s3, s4 = (
                st.columns(4)
            )

            s1.metric(
                "Mean |Residual|",
                stats[
                    "mean_abs_residual"
                ],
            )

            s2.metric(
                "Median |Residual|",
                stats[
                    "median_abs_residual"
                ],
            )

            s3.metric(
                "Maximum |Residual|",
                (
                    f"{stats['max_abs_residual']:,}"
                ),
            )

            s4.metric(
                "Zero Residuals",
                (
                    f"{stats['zero_residual_percent']:.2f}%"
                ),
            )

            histogram_counts, edges = (
                np.histogram(
                    residuals,
                    bins=100,
                )
            )

            centres = (
                edges[:-1]
                + edges[1:]
            ) / 2

            histogram_df = (
                pd.DataFrame(
                    {
                        "Residual":
                            centres,

                        "Frequency":
                            histogram_counts,
                    }
                )
                .set_index(
                    "Residual"
                )
            )

            st.markdown(
                "#### Residual distribution"
            )

            st.bar_chart(
                histogram_df,
                use_container_width=True,
            )

        except Exception as error:

            st.exception(
                error
            )

    else:

        st.info(
            (
                "Select a WAV file to inspect "
                "its waveform and residual distribution."
            )
        )


# ============================================================
# RESULTS TAB PLACEHOLDER
# ============================================================


with results_tab:

    results_container = (
        st.container()
    )


# ============================================================
# BIT INSPECTOR
# ============================================================


with inspector_tab:

    st.markdown(
        "### Rice Codeword Inspector"
    )

    st.caption(
        (
            "Inspect how residual values are mapped into "
            "Rice quotient and remainder components."
        )
    )

    if selected_paths:

        inspector_path = (
            st.selectbox(
                "Audio file",
                selected_paths,
                format_func=lambda path:
                    Path(path).name,
                key="inspector_audio",
            )
        )

        inspector_k = (
            st.selectbox(
                "Rice K",
                (
                    k_values
                    if k_values
                    else [
                        2,
                        4,
                    ]
                ),
                key="inspector_k",
            )
        )

        mode = (
            resolve_prediction_mode(
                strategy,
                Path(
                    inspector_path
                ).name,
            )
        )

        try:

            samples, _ = read_wav(
                inspector_path
            )

            values = prediction_encode(
                samples,
                mode,
            )

            detail_df = (
                rice_code_details(
                    values,
                    inspector_k,
                    maximum_rows=(
                        inspector_rows
                    ),
                )
            )

            st.dataframe(
                detail_df,
                hide_index=True,
                use_container_width=True,
            )

            if not detail_df.empty:

                first = (
                    detail_df.iloc[
                        0
                    ]
                )

                st.markdown(
                    "#### First codeword"
                )

                st.code(
                    (
                        f"signed value = "
                        f"{first['Signed value']}\n"
                        f"mapped value = "
                        f"{first['Mapped value']}\n"
                        f"q = "
                        f"{first['q']}\n"
                        f"r = "
                        f"{first['r']}\n"
                        f"Rice code = "
                        f"{first['Rice codeword']}"
                    )
                )

        except Exception as error:

            st.exception(
                error
            )

    else:

        st.info(
            "Select an audio file first."
        )


# ============================================================
# STANDALONE .EX2 DECODER
# ============================================================


with decode_tab:

    st.markdown(
        "### Decode Existing `.ex2` File"
    )

    st.info(
        (
            "The coursework .ex2 format stores the valid "
            "Rice bit length but does not store K, the "
            "prediction method, or WAV metadata. These "
            "must therefore be supplied when decoding "
            "an existing .ex2 file."
        )
    )

    ex2_upload = (
        st.file_uploader(
            "Upload .ex2 file",
            type=[
                "ex2"
            ],
            key="ex2_decoder_upload",
        )
    )

    decoder_k = (
        st.number_input(
            "Rice K",
            min_value=0,
            max_value=20,
            value=4,
            step=1,
        )
    )

    decoder_prediction_label = (
        st.selectbox(
            "Prediction used during encoding",
            list(
                PREDICTION_LABELS.keys()
            ),
        )
    )

    decoder_prediction = (
        PREDICTION_LABELS[
            decoder_prediction_label
        ]
    )

    reference_wav = (
        st.file_uploader(
            (
                "Optional reference WAV "
                "(for metadata + lossless verification)"
            ),
            type=[
                "wav"
            ],
            key="reference_wav",
        )
    )

    if reference_wav is None:

        st.markdown(
            "#### WAV output metadata"
        )

        manual_sample_rate = (
            st.number_input(
                "Sample rate (Hz)",
                min_value=8000,
                max_value=192000,
                value=44100,
                step=100,
            )
        )

        manual_channels = 1

        st.caption(
            (
                "Exercise 2 output is "
                "16-bit mono PCM."
            )
        )

    decode_button = (
        st.button(
            "Decode .EX2 → WAV",
            use_container_width=True,
            key="decode_ex2_button",
        )
    )

    decoder_progress = (
        st.empty()
    )

    decoder_status = (
        st.empty()
    )

    if (
        decode_button
        and ex2_upload is not None
    ):

        try:

            ex2_path = (
                save_uploaded_file(
                    ex2_upload,
                    prefix="decode_",
                )
            )

            header_info = (
                inspect_ex2(
                    ex2_path
                )
            )

            st.json(
                header_info
            )

            expected_sample_count = (
                None
            )

            reference_samples = None

            if (
                reference_wav
                is not None
            ):

                reference_path = (
                    save_uploaded_file(
                        reference_wav,
                        prefix="reference_",
                    )
                )

                (
                    reference_samples,
                    wav_info,
                ) = read_wav(
                    reference_path
                )

                expected_sample_count = (
                    len(
                        reference_samples
                    )
                )

            else:

                wav_info = WavInfo(
                    channels=(
                        manual_channels
                    ),
                    sample_width=2,
                    frame_rate=int(
                        manual_sample_rate
                    ),
                    frame_count=0,
                    compression_type=(
                        "NONE"
                    ),
                    compression_name=(
                        "not compressed"
                    ),
                )

            decoded_output_path = (
                TEMP_DIR
                / (
                    f"{ex2_path.stem}"
                    "_decoded.wav"
                )
            )

            progress_bar = (
                decoder_progress.progress(
                    0.0
                )
            )

            def decode_progress(
                stage,
                current,
                total,
            ):

                fraction = (
                    current
                    / max(
                        total,
                        1,
                    )
                )

                progress_bar.progress(
                    min(
                        1.0,
                        fraction,
                    )
                )

                decoder_status.info(
                    (
                        f"Decoding: "
                        f"{fraction * 100:.1f}%"
                    )
                )

            decoded_samples = (
                decode_ex2_to_wav(
                    ex2_path=(
                        ex2_path
                    ),
                    k=int(
                        decoder_k
                    ),
                    output_path=(
                        decoded_output_path
                    ),
                    wav_info=(
                        wav_info
                    ),
                    prediction_mode=(
                        decoder_prediction
                    ),
                    expected_sample_count=(
                        expected_sample_count
                    ),
                    progress_callback=(
                        decode_progress
                    ),
                )
            )

            progress_bar.progress(
                1.0
            )

            decoder_status.success(
                "Decoding complete."
            )

            d1, d2, d3 = (
                st.columns(3)
            )

            d1.metric(
                "Decoded Samples",
                (
                    f"{len(decoded_samples):,}"
                ),
            )

            d2.metric(
                "Rice K",
                int(
                    decoder_k
                ),
            )

            d3.metric(
                "Output",
                "16-bit PCM",
            )

            if (
                reference_samples
                is not None
            ):

                verified = (
                    np.array_equal(
                        reference_samples.astype(
                            np.int16
                        ),
                        decoded_samples.astype(
                            np.int16
                        ),
                    )
                )

                if verified:

                    st.success(
                        (
                            "✓ Lossless verification passed: "
                            "decoded samples exactly match "
                            "the reference WAV."
                        )
                    )

                else:

                    st.error(
                        (
                            "Lossless verification failed. "
                            "Check K and prediction settings."
                        )
                    )

            st.audio(
                str(
                    decoded_output_path
                )
            )

            safe_download(
                "⬇ Download decoded WAV",
                decoded_output_path,
                "audio/wav",
                "download_standalone_decoded",
            )

        except Exception as error:

            decoder_status.error(
                "Decoding failed."
            )

            st.exception(
                error
            )


# ============================================================
# CONFIG TAB
# ============================================================


with config_tab:

    st.markdown(
        "### Current Experiment Configuration"
    )

    configuration_table = (
        pd.DataFrame(
            [
                {
                    "Parameter":
                        "Prediction strategy",

                    "Value":
                        strategy_label,
                },
                {
                    "Parameter":
                        "Rice K values",

                    "Value":
                        str(
                            k_values
                        ),
                },
                {
                    "Parameter":
                        "Selected audio files",

                    "Value":
                        ", ".join(
                            [
                                path.name
                                for path
                                in selected_paths
                            ]
                        ),
                },
                {
                    "Parameter":
                        "EX2 bit-length header",

                    "Value":
                        "4 bytes, big-endian",
                },
                {
                    "Parameter":
                        "Input format",

                    "Value":
                        "16-bit mono PCM WAV",
                },
                {
                    "Parameter":
                        "Signed mapping",

                    "Value":
                        (
                            "(abs(s) << 1) | "
                            "(1 if s < 0 else 0)"
                        ),
                },
            ]
        )
    )

    st.dataframe(
        configuration_table,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        "### Official Exercise 2 Method"
    )

    st.code(
        """
WAV samples
    ↓
first-order delta prediction
    ↓
signed → unsigned mapping
    ↓
Rice coding with K = 2 / 4
    ↓
packed .ex2 bitstream
    ↓
Rice decoding
    ↓
inverse first-order delta
    ↓
reconstructed WAV
    ↓
sample-for-sample equality verification
        """.strip()
    )


# ============================================================
# RUN OFFICIAL / EXPERIMENTAL BATCH
# ============================================================


if run_button:

    if not selected_paths:

        st.error(
            "Select at least one WAV file."
        )

        st.stop()

    if not k_values:

        st.error(
            "Select at least one K value."
        )

        st.stop()

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_directory = (
        OUTPUTS_DIR
        / f"run_{timestamp}"
    )

    input_directory = (
        run_directory
        / "inputs"
    )

    encoded_directory = (
        run_directory
        / "encoded"
    )

    decoded_directory = (
        run_directory
        / "decoded"
    )

    results_directory = (
        run_directory
        / "results"
    )

    for directory in [
        input_directory,
        encoded_directory,
        decoded_directory,
        results_directory,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    run_inputs = (
        copy_inputs_to_run(
            selected_paths,
            input_directory,
        )
    )

    tasks = [
        (
            wav_path,
            int(k),
        )
        for wav_path
        in run_inputs
        for k
        in sorted(
            k_values
        )
    ]

    results = []

    global_progress = (
        progress_placeholder.progress(
            0.0
        )
    )

    status_metric.metric(
        "Codec Status",
        "Running",
    )

    for task_index, (
        wav_path,
        k,
    ) in enumerate(
        tasks
    ):

        live_task_placeholder.info(
            (
                f"Task "
                f"{task_index + 1}"
                f" / "
                f"{len(tasks)}"
                f": "
                f"{wav_path.name}, "
                f"K={k}"
            )
        )

        def progress_callback(
            stage,
            current,
            total,
            task_index=task_index,
        ):

            fraction = (
                current
                / max(
                    total,
                    1,
                )
            )

            if stage == "encode":

                local_fraction = (
                    0.5
                    * fraction
                )

                stage_text = (
                    "Rice encoding"
                )

            else:

                local_fraction = (
                    0.5
                    + (
                        0.5
                        * fraction
                    )
                )

                stage_text = (
                    "Rice decoding"
                )

            overall_fraction = (
                task_index
                + local_fraction
            ) / len(
                tasks
            )

            global_progress.progress(
                min(
                    1.0,
                    overall_fraction,
                )
            )

            status_placeholder.info(
                (
                    f"{stage_text}: "
                    f"{wav_path.name}, "
                    f"K={k} "
                    f"— "
                    f"{fraction * 100:.1f}%"
                )
            )

        try:

            result = compress_wav(
                filepath=(
                    wav_path
                ),
                k=k,
                encoded_dir=(
                    encoded_directory
                ),
                decoded_dir=(
                    decoded_directory
                ),
                strategy=(
                    strategy
                ),
                progress_callback=(
                    progress_callback
                ),
            )

            results.append(
                result
            )

            status_placeholder.success(
                (
                    f"✓ {wav_path.name}, "
                    f"K={k} "
                    f"lossless verified"
                )
            )

        except Exception as error:

            status_placeholder.error(
                (
                    f"Failed: "
                    f"{wav_path.name}, "
                    f"K={k}"
                )
            )

            st.exception(
                error
            )

            st.stop()

    global_progress.progress(
        1.0
    )

    status_metric.metric(
        "Codec Status",
        "Complete",
    )

    live_task_placeholder.success(
        (
            "✓ All Rice coding tasks "
            "completed successfully."
        )
    )

    # --------------------------------------------------------
    # RESULTS CSV
    # --------------------------------------------------------

    result_df = (
        results_dataframe(
            results
        )
    )

    compression_csv = (
        results_directory
        / "compression_results.csv"
    )

    result_df.to_csv(
        compression_csv,
        index=False,
    )

    # --------------------------------------------------------
    # COURSEWORK FORMAT TABLE
    # --------------------------------------------------------

    coursework_table = (
        build_coursework_table(
            result_df
        )
    )

    coursework_csv = None

    if coursework_table is not None:

        coursework_csv = (
            results_directory
            / (
                "coursework_required_"
                "exercise2_table.csv"
            )
        )

        coursework_table.to_csv(
            coursework_csv,
            index=False,
        )

    # --------------------------------------------------------
    # LOSSLESS TABLE
    # --------------------------------------------------------

    lossless_df = (
        build_lossless_table(
            result_df
        )
    )

    lossless_csv = (
        results_directory
        / "lossless_verification.csv"
    )

    lossless_df.to_csv(
        lossless_csv,
        index=False,
    )

    # --------------------------------------------------------
    # RESIDUAL STATISTICS
    # --------------------------------------------------------

    residual_rows = []

    for wav_path in run_inputs:

        mode = (
            resolve_prediction_mode(
                strategy,
                wav_path.name,
            )
        )

        residual_rows.append(
            residual_statistics(
                wav_path,
                mode,
            )
        )

    residual_df = (
        pd.DataFrame(
            residual_rows
        )
    )

    residual_csv = (
        results_directory
        / "residual_statistics.csv"
    )

    residual_df.to_csv(
        residual_csv,
        index=False,
    )

    # --------------------------------------------------------
    # CONFIG JSON
    # --------------------------------------------------------

    configuration = {
        "exercise":
            "Exercise 2",

        "strategy":
            strategy,

        "strategy_label":
            strategy_label,

        "k_values":
            sorted(
                [
                    int(k)
                    for k
                    in k_values
                ]
            ),

        "input_files":
            [
                path.name
                for path
                in run_inputs
            ],

        "ex2_format":
            (
                "4-byte big-endian valid-bit length "
                "+ packed Rice bits"
            ),

        "lossless_tests":
            int(
                result_df[
                    "lossless_verified"
                ].sum()
            ),

        "all_lossless":
            bool(
                result_df[
                    "lossless_verified"
                ].all()
            ),
    }

    config_path = (
        results_directory
        / "run_config.json"
    )

    save_run_configuration(
        config_path,
        configuration,
    )

    st.session_state[
        "exercise_2_last_run"
    ] = {
        "run_directory":
            str(
                run_directory
            ),

        "results":
            result_df.to_dict(
                orient="records"
            ),

        "coursework_table":
            (
                coursework_table.to_dict(
                    orient="records"
                )
                if coursework_table
                is not None
                else None
            ),

        "residual_statistics":
            residual_df.to_dict(
                orient="records"
            ),

        "lossless":
            lossless_df.to_dict(
                orient="records"
            ),

        "strategy":
            strategy,

        "strategy_label":
            strategy_label,
    }


# ============================================================
# DISPLAY CURRENT/PREVIOUS RESULTS
# ============================================================


run_state = (
    st.session_state.get(
        "exercise_2_last_run"
    )
)

if run_state:

    display_results_df = (
        pd.DataFrame(
            run_state[
                "results"
            ]
        )
    )

    run_path = Path(
        run_state[
            "run_directory"
        ]
    )

    with results_container:

        st.markdown(
            "### Rice Compression Results"
        )

        show_run_summary(
            display_results_df
        )

        st.dataframe(
            display_results_df[
                [
                    "file",
                    "K",
                    "method",
                    "original_size_bytes",
                    "compressed_size_bytes",
                    "compression_ratio",
                    "percent_compression",
                    "bits_per_sample",
                    "lossless_verified",
                    "processing_time_seconds",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        chart_df = (
            display_results_df[
                [
                    "file",
                    "K",
                    "percent_compression",
                ]
            ].copy()
        )

        chart_df[
            "Experiment"
        ] = (
            chart_df[
                "file"
            ]
            + " · K="
            + chart_df[
                "K"
            ].astype(str)
        )

        st.markdown(
            "#### Compression percentage"
        )

        st.bar_chart(
            chart_df.set_index(
                "Experiment"
            )[
                [
                    "percent_compression"
                ]
            ],
            use_container_width=True,
        )

        st.caption(
            (
                "Positive values indicate compression. "
                "Negative values indicate expansion."
            )
        )

        # ----------------------------------------------------
        # COURSEWORK TABLE
        # ----------------------------------------------------

        coursework_records = (
            run_state.get(
                "coursework_table"
            )
        )

        if coursework_records:

            st.markdown(
                "### Coursework Required Table"
            )

            coursework_df = (
                pd.DataFrame(
                    coursework_records
                )
            )

            st.dataframe(
                coursework_df,
                hide_index=True,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # LOSSLESS
        # ----------------------------------------------------

        st.markdown(
            "### Lossless Verification"
        )

        lossless_display = (
            pd.DataFrame(
                run_state[
                    "lossless"
                ]
            )
        )

        st.dataframe(
            lossless_display,
            hide_index=True,
            use_container_width=True,
        )

        if (
            not lossless_display.empty
            and
            bool(
                lossless_display[
                    "lossless_verified"
                ].all()
            )
        ):

            st.success(
                (
                    "✓ Every decoded WAV is "
                    "sample-for-sample identical "
                    "to its original input."
                )
            )

        # ----------------------------------------------------
        # RESIDUAL STATS
        # ----------------------------------------------------

        st.markdown(
            "### Residual Statistics"
        )

        residual_display = (
            pd.DataFrame(
                run_state[
                    "residual_statistics"
                ]
            )
        )

        st.dataframe(
            residual_display,
            hide_index=True,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # INDIVIDUAL RESULT DOWNLOADS
        # ----------------------------------------------------

        st.markdown(
            "### Generated Codec Files"
        )

        for index, row in (
            display_results_df.iterrows()
        ):

            encoded_path = Path(
                row[
                    "encoded_ex2"
                ]
            )

            decoded_path = Path(
                row[
                    "decoded_wav"
                ]
            )

            with st.expander(
                (
                    f"{row['file']} "
                    f"· K={int(row['K'])} "
                    f"· "
                    f"{row['percent_compression']:.2f}%"
                )
            ):

                a, b, c = (
                    st.columns(3)
                )

                a.metric(
                    "Original Size",
                    human_bytes(
                        row[
                            "original_size_bytes"
                        ]
                    ),
                )

                b.metric(
                    ".EX2 Size",
                    human_bytes(
                        row[
                            "compressed_size_bytes"
                        ]
                    ),
                )

                c.metric(
                    "Lossless",
                    (
                        "PASS"
                        if row[
                            "lossless_verified"
                        ]
                        else "FAIL"
                    ),
                )

                st.audio(
                    str(
                        decoded_path
                    )
                )

                download_left, download_right = (
                    st.columns(2)
                )

                with download_left:

                    safe_download(
                        (
                            "⬇ Download "
                            f"{encoded_path.name}"
                        ),
                        encoded_path,
                        "application/octet-stream",
                        (
                            f"ex2_"
                            f"{index}"
                        ),
                    )

                with download_right:

                    safe_download(
                        (
                            "⬇ Download "
                            f"{decoded_path.name}"
                        ),
                        decoded_path,
                        "audio/wav",
                        (
                            f"wav_"
                            f"{index}"
                        ),
                    )

        # ----------------------------------------------------
        # CSV DOWNLOADS
        # ----------------------------------------------------

        st.markdown(
            "### Result Files"
        )

        results_dir = (
            run_path
            / "results"
        )

        csv_paths = [
            results_dir
            / "compression_results.csv",

            results_dir
            / "coursework_required_exercise2_table.csv",

            results_dir
            / "lossless_verification.csv",

            results_dir
            / "residual_statistics.csv",
        ]

        for csv_index, path in enumerate(
            csv_paths
        ):

            if path.is_file():

                safe_download(
                    (
                        "⬇ "
                        + path.name
                    ),
                    path,
                    "text/csv",
                    (
                        f"csv_download_"
                        f"{csv_index}"
                    ),
                )

        st.caption(
            "Run directory:"
        )

        st.code(
            str(
                run_path.resolve()
            )
        )

else:

    with results_container:

        st.markdown(
            """
            <div class="empty-monitor">

                <div class="monitor-icon">
                    1010
                </div>

                <div class="monitor-title">
                    Rice codec ready
                </div>

                <div class="monitor-text">
                    Select your WAV files and K values,
                    then press
                    <strong>Run Rice Encode + Decode</strong>.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )
