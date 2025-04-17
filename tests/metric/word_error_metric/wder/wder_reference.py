# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2024 Argmax, Inc. All Rights Reserved.

# This script is a helper to calculate reference WDER from diarizationlm library
# See https://github.com/google/speaker-id/blob/master/DiarizationLM/diarizationlm/metrics.py
# Doing it like this as word-levenhstein dependency of diarizationlm was causing troubles to install
import argparse

import diarizationlm


def calculate_wder(
    hypothesis_text, hypothesis_speaker, reference_text, reference_speaker
):
    # Prepare the input in the format expected by diarizationlm
    json_dict = {
        "utterances": [
            {
                "utterance_id": "utt1",
                "hyp_text": hypothesis_text,
                "hyp_spk": hypothesis_speaker,
                "ref_text": reference_text,
                "ref_spk": reference_speaker,
            }
        ]
    }

    # Compute metrics
    result = diarizationlm.compute_metrics_on_json_dict(json_dict)

    # Return just the WDER value
    return result["WDER"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WDER using diarizationlm")
    parser.add_argument(
        "--hypothesis-text", required=True, help="The hypothesized transcription text"
    )
    parser.add_argument(
        "--hypothesis-speaker", required=True, help="The hypothesized speaker labels"
    )
    parser.add_argument(
        "--reference-text", required=True, help="The reference transcription text"
    )
    parser.add_argument(
        "--reference-speaker", required=True, help="The reference speaker labels"
    )

    args = parser.parse_args()

    wder = calculate_wder(
        args.hypothesis_text,
        args.hypothesis_speaker,
        args.reference_text,
        args.reference_speaker,
    )
    print(f"{wder}")
