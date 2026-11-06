"""Plot the encoder11.6 ASR curve from an attack-results CSV file."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_asr(csv_path: Path) -> tuple[list[int], list[float]]:
    """Load the round and ASR columns, preserving their CSV order."""
    rounds: list[int] = []
    asr_values: list[float] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"round", "asr"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                rounds.append(int(row["round"]))
                asr_values.append(float(row["asr"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid round/ASR value on CSV row {row_number}") from exc

    if not rounds:
        raise ValueError("The CSV file contains no data rows")
    return rounds, asr_values


def plot_asr(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create PNG and PDF versions of the encoder11.6 ASR curve."""
    rounds, asr_values = load_asr(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.plot(
        rounds,
        asr_values,
        color="#F97316",
        linewidth=2.4,
        marker="o",
        markersize=3.8,
        markeredgewidth=0,
    )
    axis.set_title("encoder11.6", fontsize=15, fontweight="bold", loc="left")
    axis.set_xlabel("Communication Round")
    axis.set_ylabel("ASR")
    axis.set_xlim(min(rounds), max(rounds))
    axis.set_ylim(0, 1.02)
    axis.set_xticks(range(min(rounds), max(rounds) + 1, 5))
    axis.set_yticks([value / 10 for value in range(0, 11)])
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()

    png_path = output_dir / "encoder11.6_asr.png"
    pdf_path = output_dir / "encoder11.6_asr.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV containing round and asr columns")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="Directory for the generated PNG and PDF files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    png_file, pdf_file = plot_asr(args.csv_path, args.output_dir)
    print(f"Saved {png_file}")
    print(f"Saved {pdf_file}")
