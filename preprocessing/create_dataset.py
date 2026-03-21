import re
import pandas as pd
from pathlib import Path

import hydra
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

def get_annot(folder_path: Path) -> pd.DataFrame:
    """Scans for .json files and stores their absolute paths."""
    data = []
    for json_path in folder_path.glob("*.json"):
        data.append({
            "slide_id": json_path.stem,
            "annot_path": str(json_path.absolute())
        })
    
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["slide_id", "annot_path"]).set_index("slide_id")
    return df.set_index("slide_id")

def get_slides(folder_path: Path, pattern: re.Pattern) -> pd.DataFrame:
    """Scans for .czi files and stores their absolute paths."""
    data = []
    for slide_path in folder_path.glob("*.czi"):
        if not pattern.search(slide_path.name):
            continue

        data.append({
            "slide_id": slide_path.stem,
            "slide_path": str(slide_path.absolute())
        })

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["slide_id", "slide_path"]).set_index("slide_id")
    return df.set_index("slide_id")

def create_dataset(
    slide_folder: str, annot_folder: str, pattern_str: str
) -> tuple[pd.DataFrame, list[str], list[str]]:
    
    slide_path = Path(slide_folder)
    annot_path = Path(annot_folder)
    pattern = re.compile(pattern_str)

    # 1. Gather Data
    slides_df = get_slides(slide_path, pattern)
    annot_df = get_annot(annot_path)

    # 2. Join on slide_id
    dataset_df = slides_df.join(annot_df, how="outer")

    # 3. Track Missing (for the text reports)
    missing_slides = dataset_df[dataset_df["slide_path"].isna()].index.to_list()
    missing_labels = dataset_df[dataset_df["annot_path"].isna()].index.to_list()

    # 4. Filter for only complete pairs
    dataset_df = dataset_df.dropna(subset=["slide_path", "annot_path"])
    
    # Ensure only the two path columns are kept
    dataset_df = dataset_df[["slide_path", "annot_path"]]

    return dataset_df, missing_slides, missing_labels

@with_cli_args([]) 
@hydra.main(config_path="conf", config_name="default", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    print("🚀 Creating dataset from CZI and JSON pairs...")
    
    dataset, missing_slides, missing_labels = create_dataset(
        config.dataset.folder,
        config.dataset.labels,
        config.dataset.regex_pattern,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save CSV
    csv_path = output_dir / "dataset.csv"
    dataset.to_csv(csv_path, index=True)
    logger.log_artifact(str(csv_path))
    
    print(f"💾 CSV created at: {csv_path}")

    # Save Missing Reports
    def _save_report(items, filename, title):
        if items:
            p = output_dir / filename
            p.write_text("\n".join(items) + "\n")
            logger.log_artifact(str(p))
            print(f"⚠️ {title}: {len(items)} items listed in {filename}")

    _save_report(missing_slides, "missing_slides.txt", "Labels without Images")
    _save_report(missing_labels, "missing_labels.txt", "Images without Labels")
    
    print(f"\n✅ Success! {len(dataset)} paired files identified.")

if __name__ == "__main__":
    main()