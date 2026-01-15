import logging
from pathlib import Path

import pypsa

from scripts.utilities.logging_config import setup_logging
from scripts.utilities.network_io import load_network


def main():
    snk = globals().get("snakemake")
    log_path = snk.log[0] if snk and hasattr(snk, 'log') and snk.log else "export_network_to_csv"
    logger = setup_logging(log_path)

    try:
        if not snk:
            raise RuntimeError("This script is intended to be called by Snakemake.")

        network_path = Path(snk.input.network)
        marker_path = Path(snk.output.marker)
        export_dir = Path(snk.params.export_dir) if "export_dir" in snk.params else marker_path.parent

        export_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading network from {network_path}")
        network = load_network(network_path.as_posix())

        logger.info(f"Exporting network tables to {export_dir}")
        network.export_to_csv_folder(export_dir.as_posix())

        # Write a marker file so Snakemake has a static output
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("export complete\n", encoding="utf-8")
        logger.info(f"Wrote export completion marker to {marker_path}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to export network to CSV: {e}")
        raise


if __name__ == "__main__":
    main()


