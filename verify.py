from contextlib import ExitStack
from pathlib import Path

import matplotlib.pyplot as plt
import rasterio
import yaml

from download_from_cdse import query_id
from preprocess import (
    apply_cloud_mask,
    apply_rgb_compo,
    digital_numbers_to_reflectance,
)

with open("cdse_query.yaml", encoding="utf-8") as f:
    query = yaml.safe_load(f)

dossier = Path(query["data_folder"]) / query_id(query, temporal_id=False)
dossier_2018 = dossier / "2018-01-01-2019-01-01"

images = sorted(
    dossier_2018.glob("B02/SENTINEL2_L2A_B02_*Z.tif"),
    key=lambda p: p.name,
)

if not images:
    raise FileNotFoundError(f"Aucune image B02 trouvée pour 2018 dans {dossier_2018}")

for debut in range(0, len(images), 4):
    page = images[debut:debut + 4]
    fig, axes = plt.subplots(2, len(page), figsize=(3 * len(page), 6),
                             squeeze=False)

    for col, fichier_bleu in enumerate(page):
        date = fichier_bleu.stem.split("_")[-1].removesuffix("Z")
        dossier_date = fichier_bleu.parents[1]

        with ExitStack() as stack:
            rasters = {
                bande: stack.enter_context(rasterio.open(
                    dossier_date / bande / f"SENTINEL2_L2A_{bande}_{date}Z.tif"
                ))
                for bande in ("B02", "B03", "B04", "B08", "B11")
            }
            original = apply_rgb_compo(
                rasters, preprocess=digital_numbers_to_reflectance
            )
            masque = apply_cloud_mask(
                rasters, preprocess=digital_numbers_to_reflectance
            )

        comparaison = original.copy()
        comparaison[masque] = [255, 0, 0]

        axes[0, col].imshow(original)
        axes[0, col].set_title(date)
        axes[1, col].imshow(comparaison)

        for ax in axes[:, col]:
            ax.axis("off")

    fig.suptitle(f"Données 2018 — En haut : RGB original · En bas : pixels exclus en rouge ({debut + 1}-{debut + len(page)} / {len(images)})")
    fig.tight_layout()
    plt.show()
    plt.close(fig)