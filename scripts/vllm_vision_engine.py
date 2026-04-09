from __future__ import annotations

import os
from typing import List

from PIL import Image

from scripts.vllm_engine import VllmVisionClient


SYSTEM_PROMPT = (
        "You are a medical AI assistant with expertise in medical genetics, "
        "dysmorphology, and image-based phenotype recognition. Your task is to "
        "analyze facial and neck images to identify any observable phenotypes "
        "related to known human diseases. These phenotypes can include "
        "abnormalities in morphology, development, skin, face, hair, eyes, ears, "
        "nose, mouth, or neck. You must use the Human Phenotype Ontology (HPO) "
        "to normalize all detected phenotypes."
    )
USER_PROMPT = (
    "Analyze the attached facial image. "
    "Output only observable characteristics without implying any underlying "
    "cause, disease, or mechanism. All phenotypes should be separated by ||. "
    "Please do not provide any additional texts or formatting. "
    "The response should be like this: Phenotype A || Phenotype B || Phenotype C"
)


class PhenoVisionClient:
    """
    Batched vision inference wrapper around VllmVisionClient.

    All modes submit images to the GPU in batches of max_batch_size —
    no single-image sequential calls anywhere. All modes share the same
    core _generate_from_pil_batch() method.

    Mode 1 — list of file paths (original pipeline):
        client.generate_from_paths_batch(image_paths) -> list[str]

    Mode 2 — image-only directory (no text available):
        client.generate_from_dir(image_dir) -> dict[filename, result]

    Mode 3 — HuggingFace Dataset with 'image_restored' PIL key:
        client.generate_from_dataset(dataset) -> list[str]
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_batch_size: int = 4,
    ):
        self._client = VllmVisionClient(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_batch_size=max_batch_size,
        )
        self.max_batch_size = max_batch_size

    # ------------------------------------------------------------------
    # Internal: shared batched PIL inference core
    # ------------------------------------------------------------------

    def _generate_from_pil_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Core batched inference over a list of PIL images.
        Submits in chunks of max_batch_size to avoid OOM.
        Returns list[str] aligned with input — each image treated independently.
        """
        prompt  = self._client._build_prompt(SYSTEM_PROMPT, USER_PROMPT)
        results = []

        for i in range(0, len(images), self.max_batch_size):
            chunk  = images[i:i + self.max_batch_size]
            inputs = [
                {
                    "prompt":           prompt,
                    "multi_modal_data": {"image": img},
                }
                for img in chunk
            ]
            outs = self._client.llm.generate(inputs, self._client.sampling_params)
            results.extend(o.outputs[0].text.strip() for o in outs)

        return results

    # ------------------------------------------------------------------
    # Mode 1 — list of file paths (original pipeline)
    # ------------------------------------------------------------------

    def generate_from_paths_batch(self, image_paths: List[str]) -> List[str]:
        """
        Batched inference from a list of image file paths.
        Replaces the old sequential generate_from_path() loop.

        Parameters
        ----------
        image_paths : list of file path strings

        Returns
        -------
        list[str] : phenotype strings aligned with image_paths,
                    each image treated independently.
        """
        if not image_paths:
            return []

        images = [self._client._preprocess_image(p) for p in image_paths]
        return self._generate_from_pil_batch(images)

    # ------------------------------------------------------------------
    # Mode 2 — image-only directory (no text available)
    # ------------------------------------------------------------------

    def generate_from_dir(
        self,
        image_dir: str,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ) -> dict:
        """
        Batched inference over all images in a directory.
        No clinical text is available — flagged in output.

        Returns
        -------
        dict : {filename_without_ext: {"vision_phenotypes": str,
                                       "text_available": False}}
        """
        image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(extensions)
        ])

        if not image_paths:
            print(f"[WARNING] No images found in {image_dir}")
            return {}

        images = [self._client._preprocess_image(p) for p in image_paths]
        outs   = self._generate_from_pil_batch(images)

        return {
            os.path.splitext(os.path.basename(p))[0]: {
                "vision_phenotypes": out,
                "text_available":    False,  # flag: no clinical note
            }
            for p, out in zip(image_paths, outs)
        }

    # ------------------------------------------------------------------
    # Mode 3 — HuggingFace Dataset with 'image_restored' PIL key
    # ------------------------------------------------------------------

    def generate_from_dataset(
        self,
        dataset,  # HuggingFace Dataset or list[dict] with 'image_restored' PIL Image
    ) -> List[str]:
        """
        Batched inference over a HuggingFace dataset.
        Each sample must have an 'image_restored' key containing a PIL Image.

        Returns
        -------
        list[str] : phenotype strings aligned with dataset order,
                    each sample treated independently.
        """
        images = [dataset[i]["image_restored"] for i in range(len(dataset))]
        return self._generate_from_pil_batch(images)