from __future__ import annotations

from typing import List

from scripts.vllm_engine import VllmVisionClient


# ---------------------------------------------------------------------------
# Prompts (unchanged from original)
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a medical AI assistant with expertise in medical genetics, "
    "dysmorphology, and image-based phenotype recognition. Your task is to "
    "analyze facial and neck images to identify any observable phenotypes "
    "related to known human diseases. These phenotypes can include "
    "abnormalities in morphology, development, skin, face, hair, eyes, ears, "
    "nose, mouth, or neck. You must use the Human Phenotype Ontology (HPO) "
    "to normalize all detected phenotypes."
)

USER_QUERY = (
    "Analyze the attached facial image. "
    "Output only observable characteristics without implying any underlying "
    "cause, disease, or mechanism. All phenotypes should be separated by ||. "
    "Please do not provide any additional texts or formatting. "
    "The response should be like this: Phenotype A || Phenotype B || Phenotype C"
)


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class LLaMA_Generator:
    """
    Vision phenotype generator backed by vLLM.

    Parameters
    ----------
    model_path           : path to the (merged) vision-LLM weights
    gpu_memory_utilization: fraction of GPU VRAM for this model
                            (~0.78 when used alone; lower when sharing GPU)
    """

    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.78,
    ):
        self._client = VllmVisionClient(
            model_path=model_path,
            temperature=0.8,
            max_tokens=1024,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    # ------------------------------------------------------------------
    # Public API (same signatures as the original HuggingFace version)
    # ------------------------------------------------------------------

    def generate_descriptions(
        self,
        img_path: str,
        *,
        max_new_tokens: int = 1024,
        top_p: float = 1.0,
        temperature: float = 0.8,
    ) -> str:
        """
        Run vision inference on a single image.

        Parameters
        ----------
        img_path       : path to the image file
        max_new_tokens : ignored (kept for API compatibility; use the
                         client's max_tokens instead)
        top_p          : ignored (vLLM top_p can be set on the client)
        temperature    : ignored (uses client default)

        Returns
        -------
        str : pipe-separated phenotype list, e.g.
              "Hypertelorism || Prominent ear || Short stature"
        """
        raw = self._client.chat(
            system=SYSTEM_MESSAGE,
            user=USER_QUERY,
            image_path=img_path,
        )
        return raw.replace("assistant", "").strip()

    def generate_descriptions_batch(
        self,
        img_paths: List[str],
    ) -> List[str]:
        """
        Batched vision inference.

        Parameters
        ----------
        img_paths : list of image file paths

        Returns
        -------
        list[str] : one output string per image
        """
        users = [USER_QUERY] * len(img_paths)
        raws  = self._client.chat_batch(
            system=SYSTEM_MESSAGE,
            users=users,
            image_paths=img_paths,
        )
        return [r.replace("assistant", "").strip() for r in raws]

    @staticmethod
    def process_output(output: str) -> List[str]:
        """Parse the pipe-separated phenotype string into a list."""
        if not output or not output.strip():
            return []
        if "||" in output:
            return [x.replace("*", "").strip() for x in output.split("||")]
        return [x.replace("*", "").strip() for x in output.split("\n") if x.strip()]

    def convert_hpo(self, output_list: List[str]):
        """Placeholder for HPO normalization (handled downstream)."""
        pass
