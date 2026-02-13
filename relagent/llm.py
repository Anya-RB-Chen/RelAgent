"""
LLM generation via vLLM for Entity Extraction (EE) and Relationship Reasoning (REL).
Uses templates from template.py.
"""
from typing import List, Optional

from relagent.template import EE_TEMPLATE, REL_TEMPLATE


class RelAgentLLM:
    """
    vLLM-backed LLM for RelAgent: EE and REL generation using EE_TEMPLATE and REL_TEMPLATE.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_chat: bool = True,
        **vllm_kwargs,
    ):
        """
        Args:
            model: HuggingFace model name or local path for vLLM.
            max_tokens: Max new tokens per generation.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.
            use_chat: If True, use llm.chat() so the model's chat template is applied (recommended for instruct models).
            **vllm_kwargs: Passed to vllm.LLM (e.g. tensor_parallel_size, gpu_memory_utilization).
        """
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_chat = use_chat
        self._llm = None
        self._sampling_params = None
        self._vllm_kwargs = vllm_kwargs

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams
        self._llm = LLM(model=self.model_name, **self._vllm_kwargs)
        self._sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def generate(self, prompts: List[str], use_tqdm: bool = False) -> List[str]:
        """Generate completions for a list of prompts. Returns list of generated text strings."""
        if not prompts:
            return []
        self._ensure_loaded()
        if self.use_chat:
            conversations = [[{"role": "user", "content": p}] for p in prompts]
            outputs = self._llm.chat(conversations, self._sampling_params, use_tqdm=use_tqdm)
        else:
            outputs = self._llm.generate(prompts, self._sampling_params, use_tqdm=use_tqdm)
        return [out.outputs[0].text for out in outputs]

    def generate_ee(self, smiles: str, caption: str, n: int = 1) -> List[str]:
        """
        Entity Extraction: generate n candidate entity lists (JSON with name, SMILES).
        Uses EE_TEMPLATE.
        """
        prompt = EE_TEMPLATE.format(smiles=smiles, caption=caption)
        prompts = [prompt] * n
        return self.generate(prompts, use_tqdm=(n > 1))

    def generate_rel(self, prompts: List[str], use_tqdm: bool = True) -> List[str]:
        """
        Relationship Reasoning: generate one completion per REL prompt.
        Each prompt should already be formatted with REL_TEMPLATE (smiles, caption, entities, relationships).
        """
        return self.generate(prompts, use_tqdm=use_tqdm)
