"""
LLM generation via vLLM for Entity Extraction (EE) and Relationship Reasoning (REL).
Uses templates from template.py.
"""
from typing import List, Optional

from relagent.template import EE_TEMPLATE, REL_TEMPLATE


class RelAgentLLM:
    """
    vLLM-backed LLM for RelAgent: EE and REL generation using EE_TEMPLATE and REL_TEMPLATE.
    Supports separate models for EE and REL tasks.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        ee_model: Optional[str] = None,
        rel_model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_chat: bool = True,
        **vllm_kwargs,
    ):
        """
        Args:
            model: HuggingFace model name or local path for vLLM (used for both EE and REL if ee_model/rel_model not set).
            ee_model: Optional separate model for Entity Extraction. If None, uses `model`.
            rel_model: Optional separate model for Relationship Reasoning. If None, uses `model` or `ee_model`.
            max_tokens: Max new tokens per generation.
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.
            use_chat: If True, use llm.chat() so the model's chat template is applied (recommended for instruct models).
            **vllm_kwargs: Passed to vllm.LLM (e.g. tensor_parallel_size, gpu_memory_utilization).
        """
        if model is None and ee_model is None and rel_model is None:
            raise ValueError("At least one of model, ee_model, or rel_model must be provided")
        
        self.ee_model_name = ee_model or model
        self.rel_model_name = rel_model or self.ee_model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_chat = use_chat
        self._llm_ee = None
        self._llm_rel = None
        self._sampling_params = None
        self._vllm_kwargs = vllm_kwargs

    def _ensure_loaded_ee(self):
        if self._llm_ee is not None:
            return
        from vllm import LLM, SamplingParams
        self._llm_ee = LLM(model=self.ee_model_name, **self._vllm_kwargs)
        if self._sampling_params is None:
            self._sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

    def _ensure_loaded_rel(self):
        if self._llm_rel is not None:
            return
        from vllm import LLM
        # Reuse sampling params if already created
        if self._sampling_params is None:
            from vllm import SamplingParams
            self._sampling_params = SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        # Only load REL model if different from EE model
        if self.rel_model_name != self.ee_model_name:
            self._llm_rel = LLM(model=self.rel_model_name, **self._vllm_kwargs)
        else:
            # Reuse EE model
            self._ensure_loaded_ee()
            self._llm_rel = self._llm_ee

    def generate(self, prompts: List[str], use_tqdm: bool = False, llm_instance=None) -> List[str]:
        """Generate completions for a list of prompts. Returns list of generated text strings."""
        if not prompts:
            return []
        if llm_instance is None:
            raise ValueError("llm_instance must be provided")
        if self.use_chat:
            conversations = [[{"role": "user", "content": p}] for p in prompts]
            outputs = llm_instance.chat(conversations, self._sampling_params, use_tqdm=use_tqdm)
        else:
            outputs = llm_instance.generate(prompts, self._sampling_params, use_tqdm=use_tqdm)
        return [out.outputs[0].text for out in outputs]

    def generate_ee(self, smiles: str, caption: str, n: int = 1) -> List[str]:
        """
        Entity Extraction: generate n candidate entity lists (JSON with name, SMILES).
        Uses EE_TEMPLATE.
        """
        self._ensure_loaded_ee()
        prompt = EE_TEMPLATE.format(smiles=smiles, caption=caption)
        prompts = [prompt] * n
        return self.generate(prompts, use_tqdm=(n > 1), llm_instance=self._llm_ee)

    def generate_rel(self, prompts: List[str], use_tqdm: bool = True) -> List[str]:
        """
        Relationship Reasoning: generate one completion per REL prompt.
        Each prompt should already be formatted with REL_TEMPLATE (smiles, caption, entities, relationships).
        """
        self._ensure_loaded_rel()
        return self.generate(prompts, use_tqdm=use_tqdm, llm_instance=self._llm_rel)
