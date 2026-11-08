# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import json
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias, cast

import huggingface_hub
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)

from .config import QuantizationMethod, Settings
from .system import empty_cache
from .utils import Prompt, batchify, format_exception, print


def _is_snapshot_complete(snapshot_dir: Path) -> bool:
    if not (snapshot_dir / "config.json").is_file():
        return False
    if any(snapshot_dir.glob("*.partial")) or any(snapshot_dir.glob("*.incomplete")):
        return False
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_file = snapshot_dir / index_name
        if index_file.is_file():
            try:
                data = json.loads(index_file.read_text(encoding="utf-8"))
                shards = set((data.get("weight_map") or {}).values())
                return bool(shards) and all(
                    (snapshot_dir / shard).is_file() for shard in shards
                )
            except Exception:
                return False
    return bool(
        list(snapshot_dir.glob("*.safetensors")) or list(snapshot_dir.glob("*.bin"))
    )


def _find_complete_hf_snapshot(model: str, revision: str | None = None) -> Path | None:
    if Path(model).exists():
        return Path(model)
    hub_dir = (
        Path(huggingface_hub.constants.HF_HUB_CACHE)
        / f"models--{model.replace('/', '--')}"
    )
    snapshots_dir = hub_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None
    candidates: list[Path] = []
    if revision:
        candidates.append(snapshots_dir / revision)
    ref_main = hub_dir / "refs" / "main"
    if ref_main.is_file():
        ref_rev = ref_main.read_text(encoding="utf-8").strip()
        if ref_rev:
            candidates.append(snapshots_dir / ref_rev)
    candidates.extend(
        sorted(
            (p for p in snapshots_dir.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )
    seen: set[Path] = set()
    for cand in candidates:
        if cand in seen or not cand.is_dir():
            continue
        seen.add(cand)
        if _is_snapshot_complete(cand):
            return cand
    return None


def _download_from_modelscope(model: str) -> Path:
    api_url = f"https://www.modelscope.cn/api/v1/models/{model}/repo/files"
    req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        payload = json.loads(resp.read().decode("utf-8"))
    raw_files = (payload.get("Data") or {}).get("Files") or []
    files = [
        f
        for f in raw_files
        if f.get("Type") == "blob"
        and f.get("Path")
        and f.get("Path") not in {".gitattributes"}
    ]
    if not files:
        raise OSError(f"ModelScope repository {model} contains no downloadable files.")

    commit_id = str(files[0].get("Revision") or "main")
    hub_dir = (
        Path(huggingface_hub.constants.HF_HUB_CACHE)
        / f"models--{model.replace('/', '--')}"
    )
    snapshot_dir = hub_dir / "snapshots" / commit_id
    refs_dir = hub_dir / "refs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(commit_id, encoding="utf-8")

    print(
        f"* [yellow]Hugging Face Hub rate-limited or unreachable; staging [bold]{model}[/] from ModelScope ({len(files)} files)...[/]"
    )
    for idx, f in enumerate(files, 1):
        rel_path = str(f["Path"])
        expected_size = int(f.get("Size") or 0)
        dest = snapshot_dir / rel_path
        partial = snapshot_dir / f"{rel_path}.partial"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if (
            dest.is_file()
            and not partial.exists()
            and (expected_size <= 0 or dest.stat().st_size == expected_size)
        ):
            continue
        size_mb = expected_size / (1024 * 1024)
        print(
            f"  * Downloading [bold]{rel_path}[/] ({idx}/{len(files)}, {size_mb:.1f} MB)..."
        )
        file_url = (
            f"https://www.modelscope.cn/models/{model}/resolve/master/"
            f"{urllib.parse.quote(rel_path)}"
        )
        subprocess.run(
            [
                "curl",
                "-fL",
                "--retry",
                "12",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "-C",
                "-",
                "--silent",
                "--show-error",
                "-o",
                str(partial),
                file_url,
            ],
            check=True,
        )
        if expected_size > 0 and partial.stat().st_size != expected_size:
            raise OSError(
                f"Incomplete download for {rel_path}: got {partial.stat().st_size} bytes, expected {expected_size}."
            )
        partial.replace(dest)

    return snapshot_dir


def ensure_model_cached(model: str, revision: str | None = None) -> str:
    local_snap = _find_complete_hf_snapshot(model, revision)
    if local_snap is not None:
        return str(local_snap)
    try:
        huggingface_hub.model_info(model, revision=revision, timeout=10)
        return model
    except Exception:
        if "/" in model and not Path(model).exists():
            return str(_download_from_modelscope(model))
        raise


def get_model_class(
    model: str,
) -> type[AutoModelForImageTextToText] | type[AutoModelForCausalLM]:
    local_snap = _find_complete_hf_snapshot(model)
    target = str(local_snap) if local_snap is not None else model
    configs = PretrainedConfig.get_config_dict(target)

    if any(("vision_config" in config) for config in configs):
        return AutoModelForImageTextToText
    else:
        return AutoModelForCausalLM


def check_model_resident(model: PreTrainedModel | PeftModel) -> None:
    """
    Ensure that all of the model's weights are resident in memory.
    """
    offloaded = [name for name, param in model.named_parameters() if param.is_meta]

    if not offloaded:
        return

    raise Exception(
        "The model does not fit into the available device memory: "
        f"{len(offloaded)} weights (such as {', '.join(offloaded[:3])}) had to "
        "be offloaded to CPU or disk. Heretic cannot process offloaded models "
        "because it requires direct, persistent access to all weights, whereas "
        "offloaded weights only exist temporarily during forward passes.\n"
        "Possible remedies:\n"
        "* Free up memory, e.g. by closing other applications.\n"
        "* Change the device map (device_map/max_memory settings) so that all "
        "weights fit into device memory.\n"
        '* Enable 4-bit quantization (quantization = "bnb_4bit").'
    )


def _collapse_flattened_residual_streams(
    hidden_states: tuple[FloatTensor, ...],
    config: Any,
) -> tuple[FloatTensor, ...]:
    """
    Average flattened multi-stream hidden states to hidden_size.

    Some models (e.g. Qwen4-Exp) flatten their residual streams into the
    component dimension instead of exposing a separate stream dimension, so
    their hidden states have shape (prompt, position, streams * hidden_size).
    The final entry is the merged model output (hidden_size) and is unchanged.
    """
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        config = get_text_config()
    hidden_size = getattr(config, "hidden_size", None)
    streams = getattr(config, "hc_count", 1)
    if hidden_size is None or streams <= 1:
        return hidden_states
    width = hidden_size * streams
    return tuple(
        h.unflatten(-1, (streams, hidden_size)).mean(-2) if h.shape[-1] == width else h
        for h in hidden_states
    )


# Some chat templates (e.g. gpt-oss) insert the current date into the prompt,
# which makes the residuals, and therefore the modified model, depend on the day
# Heretic is run. Rendering templates with a fixed date keeps results reproducible.
def strftime_fixed_date(format: str) -> str:
    return datetime(2026, 1, 1, tzinfo=timezone.utc).strftime(format)


def chat_template_supports_system_role(tokenizer: PreTrainedTokenizerBase) -> bool:
    # Some chat templates (e.g. Gemma 2 and its fine-tunes) reject the system role
    # outright, raising when one is present; others silently drop it. The only
    # reliable way to tell is to render a probe chat and check whether a sentinel
    # system message survives into the output. This is also how llama.cpp's template
    # engine (minja) detects system-role support.
    sentinel = "HERETIC_SYSTEM_ROLE_PROBE"
    try:
        rendered = cast(
            str,
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": sentinel},
                    {"role": "user", "content": "Hi"},
                ],
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
    except Exception:
        return False
    return sentinel in rendered


# The list contains one element per layer.
# Each element maps from the component name to a (possibly sparse) mapping
# from the module index to an (input, output) tuple containing the I/O
# tensors of shape (prompt, component).
ModuleIO: TypeAlias = list[dict[str, dict[int, tuple[Tensor, Tensor]]]]


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase
    # Set for multimodal models, None for text-only ones.
    processor: ProcessorMixin | None
    peft_config: LoraConfig
    dtype: torch.dtype
    # Whether the model's chat template accepts a system role (see __init__).
    supports_system_role: bool

    def __init__(self, settings: Settings):
        self.settings = settings
        self.needs_reload = False

        self.revision_kwargs = {}
        if settings.model_commit is not None:
            self.revision_kwargs["revision"] = settings.model_commit

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        resolved_model = ensure_model_cached(settings.model, settings.model_commit)
        load_revision_kwargs = (
            self.revision_kwargs if resolved_model == settings.model else {}
        )

        # PreTrainedTokenizerBase is the "base class for all tokenizer backends"
        # according to the documentation.
        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                resolved_model,
                revision=load_revision_kwargs.get("revision"),
            ),
        )

        # Multimodal models have a processor we'll want to save.
        self.processor = None
        if get_model_class(resolved_model) == AutoModelForImageTextToText:
            self.processor = AutoProcessor.from_pretrained(
                resolved_model,
                revision=load_revision_kwargs.get("revision"),
            )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        # Detect once whether this tokenizer's chat template accepts a system
        # role. Templates that don't (e.g. Gemma 2) need the system prompt folded
        # into the first user message instead (see _apply_system_role).
        self.supports_system_role = chat_template_supports_system_role(self.tokenizer)

        self.model = None  # ty: ignore[invalid-assignment]
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )

        self.trusted_models = set()

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]...")

            try:
                quantization_config = self._get_quantization_config(dtype)

                extra_kwargs = {}
                # Only include quantization_config if it's not None
                # (some models like gpt-oss have issues with explicit None).
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config

                self.model = get_model_class(resolved_model).from_pretrained(
                    resolved_model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if settings.model in self.trusted_models
                    else None,
                    **load_revision_kwargs,
                    **extra_kwargs,
                )
                self.model.config._name_or_path = settings.model

                self.dtype = self.model.dtype

                # If we reach this point and the model requires trust_remote_code,
                # the user must have agreed when prompted to execute remote code,
                # because from_pretrained raises an exception otherwise.
                self.trusted_models.add(settings.model)

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                    # The KV cache is deliberately left enabled here so we can
                    # catch dtype errors related to the cache, which we will need
                    # for generating responses.
                    # use_cache=False,
                )
            except Exception as error:
                self.model = None  # ty: ignore[invalid-assignment]
                empty_cache()

                formatted = format_exception(error)
                if "\n" in formatted:
                    print(f"* [red]Failed:\n{formatted}[/]")
                else:
                    print(f"* [red]Failed ({formatted})[/]")

                continue

            if settings.quantization == QuantizationMethod.BNB_4BIT:
                print("* Quantized to 4-bit precision")

            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        check_model_resident(self.model)

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")

        all_components = {}
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                if component not in all_components:
                    all_components[component] = 0
                all_components[component] += len(modules)

        print("* Modifiable components:")
        for component, count in all_components.items():
            print(f"  * [bold]{component}[/]: [bold]{count}[/] modules total")

    def apply_lora(self, lora_rank: int):
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PreTrainedModel)

        # Always use LoRA adapters for modification (faster reload, no weight modification).
        # Collect actual leaf module names from the model for LoRA targeting.
        # This is more robust than splitting component keys (e.g. "attn.o_proj" -> "o_proj")
        # because hybrid models like Qwen3.5 MoE have modules with different names
        # across layers (e.g. "o_proj" on attention layers, "out_proj" on linear attention layers).
        target_modules_set: set[str] = set()

        module_id_to_full_name = {
            id(module): module_name
            for module_name, module in self.model.named_modules()
        }

        for layer_index in range(len(self.get_layers())):
            for modules in self.get_layer_modules(layer_index).values():
                for module in modules:
                    full_name = module_id_to_full_name.get(id(module))
                    if full_name is not None:
                        target_modules_set.add(full_name)

        target_modules = sorted(target_modules_set)

        self.peft_config = LoraConfig(
            r=lora_rank,
            target_modules=target_modules,
            lora_alpha=lora_rank,  # Apply adapter at full strength.
            lora_dropout=0,
            bias="none",
            # Even if we're using AutoModelForImageTextToText, this is still correct,
            # as VL models are typically just causal LMs with an added image encoder.
            task_type="CAUSAL_LM",
        )

        # self.peft_config is a LoraConfig object rather than a dictionary,
        # so the result is a PeftModel rather than a PeftMixedModel.
        self.model = cast(PeftModel, get_peft_model(self.model, self.peft_config))

        # PEFT initializes A randomly from the global RNG. B is zero, so this doesn't
        # change the model, but zeroing A too keeps saved adapters reproducible.
        self._zero_lora_weights()

    def _zero_lora_weights(self):
        for name, module in self.model.named_modules():
            if ("lora_A" in name or "lora_B" in name) and hasattr(module, "weight"):
                if module.weight.device.type == "meta":
                    module.to_empty(device=self.model.device)
                torch.nn.init.zeros_(module.weight)

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Guard against calling this method at the wrong time.
        assert isinstance(self.model, PeftModel)

        has_offloaded_params = any(
            param.device.type == "meta" for param in self.model.parameters()
        )

        # Check if we need special handling for quantized or CPU-offloaded models
        if (
            self.settings.quantization == QuantizationMethod.BNB_4BIT
            or has_offloaded_params
        ):
            # Quantized or CPU-offloaded models need special handling - we must reload
            # the base model in full precision on CPU to merge the LoRA adapters

            # Get the adapter state dict and dtype before freeing the working model
            model_dtype = self.model.dtype
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            self.model = None  # ty: ignore[invalid-assignment]
            self.needs_reload = True
            empty_cache()

            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            resolved_model = ensure_model_cached(
                self.settings.model, self.settings.model_commit
            )
            load_revision_kwargs = (
                self.revision_kwargs if resolved_model == self.settings.model else {}
            )
            base_model = get_model_class(resolved_model).from_pretrained(
                resolved_model,
                torch_dtype=model_dtype,
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **load_revision_kwargs,
            )
            base_model.config._name_or_path = self.settings.model

            # Apply LoRA adapters to the CPU model
            print("* Applying LoRA adapters...")
            peft_model = cast(PeftModel, get_peft_model(base_model, self.peft_config))

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = self.model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self) -> bool:
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.

        Returns True if the fast path was taken.
        """

        # If a prior model load was interrupted/cancelled mid-process, self.model will be None.
        current_model = None
        if self.model is not None:
            current_model = getattr(self.model.config, "name_or_path", None)

        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation).
            # Zeroing A as well ensures that adapter weights of modules not modified
            # by the next trial don't depend on what previous trials did.
            self._zero_lora_weights()
            return True

        # Purge existing model object from memory to make space.
        self.model = None  # ty: ignore[invalid-assignment]
        empty_cache()

        quantization_config = self._get_quantization_config(
            str(self.dtype).split(".")[-1]
        )

        # Build kwargs, only include quantization_config if it's not None.
        extra_kwargs = {}
        if quantization_config is not None:
            extra_kwargs["quantization_config"] = quantization_config

        resolved_model = ensure_model_cached(
            self.settings.model, self.settings.model_commit
        )
        load_revision_kwargs = (
            self.revision_kwargs if resolved_model == self.settings.model else {}
        )

        self.model = get_model_class(resolved_model).from_pretrained(
            resolved_model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **load_revision_kwargs,
            **extra_kwargs,
        )
        self.model.config._name_or_path = self.settings.model

        check_model_resident(self.model)

        self.needs_reload = False

        return False

    def get_layers(self) -> ModuleList:
        model = self.model

        # Unwrap PeftModel (always true after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Most multimodal models.
        with suppress(Exception):
            return model.model.language_model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

        # Text-only models.
        return model.model.layers  # ty: ignore[unresolved-attribute, invalid-return-type]

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Standard self-attention out-projection (most models).
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_proj)  # ty: ignore[unresolved-attribute]

        # Qwen3.5 MoE hybrid layers use GatedDeltaNet (linear attention) instead of
        # standard self-attention, so self_attn.o_proj doesn't exist on those layers.
        with suppress(Exception):
            try_add("attn.o_proj", layer.linear_attn.out_proj)  # ty: ignore[unresolved-attribute]

        # DeepSeek V4.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.o_b_proj)  # ty: ignore[unresolved-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty: ignore[unresolved-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Shared experts of MoE models (e.g. DeepSeek V4 and GLM 5.3).
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.shared_experts.down_proj)  # ty: ignore[unresolved-attribute]

        # Shared expert of MoE models (e.g. Qwen4-Exp).
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.shared_expert.down_proj)  # ty: ignore[unresolved-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # LFM dense operator blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.conv.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            try_add("mlp.down_proj", layer.feed_forward.w2)  # ty: ignore[unresolved-attribute]

        # LFM transformer blocks.
        with suppress(Exception):
            try_add("attn.o_proj", layer.self_attn.out_proj)  # ty: ignore[unresolved-attribute]

        with suppress(Exception):
            for expert in layer.feed_forward.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.w2)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty: ignore[unresolved-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[not-iterable, unresolved-attribute]
                try_add("mlp.down_proj", expert.output_linear)  # ty: ignore[unresolved-attribute]

        # We need at least one module across all components.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No modifiable modules found in layer"

        return modules

    def get_modifiable_components(self) -> list[str]:
        components: set[str] = set()

        # Scan all layers because hybrid models (e.g. Qwen3.5 MoE) have different
        # components on different layers (some have self_attn, others linear_attn).
        for layer_index in range(len(self.get_layers())):
            components.update(self.get_layer_modules(layer_index).keys())

        return sorted(components)

    def _apply_system_role(
        self, messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        # Templates that support the system role get the messages unchanged, so the
        # rendered chat (and therefore the resulting weights) is byte-for-byte
        # identical to before. For templates that reject it (e.g. Gemma 2), fold each
        # system message into the next user message (what Gemma's own later templates
        # do), so a system role anywhere in the chat is removed before rendering, not
        # just a leading one. Empty system prompts are dropped; a system message with
        # no following user message becomes a user turn.
        if self.supports_system_role:
            return messages
        result: list[dict[str, str]] = []
        pending: list[str] = []
        for message in messages:
            if message["role"] == "system":
                if message["content"]:
                    pending.append(message["content"])
                continue
            if pending and message["role"] == "user":
                message = dict(message)
                message["content"] = "\n\n".join([*pending, message["content"]])
                pending = []
            result.append(message)
        if pending:
            result.append({"role": "user", "content": "\n\n".join(pending)})
        return result

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        chats = [
            self._apply_system_role(
                [
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user},
                ]
            )
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
                # Overrides the function that templates call to get the current date.
                strftime_now=strftime_fixed_date,
            ),
        )

        if self.settings.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [
                prompt + self.settings.response_prefix for prompt in chat_prompts
            ]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            responses.extend(
                self.get_responses(
                    batch,
                    skip_special_tokens=skip_special_tokens,
                )
            )

        return responses

    def get_residuals(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            # KV cache is unnecessary here because we only need the hidden states
            # for the first generated token.
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        # Models with Hyper-Connections (e.g. DeepSeek V4 and GLM 5.3) have several
        # residual streams, so their hidden states have shape
        # (prompt, position, stream, component). We use the mean of the streams.
        # The last entry has no stream dimension, because Transformers replaces it
        # with the model output, where the streams have already been merged.
        hidden_states = tuple(
            (
                layer_hidden_states.mean(dim=2)
                if layer_hidden_states.dim() == 4
                else layer_hidden_states
            )
            for layer_hidden_states in hidden_states
        )

        hidden_states = _collapse_flattened_residual_streams(
            cast(tuple[FloatTensor, ...], hidden_states),
            self.model.config,
        )

        target_device = (
            torch.device("cpu")
            if self.settings.offload_outputs_to_cpu
            else hidden_states[0].device
        )

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [
                layer_hidden_states[:, -1, :].to(target_device)
                for layer_hidden_states in hidden_states
            ],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        residuals = residuals.to(torch.float32)

        if 0 <= winsorization_quantile < 1:
            # Apply symmetric winsorization to each layer of the per-prompt residuals.
            abs_residuals = torch.abs(residuals)
            # Get the (prompt, layer, 1) quantiles of the (prompt, layer, component) residuals.
            thresholds = torch.quantile(
                abs_residuals,
                winsorization_quantile,
                dim=2,
                keepdim=True,
            )
            residuals = torch.clamp(residuals, -thresholds, thresholds)

        if self.settings.offload_outputs_to_cpu:
            residuals = residuals.cpu()
            empty_cache()

        return residuals

    def get_residuals_batched(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(
                self.get_residuals(
                    batch,
                    winsorization_quantile=winsorization_quantile,
                )
            )

        return torch.cat(residuals, dim=0)

    def get_residuals_mean(
        self,
        prompts: list[Prompt],
        winsorization_quantile: float = 1.0,
    ) -> Tensor:
        if not prompts:
            raise ValueError("prompts must not be empty")

        running_sum = None
        total_count = 0

        for batch in batchify(prompts, self.settings.batch_size):
            batch_residuals = self.get_residuals(
                batch,
                winsorization_quantile=winsorization_quantile,
            )

            # Accumulate in high precision on CPU to reduce peak VRAM usage.
            batch_sum = batch_residuals.sum(dim=0, dtype=torch.float64).cpu()

            if running_sum is None:
                running_sum = batch_sum
            else:
                running_sum += batch_sum

            total_count += batch_residuals.shape[0]

        assert running_sum is not None

        return (running_sum / total_count).to(torch.float32)

    def get_module_io(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # The list contains one element per layer.
        # Each element maps from the component name to a (possibly sparse) mapping
        # from the module index to an (input, output) tuple containing the I/O
        # tensors of shape (prompt, component).
        module_io: ModuleIO = []

        def get_hook(
            layer_index: int,
            component: str,
            module_index: int,
        ) -> Callable[[Module, tuple[Tensor, ...], Tensor], None]:
            def hook(
                module: Module,
                inputs: tuple[Tensor, ...],
                outputs: Tensor,
            ) -> None:
                if len(module_io) == layer_index:
                    # First invocation of the hook for this layer.
                    module_io.append({})

                # Layers are invoked in order during inference,
                # so this should always hold.
                assert len(module_io) == layer_index + 1

                if component not in module_io[layer_index]:
                    module_io[layer_index][component] = {}

                # Each module should be invoked at most once per inference step.
                assert module_index not in module_io[layer_index][component]

                # inputs[0] and outputs have shape (prompt, position, component),
                # so this extracts the input/output at the end of each prompt.
                # Move to CPU to decouple from device assignments, which can
                # change between model reloads in multi-GPU configurations.
                input = inputs[0][:, -1, :].detach().clone().cpu()
                output = outputs[:, -1, :].detach().clone().cpu()

                # The modules associated with a component (e.g. expert MLPs)
                # are not necessarily invoked in order, nor are all of them
                # necessarily invoked in each inference step, so we cannot
                # use a list here.
                module_io[layer_index][component][module_index] = (input, output)

            return hook

        hook_handles: list[RemovableHandle] = []

        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                for module_index, module in enumerate(modules):
                    hook_handles.append(
                        module.register_forward_hook(
                            get_hook(layer_index, component, module_index)
                        )
                    )

        self.generate(
            prompts,
            max_new_tokens=1,
            use_cache=False,
        )

        for hook_handle in hook_handles:
            hook_handle.remove()

        return module_io

    def get_module_io_batched(
        self,
        prompts: list[Prompt],
    ) -> ModuleIO:
        # Aggregating batch results is more complicated for module I/O
        # than for other get_*_batched methods, because the structure of the results
        # might differ between batches, as whether individual modules activate
        # can depend on the prompt (in particular for MoE models).
        # In practice, inhomogeneous results should be very rare, but to be fully
        # generic, this logic is required.
        module_io_batches: list[ModuleIO] = [
            self.get_module_io(batch)
            for batch in batchify(prompts, self.settings.batch_size)
        ]

        module_io: ModuleIO = []

        for layer_index in range(len(self.get_layers())):
            module_io.append({})

            for module_io_batch in module_io_batches:
                for component, io_map in module_io_batch[layer_index].items():
                    if component not in module_io[layer_index]:
                        module_io[layer_index][component] = {}

                    for module_index in io_map:
                        if module_index not in module_io[layer_index][component]:
                            # This is a placeholder; the actual aggregation happens below.
                            # We need to iterate over the batches twice because we don't
                            # know in advance which components and module indices are present.
                            module_io[layer_index][component][module_index] = (
                                torch.empty(0),
                                torch.empty(0),
                            )

            for component, io_map in module_io[layer_index].items():
                for module_index in io_map:
                    inputs_outputs = [
                        module_io_batch[layer_index][component][module_index]
                        for module_io_batch in module_io_batches
                        if component in module_io_batch[layer_index]
                        and module_index in module_io_batch[layer_index][component]
                    ]
                    input = torch.cat(
                        [input_output[0] for input_output in inputs_outputs],
                        dim=0,
                    )
                    output = torch.cat(
                        [input_output[1] for input_output in inputs_outputs],
                        dim=0,
                    )

                    # The key already exists, and replacing existing values
                    # in a dictionary while iterating over the same dictionary
                    # is safe in Python.
                    module_io[layer_index][component][module_index] = (input, output)

        return module_io

    def get_logits(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the raw logits over the vocabulary
        # at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_logits=True,
            return_dict_in_generate=True,
            use_cache=False,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # Use raw logits, not processed generation scores; processors can insert
        # -inf for suppressed tokens, which can make KL divergence evaluate to NaN.
        # This cast is valid because we passed output_logits=True above.
        logits = cast(tuple[FloatTensor], outputs.logits)[0]

        # The returned tensor has shape (prompt, token).
        if self.settings.offload_outputs_to_cpu:
            del outputs
            logits = logits.cpu()
            empty_cache()

        return logits

    def get_logits_batched(self, prompts: list[Prompt]) -> Tensor:
        logits = []

        for batch in batchify(prompts, self.settings.batch_size):
            logits.append(self.get_logits(batch))

        return torch.cat(logits, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat = self._apply_system_role(chat)
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        # This cast is valid because str is the return type
        # when passing a sequence of token IDs.
        return cast(
            str,
            self.tokenizer.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        )
