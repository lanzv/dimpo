from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
from transformers import AutoModelForCausalLM
import logging
from .llama.lowdim_attn import LlamaLowDimAttention
from .qwen3.lowdim_attn import Qwen3LowDimAttention
from .qwen2.lowdim_attn import Qwen2LowDimAttention
from .lowdim_modules import LowDimDimPO




def parse_lowdim_attn_layers(lowdim_attn_layers):
    lowdim_attn_layers = str(lowdim_attn_layers)
    if lowdim_attn_layers == "None" or lowdim_attn_layers == None: 
        lowdim_attn_layers = None
        logging.info(f"All attention layers were repalced by LowDimAttention layer.")
    elif len(lowdim_attn_layers) == 0:
        lowdim_attn_layers = []
        logging.info(f"No attention layer was repalced by LowDimAttention layer.")
    else:
        lowdim_attn_layers = [int(i) for i in lowdim_attn_layers.split(' ')]
        logging.info(f"The following attention layers were replaced by LowDimAttention layer: {lowdim_attn_layers}")
    return lowdim_attn_layers


def get_lowdim_attention(module, lowdim_model):
    if isinstance(module, LlamaAttention):
        return LlamaLowDimAttention(
            base_attention=module,
            lowdim_model=lowdim_model
        )
    elif isinstance(module, Qwen3Attention):
        return Qwen3LowDimAttention(
            base_attention=module,
            lowdim_model=lowdim_model
        )
    elif isinstance(module, Qwen2Attention):
        return Qwen2LowDimAttention(
            base_attention=module,
            lowdim_model=lowdim_model
        )
    raise Exception("Not supported attention module")




class AutoLowDimAttentionsModel:
    def from_pretrained(model_path, lowdim_attentions_path, device_map="auto", lowdim_attn_layers = None):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map)

        return AutoLowDimAttentionsModel.inject_lowdim_attentions(
                model=model,
                lowdim_attentions_path=lowdim_attentions_path,
                lowdim_attn_layers=lowdim_attn_layers
            )
    
    def inject_lowdim_attentions(model, lowdim_attentions_path, lowdim_attn_layers = None):
        # Inject LowDim attentions
        attention_layer_counter = 0
        for name, module in model.named_modules():
            if isinstance(module, LlamaAttention) or isinstance(module, Qwen3Attention) or isinstance(module, Qwen2Attention):
                if lowdim_attn_layers is None or attention_layer_counter in lowdim_attn_layers:
                    new_attention = get_lowdim_attention(
                        module=module, 
                        lowdim_model=LowDimDimPO.load_from_disk(target_dir=lowdim_attentions_path, layer_idx=module.layer_idx, dtype=model.dtype)
                    )
                    new_attention.lowdim_eval()

                    # Replace the module in its parent
                    parent_module = model.get_submodule(name.rsplit('.', 1)[0])
                    setattr(parent_module, name.rsplit('.', 1)[1], new_attention)
                attention_layer_counter += 1
        if attention_layer_counter == 0:
            raise Exception(f"{attention_layer_counter} attentions layers found! Only these architectures are supported: Llama3")
        
        return model