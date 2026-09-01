# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops.nms import nms
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.runner.amp import autocast
from torch import Tensor
import transformers
import math

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.models.layers.transformer.utils import coordinate_to_encoding, MLP
from llava.constants import IGNORE_INDEX
from fairscale.nn.checkpoint import checkpoint_wrapper

from ..layers.TextGuided import TextGuidedSS2D2
from ..moe.prompt_router import MoAELLMPromptAdapter
from ..utils import VisionMask

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_sineembed_for_position_2d(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=pos_tensor.dtype, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, 0] * scale
    y_embed = pos_tensor[:, 1] * scale
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=1)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, 2] * scale
        pos_w = w_embed[:, None] / dim_t
        pos_w = torch.stack((pos_w[:, 0::2].sin(), pos_w[:, 1::2].cos()), dim=2).flatten(1)

        h_embed = pos_tensor[:, 3] * scale
        pos_h = h_embed[:, None] / dim_t
        pos_h = torch.stack((pos_h[:, 0::2].sin(), pos_h[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=1)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)




def convert_grounding_to_cls_scores(logits: Tensor,
                                    positive_maps: List[dict]) -> Tensor:
    """Convert logits to class scores."""
    assert len(positive_maps) == logits.shape[0]  # batch size

    scores = torch.zeros(logits.shape[0], logits.shape[1],
                         len(positive_maps[0])).to(logits.device)
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            for label_j in positive_map:
                scores[:, :, label_j -
                       1] = logits[:, :,
                                   torch.LongTensor(positive_map[label_j]
                                                    )].mean(-1)
        else:
            for i, positive_map in enumerate(positive_maps):
                for label_j in positive_map:
                    scores[i, :, label_j - 1] = logits[
                        i, :, torch.LongTensor(positive_map[label_j])].mean(-1)
    return scores


def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.
    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption) # ['cat', 'a remote', 'the picture']
    """
    try:
        import nltk
        # nltk.download('punkt', download_dir='~/nltk_data')
        # nltk.download('averaged_perceptron_tagger', download_dir='~/nltk_data')
    except ImportError:
        raise RuntimeError('nltk is not installed, please install it by: '
                           'pip install nltk.')

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    """Remove punctuation from a text.
    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    punctuation = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\'', '\"', '’',
        '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    for p in punctuation:
        text = text.replace(p, '')
    return text.strip()


def run_ner(caption: str) -> Tuple[list, list]:
    """Run NER on a caption and return the tokens and noun phrases.
    Args:
        caption (str): The input caption.

    Returns:
        Tuple[List, List]: A tuple containing the tokens and noun phrases.
            - tokens_positive (List): A list of token positions.
            - noun_phrases (List): A list of noun phrases.
    """
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
    print('noun_phrases:', noun_phrases)
    relevant_phrases = noun_phrases
    labels = noun_phrases

    tokens_positive = []
    for entity, label in zip(relevant_phrases, labels):
        try:
            # search all occurrences and mark them as different entities
            # TODO: Not Robust
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except Exception:
            print('noun entities:', noun_phrases)
            print('entity:', entity)
            print('caption:', caption.lower())
    return tokens_positive, noun_phrases


def get_rec_phrase(caption: str):
    import nltk
    from nltk import pos_tag, word_tokenize
    from nltk.tree import Tree
    from nltk.corpus import stopwords
    # 下载nltk所需的数据
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('punkt')
    # nltk.download('stopwords')
    tokens = word_tokenize(caption)
    pos_tags = pos_tag(tokens)
    
    # 定义简单的语法，名词性短语NP作为主语
    grammar = r"""
      NP: {<DT>?<JJ>*<NN|NNS>}   # 定义名词性短语
    """
    
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(pos_tags)

    # 遍历解析树，寻找名词短语NP
    noun_phrases = None
    for subtree in tree:
        if isinstance(subtree, Tree) and subtree.label() == 'NP':
            subject = " ".join([word for word, pos in subtree.leaves()])
            if subject.lower() not in stopwords.words('english'):  # 排除停用词
                # print("主语:", subject)
                noun_phrases = subject
                break
    if noun_phrases is None:
        noun_phrases = caption
    tokens_positive = []
    for m in re.finditer(noun_phrases, caption):
        tokens_positive.append([[m.start(), m.end()]])
    if len(tokens_positive) == 0:
        return [[[0, len(caption)-1]]], [caption]
    
    return tokens_positive, [noun_phrases]



def create_positive_map(tokenized,
                        tokens_positive: list,
                        max_num_entities: int = 256) -> Tensor:
    """construct a map such that positive_map[i,j] = True
    if box i is associated to token j

    Args:
        tokenized: The tokenized input.
        tokens_positive (list): A list of token ranges
            associated with positive boxes.
        max_num_entities (int, optional): The maximum number of entities.
            Defaults to 256.

    Returns:
        torch.Tensor: The positive map.

    Raises:
        Exception: If an error occurs during token-to-char mapping.
    """
    positive_map = torch.zeros((len(tokens_positive), max_num_entities),
                               dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print('beg:', beg, 'end:', end)
                print('token_positive:', tokens_positive)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos:end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def create_positive_map_label_to_token(positive_map: Tensor,
                                       plus: int = 0) -> dict:
    """Create a dictionary mapping the label to the token.
    Args:
        positive_map (Tensor): The positive map tensor.
        plus (int, optional): Value added to the label for indexing.
            Defaults to 0.

    Returns:
        dict: The dictionary mapping the label to the token.
    """
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(
            positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token



def clean_label_name(name: str) -> str:
    name = re.sub(r'\(.*\)', '', name)
    name = re.sub(r'_', ' ', name)
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """Yield successive n-sized chunks from lst."""
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class GroundingDINO(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self,
                 lmm=None,
                 lmm_max_token_length=512,
                 lmm_layers=1,
                 lmm_region_loss_weight=1.0,
                 lmm_image_loss_weight=1.0,
                 lmm_connector=None,
                 lmm_connector_prefix='mm_projector', # 'vision_projector.0'
                 pretrain_ckpt=None,
                 freeze_backbone=False,
                 freeze_lm=False,
                 language_model=None,
                 use_dn=True,
                 use_plora=False,
                 use_lora=False,
                 use_mogae=True,   
                 use_lmm_cross_attn=False,
                 num_lmm_new_layers=0,
                 lmm_new_layer_insert_type='all',
                 feature_map_size=27,
                 lora_r=64, 
                 lora_alpha=128, 
                 lora_dropout=0,
                 mogae_bottleneck=256,
                 mogae_dropout=0.0,
                 mogae_num_adapters=4,
                 mogae_router_input_dim=None,
                 mogae_adapter_scalar="learnable_scalar",
                 mogae_adapter_layernorm_option="none",
                 mogae_sparse=True,
                 mogae_init_option="lora",
                 use_constrast_conv=False,
                 fsdp=False,
                 num_region_caption=0,
                 use_p5_input=True,
                 use_p4_input=True,
                 use_vision_mask=False,  #改
                 vision_mask_level=-2,    #改  取feature的倒数第二层
                 use_text_guided_refine=False,  #改
                 text_guided_level=-2,  #改
                 text_guided_cfg=None,   #改
                 use_tgcr_loss=False,
                 tgcr_loss_weight=1e-5,
                 use_query_input=False,
                 use_image_level_cross_attn=False,
                 use_pretrained_projector=True,
                 use_prompt_adapter=False,
                 prompt_adapter_cfg=None,
                 prompt_adapter_importance_loss_weight=0.0,
                 prompt_adapter_detach_inputs=False,
                 mini_query=False,
                 vis=False,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.lmm = lmm
        self.lmm_max_token_length = lmm_max_token_length
        self.lmm_layers = lmm_layers
        self.lmm_region_loss_weight = lmm_region_loss_weight
        self.lmm_image_loss_weight = lmm_image_loss_weight
        self.use_vision_mask = use_vision_mask   #改
        self.vision_mask_level = vision_mask_level  #改
        self.use_text_guided_refine = use_text_guided_refine  #改
        self.text_guided_level = text_guided_level    #改
        self.text_guided_cfg = {} if text_guided_cfg is None else copy.deepcopy(text_guided_cfg)  #改
        self.use_tgcr_loss = use_tgcr_loss
        self.tgcr_loss_weight = tgcr_loss_weight
        self.freeze_backbone = freeze_backbone
        self.freeze_lm = freeze_lm
        self.num_region_caption = num_region_caption
        self.use_lmm_cross_attn = use_lmm_cross_attn
        self.use_plora = use_plora
        self.use_mogae = use_mogae
        self.feature_map_size = feature_map_size
        self.use_constrast_conv = use_constrast_conv
        self.vis = vis
        self.use_p5_input = use_p5_input
        self.use_p4_input = use_p4_input
        self.use_query_input = use_query_input
        self.use_image_level_cross_attn = use_image_level_cross_attn
        self.use_prompt_adapter = use_prompt_adapter
        self.prompt_adapter_cfg = (
            {} if prompt_adapter_cfg is None else copy.deepcopy(prompt_adapter_cfg)
        )
        self.prompt_adapter_importance_loss_weight = (
            prompt_adapter_importance_loss_weight
        )
        self.prompt_adapter_detach_inputs = prompt_adapter_detach_inputs
        self.latest_prompt_adapter_outputs = None
        self.latest_prompt_adapter_importance_loss = None
        self.mini_query = mini_query
        super().__init__(*args, **kwargs)
        self.bbox_head.use_dn = use_dn
        self.use_dn = use_dn
        if not use_dn:
            self.dn_query_generator.requires_grad_(False)
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.neck.requires_grad_(False)
        if self.freeze_lm:
            self.language_model.requires_grad_(False)
        if pretrain_ckpt is not None:
            state_dict = torch.load(pretrain_ckpt, 'cpu')['state_dict']
            msg = self.load_state_dict(state_dict, False)
            print(msg)
        if lmm is not None:
            from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
            from llava.model.multimodal_projector.builder import vision_projector_with_pos_proj
            if fsdp:
                self.lmm = LlavaQwenForCausalLM.from_pretrained(lmm)
            else:
                self.lmm = LlavaQwenForCausalLM.from_pretrained(lmm).half()
            self.lmm = checkpoint_wrapper(self.lmm)
            self.lmm.requires_grad_(False)
            self.lmm.config.use_cache = False

            if self.use_constrast_conv:
                self.img_sep = nn.Embedding(1, self.lmm.config.hidden_size)

            if use_plora:
                self.lmm.model.build_plora(num_lmm_new_layers, lmm_new_layer_insert_type, lora_r, lora_alpha, lora_dropout)
            if use_lora:
                self.lmm.model.build_lora(lora_r, lora_alpha, lora_dropout)
                # self.lmm.enable_input_require_grads()
            if use_mogae and num_lmm_new_layers > 0:
                router_input_dim = (
                    self.lmm.config.hidden_size
                    if mogae_router_input_dim is None
                    else mogae_router_input_dim
                )
                self.lmm.model.build_mogae(
                    num_layers=num_lmm_new_layers,
                    insert_type=lmm_new_layer_insert_type,
                    bottleneck=mogae_bottleneck,
                    dropout=mogae_dropout,
                    init_option=mogae_init_option,
                    adapter_scalar=mogae_adapter_scalar,
                    adapter_layernorm_option=mogae_adapter_layernorm_option,
                    num_adapters=mogae_num_adapters,
                    router_input_dim=router_input_dim,
                    sparse=mogae_sparse,
                )
            if (use_lmm_cross_attn and num_region_caption > 0) or use_image_level_cross_attn:
                self.lmm.model.build_cross_attention(num_lmm_new_layers, lmm_new_layer_insert_type, self.num_feature_levels)
                self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)

            self.lmm_tokenizer = transformers.AutoTokenizer.from_pretrained(
                lmm,
                cache_dir=None, 
                model_max_length=self.lmm_max_token_length, 
                padding_side="right")
            if hasattr(self.lmm.model, 'mm_projector'):
                del self.lmm.model.mm_projector
            if hasattr(self.lmm.model, 'vision_tower'):
                del self.lmm.model.vision_tower
            
            self.lmm.config.tokenizer_padding_side = self.lmm_tokenizer.padding_side
            self.lmm.config.tokenizer_model_max_length = self.lmm_max_token_length

            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', 'mlp2x_gelu')
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(256, self.lmm.config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(self.lmm.config.hidden_size, self.lmm.config.hidden_size))
                vision_projector = nn.Sequential(*modules)
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            vision_projector = vision_projector_with_pos_proj(self.lmm.config.hidden_size, [vision_projector])
            mm_projector_weights = torch.load(lmm_connector, map_location='cpu')
            if use_pretrained_projector:
                vision_projector.load_state_dict(get_w(mm_projector_weights, lmm_connector_prefix))
            # torch.nn.init.zeros_(vision_projector.pos_proj.weight)
            # torch.nn.init.zeros_(vision_projector.pos_proj.bias)
            if self.use_p5_input or self.use_p4_input:
                self.connector = vision_projector
            if self.num_region_caption > 0 or self.use_query_input:
                self.region_connector = copy.deepcopy(vision_projector)
            if self.use_prompt_adapter:
                prompt_cfg = copy.deepcopy(self.prompt_adapter_cfg)
                prompt_cfg.setdefault('vis1_dim', self.embed_dims)
                prompt_cfg.setdefault('vis2_dim', self.embed_dims)
                prompt_cfg.setdefault('text_dim', self.lmm.config.hidden_size)
                prompt_cfg.setdefault('llm_dim', self.lmm.config.hidden_size)
                self.prompt_adapter = MoAELLMPromptAdapter(**prompt_cfg)
                # Start from the original behavior and let training enable the
                # prompt adapter only when it becomes useful.
                self.prompt_adapter_alpha = nn.Parameter(torch.tensor(0.0))
            
            yv, xv = torch.meshgrid([torch.range(0, 1, 1/self.feature_map_size), torch.range(0, 1, 1/self.feature_map_size)])
            grid = torch.stack((xv, yv), 2).view(self.feature_map_size+1, self.feature_map_size+1, 2)
            self.grid_box = torch.cat([grid[:-1, :-1], grid[1:, 1:]], dim=-1).flatten(0, 1)

            if self.use_p4_input:
                self.image_seperate = nn.Embedding(1, self.lmm.config.hidden_size)
                yv, xv = torch.meshgrid([torch.range(0, 1, 1/20), torch.range(0, 1, 1/20)])
                grid = torch.stack((xv, yv), 2).view(20+1, 20+1, 2)
                self.p5_grid_box = torch.cat([grid[:-1, :-1], grid[1:, 1:]], dim=-1).flatten(0, 1)

            self.img_idx = 0


    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        #改  实例化
        if self.use_vision_mask:
            self.vision_mask = VisionMask(
                img_dim=self.embed_dims,
                text_dim=self.language_model.language_backbone.body.language_dim,
                embed_dim=self.embed_dims,
            )
        if self.use_text_guided_refine:   #改  实例化
            tg_cfg = copy.deepcopy(self.text_guided_cfg)
            tg_cfg.setdefault('d_model', self.embed_dims)
            tg_cfg.setdefault('channel_first', False)
            tg_cfg.setdefault('forward_type', 'm0')
            tg_cfg.setdefault('with_initial_state', False)
            self.text_guided_refine = TextGuidedSS2D2(**tg_cfg)
            # Start from the original behavior and let training learn how much
            # text-guided refinement to use.
            self.text_guided_alpha = nn.Parameter(torch.tensor(0.0))
        #改#改



    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)
###############改
    def _prepare_text_dict(self, text_dict: Dict) -> Dict:
        """Keep the raw BERT features for VisionMask and map text to detector dim."""
        if self.use_vision_mask and 'raw_embedded' not in text_dict:
            text_dict['raw_embedded'] = text_dict['embedded']
        if self.text_feat_map is not None:
            mapped = self.text_feat_map(text_dict['embedded'])
            text_dict['embedded'] = mapped
            if self.use_text_guided_refine:
                text_dict['embedded_for_guidance'] = (
                    mapped *
                    text_dict['text_token_mask'].unsqueeze(-1).to(mapped.dtype))
        return text_dict

    def _apply_vision_mask(self, encoder_outputs_dict: Dict,
                           text_dict: Dict) -> Dict:
        """Enhance one encoder feature level with text-guided spatial gating."""
        if not self.use_vision_mask:
            return encoder_outputs_dict

        memory = encoder_outputs_dict['memory']
        memory_mask = encoder_outputs_dict['memory_mask']
        spatial_shapes = encoder_outputs_dict['spatial_shapes']
        raw_text_embeds = text_dict['raw_embedded']
        text_token_mask = text_dict['text_token_mask']

        num_levels = spatial_shapes.shape[0]
        level_idx = self.vision_mask_level
        if level_idx < 0:
            level_idx = num_levels + level_idx
        if level_idx < 0 or level_idx >= num_levels:
            raise ValueError(
                f'vision_mask_level {self.vision_mask_level} is out of range '
                f'for {num_levels} feature levels')

        level_token_counts = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        start = int(sum(level_token_counts[:level_idx]))
        end = int(start + level_token_counts[level_idx])
        H = int(spatial_shapes[level_idx, 0].item())
        W = int(spatial_shapes[level_idx, 1].item())

        level_memory = memory[:, start:end, :]
        level_memory = level_memory.transpose(1, 2).reshape(
            memory.shape[0], memory.shape[-1], H, W)

        level_pad_mask = None
        if memory_mask is not None:
            level_pad_mask = memory_mask[:, start:end].reshape(
                memory.shape[0], 1, H, W)
            level_memory = level_memory.masked_fill(level_pad_mask, 0)

        guided_vision_feature, routing_mask, token_attention = self.vision_mask(
            level_memory, raw_text_embeds, text_token_mask)
        guided_flat = guided_vision_feature.flatten(2).transpose(1, 2)

        if level_pad_mask is not None:
            guided_flat = torch.where(
                level_pad_mask.flatten(2).transpose(1, 2),
                memory[:, start:end, :], guided_flat)

        new_memory = memory.clone()
        new_memory[:, start:end, :] = guided_flat
        encoder_outputs_dict['memory'] = new_memory
        self.latest_routing_mask = routing_mask
        self.latest_token_attention = token_attention
        return encoder_outputs_dict

    def _apply_text_guided_refine(self, encoder_outputs_dict: Dict,
                                  text_dict: Dict) -> Dict:
        """Refine one encoder feature level with TextGuidedSS2D2."""
        if not self.use_text_guided_refine:
            return encoder_outputs_dict

        memory = encoder_outputs_dict['memory']
        memory_mask = encoder_outputs_dict['memory_mask']
        spatial_shapes = encoder_outputs_dict['spatial_shapes']
        guide = text_dict['embedded_for_guidance']

        num_levels = spatial_shapes.shape[0]
        level_idx = self.text_guided_level
        if level_idx < 0:
            level_idx = num_levels + level_idx
        if level_idx < 0 or level_idx >= num_levels:
            raise ValueError(
                f'text_guided_level {self.text_guided_level} is out of range '
                f'for {num_levels} feature levels')

        level_token_counts = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
        start = int(sum(level_token_counts[:level_idx]))
        end = int(start + level_token_counts[level_idx])
        H = int(spatial_shapes[level_idx, 0].item())
        W = int(spatial_shapes[level_idx, 1].item())

        level_memory = memory[:, start:end, :].contiguous().view(
            memory.shape[0], H, W, memory.shape[-1])

        level_pad_mask = None
        if memory_mask is not None:
            level_pad_mask = memory_mask[:, start:end].contiguous().view(
                memory.shape[0], H, W, 1)
            level_memory = level_memory.masked_fill(level_pad_mask, 0)

        guided_out, _ = self.text_guided_refine(level_memory, guide)
        blended = level_memory + self.text_guided_alpha * (
            guided_out - level_memory)
        guided_flat = blended.contiguous().view(
            memory.shape[0], H * W, memory.shape[-1])

        if level_pad_mask is not None:
            guided_flat = torch.where(
                level_pad_mask.view(memory.shape[0], H * W, 1),
                memory[:, start:end, :], guided_flat)

        new_memory = memory.clone()
        new_memory[:, start:end, :] = guided_flat
        encoder_outputs_dict['memory'] = new_memory
        self.latest_text_guided_out = guided_out
        return encoder_outputs_dict

    def _loss_text_guided_consistency(self, visual_memory: Tensor,
                                      text_dict: Dict,
                                      memory_mask: Optional[Tensor] = None
                                      ) -> Tensor:
        """Align pooled visual memory with pooled text guidance features."""
        text_features = text_dict.get('embedded_for_guidance',
                                      text_dict.get('embedded', None))
        if text_features is None or visual_memory.shape[-1] != text_features.shape[-1]:
            return visual_memory.sum() * 0

        if memory_mask is None:
            visual_mask = visual_memory.new_ones(
                visual_memory.shape[:2] + (1,))
        else:
            visual_mask = (~memory_mask).unsqueeze(-1).to(visual_memory.dtype)
        visual_pool = (visual_memory * visual_mask).sum(dim=1)
        visual_pool = visual_pool / visual_mask.sum(dim=1).clamp_min(1.0)

        text_mask = text_dict['text_token_mask'].unsqueeze(-1).to(
            text_features.dtype)
        text_pool = (text_features * text_mask).sum(dim=1)
        text_pool = text_pool / text_mask.sum(dim=1).clamp_min(1.0)

        visual_pool = F.normalize(visual_pool.float(), dim=-1)
        text_pool = F.normalize(text_pool.detach().float(), dim=-1)
        return 1 - (visual_pool * text_pool).sum(dim=-1).mean()

    def _resolve_level_index(self, level_idx: int, num_levels: int) -> int:
        if level_idx < 0:
            level_idx = num_levels + level_idx
        if level_idx < 0 or level_idx >= num_levels:
            raise ValueError(
                f'level_idx {level_idx} is out of range for {num_levels} levels')
        return level_idx

    def _pool_memory_level(self, memory: Tensor, memory_mask: Optional[Tensor],
                           level_start_index: Tensor, level_idx: int) -> Tensor:
        num_levels = level_start_index.shape[0]
        level_idx = self._resolve_level_index(level_idx, num_levels)
        start = int(level_start_index[level_idx].item())
        if level_idx + 1 < num_levels:
            end = int(level_start_index[level_idx + 1].item())
        else:
            end = int(memory.shape[1])

        level_memory = memory[:, start:end, :]
        if memory_mask is None:
            valid_mask = torch.ones(
                level_memory.shape[:2], device=level_memory.device, dtype=torch.bool)
        else:
            valid_mask = ~memory_mask[:, start:end]

        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(level_memory.dtype)
        pooled = (level_memory * valid_mask.unsqueeze(-1).to(level_memory.dtype)).sum(dim=1)
        return pooled / denom

    def _get_image_prompt_texts(self, batch_data_samples: SampleList) -> List[str]:
        prompt_texts = []
        for data_samples in batch_data_samples:
            ori_conv = getattr(data_samples, 'ori_conv', None)
            prompt_text = ''
            if ori_conv is not None and len(ori_conv) > 0:
                prompt_text = ori_conv[0].get('value', '')
            prompt_text = prompt_text.replace('<image>', ' ').strip()
            prompt_text = re.sub(r'\s+', ' ', prompt_text)
            if prompt_text == '':
                prompt_text = 'Describe this image in detail.'
            prompt_texts.append(prompt_text)
        return prompt_texts

    def _pool_image_prompt_text(self, batch_data_samples: SampleList,
                                device: torch.device) -> Tensor:
        prompt_texts = self._get_image_prompt_texts(batch_data_samples)
        tokenized = self.lmm_tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.lmm_max_token_length,
            return_tensors='pt')
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        input_embeds = self.lmm.get_model().embed_tokens(input_ids)
        valid_mask = attention_mask.bool()
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(input_embeds.dtype)
        pooled = (input_embeds * valid_mask.unsqueeze(-1).to(input_embeds.dtype)).sum(dim=1)
        return pooled / denom

    def _build_image_prompt_embeds(self, decoder_inputs_dict: Dict,
                                   batch_data_samples: SampleList) -> Optional[Tensor]:
        self.latest_prompt_adapter_outputs = None
        self.latest_prompt_adapter_importance_loss = None
        if not self.use_prompt_adapter:
            return None

        refined_level = self.text_guided_level if self.use_text_guided_refine else -2
        memory = decoder_inputs_dict['memory']
        memory_mask = decoder_inputs_dict['memory_mask']
        level_start_index = decoder_inputs_dict['level_start_index']
        device = memory.device

        vis_feat1 = self._pool_memory_level(
            memory, memory_mask, level_start_index, refined_level)
        vis_feat2 = self._pool_memory_level(
            memory, memory_mask, level_start_index, -3)
        text_feat = self._pool_image_prompt_text(batch_data_samples, device)

        if self.prompt_adapter_detach_inputs:
            vis_feat1 = vis_feat1.detach()
            vis_feat2 = vis_feat2.detach()
            text_feat = text_feat.detach()

        prompt_outputs = self.prompt_adapter(vis_feat1, vis_feat2, text_feat)
        self.latest_prompt_adapter_outputs = prompt_outputs

        if (self.prompt_adapter_importance_loss_weight > 0
                and prompt_outputs['importance_loss'].detach().item() > 0):
            self.latest_prompt_adapter_importance_loss = (
                prompt_outputs['importance_loss']
                * self.prompt_adapter_importance_loss_weight)
        else:
            self.latest_prompt_adapter_importance_loss = None

        final_prompt = prompt_outputs['final_prompt']
        if final_prompt.shape[1] == 0:
            return None
        return self.prompt_adapter_alpha * final_prompt
#########################改
    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokens_positive, entities = get_rec_phrase(original_caption)
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = self.get_positive_map(
                    tokenized, tokens_positive)
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        encoder_outputs_dict = self._apply_vision_mask(encoder_outputs_dict,
                                                       text_dict)    #进行mask操作
        encoder_outputs_dict = self._apply_text_guided_refine(
            encoder_outputs_dict, text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],   #  memory_text：编码后的文本特征
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,           #  memory：编码后的视觉特征
            memory_mask=feat_mask,   #  memory_text：编码后的文本特征
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training and self.use_dn:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        if 'tokens_positive' in batch_data_samples[0]: # grounding data
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []
            positive_map_dicts = []
            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]
                positive_map_dict, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                positive_map_dicts.append(positive_map_dict)
            new_text_prompts = text_prompts
        else: # detection data
            new_text_prompts = []
            positive_maps = []
            positive_map_dicts = []
            if len(set(text_prompts)) == 1:
                # All the text prompts are the same,
                # so there is no need to calculate them multiple times.
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    positive_map_dict, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    positive_map_dicts.append(positive_map_dict)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    positive_map_dict, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)
                    positive_map_dicts.append(positive_map_dict)

        if self.freeze_backbone:
            with torch.no_grad():
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)
        if self.freeze_lm:
            with torch.no_grad():
                text_dict = self.language_model(new_text_prompts)
        else:
            text_dict = self.language_model(new_text_prompts)
        text_dict = self._prepare_text_dict(text_dict)   #改

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)
        
        head_inputs_dict, decoder_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses, all_stage_assign_result, all_layers_matching_cls_scores, all_layers_matching_bbox_preds = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        if self.use_tgcr_loss:
            loss_tgcr = self._loss_text_guided_consistency(
                decoder_inputs_dict['memory'],
                text_dict,
                decoder_inputs_dict.get('memory_mask', None))
            losses['loss_tgcr'] = loss_tgcr * self.tgcr_loss_weight
        
        if self.lmm is not None:
            # region-level description
            if self.num_region_caption > 0:
                for i in range(1, self.lmm_layers + 1):
                    matching_bbox_preds = all_layers_matching_bbox_preds[-i].clone().detach() # [bs, num_query, 4]
                    matching_bbox_preds = bbox_cxcywh_to_xyxy(matching_bbox_preds)
                    assign_results = all_stage_assign_result[-i]
                    
                    # prepare object queries for LMM input
                    selected_queries, selected_boxes, query_lens = [], [], []
                    for j in range(matching_bbox_preds.shape[0]):
                        if len(batch_data_samples[j].region_conversations['conversations']) == 0:
                            query_lens.append(0)
                            continue
                        
                        index_mappting = torch.zeros(len(batch_data_samples[j].gt_instances.bboxes), device=batch_data_samples[j].gt_instances.bboxes.device, dtype=torch.long)
                        index_mappting[assign_results[j].gt_inds[assign_results[j].gt_inds > 0] - 1] = torch.arange(0, len(assign_results[j].gt_inds), device=index_mappting.device)[assign_results[j].gt_inds > 0]
                        query_index = index_mappting[batch_data_samples[j].region_conversations['box_index']]
                        matched_query_boxes = matching_bbox_preds[j][query_index]
                        features = head_inputs_dict['hidden_states'][-i][j][query_index]
                        selected_boxes.append(matched_query_boxes)
                        selected_queries.append(features)
                        query_lens.append(len(features))
                    
                    selected_boxes = torch.cat(selected_boxes).unsqueeze(1) # (total_queries, 1, 4)
                    selected_queries = torch.cat(selected_queries).unsqueeze(1)
                    if len(selected_queries) == 0:
                        losses[f'd{i}.loss_lmm_region'] = torch.sum(head_inputs_dict['hidden_states'][-i]) * 0
                        continue

                    cross_attention_input = None
                    if self.use_lmm_cross_attn:
                        repeat_num = torch.tensor(query_lens, device=selected_queries.device)
                        new_valid_ratios = torch.repeat_interleave(decoder_inputs_dict['valid_ratios'], repeat_num, dim=0)
                        new_memory = torch.repeat_interleave(decoder_inputs_dict['memory'], repeat_num, dim=0)
                        if decoder_inputs_dict['memory_mask'] is not None:
                            new_memory_mask = torch.repeat_interleave(decoder_inputs_dict['memory_mask'], repeat_num, dim=0)
                        else:
                            new_memory_mask = decoder_inputs_dict['memory_mask']
                        
                        reference_points = box_xyxy_to_cxcywh(selected_boxes)
                        reference_points_input = \
                            reference_points[:, :, None] * torch.cat(
                                [new_valid_ratios, new_valid_ratios], -1)[:, None]

                        query_sine_embed = coordinate_to_encoding(
                            reference_points_input[:, :, 0, :])
                        query_pos = self.ref_point_head(query_sine_embed)

                        cross_attention_input = {
                            'query_pos': query_pos,
                            'memory': new_memory,
                            'memory_mask': new_memory_mask,
                            'spatial_shapes': decoder_inputs_dict['spatial_shapes'],  # batch size irrelevant
                            'level_start_index': decoder_inputs_dict['level_start_index'],  # batch size irrelevant
                            'reference_points_input': reference_points_input,
                            'half': False,
                        }

                    features = self.region_connector([selected_queries])
                    features = features + self.region_connector.forward_pos(gen_sineembed_for_position(selected_boxes)).half()
                    
                    # prepare LMM input
                    region_conversations = []
                    for data_samples in batch_data_samples:
                        region_conversations += data_samples.region_conversations['conversations']
                    input_ids = [
                        torch.tensor(conv['input_id'], dtype=torch.long, device=head_inputs_dict['hidden_states'].device) for conv in region_conversations
                    ]
                    labels = [
                        torch.tensor(conv['label'], dtype=torch.long, device=head_inputs_dict['hidden_states'].device) for conv in region_conversations
                    ]
                    if self.lmm_tokenizer.pad_token_id is None:
                    # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
                        self.lmm_tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
                    input_ids = torch.nn.utils.rnn.pad_sequence(
                        input_ids,
                        batch_first=True,
                        padding_value=self.lmm_tokenizer.pad_token_id)
                    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                                batch_first=True,
                                                                padding_value=IGNORE_INDEX)
                    input_ids = input_ids[:, :40]
                    labels = labels[:, :40]
                    attention_mask=input_ids.ne(self.lmm_tokenizer.pad_token_id)
                    lmm_imput_dict = dict(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    
                    lmm_imput_dict['input_ids'] = lmm_imput_dict['input_ids'].to(features.device)
                    lmm_imput_dict['labels'] = lmm_imput_dict['labels'].to(features.device)
                    lmm_imput_dict['attention_mask'] = lmm_imput_dict['attention_mask'].to(features.device)
                    lmm_imput_dict['image_queries'] = features
                    lmm_imput_dict['query_masks'] = torch.ones((len(input_ids), 1), device=features.device, dtype=torch.bool)
                    lmm_imput_dict['cross_attention_input'] = cross_attention_input
                    self.lmm.eval()
                    with autocast(enabled=True):
                        loss_lmm = self.lmm.detection_forward(**lmm_imput_dict)
                    losses[f'd{i}.loss_lmm_region'] = loss_lmm.loss * self.lmm_region_loss_weight

            if self.use_p5_input:
                # image-level description
                input_ids = [
                    torch.tensor(data_samples.conversations['input_id'], dtype=torch.long, device=head_inputs_dict['hidden_states'].device) for data_samples in batch_data_samples
                ]
                labels = [
                    torch.tensor(data_samples.conversations['label'], dtype=torch.long, device=head_inputs_dict['hidden_states'].device) for data_samples in batch_data_samples
                ]
                if self.use_constrast_conv:
                    constrast_input_ids = [
                        torch.tensor(data_samples.contrast_conv[1]['input_id'], dtype=torch.long, device=head_inputs_dict['hidden_states'].device) for data_samples in batch_data_samples if 'contrast_conv' in data_samples
                    ]
                    constrast_labels = [
                        torch.tensor(data_samples.contrast_conv[1]['label'], dtype=torch.long, device=head_inputs_dict['hidden_states'].device) for data_samples in batch_data_samples if 'contrast_conv' in data_samples
                    ]
                    input_ids += constrast_input_ids
                    labels += constrast_labels
                if self.lmm_tokenizer.pad_token_id is None:
                # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
                    self.lmm_tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    input_ids,
                    batch_first=True,
                    padding_value=self.lmm_tokenizer.pad_token_id)
                labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                            batch_first=True,
                                                            padding_value=IGNORE_INDEX)
                input_ids = input_ids[:, :self.lmm_max_token_length]
                labels = labels[:, :self.lmm_max_token_length]
                attention_mask=input_ids.ne(self.lmm_tokenizer.pad_token_id)
                lmm_imput_dict = dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )

                image_queries = []
                query_masks = []
                cross_attention_input = None
                        
                if not self.use_p4_input:
                    for i in range(len(decoder_inputs_dict['memory'])):
                        p5_feature_map = decoder_inputs_dict['memory'][i][decoder_inputs_dict['level_start_index'][-2]: decoder_inputs_dict['level_start_index'][-1]]
                        H, W = int(decoder_inputs_dict['spatial_shapes'][-2][0].data), int(decoder_inputs_dict['spatial_shapes'][-2][1].data)
                        p5_feature_map = p5_feature_map.reshape(H, W, decoder_inputs_dict['memory'].shape[-1]).permute(2, 0, 1)
                        p5_feature_map = p5_feature_map[:, :int(torch.round(H * decoder_inputs_dict['valid_ratios'][i, -2, 1])), :int(torch.round(W * decoder_inputs_dict['valid_ratios'][i, -2, 0]))]
                        p5_feature_map = F.interpolate(p5_feature_map[None], size=(self.feature_map_size, self.feature_map_size), mode='bilinear')[0]
                        # if flips[i]:
                        #     p5_feature_map = p5_feature_map.flip(-1)
                        p5_feature_map = self.connector([p5_feature_map.permute(1, 2, 0)]).flatten(0, 1)
                        p5_feature_map = p5_feature_map + self.connector.forward_pos(gen_sineembed_for_position_2d(self.grid_box.to(p5_feature_map.device)))
                        image_queries.append(p5_feature_map.half())
                        query_masks.append(torch.ones((len(image_queries[-1])), device=p5_feature_map.device, dtype=torch.bool))

                else:
                    for i in range(len(decoder_inputs_dict['memory'])):
                        p5_feature_map = decoder_inputs_dict['memory'][i][decoder_inputs_dict['level_start_index'][-2]: decoder_inputs_dict['level_start_index'][-1]]
                        H, W = int(decoder_inputs_dict['spatial_shapes'][-2][0].data), int(decoder_inputs_dict['spatial_shapes'][-2][1].data)
                        p5_feature_map = p5_feature_map.reshape(H, W, decoder_inputs_dict['memory'].shape[-1]).permute(2, 0, 1)
                        p5_feature_map = p5_feature_map[:, :int(torch.round(H * decoder_inputs_dict['valid_ratios'][i, -2, 1])), :int(torch.round(W * decoder_inputs_dict['valid_ratios'][i, -2, 0]))]
                        p5_feature_map = F.interpolate(p5_feature_map[None], size=(20, 20), mode='bilinear')[0]
                        # if flips[i]:
                        #     p5_feature_map = p5_feature_map.flip(-1)
                        p5_feature_map = self.connector([p5_feature_map.permute(1, 2, 0)]).flatten(0, 1)
                        p5_feature_map = p5_feature_map + self.connector.forward_pos(gen_sineembed_for_position_2d(self.p5_grid_box.to(p5_feature_map.device)))

                        p4_feature_map = decoder_inputs_dict['memory'][i][decoder_inputs_dict['level_start_index'][-3]: decoder_inputs_dict['level_start_index'][-2]]
                        H, W = int(decoder_inputs_dict['spatial_shapes'][-3][0].data), int(decoder_inputs_dict['spatial_shapes'][-3][1].data)
                        p4_feature_map = p4_feature_map.reshape(H, W, decoder_inputs_dict['memory'].shape[-1]).permute(2, 0, 1)
                        p4_feature_map = p4_feature_map[:, :int(torch.round(H * decoder_inputs_dict['valid_ratios'][i, -3, 1])), :int(torch.round(W * decoder_inputs_dict['valid_ratios'][i, -3, 0]))]
                        p4_feature_map = F.interpolate(p4_feature_map[None], size=(self.feature_map_size, self.feature_map_size), mode='bilinear')[0]
                        # if flips[i]:
                        #     p4_feature_map = p4_feature_map.flip(-1)
                        p4_feature_map = self.connector([p4_feature_map.permute(1, 2, 0)]).flatten(0, 1)
                        p4_feature_map = p4_feature_map + self.connector.forward_pos(gen_sineembed_for_position_2d(self.grid_box.to(p4_feature_map.device)))

                        image_queries.append(torch.cat([p5_feature_map, self.image_seperate.weight, p4_feature_map]).half())
                        query_masks.append(torch.ones((len(image_queries[-1])), device=p5_feature_map.device, dtype=torch.bool))


                if self.use_image_level_cross_attn:
                    reference_points_input = self.grid_box.clone().unsqueeze(0).repeat(len(decoder_inputs_dict['memory']), 1, 1).to(decoder_inputs_dict['valid_ratios'].dtype).to(decoder_inputs_dict['valid_ratios'].device)
                    reference_points_input = reference_points_input.unsqueeze(2).repeat(1, 1, decoder_inputs_dict['valid_ratios'].shape[1], 1)
                    # reference_points_input = \
                    #     reference_points[:, :, None] * decoder_inputs_dict['valid_ratios'][:, None]

                    query_sine_embed = coordinate_to_encoding(
                        reference_points_input[:, :, 0, :])
                    query_pos = self.ref_point_head(query_sine_embed)
                    cross_attention_input = {
                            'query_pos': query_pos,
                            'memory': decoder_inputs_dict['memory'],
                            'memory_mask': decoder_inputs_dict['memory_mask'],
                            'spatial_shapes': decoder_inputs_dict['spatial_shapes'],
                            'level_start_index': decoder_inputs_dict['level_start_index'],
                            'reference_points_input': reference_points_input,
                            'half': False,
                        }

                lmm_imput_dict['input_ids'] = lmm_imput_dict['input_ids'].to(decoder_inputs_dict['memory'].device)
                lmm_imput_dict['labels'] = lmm_imput_dict['labels'].to(decoder_inputs_dict['memory'].device)
                lmm_imput_dict['attention_mask'] = lmm_imput_dict['attention_mask'].to(decoder_inputs_dict['memory'].device)
                lmm_imput_dict['image_queries'] = image_queries
                lmm_imput_dict['query_masks'] = query_masks
                lmm_imput_dict['cross_attention_input'] = cross_attention_input
                prompt_embeds = self._build_image_prompt_embeds(
                    decoder_inputs_dict, batch_data_samples)
                if prompt_embeds is not None:
                    lmm_imput_dict['prompt_embeds'] = prompt_embeds
                self.lmm.eval()
                with autocast(enabled=True):
                    loss_lmm = self.lmm.detection_forward(**lmm_imput_dict)
                losses['loss_lmm_image'] = loss_lmm.loss * self.lmm_image_loss_weight
                if self.latest_prompt_adapter_importance_loss is not None:
                    losses['loss_prompt_importance'] = (
                        self.latest_prompt_adapter_importance_loss)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        is_rec_tasks = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positive = data_samples.get('tokens_positive', None)
            is_rec_tasks.append(tokens_positive == -1)
            tokens_positives.append(tokens_positive)

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                text_dict = self._prepare_text_dict(text_dict)

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict, decoder_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)
            results_list = [results_list[0].cat(results_list)]
            # is_rec_tasks = [False] * len(results_list)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            text_dict = self._prepare_text_dict(text_dict)

            # is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                # if token_positive_maps[i] is not None:
                #     is_rec_tasks.append(False)
                # else:
                #     is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict, decoder_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        for i, (data_sample, pred_instances, entity, is_rec_task) in enumerate(zip(
                batch_data_samples, results_list, entities, is_rec_tasks)):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(text_prompts[i])
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
                if is_rec_task:
                    H, W = data_sample.ori_shape
                    bias = 0
                    if 'left' in data_sample.text:
                        centers = (pred_instances.bboxes[:, 2] + pred_instances.bboxes[:, 0]) / 2 / W
                        bias = bias + (1 - centers) * 0.7 + 1
                    if 'right' in data_sample.text:
                        centers = (pred_instances.bboxes[:, 2] + pred_instances.bboxes[:, 0]) / 2 / W
                        bias = bias + centers * 0.7 + 1
                    if 'top' in data_sample.text or 'upper' in data_sample.text:
                        centers = (pred_instances.bboxes[:, 3] + pred_instances.bboxes[:, 1]) / 2 / H
                        bias = bias + (1 - centers) * 0.7 + 1
                    if 'under' in data_sample.text or 'bottom' in data_sample.text:
                        centers = (pred_instances.bboxes[:, 3] + pred_instances.bboxes[:, 1]) / 2 / H
                        bias = bias + centers * 0.7 + 1

                    scores = pred_instances.scores + bias
                    scores, inds = torch.sort(scores, descending=True)
                    pred_instances.scores = scores
                    pred_instances.bboxes = pred_instances.bboxes[inds]
            data_sample.pred_instances = pred_instances
            data_sample.object_query = head_inputs_dict['hidden_states'][-1][i]

        return batch_data_samples
    

    def predict_encoder_only(self, batch_inputs, batch_data_samples, resize: bool = True):
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        if isinstance(text_prompts[0], list):
            # chunked text prompts, only bs=1 is supported
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            entities = [[item for lst in entities[0] for item in lst]]

            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]
                text_dict = self.language_model(text_prompts_once)
                text_dict = self._prepare_text_dict(text_dict)

                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                head_inputs_dict, decoder_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))
            text_dict = self._prepare_text_dict(text_dict)

            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            head_inputs_dict, decoder_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)
        
        new_selected_queries = []
        for k, data_sample in enumerate(batch_data_samples):
            p5_feature_map = decoder_inputs_dict['memory'][k][decoder_inputs_dict['level_start_index'][2]: decoder_inputs_dict['level_start_index'][3]]
            H, W = int(decoder_inputs_dict['spatial_shapes'][2][0].data), int(decoder_inputs_dict['spatial_shapes'][2][1].data)
            p5_feature_map = p5_feature_map.reshape(H, W, decoder_inputs_dict['memory'].shape[-1]).permute(2, 0, 1)
            p5_feature_map = p5_feature_map[:, :int(torch.round(H * decoder_inputs_dict['valid_ratios'][k, 2, 1])), :int(torch.round(W * decoder_inputs_dict['valid_ratios'][k, 2, 0]))]
            if resize:
                p5_feature_map = F.interpolate(p5_feature_map[None], size=(27, 27), mode='bilinear')[0].flatten(1).permute(1, 0)
                new_selected_queries.append(p5_feature_map)
            else:
                new_selected_queries.append(p5_feature_map)
            
        return new_selected_queries
