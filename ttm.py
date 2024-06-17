"""Token Turing Machines.
https://arxiv.org/abs/2211.09119
https://github.com/google-research/scenic/tree/main/scenic/projects/token_turing

Memorizing Transformers.
https://github.com/lucidrains/memorizing-transformers-pytorch
https://arxiv.org/pdf/2203.08913.pdf 
"""
import torch
from torch import nn
from typing import Callable, Optional, Sequence, Tuple

class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses a MLP with gelu inbetween. It also uses softmax
    instead of sigmoid. We confirmed that this version works better in general.

    Attributes:
        num_tokens: Number of tokens.
        bottleneck_dim: The size of hidden units in the MLP for spatial attention.
        dropout_rate: Dropout rate.
    """
    def __init__(self, num_tokens: int, input_shape: int, 
                 bottleneck_dim: int = 64, dropout_rate: float = 0.) -> None:
        super().__init__()  
        self.mlp1 = MLPBlock(in_dim=input_shape, mlp_dim=bottleneck_dim, dropout=dropout_rate, out_dim=num_tokens)
        self.layernorm = nn.LayerNorm(input_shape)

    def forward(self, inputs):
        """Applies learnable tokenization to the 2D inputs.
        Args:
            inputs: Inputs of shape `[bs, hw, c]`.
        Returns:
            Output of shape `[bs, n_token, c]`.
        """

        selected = self.layernorm(inputs)
        selected = self.mlp1(selected)
        selected = torch.transpose(selected, 1, 2)  # Shape: [bs, n_token, hw].
        selected = torch.softmax(selected, dim=-1)

        feat = torch.einsum('...si,...id->...sd', selected, inputs)
        return feat

class MLPBlock(nn.Sequential):
    """Transformer MLP block."""
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, out_dim: Optional[int] = None):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        if out_dim is not None:
            self.linear_2 = nn.Linear(mlp_dim, out_dim)
        else:
            self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)

class TokenAddEraseWrite(nn.Module):
    def __init__(self, num_tokens: int = 8, bottleneck_dim: int = 64, dropout_rate: float = 0., 
                 input_dim: int = 768, memory_size: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate

        self.layernorm1 = nn.LayerNorm(input_dim)
        self.mlp1 = MLPBlock(in_dim=input_dim, mlp_dim=self.bottleneck_dim, dropout=self.dropout_rate, out_dim=self.num_tokens)
        
        self.layernorm2 = nn.LayerNorm(input_dim)
        self.mlp2 = MLPBlock(in_dim=num_tokens, mlp_dim=self.bottleneck_dim, dropout=self.dropout_rate, out_dim=self.num_tokens)
        self.mlp3 = MLPBlock(in_dim=input_dim, mlp_dim=self.bottleneck_dim, dropout=self.dropout_rate)
        
        self.layernorm3 = nn.LayerNorm(input_dim)
        self.mlp4 = MLPBlock(in_dim=num_tokens, mlp_dim=self.bottleneck_dim, dropout=self.dropout_rate, out_dim=self.num_tokens)
        self.mlp5 = MLPBlock(in_dim=input_dim, mlp_dim=self.bottleneck_dim, dropout=self.dropout_rate)

    def forward(self, memory, control_inputs, training=False):
        """
        Args:
            memory: Inputs of shape `[bs, memory_size, c]`.
            control_inputs: Inputs of shape `[bs, memory_size + input_size, c]`.
        """
        # print(memory.shape, control_inputs.shape, self.num_tokens)
        selected = self.layernorm1(memory)
        # print(selected.shape)
        selected = self.mlp1(selected)
        # print(selected.shape)
        selected = torch.transpose(selected, 1, 2) # Shape: [bs, n_token, hw].
        selected = torch.softmax(selected, dim=-1)
        # print(selected.shape)

        et = self.layernorm2(control_inputs)
        # print(et.shape)
        et = torch.transpose(et, 1, 2) # Shape: [bs, c, hw].
        et = self.mlp2(et)  # Shape: [bs, c, n_token].
        et = torch.transpose(et, 1, 2)  # Shape: [bs, n_token, c].
        et = self.mlp3(et)

        wet = torch.unsqueeze(selected, -1) * torch.unsqueeze(et, 2)  # Shape: [bs, n_token, hw, c].
        wet = 1 - wet
        wet = torch.prod(wet, dim=1)  # Shape: [bs, hw, c].
        
        # print(wet.shape, memory.shape, control_inputs.shape, selected.shape, et.shape)
        output = memory * wet

        at = self.layernorm3(control_inputs)
        at = torch.transpose(at, 1, 2)  # Shape: [bs, c, hw]
        at = self.mlp4(at)  # Shape: [bs, c, n_token]
        at = torch.transpose(at, 1, 2)  # Shape: [bs, n_token, c]
        at = self.mlp5(at)

        wat = torch.unsqueeze(selected, -1) * torch.unsqueeze(at, 2)  # Shape: [bs, n_token, hw, c].
        wat = 1 - wat
        wat = torch.mean(wat, dim=1)  # Shape: [bs, hw, c].

        output += wat
        return output  # Shape: [bs, hw, c]

class TokenTuringMachineUnit(nn.Module):
    """Token write operations motivated by the `write' in Neural Turing Machines.
    Instead of directly using the token summarization (with TokenLearner), it uses
    a similar but different mechanism to (soft-)select memory elements to zero out
    and write to them. This can be used as an alternative write operation in the
    TTM, particularly when the memory size is huge.
    """
    def __init__(self, process_size: int = 8, memory_size: int = 64, memory_mode: str = 'TL-AddErase',
                 processing_unit: str = 'transformer', num_layers: int = 1, mlp_dim: int = 512,
                 num_heads: int = 8, dropout_rate: float = 0., input_dim: int = 768, process_len: int = 4):
        super().__init__()

        self.process_size = process_size
        self.memory_size = memory_size
        self.memory_mode = memory_mode
        self.processing_unit = processing_unit
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim
        self.process_len = process_len

        # self.token_learner_model = nn.TransformerEncoder(
        #     encoder_layer=nn.TransformerEncoderLayer(
        #         d_model=self.input_dim, 
        #         nhead=self.num_heads, 
        #         dim_feedforward=self.mlp_dim, 
        #         dropout=self.dropout_rate), 
        #     num_layers=self.num_layers)
        self.token_learner_model = TokenLearnerModuleV11(
            num_tokens=self.process_size,
            input_shape=self.input_dim,
            bottleneck_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate)
        
        self.vit_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.input_dim, 
                nhead=self.num_heads, 
                dim_feedforward=self.mlp_dim, 
                dropout=self.dropout_rate), 
            num_layers=self.num_layers)
        
        self.token_add_erase_write = TokenAddEraseWrite(
            num_tokens=self.process_size, 
            bottleneck_dim=self.mlp_dim, 
            dropout_rate=self.dropout_rate, 
            input_dim=self.input_dim,
            memory_size=self.memory_size,
            )

    def forward(self, memory_tokens, input_tokens, train=False):
        """Applies Token Turing Machine unit.
        Args:
            memory_tokens: Inputs of shape '[bs, memory_size, c]'.
            input_tokens: Inputs of shape '[bs, n_token, c]'.
            train: Weather we are in the training mode.
        Returns:
            Tuple of shape '([bs, memory_size, c], [bs, process_size, c])'
        """
        all_tokens = torch.cat([memory_tokens, input_tokens], dim=1)

        # if self.memory_mode == 'TL-AddErase':
        all_tokens = self.token_learner_model(all_tokens)
    
        # if self.processing_unit == 'transformer':
        output_tokens = self.vit_encoder(all_tokens)

        # mem_out_tokens = torch.cat([memory_tokens, input_tokens, output_tokens], dim=1)
        
        # f(input entities, memory entities) -> (relevant entities)
        # if self.memory_mode == 'TL-AddErase':
        mem_out_tokens = self.token_add_erase_write(memory_tokens, output_tokens, train)

        return (mem_out_tokens, output_tokens)

class TokenTuringMachineEncoder(nn.Module):
    """Token Turing Machine main model encoder.
    It implements https://arxiv.org/abs/2211.09119. It essentially repeats
    TokenTuringMachineUnit for the number of steps (of the input tensor).

    This version is for the training and inference with a fixed shaped, static
    input tensor. One will need to modify/extend this module together with the
    data pipeline for the streaming inference implementation.

    Attributes:
        process_size: Number of tokens for the Transformer to process.
        memory_size: The number of memory tokens in the TTM.
        memory_mode: Specifies the token summarization method to use. Supports
            'TL-AddErase'.
        processing_unit: Specifies which processing unit module to use. Supports
            'transformer'
        num_layers: Number of layers in the processing unit.
    """
    def __init__(self, process_size: int = 8, memory_size: int = 64, memory_mode: str = 'TL-AddErase',
                 processing_unit: str = 'transformer', num_layers: int = 1, mlp_dim: int = 512,
                 num_heads: int = 8, dropout_rate: float = 0., input_dim: int = 768):
        super().__init__()

        self.process_size = process_size
        self.memory_size = memory_size
        self.memory_mode = memory_mode
        self.processing_unit = processing_unit
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.input_dim = input_dim

        self.memory_tokens = nn.Parameter(torch.zeros((self.memory_size, input_dim)))
        nn.init.xavier_uniform_(self.memory_tokens)
        # initialize
        
        self.token_turing_machine_unit = TokenTuringMachineUnit(
            process_size=self.process_size,
            memory_size=self.memory_size,
            memory_mode=self.memory_mode,
            processing_unit=self.processing_unit,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            input_dim=self.input_dim)
        
    def forward(self, input_tokens, train=False):
        """Applies Token Turing Machine model.
        Args:
            input_tokens: Inputs of shape `[bs, num_steps, n_tokens, c]`.
            train: Weather we are in the training mode.
        Returns:
            Tensor of shape `[bs, num_steps, process_size, c]`.
        """
        bs, num_steps, n_tokens, c = input_tokens.shape

        output_tokens_list = []
        # memory_tokens = torch.zeros([bs, self.memory_size, c]).to(input_tokens.device)
        # print(memory_tokens.shape)

        for i in range(num_steps):
            step_tokens = input_tokens[:, i, :, :]
            # print(memory_tokens.shape, step_tokens.shape)
            if i == 0:
                memory_tokens, output_tokens = self.token_turing_machine_unit(self.memory_tokens.unsqueeze(0).repeat(bs, 1, 1), step_tokens)
            else:
                memory_tokens, output_tokens = self.token_turing_machine_unit(memory_tokens, step_tokens, train)

            output_tokens = torch.unsqueeze(output_tokens, dim=1)
            output_tokens_list.append(output_tokens)

        output_tokens = torch.cat(output_tokens_list, dim=1)
        # self.memory_tokens = memory_tokens.detach()
        return output_tokens
  
if __name__=="__main__":
    bs, num_steps, n_tokens, c = 100, 2, 3, 768
    input_tokens = torch.randn((bs, num_steps, n_tokens, c))
    ttm_encoder = TokenTuringMachineEncoder(process_size=n_tokens)
    output_tokens = ttm_encoder(input_tokens)
    print(output_tokens.shape)