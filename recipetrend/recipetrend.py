import torch
import torch.nn as nn

from typing import Optional, List, Tuple, Union

import random

from utils.modeling_outputs import CustomBaseModelOutputWithPoolingAndCrossAttentions

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from transformers import BertModel

class RecipeTrend(BertModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.config.hidden_size = 199492

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.l_ok = 0
        self.h_ok = 0
        self.b_ok = 0

        self.max_seq_length = self.config.max_position_embeddings
        self.hidden_size = self.config.hidden_size

        self.low_r = self.max_seq_length // 4 
        self.high_r = self.max_seq_length // 4
    
        self.LPA = self.createLPAilter((self.max_seq_length, self.hidden_size), self.low_r)
        self.HPA = self.createHPAilter((self.max_seq_length, self.hidden_size), self.high_r)
        self.BSA = [self.createBSAilter((self.max_seq_length, self.hidden_size), i, 2)
                for i in range(min(self.max_seq_length, self.hidden_size) // 2 + 1)]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        ## labels
        mlm_labels = self.encoder(
            embedding_output,
            # attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = mlm_labels[0]

        ## augmentation
        output_aug_l1, output_aug_l2, output_aug_h1, output_aug_h2, output_aug_b1, output_aug_b2 = [None for i in range(6)]

        if self.l_ok:

            self.LPA = self.LPA.resize_(embedding_output.shape[1:])

            input_emb_aug_l1 = self.fft_2(embedding_output, self.LPA)
            input_emb_aug_l2 = self.fft_2(embedding_output, self.LPA)
            input_emb_aug_l1 = self.dropout(input_emb_aug_l1)
            input_emb_aug_l2 = self.dropout(input_emb_aug_l2)   
            input_emb_aug_l1 = self.encoder(
                input_emb_aug_l1,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output_aug_l1 = input_emb_aug_l1[-1]

            input_emb_aug_l2 = self.encoder(
                input_emb_aug_l2,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output_aug_l2 = input_emb_aug_l2[-1]

        if self.h_ok:

            self.HPA = self.HPA.resize_(embedding_output.shape[1:])

            input_emb_aug_h1 = self.fft_2(embedding_output, self.HPA)
            input_emb_aug_h2 = self.fft_2(embedding_output, self.HPA)
            input_emb_aug_h1 = self.dropout(input_emb_aug_h1)
            input_emb_aug_h2 = self.dropout(input_emb_aug_h2)   
            input_emb_aug_h1 = self.encoder(
                input_emb_aug_h1,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output_aug_h1 = input_emb_aug_h1[-1]

            input_emb_aug_h2 = self.encoder(
                input_emb_aug_h2,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output_aug_h2 = input_emb_aug_h2[-1]

        if self.b_ok:

            input_emb_aug_b1 = self.fft_2(embedding_output, random.choice(self.BSA).resize_(embedding_output.shape[1:]))
            input_emb_aug_b2 = self.fft_2(embedding_output, random.choice(self.BSA).resize_(embedding_output.shape[1:]))
            input_emb_aug_b1 = self.dropout(input_emb_aug_b1)
            input_emb_aug_b2 = self.dropout(input_emb_aug_b2)   
            input_emb_aug_b1 = self.encoder(
                input_emb_aug_b1,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output_aug_b1 = input_emb_aug_b1[-1]

            input_emb_aug_b2 = self.encoder(
                input_emb_aug_b2,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output_aug_b2 = input_emb_aug_b2[-1]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
            

        return CustomBaseModelOutputWithPoolingAndCrossAttentions(
            output_aug_l1,
            output_aug_l2,
            output_aug_h1,
            output_aug_h2,
            output_aug_b1,
            output_aug_b2,
            # last_hidden_state=sequence_output,
            logits=sequence_output,
            labels=mlm_labels,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions
        )

    def my_fft(self, seq):
        f = torch.fft.rfft(seq, dim=1)
        amp = torch.absolute(f)
        phase = torch.angle(f)
        return amp, phase

    def fft_2(self, x, filter):
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        return torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift.cuda() * filter.cuda())))

    def createBSAilter(self, shape, bandCenter, bandWidth):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        bsFilter = torch.zeros((rows, cols))

        if min(rows, cols) // 2 == bandCenter:
            bsFilter[d < (bandCenter - bandWidth / 2)] = 1
        else:
            bsFilter[d > (bandCenter + bandWidth / 2)] = 1
            bsFilter[d < (bandCenter - bandWidth / 2)] = 1

        return bsFilter

    def createLPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt() 

        lpFilter = torch.ones((rows, cols))
        lpFilter[d > bandCenter] = 0

        return lpFilter

    def createHPAilter(self, shape, bandCenter):
        rows, cols = shape

        xx = torch.arange(0, cols, 1)
        yy = torch.arange(0, rows, 1)
        x = xx.repeat(rows, 1)
        y = yy.repeat(cols, 1).T

        x = x - cols // 2
        y = y - rows // 2

        d = (x.pow(2) + y.pow(2)).sqrt()

        hpFilter = torch.ones((rows, cols))
        hpFilter[d < bandCenter] = 0

        return hpFilter