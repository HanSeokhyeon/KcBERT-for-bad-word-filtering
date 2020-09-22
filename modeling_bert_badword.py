from transformers.modeling_bert import *
from transformers.modeling_bert import _TOKENIZER_FOR_DOC, _CONFIG_FOR_DOC


class PuriPooler(nn.Module):
    def __init__(self, config):
        super(PuriPooler, self).__init__()

        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # we only use activation funtion
        # to minimize the impact on puri attention probs
        pooled_output = self.activation(hidden_states)
        return pooled_output


class PuriAttention(nn.Module):
    """This layer is important role for purify toxic expression
    You can select the output of any encoder layer and embedding output to use query, key, value
    and we just use layer normalization when choosing multiple layer output.
    embedding output : 0
    1~12 encoder layer : 1~12

    Params:
        `config` : a BertConfig class instance with the configuration to build a new model.

    Input:
        `query_hidden_states` : average of values passed through the selected layer for query
        `key_hidden_states` : average of values passed through the selected layer for key
        `value_hidden_states` : average of values passed through the selected layer for value
        `attention_mask` : an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `query_att` : choose to use query_attention matrix when update attention_probs. Default: `False`.
        `key_att` : choose to use key_attention matrix when update attention_probs. Default: `False`.
        `multi_head` : choose to apply multi-head attention. Default: `True`.
        `dropout` : choose to apply dropout to attention_probs. Default: `False`.
        `pooler` : choose to apply tanh activation function. Default: `True`.

    Outputs:
        `attention_output` : final computation of cls token
        `cls_info` : attention_scores, attention_probs values passed through puri layer.

    """

    def __init__(self, config):
        super(PuriAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.pooler = PuriPooler(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_hidden_states, key_hidden_states, value_hidden_states, attention_mask,
                query_att=False, key_att=False, multi_head=True, dropout=False, pooler=True):
        # to save weight information
        cls_info = {}

        # choose to use query_attention matrix when update attention_probs
        # all of hidden_states sizes are [32, 128, 768] and it belongs to selected encoder layers output
        if query_att:
            mixed_query_layer = self.query(query_hidden_states)
        else:
            mixed_query_layer = query_hidden_states[:, :, :]
        if key_att:
            mixed_key_layer = self.key(key_hidden_states)
        else:
            mixed_key_layer = key_hidden_states[:, :, :]
        mixed_value_layer = value_hidden_states[:, :, :]

        # block attention score cls to itself
        attention_mask[:, :, :, 0] -= 10000

        # apply multi-head attention
        if multi_head:
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            value_layer = self.transpose_for_scores(mixed_value_layer)

            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            # attention_scores size becomes [batch, num_head, 1(cls), seq_length]
            attention_scores = attention_scores[:, :, 0:1, :]
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # choose to apply dropout to attention_probs
            if dropout:
                attention_probs = self.dropout(attention_probs)

            # context_layer size is [batch, num_head, 1(cls), head_size]
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            # context_layer size is [batch, 1(cls), all_head_size]
            attention_output = context_layer.view(*new_context_layer_shape)

            # attention size becomes [batch, 1(cls), num_head, seq_length]
            attention_probs = attention_probs.permute(0, 2, 1, 3).contiguous()
            attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()

        # apply single-head attention
        else:
            # Take the dot product between "query" and "key" to get the raw attention scores.
            attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
            # attention_scores size becomes [batch, 1(cls), seq_length]
            attention_scores = attention_scores[:, 0:1, :]
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # reduce rank of attention mask
            attention_mask = attention_mask.squeeze(1)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # choose to apply dropout to attention_probs
            if dropout:
                attention_probs = self.dropout(attention_probs)

            # attention_output size is [batch, 1(cls), all_head_size]
            attention_output = torch.matmul(attention_probs, mixed_value_layer)

        cls_info['scores'] = attention_scores
        cls_info['probs'] = attention_probs

        # reduce the rank of attention_output
        attention_output = attention_output[:, 0]

        # apply tanh activation function
        if pooler:
            attention_output = self.pooler(attention_output)

        return attention_output, cls_info


class BertForBadWordFiltering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.puri = PuriAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="bert-base-uncased",
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )

    # select layers for input of puri attention
    def select_layers(self, output_layers, selected_layers):
        # selected_layers is indices of index which 0 index means embedding_output
        mean_output_layers = output_layers[selected_layers[0]]
        if len(selected_layers) > 1:
            for idx in selected_layers[1:]:
                mean_output_layers = torch.add(mean_output_layers, output_layers[idx])
            mean_output_layers = self.LayerNorm(mean_output_layers)
        return mean_output_layers

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        front_pooler=True,
        query=[0], key=[0], value=[0],
        query_att=False,
        key_att=False,
        multi_head=True,
        dropout=False,
        back_pooler=True
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_all_encoded_layers=True
        )

        output_layers, pooled_output, attention_mask, embedding_output = outputs
        output_layers.insert(0, embedding_output)

        query_hidden_states = self.select_layers(output_layers, query)
        key_hidden_states = self.select_layers(output_layers, key)
        value_hidden_states = self.select_layers(output_layers, value)

        if front_pooler:
            query_hidden_states = torch.cat((pooled_output.unsqueeze(1), query_hidden_states[:, 1:, :]), 1)
        if dropout:
            query_hidden_states = self.dropout(query_hidden_states)
            key_hidden_states = self.dropout(key_hidden_states)
            value_hidden_states = self.dropout(value_hidden_states)

        cls_output, cls_info = self.puri(query_hidden_states, key_hidden_states, value_hidden_states, attention_mask,
                                         query_att, key_att, multi_head, dropout, back_pooler)
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=cls_info,
        )
