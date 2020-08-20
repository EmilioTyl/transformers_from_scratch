import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_modules import TransformerModule
from .model_utils import device_selection
import pdb

class ClassSequenceTransformer(nn.Module):
    def __init__(self, num_classes, embeding_size, transformer_heads, depth,
                vocab_size, max_sequence, dropout_flag=False):
        super().__init__()
        self.num_classes = num_classes
        self.embeding_size = embeding_size
        self.transformer_heads = transformer_heads
        self.depth = depth
        self.vocab_size = vocab_size
        self.max_sequence = max_sequence
        self.dropout_flag = dropout_flag

        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embeding_size)
        self.pos_embedding = nn.Embedding(num_embeddings=max_sequence, embedding_dim=embeding_size)
        transformers_list = [TransformerModule(k=embeding_size, heads=transformer_heads, hidden_layer_mult=4) for _ in range(self.depth)] 
        self.transfomer = nn.Sequential(*transformers_list)

        self.output_ff = nn.Linear(self.embeding_size, self.num_classes)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        word = self.word_embedding(x)
        b, t, k =  word.size()
        # Determine sequence length, 
        # create embeding for each word of the sequence, expand with a batch dimension 1, and make batch copies
        pos = self.pos_embedding(torch.arange(t, device=device_selection()))[None, :, :].expand(b, t, k)

        mix = word + pos
        # if self.dropout_flag: 
        #     mix = self.droput(mix)
        transformer_output = self.transfomer(mix)

        #Global Avg in the sequence dimension
        averged_output_vector = transformer_output.mean(dim=1)
        output = self.output_ff(averged_output_vector)
        output = F.log_softmax(output, dim=1)
        return output


class GenerationCharacterTransformer(nn.Module):
    def __init__(self, embedding_size, transformer_heads, depth, max_sequence, token_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.transformer_heads = transformer_heads
        self.depth = depth
        self.max_sequence = max_sequence
        self.token_size = token_size

        self.character_embeding = nn.Embedding(num_embeddings=token_size, embedding_dim=embedding_size)
        self.pos_embedding = nn.Embedding(num_embeddings=max_sequence, embedding_dim=embedding_size)
        #With mask
        transformers_list = [TransformerModule(k=embedding_size, heads=self.transformer_heads, hidden_layer_mult=4, mask=True) for _ in range(self.depth)]
        self.transformer = nn.Sequential(*transformers_list)

        self.output_ff = nn.Linear(self.embedding_size, token_size)

    def forward(self, x):
        # x sequence is size b(batch), t(sequence), embeding(k)
        characters = self.character_embeding(x)
        b, t, k = characters.size()
        pos = self.pos_embedding(torch.arange(t, device=device_selection()))[None,:,:].expand(b, t, k)

        mix = characters + pos
        #Flaten in the batch dimension
        mix = mix.view(b*t, k)
        output = self.output_ff(mix)
        #Recover dimension sequence
        output = output.view(b, t, self.token_size)
        return F.log_softmax(output, dim=2)