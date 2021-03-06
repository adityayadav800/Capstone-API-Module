from os.path import isfile
import tensorflow as tf
import numpy as np
import pickle
import time
import re

tokenizer_one = pickle.load(open("tokenizer_hi","rb"))
tokenizer_two = pickle.load(open("tokenizer_en","rb"))

# creating the embedding Matrices
# Define the vocab sizes

input_vocab_size = len(tokenizer_one.word_index) + 2
target_vocab_size = len(tokenizer_two.word_index) + 2
# Load the pretrained embeddings

words_en, embeddings_en = pickle.load(
    open('polyglot-en.pkl', 'rb'), encoding='latin1')
words_hi, embeddings_hi = pickle.load(
    open('polyglot-hi.pkl', 'rb'), encoding='latin1')

# English embedding matrix

embeddings_index_en = {}

word_index_en = tokenizer_two.word_index
for i in range(len(words_en)):
    embeddings_index_en[words_en[i].lower()] = embeddings_en[i]

embedding_matrix_en = np.zeros((target_vocab_size, 64))

for word, i in word_index_en.items():
    embedding_vector = embeddings_index_en.get(word)
    if embedding_vector is not None:
        embedding_matrix_en[i] = embedding_vector


# Hindi embedding matrix
import random
embeddings_index_hi = {}
word_index_hi = tokenizer_one.word_index
for i in range(len(words_hi)):
    embeddings_index_hi[words_hi[i]] = embeddings_hi[i]

embedding_matrix_hi = np.zeros((input_vocab_size, 64))
    
for word, i in word_index_hi.items():
    embedding_vector = embeddings_index_hi.get(word)
    if embedding_vector is not None:
        embedding_matrix_hi[i] = embedding_vector

embedding_matrix_one = embedding_matrix_hi
embedding_matrix_two = embedding_matrix_en



# Utility functions for the model

def positional_encoding(max_pos, d_model):
    """ Returns the positional encoding for all positions

    Args:
        max_pos: (int) size of required positional embeddings equal to 
        the vocab size
        d_model: (int) model size equal to the embedding size
    
    Returns:
        pe: (tensor of type float32, shape = 
        (1, max_pos, d_model)) positional encodings of type float32
    """

    theta = np.expand_dims(np.arange(max_pos), 1) / (np.power(10000, 
    (2 * np.expand_dims(np.arange(d_model), 0) // 2) / d_model))

    # sin(i) for all even i
    theta[:, 0::2] = np.sin(theta[:, 0::2]) 
    
    # cos(i) for all odd i
    theta[:, 1::2] = np.cos(theta[:, 1::2]) 
    
    pe = np.expand_dims(theta, 0)
    
    return tf.cast(pe, dtype=tf.float32)

def create_padding_mask(seq):
    """ Creates a padding mask for the self attention layer in the decoder
    """

    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    # (seq_len, seq_len)
    return mask  

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
        
    Returns:
        output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  

    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)  

    return output, attention_weights

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)

def point_wise_feed_forward_network(d_model, dff):

    # (batch_size, seq_len, dff)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
        ])

def create_masks(inp, tar):
# Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

#Define the Model layers

# Multihead Attention keras layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
            
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is :
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
        
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        # (batch_size, seq_len, d_model)
        q = self.wq(q)  
        k = self.wk(k)
        v = self.wv(v)
        
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)

        # (batch_size, num_heads, seq_len_k, depth)  
        k = self.split_heads(k, batch_size)

        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))  

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights

# Encoder keras Layer

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)  
        
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)  
        
        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, 
            d_model, 
            weights = [embedding_matrix_one], 
            trainable = False)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)
            
    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]
        
        # adding embedding and position encoding.
        # (batch_size, input_seq_len, d_model)
        x = self.embedding(x)  
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        
        # (batch_size, input_seq_len, d_model)
        return x  

# Decoder keras layer

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
        
    def call(self, x, enc_output, training, 
            look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  
        attn2 = self.dropout2(attn2, training=training)
        
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)  
        
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  
        ffn_output = self.dropout3(ffn_output, training=training)
        
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)  
        
        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size, 
            d_model, 
            weights = [embedding_matrix_two], 
            trainable=False)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
def call(self, x, enc_output, training, 
        look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    # (batch_size, target_seq_len, d_model)
    x = self.embedding(x) 
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
        x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                look_ahead_mask, padding_mask)
        
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


# Define the Model

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, training, enc_padding_mask)  
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)  
        
        return final_output, attention_weights

# Define the Learning Rate

# Custom Learning rate scheduler

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
