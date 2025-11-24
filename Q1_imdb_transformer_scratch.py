# imdb_transformer_scratch.py
# Run: python imdb_transformer_scratch.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ---------------------------
# Hyperparameters / design
# ---------------------------
vocab_size = 20000
maxlen = 256
embed_dim = 128
num_heads = 4
ff_dim = 256
num_layers = 3
dropout_rate = 0.1
batch_size = 64
epochs = 6      # increase to 8-12 for better final acc
learning_rate = 1e-4

# ---------------------------
# Dataset: IMDB (integer tokenized)
# ---------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Create a validation split from train
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_val   = pad_sequences(x_val,   maxlen=maxlen, padding='post', truncating='post')
x_test  = pad_sequences(x_test,  maxlen=maxlen, padding='post', truncating='post')

# ---------------------------
# Sinusoidal positional encoding
# ---------------------------
def get_positional_encoding(maxlen, d_model):
    pos = np.arange(maxlen)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2*(i//2))/np.float32(d_model))
    angle_rads = pos * angle_rates
    pe = np.zeros((maxlen, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(pe, dtype=tf.float32)

pos_encoding = get_positional_encoding(maxlen, embed_dim)

# ---------------------------
# Custom Multi-Head Self-Attention
# ---------------------------
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = embed_dim // num_heads
        # projection layers
        self.wq = layers.Dense(embed_dim)
        self.wk = layers.Dense(embed_dim)
        self.wv = layers.Dense(embed_dim)
        self.dense = layers.Dense(embed_dim)
        self.scale = tf.math.sqrt(tf.cast(self.proj_dim, tf.float32))
        self.last_attn = None

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0,2,1,3])

    def call(self, x, mask=None, return_attention=False):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        scores = tf.matmul(q, k, transpose_b=True) / self.scale
        if mask is not None:
            scores += (mask * -1e9)
        attn = tf.nn.softmax(scores, axis=-1)
        self.last_attn = attn
        out = tf.matmul(attn, v)  # (batch, heads, seq_len, proj_dim)
        out = tf.transpose(out, perm=[0,2,1,3])
        out = tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1], self.embed_dim))
        out = self.dense(out)
        if return_attention:
            return out, attn
        return out

# ---------------------------
# Transformer encoder block
# ---------------------------
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        attn_output = self.mha(x, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2

# ---------------------------
# Build model
# ---------------------------
inputs = layers.Input(shape=(maxlen,), dtype='int32')
embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=False)
x = embedding_layer(inputs)
x = x + pos_encoding[tf.newaxis, :maxlen, :]

encoders = []
for _ in range(num_layers):
    enc = TransformerEncoder(embed_dim, num_heads, ff_dim, dropout_rate)
    x = enc(x)
    encoders.append(enc)

x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(dropout_rate)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(dropout_rate)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=epochs,
    batch_size=batch_size
)

# ---------------------------
# Evaluate & plots
# ---------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Plot loss & accuracy
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Accuracy vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.tight_layout()
plt.show()

# Attention heatmap on a sample
sample_idx = 10
sample = x_test[sample_idx:sample_idx+1]
_ = model(sample, training=False)  # forward to populate last_attn
attn = encoders[0].mha.last_attn  # (batch, heads, seq_len, seq_len)
if attn is not None:
    attn_avg = tf.reduce_mean(attn, axis=1).numpy()[0]  # average heads
    seq_show = min(80, attn_avg.shape[0])
    plt.figure(figsize=(6,5))
    plt.imshow(attn_avg[:seq_show, :seq_show], aspect='auto')
    plt.colorbar()
    plt.title('Avg attention (encoder 0) — first 80 positions')
    plt.xlabel('Key position'); plt.ylabel('Query position')
    plt.show()
else:
    print("Attention not available.")
