import keras
from keras.layers import Layer, Dropout
from keras import ops

from tkan import TKAN
from keras_sig import SigLayer
from grns import GRKAN

class SigTKAN(Layer):
    """
    SigTKAN Layer - Combines TKAN recurrence with SigKAN-style signature processing.
    
    Args:
        units: int, dimensionality of TKAN output.
        sig_level: float, noise level for signature transform.
        dropout: float, dropout rate.
        mode: 'manual' (loop time steps) or 'auto' (standard RNN mode).
        return_sequences: whether to return sequence or last time step.
        use_hidden_state: whether to propagate hidden state through time.
        **tkan_kwargs: forwarded to TKAN (e.g., sub_kan_output_dim, sub_kan_input_dim, etc.)
    """

    def __init__(self, units, sig_level, dropout=0., mode='manual', **tkan_kwargs):
        layer_kwargs = {}
        if 'name' in tkan_kwargs:
            layer_kwargs['name'] = tkan_kwargs.pop('name')

        # Appeler super().__init__ d'abord pour que Keras puisse tracer proprement
        super().__init__(**layer_kwargs)

        # Nettoyage des kwargs
        self.return_sequences = tkan_kwargs.pop('return_sequences', True)
        self.use_hidden_state = tkan_kwargs.pop('use_hidden_state', True)

        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        self.mode = mode.lower()
        
        self.sig_layer = SigLayer(self.sig_level)
        self.sig_to_weight = GRKAN(units, activation='softmax', dropout=dropout)
        self.dropout = Dropout(dropout)

        self.tkan_kwargs = tkan_kwargs
        self.tkan_layer = None  # construit dans build()

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, features), got {input_shape}")

        _, seq_length, n_features = input_shape
        name = self.name or "sigtkan"

        self.time_weighting_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weighting_kernel",
            initializer="ones"
        )

        if self.use_hidden_state:
            self.hidden_init_kernel = self.add_weight(
                shape=(n_features, self.units),
                name=f"{name}_hidden_init_kernel",
                initializer='glorot_uniform'
            )

        self.tkan_layer = TKAN(
            self.units,
            dropout=self.dropout_rate,
            return_sequences=False,  # boucle manuelle = pas de return_sequences ici
            **self.tkan_kwargs
        )

        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):    
        # Boucle manuelle
        weighted_inputs = self.time_weighting_kernel * inputs
        sig = self.sig_layer(weighted_inputs)
        weights = self.sig_to_weight(sig)

        if self.use_hidden_state:
            hidden_state = ops.matmul(inputs[:, 0, :], self.hidden_init_kernel)

        outputs = []
        seq_length = ops.shape(inputs)[1]

        for t in range(seq_length):
            x_t = inputs[:, t, :]
            if self.use_hidden_state:
                combined = ops.concatenate([x_t, hidden_state], axis=-1)
                tkan_input = ops.expand_dims(combined, axis=1)
            else:
                tkan_input = ops.expand_dims(x_t, axis=1)

            out_t = self.tkan_layer(tkan_input, training=training, **kwargs)
            out_t = self.dropout(out_t, training=training)
            weighted_out_t = out_t * weights

            if self.use_hidden_state:
                hidden_state = weighted_out_t

            outputs.append(weighted_out_t)

        output_seq = ops.stack(outputs, axis=1)
        return output_seq if self.return_sequences else output_seq[:, -1, :]

    def compute_output_shape(self, input_shape):
        batch_size, seq_len, _ = input_shape
        return (batch_size, seq_len, self.units) if self.return_sequences else (batch_size, self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sig_level": self.sig_level,
            "dropout": self.dropout_rate,
            "mode": self.mode,
            "return_sequences": self.return_sequences,
            "use_hidden_state": self.use_hidden_state,
        })
        config.update(self.tkan_kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
