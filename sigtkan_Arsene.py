import keras
from keras.layers import Layer, Dropout
from keras import ops

# Import existing components
from tkan import TKAN
from keras_sig import SigLayer
from grns import GRKAN


class SigTKAN(Layer):
    """
    SigTKAN Layer - Combines TKAN recurrence with SigKAN-style signature processing
        
    Args:
        units: dimensionality of the TKAN output space.
        sig_level: float, noise level for SigLayer.
        dropout: float between 0 and 1, dropout fraction.
        **tkan_kwargs: all other keyword args forwarded to TKAN
    """
    def __init__(self, units, sig_level, dropout=0., **tkan_kwargs):

        layer_kwargs = {}
        if 'name' in tkan_kwargs:
            layer_kwargs['name'] = tkan_kwargs.pop('name')
        
        super().__init__(**layer_kwargs)
        
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout

        self.return_sequences = tkan_kwargs.get('return_sequences', True)
        self.use_hidden_state = tkan_kwargs.get('use_hidden_state', True)
        
        self.sig_layer = SigLayer(self.sig_level)
        # Sans la boucle manuelle, on utilise return_sequence = True (par défault)
        # self.tkan_layer = TKAN(units, dropout=dropout, **tkan_kwargs)

        # Avec la boucle manuelle, on fait tkan_alyer = none et on construit dans build
        self.tkan_layer = None


        self.sig_to_weight = GRKAN(units, activation='softmax', dropout=dropout)
        self.dropout = Dropout(dropout)
        
        self.tkan_kwargs = tkan_kwargs

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq_length, features), got shape {input_shape}")
            
        _, seq_length, n_features = input_shape
        name = self.name if self.name else "sigtkan"
        
        self.time_weighting_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weighting_kernel",
            initializer='ones'
        )

        # Pour la boucle manuelle
        if self.use_hidden_state:
            self.hidden_init_kernel = self.add_weight(
                shape=(n_features, self.units),
                name=f"{name}_hidden_init_kernel",
                initializer='glorot_uniform'
            )

        self.tkan_layer = TKAN(self.units, dropout=self.dropout_rate, return_sequences=False, **self.tkan_kwargs)

        super().build(input_shape)

    # méthode call sans la boucle manuelle
    def call_automatic_loop(self, inputs, training=None, **kwargs):
        """Forward pass - following SigKAN structure exactly"""
        # Apply time weighting (same as SigKAN)
        weighted_inputs = self.time_weighting_kernel * inputs
        
        sig = self.sig_layer(weighted_inputs)
        
        weights = self.sig_to_weight(sig)
        tkan_out = self.tkan_layer(weighted_inputs, training=training, **kwargs)
        
        tkan_out = self.dropout(tkan_out, training=training)
        
        if len(ops.shape(tkan_out)) == 3:  # return_sequences=True: (batch, seq, units)
            weights_expanded = ops.expand_dims(weights, axis=1)
            return tkan_out * weights_expanded
        else:  # return_sequences=False: (batch, units)
            return tkan_out * weights

    # méthode call avec boucle manuelle
    def call(self, inputs, training=None, **kwargs):
        weighted_inputs = self.time_weighting_kernel * inputs
        sig = self.sig_layer(weighted_inputs)
        weights = self.sig_to_weight(sig)

        if self.use_hidden_state:
            hidden_state = ops.matmul(inputs[:, 0, :], self.hidden_init_kernel)

        outputs = []
        seq_length = ops.shape(inputs)[1]

        # boucle
        for t in range(seq_length):
            x_t = inputs[:, t, :]

            if self.use_hidden_state:
                combined = ops.concatenate([x_t, hidden_state], axis=-1)
                tkan_input = ops.expand_dims(combined, axis=1)
            else:
                tkan_input = ops.expand_dims(x_t, axis=1)

            # tkan sur un seul pas de temps
            out_t = self.tkan_layer(tkan_input, training=training, **kwargs)
            out_t = self.dropout(out_t, training=training)
            weighted_out_t = out_t * weights

            if self.use_hidden_state:
                hidden_state = weighted_out_t

            outputs.append(weighted_out_t)

        output_seq = ops.stack(outputs, axis=1)
        return output_seq if self.return_sequences else output_seq[:, -1, :]

    def compute_output_shape(self, input_shape):
        # avec boucle auto
        # return self.tkan_layer.compute_output_shape(input_shape)

        # avec boucle manuelle
        batch_size, seq_length, _ = input_shape
        if self.return_sequences:
            return (batch_size, seq_length, self.units)
        else:
            return (batch_size, self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sig_level": self.sig_level,
            "dropout": self.dropout_rate,
        })
        config.update(self.tkan_kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)