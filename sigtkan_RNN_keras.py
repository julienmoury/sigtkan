import keras
from keras.layers import Layer, Dropout
from keras import ops
from keras_sig import SigLayer

from grns import GRKAN
from tkan import TKAN


class SigTKAN(Layer):
    """
    SigTKAN Layer
    Args:
        units: dimensionality of the TKAN output space.
        sig_level: float, noise level for SigLayer.
        dropout: float between 0 and 1, dropout fraction.
        **tkan_kwargs: all other keyword args forwarded to TKAN (return_sequences).
    """
    def __init__(self, units, sig_level, dropout=0., **tkan_kwargs):
        layer_kwargs = {}
        if 'name' in tkan_kwargs:
            layer_kwargs['name'] = tkan_kwargs.pop('name')
        
        super().__init__(**layer_kwargs)
        
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        
        self.sig_layer = SigLayer(self.sig_level)
        self.tkan_layer = TKAN(units, dropout=dropout, **tkan_kwargs)
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
        
        self.sig_layer.build(input_shape)
        self.tkan_layer.build(input_shape)
        sig_output_shape = self.sig_layer.compute_output_shape(input_shape)
        # poids signatures
        self.sig_to_weight.build(sig_output_shape)
        # dropout
        tkan_output_shape = self.tkan_layer.compute_output_shape(input_shape)
        self.dropout.build(tkan_output_shape)
        
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):

        # poids en fction du temps
        weighted_inputs = self.time_weighting_kernel * inputs

        # Calcul signatures
        sig = self.sig_layer(weighted_inputs)
        
        # les poids
        weights = self.sig_to_weight(sig)

        tkan_out = self.tkan_layer(weighted_inputs, training=training, **kwargs)
        
        # le dropout
        tkan_out = self.dropout(tkan_out, training=training)
        
        # applique les poids en prenant en compte les dimensiosn
        if len(ops.shape(tkan_out)) == 3:  # return_sequences=True: (batch, seq, units)
            # Format: (batch, units) -> vers (batch, 1, units)
            weights_expanded = ops.expand_dims(weights, axis=1)
            return tkan_out * weights_expanded
        else:  # return_sequences=False: (batch, units)
            # Format : (batch, units) - on peut multiplier direct
            return tkan_out * weights

    def compute_output_shape(self, input_shape):
        # le tkan gère les dimensions directement
        return self.tkan_layer.compute_output_shape(input_shape)

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