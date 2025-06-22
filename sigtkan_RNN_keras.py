import keras
from keras.layers import Layer, Dropout
from keras import ops
from keras_sig import SigLayer

from grns import GRKAN
from tkan import TKAN


class SigTKAN(Layer):
    """
    SigTKAN Layer - Combines TKAN recurrence with SigKAN-style signature processing
        
    Args:
        units: dimensionality of the TKAN output space.
        sig_level: float, noise level for SigLayer.
        dropout: float between 0 and 1, dropout fraction.
        **tkan_kwargs: all other keyword args forwarded to TKAN (e.g., sub_kan_configs, return_sequences, etc.).
    """
    def __init__(self, units, sig_level, dropout=0., **tkan_kwargs):
        # Extract name and other Layer-specific kwargs safely
        layer_kwargs = {}
        if 'name' in tkan_kwargs:
            layer_kwargs['name'] = tkan_kwargs.pop('name')
        
        super().__init__(**layer_kwargs)
        
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        
        # Core components (mirroring SigKAN structure)
        self.sig_layer = SigLayer(self.sig_level)
        self.tkan_layer = TKAN(units, dropout=dropout, **tkan_kwargs)
        self.sig_to_weight = GRKAN(units, activation='softmax', dropout=dropout)
        self.dropout = Dropout(dropout)
        
        # Store config for serialization
        self.tkan_kwargs = tkan_kwargs

    def build(self, input_shape):
        """Build the layer - following SigKAN pattern exactly"""
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq_length, features), got shape {input_shape}")
            
        _, seq_length, n_features = input_shape
        name = self.name if self.name else "sigtkan"
        
        # Time weighting kernel (exactly like SigKAN)
        self.time_weighting_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weighting_kernel",
            initializer='ones'
        )
        
        self.sig_layer.build(input_shape)
        self.tkan_layer.build(input_shape)
        # Build signature to weight mapping
        sig_output_shape = self.sig_layer.compute_output_shape(input_shape)
        self.sig_to_weight.build(sig_output_shape)        
        # Build dropout
        tkan_output_shape = self.tkan_layer.compute_output_shape(input_shape)
        self.dropout.build(tkan_output_shape)
        
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """Forward pass - following SigKAN structure exactly"""
        # Apply time weighting (same as SigKAN)
        weighted_inputs = self.time_weighting_kernel * inputs
        
        # Compute signatures
        sig = self.sig_layer(weighted_inputs)
        
        # Get attention weights from signatures
        weights = self.sig_to_weight(sig)
        
        # Apply TKAN instead of KANLinear
        tkan_out = self.tkan_layer(weighted_inputs, training=training, **kwargs)
        
        # Apply dropout
        tkan_out = self.dropout(tkan_out, training=training)
        
        # Apply attention weighting - need to handle TKAN output shapes correctly
        if len(ops.shape(tkan_out)) == 3:  # return_sequences=True: (batch, seq, units)
            # weights shape: (batch, units) -> expand to (batch, 1, units)
            weights_expanded = ops.expand_dims(weights, axis=1)
            return tkan_out * weights_expanded
        else:  # return_sequences=False: (batch, units)
            # weights shape: (batch, units) - direct multiplication
            return tkan_out * weights

    def compute_output_shape(self, input_shape):
        """Delegate to TKAN for output shape computation"""
        return self.tkan_layer.compute_output_shape(input_shape)

    def get_config(self):
        """Configuration for serialization"""
        config = super().get_config()
        config.update({
            "units": self.units,
            "sig_level": self.sig_level,
            "dropout": self.dropout_rate,
        })
        # Add TKAN config
        config.update(self.tkan_kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)