import keras
from keras.layers import Layer, Dropout
from keras import ops

from tkan import TKAN
from keras_sig import SigLayer
from grns import GRKAN

class SigTKAN(Layer):
    """
    SigTKAN Layer - Version manuelle améliorée combinant les boucles manuelles 
    avec les améliorations de la version Keras.
    
    Args:
        units: dimensionality of the TKAN output space.
        sig_level: float, noise level for SigLayer.
        dropout: float between 0 and 1, dropout fraction.
        return_sequences: whether to return full sequence or last timestep only.
        use_hidden_state: whether to maintain and propagate hidden state.
        **tkan_kwargs: all other keyword args forwarded to TKAN.
    """

    def __init__(self, units, sig_level, dropout=0., return_sequences=True, 
                 use_hidden_state=True, **tkan_kwargs):
        # Extract Layer-specific kwargs safely
        layer_kwargs = {}
        if 'name' in tkan_kwargs:
            layer_kwargs['name'] = tkan_kwargs.pop('name')

        super().__init__(**layer_kwargs)

        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        self.use_hidden_state = use_hidden_state

        # Core components (following Keras version structure)
        self.sig_layer = SigLayer(self.sig_level)
        self.sig_to_weight = GRKAN(units, activation='softmax', dropout=dropout)
        self.dropout = Dropout(dropout)

        # TKAN configured for single timestep processing
        self.tkan_layer = TKAN(
            units,
            dropout=dropout,
            return_sequences=False,  # Always False for manual processing
            **tkan_kwargs
        )

        # Store config for serialization
        self.tkan_kwargs = tkan_kwargs

    def build(self, input_shape):
        """Build the layer - simplified following Keras version pattern"""
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq_length, features), got shape {input_shape}")

        _, seq_length, n_features = input_shape
        name = self.name if self.name else "sigtkan"

        # Time weighting kernel (exactly like Keras version)
        self.time_weighting_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weighting_kernel",
            initializer='ones'
        )

        # Hidden state initialization (only if needed)
        if self.use_hidden_state:
            self.hidden_init_kernel = self.add_weight(
                shape=(n_features, self.units),
                name=f"{name}_hidden_init_kernel",
                initializer='glorot_uniform'
            )

        # Build components - let them handle their own build logic
        self.sig_layer.build(input_shape)
        
        # For TKAN, build with single timestep shape
        single_step_shape = (input_shape[0], 1, n_features + (self.units if self.use_hidden_state else 0))
        self.tkan_layer.build(single_step_shape)
        
        # Build signature processing
        sig_output_shape = self.sig_layer.compute_output_shape(input_shape)
        self.sig_to_weight.build(sig_output_shape)
        
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        """Forward pass - manual loop with Keras-style improvements"""
        # Apply time weighting (same as Keras version)
        weighted_inputs = self.time_weighting_kernel * inputs
        
        # Compute signatures for attention weights
        sig = self.sig_layer(weighted_inputs)
        weights = self.sig_to_weight(sig)
        
        seq_length = ops.shape(inputs)[1]
        
        # Initialize hidden state if needed
        if self.use_hidden_state:
            hidden_state = ops.matmul(inputs[:, 0, :], self.hidden_init_kernel)
        
        outputs = []
        
        # Manual timestep processing
        for t in range(seq_length):
            x_t = inputs[:, t, :]
            
            # Concatenate with hidden state if using it
            if self.use_hidden_state:
                x_t = ops.concatenate([x_t, hidden_state], axis=-1)
            
            # Expand dims for TKAN (expects 3D input)
            x_t = ops.expand_dims(x_t, axis=1)
            
            # Process through TKAN
            out_t = self.tkan_layer(x_t, training=training, **kwargs)
            out_t = self.dropout(out_t, training=training)
            
            # Apply signature-based attention weights
            # weights shape: (batch, units), out_t shape: (batch, units)
            weighted_out_t = out_t * weights
            
            outputs.append(weighted_out_t)
            
            # Update hidden state
            if self.use_hidden_state:
                hidden_state = weighted_out_t

        # Stack outputs and return based on return_sequences
        output_seq = ops.stack(outputs, axis=1)  # (batch, seq_length, units)
        
        if self.return_sequences:
            return output_seq
        else:
            return output_seq[:, -1, :]  # Return last timestep only

    def compute_output_shape(self, input_shape):
        """Compute output shape based on return_sequences setting"""
        batch_size, seq_len, _ = input_shape
        if self.return_sequences:
            return (batch_size, seq_len, self.units)
        else:
            return (batch_size, self.units)

    def get_config(self):
        """Configuration for serialization - following Keras version pattern"""
        config = super().get_config()
        config.update({
            "units": self.units,
            "sig_level": self.sig_level,
            "dropout": self.dropout_rate,
            "return_sequences": self.return_sequences,
            "use_hidden_state": self.use_hidden_state,
        })
        # Add TKAN config
        config.update(self.tkan_kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)