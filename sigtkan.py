import keras
from keras import ops
from keras.layers import Layer, Dense, Dropout

from keras_efficient_kan import KANLinear
from keras_sig import SigLayer
from .grns import GRKAN, GRN

class SigTKANCell(Layer):
    """
    SigTKAN Cell combining signatures with TKAN functionality
    """
    def __init__(self, units, sig_level=2, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        
        # Signature layer
        self.sig_layer = SigLayer(self.sig_level)
        
        # TKAN-like KAN layers
        self.kan_layer = KANLinear(units, dropout=dropout, use_bias=False, use_layernorm=False)
        
        # Signature to weight conversion
        self.sig_to_weight = GRKAN(units, activation='softmax', dropout=dropout)
        
        # Dropout layer
        self.dropout = Dropout(dropout)
        
        # LSTM-like gates
        self.forget_gate = Dense(units, activation='sigmoid')
        self.input_gate = Dense(units, activation='sigmoid')
        self.candidate_gate = Dense(units, activation='tanh')
        self.output_gate = Dense(units, activation='sigmoid')
        
        # State transformation
        self.state_transform = Dense(units)

    def build(self, input_shape):
        batch_size, seq_length, n_features = input_shape
        
        # Time weighting kernel for temporal attention
        self.time_weighting_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{self.name}_time_weighting_kernel",
            initializer="glorot_uniform",
        )
        
        super().build(input_shape)

    def call(self, inputs, states, training=None):
        # Previous hidden and cell states
        h_prev, c_prev = states
        
        # Apply time weighting
        weighted_inputs = self.time_weighting_kernel * inputs
        
        # Compute signature - handle variable sequence length
        try:
            sig = self.sig_layer(weighted_inputs)
        except:
            # Fallback: use mean pooling if signature fails
            sig = ops.mean(weighted_inputs, axis=1, keepdims=True)
        
        # Get attention weights from signature
        try:
            attention_weights = self.sig_to_weight(sig)
        except:
            # Fallback: uniform attention weights
            attention_weights = ops.ones((ops.shape(inputs)[0], ops.shape(inputs)[1])) / ops.cast(ops.shape(inputs)[1], 'float32')
        
        # Apply KAN transformation
        kan_output = self.kan_layer(weighted_inputs)
        kan_output = self.dropout(kan_output, training=training)
        
        # Apply signature-based attention
        attended_output = kan_output * ops.expand_dims(attention_weights, axis=-1)
        
        # Aggregate over sequence dimension
        current_input = ops.mean(attended_output, axis=1)
        
        # Combine with previous hidden state
        combined = ops.concatenate([current_input, h_prev], axis=-1)
        
        # LSTM-like gates
        forget = self.forget_gate(combined)
        input_gate_val = self.input_gate(combined)
        candidate = self.candidate_gate(combined)
        output = self.output_gate(combined)
        
        # Update cell state
        c_new = forget * c_prev + input_gate_val * candidate
        
        # Update hidden state
        h_new = output * ops.tanh(c_new)
        
        return h_new, [h_new, c_new]

class SigTKAN(Layer):
    """
    SigTKAN Layer with manual loop implementation
    """
    def __init__(self, units, sig_level=2, dropout=0.0, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        
        # SigTKAN cell
        self.cell = SigTKANCell(units, sig_level, dropout)

    def build(self, input_shape):
        self.cell.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch_size = ops.shape(inputs)[0]
        seq_length = ops.shape(inputs)[1]
        
        # Initialize states
        h_state = ops.zeros((batch_size, self.units))
        c_state = ops.zeros((batch_size, self.units))
        states = [h_state, c_state]
        
        outputs = []
        
        # Manual loop through sequence - simplified for stability
        for t in range(seq_length):
            # For signature computation, we need at least 2 timesteps
            if t == 0:
                # Use first timestep repeated for signature computation
                current_sequence = ops.expand_dims(inputs[:, 0, :], axis=1)
                current_sequence = ops.concatenate([current_sequence, current_sequence], axis=1)
            else:
                # Get current input (all timesteps up to t+1 for signature computation)
                current_sequence = inputs[:, :t+1, :]
            
            # Call cell
            output, states = self.cell(current_sequence, states, training=training)
            
            if self.return_sequences:
                outputs.append(output)
        
        if self.return_sequences:
            return ops.stack(outputs, axis=1)
        else:
            return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sig_level": self.sig_level,
            "dropout": self.dropout_rate,
            "return_sequences": self.return_sequences,
        })
        return config

class SigTKANDense(Layer):
    """
    Simple SigTKAN layer for dense-like operations
    """
    def __init__(self, units, sig_level=2, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        
        # Signature layer
        self.sig_layer = SigLayer(self.sig_level)
        
        # Dense-KAN hybrid
        self.dense_layer = Dense(units)
        self.kan_layer = KANLinear(units, dropout=dropout, use_bias=False, use_layernorm=False)
        
        # Signature to weight conversion
        self.sig_to_weight = GRN(units, activation='softmax', dropout=dropout)
        
        # Dropout
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        _, seq_length, n_features = input_shape
        
        # Time weighting kernel
        self.time_weighting_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{self.name}_time_weighting_kernel",
            initializer="glorot_uniform",
        )
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Apply time weighting
        weighted_inputs = self.time_weighting_kernel * inputs
        
        # Compute signature
        sig = self.sig_layer(weighted_inputs)
        
        # Get weights from signature
        weights = self.sig_to_weight(sig)
        
        # Apply both dense and KAN transformations
        dense_out = self.dense_layer(weighted_inputs)
        kan_out = self.kan_layer(weighted_inputs)
        
        # Combine outputs
        combined_out = dense_out + kan_out
        combined_out = self.dropout(combined_out, training=training)
        
        # Apply signature-based weighting
        return combined_out * ops.expand_dims(weights, axis=1)
