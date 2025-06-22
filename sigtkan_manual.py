import keras
from keras.layers import Layer, Dropout
from keras import ops

from tkan import TKAN
from keras_sig import SigLayer
from grns import GRKAN

class SigTKAN(Layer):
    """
    SigTKAN Layer avec boucles manuelles
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
        layer_kwargs = {}
        if 'name' in tkan_kwargs:
            layer_kwargs['name'] = tkan_kwargs.pop('name')

        super().__init__(**layer_kwargs)
        # Initialisation de la classe
        self.units = units
        self.sig_level = sig_level
        self.dropout_rate = dropout
        self.return_sequences = return_sequences
        self.use_hidden_state = use_hidden_state

        self.sig_layer = SigLayer(self.sig_level)
        self.sig_to_weight = GRKAN(units, activation='softmax', dropout=dropout)
        self.dropout = Dropout(dropout)

        # TKAN pour 1 pas de temps (car boucle manuelle ensuite)
        self.tkan_layer = TKAN(
            units,
            dropout=dropout,
            return_sequences=False,  # tjr a false car boucle manuel
            **tkan_kwargs
        )

        self.tkan_kwargs = tkan_kwargs

    def build(self, input_shape):
        # construction de la layer
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (batch, seq_length, features), got shape {input_shape}")

        _, seq_length, n_features = input_shape
        name = self.name if self.name else "sigtkan"

        # Poids en fonction du temps
        self.time_weighting_kernel = self.add_weight(
            shape=(seq_length, 1),
            name=f"{name}_time_weighting_kernel",
            initializer='ones'
        )

        # Hidden state si besoin
        if self.use_hidden_state:
            self.hidden_init_kernel = self.add_weight(
                shape=(n_features, self.units),
                name=f"{name}_hidden_init_kernel",
                initializer='glorot_uniform'
            )

        self.sig_layer.build(input_shape)
        
        # TKAN :  build avec un seul pas de temps
        single_step_shape = (input_shape[0], 1, n_features + (self.units if self.use_hidden_state else 0))
        self.tkan_layer.build(single_step_shape)
        
        # Build signature
        sig_output_shape = self.sig_layer.compute_output_shape(input_shape)
        self.sig_to_weight.build(sig_output_shape)
        
        super().build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # Time weigthing
        weighted_inputs = self.time_weighting_kernel * inputs
        
        # Calcule des signatures pour les poids
        sig = self.sig_layer(weighted_inputs)
        weights = self.sig_to_weight(sig)
        
        seq_length = ops.shape(inputs)[1]
        
        # hidden state si besoin
        if self.use_hidden_state:
            hidden_state = ops.matmul(inputs[:, 0, :], self.hidden_init_kernel)
        
        outputs = []
        
        # La boucle manuelle
        for t in range(seq_length):
            x_t = inputs[:, t, :]
            
            # On concatene avec la hidden state si besoin
            if self.use_hidden_state:
                x_t = ops.concatenate([x_t, hidden_state], axis=-1)
            
            # puis on expand pour passer au format tkan
            x_t = ops.expand_dims(x_t, axis=1)
            
            # TKAN
            out_t = self.tkan_layer(x_t, training=training, **kwargs)
            out_t = self.dropout(out_t, training=training)
            
            # On applique les poids des signatures
            # weights format: (batch, units), out_t format: (batch, units)
            weighted_out_t = out_t * weights
            
            outputs.append(weighted_out_t)
            
            # Si besoin, on update l'hidden state
            if self.use_hidden_state:
                hidden_state = weighted_out_t

        # on  stack l'output et return return_sequences
        output_seq = ops.stack(outputs, axis=1)  # (batch, seq_length, units)
        
        if self.return_sequences:
            return output_seq
        else:
            # Return le dernier timestep seulement
            return output_seq[:, -1, :]

    def compute_output_shape(self, input_shape):
        batch_size, seq_len, _ = input_shape
        if self.return_sequences:
            return (batch_size, seq_len, self.units)
        else:
            return (batch_size, self.units)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "sig_level": self.sig_level,
            "dropout": self.dropout_rate,
            "return_sequences": self.return_sequences,
            "use_hidden_state": self.use_hidden_state,
        })
        config.update(self.tkan_kwargs)
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)