"""
Member 3: User Personalization & Transfer Learning
Freezes feature extraction layers of the static model and fine-tunes
top layers on user-recorded samples for individual hand geometry adaptation.
"""

import os
import tensorflow as tf

class UserProfileManager:
    def __init__(self, profiles_dir="models/user_profiles"):
        self.profiles_dir = profiles_dir
        os.makedirs(self.profiles_dir, exist_ok=True)

    def freeze_and_adapt(self, base_model, user_data, user_labels, epochs=5, lr=1e-4):
        """
        Clones base model, freezes all layers except the final classification layers,
        and trains on user data.
        """
        # Freeze base layers
        for layer in base_model.layers[:-2]:
            layer.trainable = False
            
        base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        base_model.fit(user_data, user_labels, epochs=epochs, verbose=0)
        return base_model

    def save_profile(self, model, username):
        path = os.path.join(self.profiles_dir, f"{username}_profile.h5")
        model.save(path)
        return path

    def load_profile(self, username):
        path = os.path.join(self.profiles_dir, f"{username}_profile.h5")
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
        return None
