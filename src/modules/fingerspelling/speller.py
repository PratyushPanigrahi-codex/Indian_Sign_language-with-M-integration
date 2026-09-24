"""
Member 2: Fingerspelling (A–Z) Manager & Word Assembler
Handles letter classification and buffering letters into complete words.
"""

import time
import numpy as np

class FingerspellingManager:
    def __init__(self, model_path=None, label_encoder_path=None, pause_timeout=1.5):
        self.model = None
        self.classes = None
        self.pause_timeout = pause_timeout
        self.current_word_letters = []
        self.last_letter_time = time.time()
        
    def add_letter(self, letter):
        """Append confirmed letter to the buffer."""
        self.current_word_letters.append(letter.upper())
        self.last_letter_time = time.time()

    def check_word_completion(self):
        """
        If user paused signing for `pause_timeout` seconds,
        return assembled word and clear letter buffer.
        """
        if self.current_word_letters and (time.time() - self.last_letter_time > self.pause_timeout):
            word = "".join(self.current_word_letters)
            self.current_word_letters = []
            return word
        return None

    def get_current_buffer(self):
        """Returns string representation of currently spelled letters."""
        return "".join(self.current_word_letters)
