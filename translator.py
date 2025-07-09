# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, MarianMTModel
import logging
from functools import lru_cache

class EnglishToChineseTranslator:
    """English to Chinese translation using Helsinki-NLP models"""
    
    def __init__(self):
        self.src = "en"  # source language
        self.trg = "zh"  # target language (zh for Chinese, not ch)
        self.model_name = f"Helsinki-NLP/opus-mt-{self.src}-{self.trg}"
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the translation model and tokenizer"""
        try:
            print(f"Loading translation model: {self.model_name}")
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("Translation model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load translation model: {e}")
            # Fallback to a more general model if specific one fails
            try:
                self.model_name = "Helsinki-NLP/opus-mt-en-zh"
                self.model = MarianMTModel.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                print("Fallback translation model loaded")
            except Exception as e2:
                logging.error(f"Failed to load fallback model: {e2}")
                
    def translate(self, text):
        """Translate English text to Chinese"""
        if not self.model or not self.tokenizer or not text.strip():
            return ""
            
        try:
            # Tokenize the input text
            batch = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Generate translation
            generated_ids = self.model.generate(**batch, max_length=512, num_beams=4, early_stopping=True)
            
            # Decode the translation
            translated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return translated_text
            
        except Exception as e:
            logging.error(f"Translation failed: {e}")
            return ""

class TranslationBuffer:
    """Manages accumulated text for translation"""
    
    def __init__(self, translator):
        self.translator = translator
        self.accumulated_text = ""
        
    def add_text(self, new_text):
        """Add new transcribed text and return translation"""
        if new_text.strip():
            # Add space if there's existing text
            if self.accumulated_text:
                self.accumulated_text += " " + new_text.strip()
            else:
                self.accumulated_text = new_text.strip()
                
        return self.translate_accumulated()
        
    def translate_accumulated(self):
        """Translate the accumulated text"""
        if not self.accumulated_text:
            return ""
        return self.translator.translate(self.accumulated_text)
        
    def clear(self):
        """Clear the accumulated text"""
        self.accumulated_text = ""
        
    def get_accumulated_text(self):
        """Get the current accumulated text"""
        return self.accumulated_text
