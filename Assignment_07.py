#!/usr/bin/env python3
"""
Hugging Face Text-to-Speech (TTS) Model Inference
==================================================

This script demonstrates how to clone and perform inference with a pre-trained
Text-to-Speech model from Hugging Face. It converts input text into natural-
sounding speech and saves the output as an audio file.

Author: HaoPQ1
Date: October 31, 2025
"""

import torch
import soundfile as sf
import numpy as np
from transformers import VitsModel, AutoTokenizer
from IPython.display import Audio
import warnings
warnings.filterwarnings("ignore")

def load_tts_model(model_name="facebook/mms-tts-eng"):
    """
    Load a pre-trained TTS model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): The name of the TTS model from Hugging Face Hub
        
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer objects
        
    Challenges addressed:
    - Model compatibility: Using VITS models which are well-supported
    - Memory management: Loading models efficiently
    """
    print(f"Loading TTS model: {model_name}")
    print("This may take a few minutes on first run as the model is downloaded...")
    
    try:
        # Load the pre-trained VITS model from Hugging Face
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Model and tokenizer loaded successfully!")
        print(f"Model configuration:")
        print(f"  - Sampling rate: {model.config.sampling_rate} Hz")
        print(f"  - Model type: {type(model).__name__}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Falling back to alternative model...")
        
        # Fallback to a different model if the primary one fails
        fallback_model = "microsoft/speecht5_tts"
        try:
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
            model = SpeechT5ForTextToSpeech.from_pretrained(fallback_model)
            tokenizer = SpeechT5Processor.from_pretrained(fallback_model)
            print(f"Fallback model {fallback_model} loaded successfully!")
            return model, tokenizer
        except Exception as fallback_error:
            print(f"Fallback model also failed: {str(fallback_error)}")
            raise

def synthesize_speech(model, tokenizer, text, output_filename="output.wav"):
    """
    Convert input text to speech using the TTS model.
    
    Args:
        model: The loaded TTS model
        tokenizer: The tokenizer for the model
        text (str): Input text to convert to speech
        output_filename (str): Name of the output audio file
        
    Returns:
        tuple: (waveform, sampling_rate) - Generated audio data
        
    Challenges addressed:
    - Text preprocessing: Handling various text inputs
    - Audio quality: Ensuring proper sampling rate and format
    - Memory efficiency: Using torch.no_grad() for inference
    """
    print(f"\nSynthesizing speech for text: '{text}'")
    
    # Step 1: Prepare input text
    # Clean and preprocess the text
    text = text.strip()
    if not text:
        raise ValueError("Input text cannot be empty")
    
    print("Text preprocessed")
    
    # Step 2: Tokenize the input text
    try:
        inputs = tokenizer(text, return_tensors="pt")
        print("Text tokenized successfully")
    except Exception as e:
        print(f"Tokenization error: {str(e)}")
        raise
    
    # Step 3: Perform inference to generate the waveform
    print("Generating audio waveform...")
    try:
        with torch.no_grad():  # Disable gradient computation for efficiency
            if hasattr(model, 'generate_speech'):
                # For SpeechT5 models
                output = model.generate_speech(inputs["input_ids"])
                waveform = output.cpu().numpy()
                sampling_rate = 16000  # Default for SpeechT5
            else:
                # For VITS models
                output = model(**inputs)
                waveform = output.waveform.squeeze().cpu().numpy()
                sampling_rate = model.config.sampling_rate
        
        print("Audio waveform generated successfully")
        print(f"  - Waveform shape: {waveform.shape}")
        print(f"  - Duration: {len(waveform) / sampling_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Inference error: {str(e)}")
        raise
    
    # Step 4: Save the generated audio to file
    try:
        sf.write(output_filename, waveform, sampling_rate)
        print(f"Audio saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving audio file: {str(e)}")
        raise
    
    # Also save as 'output.wav' for the specific Vietnamese text (assignment requirement)
    if text == "Xin chào anh em đến với bài tập của khoá AI Application Engineer":
        sf.write('output.wav', waveform, sampling_rate)
        print(f"Also saved as 'output.wav' (as per assignment example)")
    
    # Ensure output.wav is always created for Vietnamese samples
    if "vietnamese" in output_filename:
        sf.write('output.wav', waveform, sampling_rate)
        print(f"Main output file 'output.wav' created")
    
    return waveform, sampling_rate

def play_audio_in_notebook(waveform, sampling_rate, display_now=False):
    """
    Display audio player in Jupyter notebook environment.
    
    Args:
        waveform (numpy.ndarray): Audio waveform data
        sampling_rate (int): Sampling rate of the audio
        display_now (bool): Whether to display immediately (for Jupyter)
        
    Returns:
        IPython.display.Audio: Audio player object for notebooks
    """
    try:
        audio_player = Audio(waveform, rate=sampling_rate)
        print("Audio player created (for Jupyter notebook)")
        
        # Option to display immediately in Jupyter
        if display_now:
            from IPython.display import display
            display(audio_player)
            
        return audio_player
    except Exception as e:
        print(f"Error creating audio player: {str(e)}")
        return None

def assignment_demo():
    """
    Assignment Demo: Step-by-Step TTS following the exact requirement structure
    """
    print("Assignment Demo: Following Step-by-Step Instructions")
    print("=" * 55)
    
    # Step 1: Install required packages (already done)
    print("Step 1: Required packages already installed")
    
    # Step 2: Import required libraries (already done above)
    print("Step 2: Libraries imported successfully")
    
    # Step 3: Clone and load the pre-trained TTS model from Hugging Face
    print("\nStep 3: Clone and load pre-trained TTS model")
    model = VitsModel.from_pretrained("facebook/mms-tts-vie")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
    print("Model cloned and loaded successfully!")
    
    # Step 4: Prepare input text
    print("\nStep 4: Prepare input text")
    text = "Xin chào anh em đến với bài tập của khoá AI Application Engineer"
    print(f"Input text: '{text}'")
    
    # Step 5: Tokenize the input text
    print("\nStep 5: Tokenize the input text")
    inputs = tokenizer(text, return_tensors="pt")
    print("Text tokenized successfully")
    
    # Step 6: Perform inference to generate the waveform
    print("\nStep 6: Perform inference to generate waveform")
    with torch.no_grad():
        output = model(**inputs).waveform
    print(f"Waveform generated - Shape: {output.shape}")
    
    # Step 7: Play the generated audio in Jupyter Notebook
    print("\nStep 7: Create audio player for Jupyter Notebook")
    audio_display = Audio(output.numpy(), rate=model.config.sampling_rate)
    
    # Optional: Save audio to file (as required)
    print("\nOptional: Save audio to file")
    import soundfile as sf
    sf.write('output.wav', output.numpy().squeeze(), model.config.sampling_rate)
    print("Audio saved as 'output.wav'")
    
    # Display audio player (for Jupyter)
    try:
        from IPython.display import display
        display(audio_display)
        print("Audio displayed in Jupyter notebook")
    except:
        print("Audio player created (display in Jupyter notebook)")
    
    print("\nAssignment Demo Complete!")
    return audio_display

def main():
    """
    Main function to demonstrate the complete TTS pipeline.
    
    This function orchestrates the entire process:
    1. Loading the model
    2. Processing sample texts
    3. Generating audio outputs
    4. Saving results
    """
    print("=" * 60)
    print("Hugging Face Text-to-Speech (TTS) Demo")
    print("=" * 60)
    
    # Configuration - Multiple Models and Languages
    MODELS_CONFIG = {
        "english": {
            "model_name": "facebook/mms-tts-eng",
            "texts": [
                "Hello, welcome to the AI Application Engineer course!",
                "This is a demonstration of text to speech synthesis using Hugging Face transformers.",
                "The quick brown fox jumps over the lazy dog."
            ]
        },
        "vietnamese": {
            "model_name": "facebook/mms-tts-vie",
            "texts": [
                "Xin chào anh em đến với bài tập của khoá AI Application Engineer",  # Đây là text chính xác từ đề bài
                "Đây là một minh hoạ về tổng hợp giọng nói từ văn bản sử dụng Hugging Face transformers.",
                "Công nghệ Text-to-Speech chuyển đổi văn bản thành giọng nói tự nhiên.",
                "Chúng ta đang học về trí tuệ nhân tạo và xử lý ngôn ngữ tự nhiên."
            ]
        }
    }
    
    try:
        # Process both English and Vietnamese models
        all_audio_files = []
        
        for language, config in MODELS_CONFIG.items():
            print(f"\nProcessing {language.upper()} Language")
            print("=" * 50)
            
            # Step 1: Load the TTS model for this language
            print(f"\nLoading {language} TTS model: {config['model_name']}")
            model, tokenizer = load_tts_model(config['model_name'])
            
            # Step 2: Process sample texts for this language
            print(f"\nGenerating {language} speech samples")
            
            for i, text in enumerate(config['texts'], 1):
                print(f"\n--- Processing {language.title()} Sample {i} ---")
                output_filename = f"{language}_sample_{i}_output.wav"
                
                try:
                    waveform, sampling_rate = synthesize_speech(
                        model, tokenizer, text, output_filename
                    )
                    all_audio_files.append({
                        'filename': output_filename,
                        'language': language,
                        'model': config['model_name'],
                        'text': text
                    })
                    
                    # Create audio player for notebook environment
                    audio_player = play_audio_in_notebook(waveform, sampling_rate, display_now=True)
                    
                except Exception as e:
                    print(f"Failed to process {language} sample {i}: {str(e)}")
                    continue
        
        # Step 3: Summary
        print("\n" + "=" * 60)
        print("Multi-Language TTS Processing Complete!")
        print("=" * 60)
        print(f"Successfully processed {len(all_audio_files)} audio samples across multiple languages")
        print(f"Generated audio files:")
        
        # Group files by language for better display
        for language in MODELS_CONFIG.keys():
            language_files = [f for f in all_audio_files if f['language'] == language]
            if language_files:
                print(f"\n  {language.title()} ({len(language_files)} files):")
                for file_info in language_files:
                    print(f"    - {file_info['filename']} - \"{file_info['text'][:50]}...\"")
        
        print(f"\nTechnical Details:")
        print(f"  - Languages supported: {', '.join(MODELS_CONFIG.keys())}")
        print(f"  - Models used:")
        for lang, config in MODELS_CONFIG.items():
            print(f"    - {lang.title()}: {config['model_name']}")
        print(f"  - Sampling rate: {sampling_rate} Hz")
        print(f"  - Audio format: WAV (16-bit)")
        
        print(f"\nNext Steps:")
        print(f"  - Play the audio files using any media player")
        print(f"  - Compare pronunciation quality between languages")
        print(f"  - Try adding more languages or custom texts")
        print(f"  - Experiment with different TTS models from Hugging Face")
        
    except Exception as e:
        print(f"\nCritical error in main execution: {str(e)}")
        print("\nTroubleshooting Tips:")
        print("1. Ensure internet connection for model download")
        print("2. Check if all required packages are installed")
        print("3. Try a different model if the current one fails")
        print("4. Verify sufficient disk space for model files")
        raise

if __name__ == "__main__":
    # Run the comprehensive multi-language demo
    main()
    
    # Run the assignment-specific demo following exact requirements
    print("\n" + "=" * 80)
    assignment_demo()