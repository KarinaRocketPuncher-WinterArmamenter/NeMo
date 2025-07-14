"""
Script to prepare data for Conformer-Transformer streaming ASR training.
Creates manifest files and can optionally train a tokenizer.

Usage:
    python prepare_streaming_asr_data.py \
        --audio_dir=/path/to/audio/files \
        --transcript_file=/path/to/transcripts.txt \
        --output_dir=/path/to/output \
        --train_tokenizer \
        --tokenizer_vocab_size=1024
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import subprocess

import librosa
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from nemo.utils import logging


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds"""
    try:
        duration = librosa.get_duration(filename=audio_path)
        return duration
    except Exception as e:
        logging.warning(f"Error reading {audio_path}: {e}")
        return -1


def create_manifest_entry(audio_path: str, text: str, duration: float = None) -> Dict:
    """Create a single manifest entry"""
    if duration is None:
        duration = get_audio_duration(audio_path)
        
    return {
        "audio_filepath": audio_path,
        "text": text,
        "duration": duration,
    }


def write_manifest(entries: List[Dict], output_path: str):
    """Write manifest entries to file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logging.info(f"Wrote {len(entries)} entries to {output_path}")


def train_tokenizer(
    texts: List[str],
    output_dir: str,
    vocab_size: int = 1024,
    model_type: str = "bpe",
):
    """Train a SentencePiece tokenizer"""
    
    # Write texts to temporary file
    temp_text_file = os.path.join(output_dir, "temp_texts.txt")
    with open(temp_text_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
            
    # Train tokenizer
    model_prefix = os.path.join(output_dir, "tokenizer")
    spm.SentencePieceTrainer.train(
        input=temp_text_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )
    
    # Clean up temp file
    os.remove(temp_text_file)
    
    logging.info(f"Trained tokenizer saved to {model_prefix}.model")
    
    # Also create vocab.txt for compatibility
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    vocab_path = os.path.join(output_dir, "vocab.txt")
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for i in range(sp.get_piece_size()):
            f.write(sp.id_to_piece(i) + '\n')
            
    logging.info(f"Vocabulary saved to {vocab_path}")


def prepare_data_from_directory(
    audio_dir: str,
    transcript_file: str,
    output_dir: str,
    test_size: float = 0.1,
    val_size: float = 0.1,
    min_duration: float = 0.1,
    max_duration: float = 20.0,
    train_tokenizer_flag: bool = False,
    tokenizer_vocab_size: int = 1024,
):
    """Prepare ASR data from directory structure"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read transcripts
    audio_to_text = {}
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '\t' in line:
                audio_id, text = line.split('\t', 1)
            else:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts
                else:
                    continue
            audio_to_text[audio_id] = text
            
    # Find audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.opus'}
    manifest_entries = []
    
    logging.info(f"Searching for audio files in {audio_dir}")
    
    for audio_path in tqdm(Path(audio_dir).rglob('*')):
        if audio_path.suffix.lower() in audio_extensions:
            # Get audio ID (filename without extension)
            audio_id = audio_path.stem
            
            # Check if we have transcript
            if audio_id in audio_to_text:
                # Get duration
                duration = get_audio_duration(str(audio_path))
                
                # Filter by duration
                if min_duration <= duration <= max_duration:
                    entry = create_manifest_entry(
                        str(audio_path),
                        audio_to_text[audio_id],
                        duration
                    )
                    manifest_entries.append(entry)
                    
    logging.info(f"Found {len(manifest_entries)} valid audio-transcript pairs")
    
    # Split data
    train_entries, test_entries = train_test_split(
        manifest_entries, test_size=test_size, random_state=42
    )
    
    train_entries, val_entries = train_test_split(
        train_entries, test_size=val_size/(1-test_size), random_state=42
    )
    
    # Write manifests
    write_manifest(train_entries, os.path.join(output_dir, "train_manifest.json"))
    write_manifest(val_entries, os.path.join(output_dir, "dev_manifest.json"))
    write_manifest(test_entries, os.path.join(output_dir, "test_manifest.json"))
    
    # Train tokenizer if requested
    if train_tokenizer_flag:
        logging.info("Training tokenizer...")
        texts = [entry['text'] for entry in train_entries]
        train_tokenizer(
            texts,
            output_dir,
            vocab_size=tokenizer_vocab_size,
        )
        
    # Create data statistics file
    stats = {
        "total_entries": len(manifest_entries),
        "train_entries": len(train_entries),
        "val_entries": len(val_entries),
        "test_entries": len(test_entries),
        "total_duration_hours": sum(e['duration'] for e in manifest_entries) / 3600,
        "train_duration_hours": sum(e['duration'] for e in train_entries) / 3600,
        "val_duration_hours": sum(e['duration'] for e in val_entries) / 3600,
        "test_duration_hours": sum(e['duration'] for e in test_entries) / 3600,
    }
    
    with open(os.path.join(output_dir, "data_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
        
    logging.info("Data preparation completed!")
    logging.info(f"Statistics: {stats}")


def prepare_librispeech_style_data(
    data_root: str,
    output_dir: str,
    splits: List[str] = ["train-clean-100", "dev-clean", "test-clean"],
    train_tokenizer_flag: bool = False,
    tokenizer_vocab_size: int = 1024,
):
    """Prepare data in LibriSpeech style format"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_train_entries = []
    
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            logging.warning(f"Split directory {split_dir} not found, skipping...")
            continue
            
        manifest_entries = []
        
        # Walk through LibriSpeech directory structure
        for speaker_dir in Path(split_dir).iterdir():
            if not speaker_dir.is_dir():
                continue
                
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                    
                # Read transcript file
                trans_file = chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                if not trans_file.exists():
                    continue
                    
                # Read transcripts
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) != 2:
                            continue
                            
                        utt_id, text = parts
                        audio_path = chapter_dir / f"{utt_id}.flac"
                        
                        if audio_path.exists():
                            duration = get_audio_duration(str(audio_path))
                            entry = create_manifest_entry(
                                str(audio_path),
                                text.lower(),  # LibriSpeech convention
                                duration
                            )
                            manifest_entries.append(entry)
                            
        # Write manifest for this split
        output_name = "train_manifest.json" if "train" in split else f"{split}_manifest.json"
        output_name = output_name.replace("dev", "val")  # Use 'val' instead of 'dev'
        
        write_manifest(manifest_entries, os.path.join(output_dir, output_name))
        
        if "train" in split:
            all_train_entries.extend(manifest_entries)
            
    # Train tokenizer on all training data
    if train_tokenizer_flag and all_train_entries:
        logging.info("Training tokenizer on all training data...")
        texts = [entry['text'] for entry in all_train_entries]
        train_tokenizer(
            texts,
            output_dir,
            vocab_size=tokenizer_vocab_size,
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR data for streaming training")
    
    # Data source options
    parser.add_argument(
        "--data_format",
        type=str,
        choices=["directory", "librispeech"],
        default="directory",
        help="Data format to process",
    )
    
    # Directory format options
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing audio files (for directory format)",
    )
    parser.add_argument(
        "--transcript_file",
        type=str,
        help="File containing transcripts (for directory format)",
    )
    
    # LibriSpeech format options
    parser.add_argument(
        "--data_root",
        type=str,
        help="Root directory of LibriSpeech-style data",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100", "dev-clean", "test-clean"],
        help="LibriSpeech splits to process",
    )
    
    # Common options
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for manifests and tokenizer",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Fraction of data for test set",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of data for validation set",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=20.0,
        help="Maximum audio duration in seconds",
    )
    
    # Tokenizer options
    parser.add_argument(
        "--train_tokenizer",
        action="store_true",
        help="Train a tokenizer on the training data",
    )
    parser.add_argument(
        "--tokenizer_vocab_size",
        type=int,
        default=1024,
        help="Vocabulary size for tokenizer",
    )
    
    args = parser.parse_args()
    
    # Process based on data format
    if args.data_format == "directory":
        if not args.audio_dir or not args.transcript_file:
            raise ValueError("--audio_dir and --transcript_file required for directory format")
            
        prepare_data_from_directory(
            audio_dir=args.audio_dir,
            transcript_file=args.transcript_file,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            train_tokenizer_flag=args.train_tokenizer,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
        )
        
    elif args.data_format == "librispeech":
        if not args.data_root:
            raise ValueError("--data_root required for librispeech format")
            
        prepare_librispeech_style_data(
            data_root=args.data_root,
            output_dir=args.output_dir,
            splits=args.splits,
            train_tokenizer_flag=args.train_tokenizer,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
        )


if __name__ == "__main__":
    main()
