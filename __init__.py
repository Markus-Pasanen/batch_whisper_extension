import os
import uuid
import torch
import whisper
import torchaudio
import logging
import comfy.model_management as mm
import comfy.model_patcher
import folder_paths

WHISPER_MODEL_SUBDIR = os.path.join("stt", "whisper")
logger = logging.getLogger(__name__)
WHISPER_PATCHER_CACHE = {}

class WhisperModelWrapper(torch.nn.Module):
    """Wraps Whisper model for ComfyUI model management"""
    def __init__(self, model_name, download_root):
        super().__init__()
        self.model_name = model_name
        self.download_root = download_root
        self.whisper_model = None
        self.model_loaded_weight_memory = 0

    def load_model(self, device):
        self.whisper_model = whisper.load_model(
            self.model_name,
            download_root=self.download_root,
            device=device
        )
        self.model_loaded_weight_memory = sum(
            p.numel() * p.element_size() for p in self.whisper_model.parameters()
        )

class WhisperPatcher(comfy.model_patcher.ModelPatcher):
    """Handles VRAM-safe loading/unloading for Whisper"""
    def patch_model(self, device_to=None, *args, **kwargs):
        target_device = self.load_device
        if self.model.whisper_model is None:
            logger.info(f"Loading Whisper model '{self.model.model_name}' to {target_device}...")
            self.model.load_model(target_device)
            self.size = self.model.model_loaded_weight_memory
        return super().patch_model(device_to=target_device, *args, **kwargs)

    def unpatch_model(self, device_to=None, unpatch_weights=True, *args, **kwargs):
        if unpatch_weights:
            logger.info(f"Offloading Whisper model '{self.model.model_name}'")
            self.model.whisper_model = None
            self.model.model_loaded_weight_memory = 0
            mm.soft_empty_cache()
        return super().unpatch_model(device_to, unpatch_weights, *args, **kwargs)

class BatchWhisperNode:
    languages_by_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio1": ("AUDIO",),
                "speaker1": ("STRING", {"default": "Speaker1"}),
                "audio2": ("AUDIO",),
                "speaker2": ("STRING", {"default": "Speaker2"}),
                "audio3": ("AUDIO",),
                "speaker3": ("STRING", {"default": "Speaker3"}),
                "model_name": ([
                    'tiny.en','tiny','base.en','base','small.en','small','medium.en','medium',
                    'large-v1','large-v2','large-v3','large','large-v3-turbo','turbo'
                ], {"default": "large-v3-turbo"}),
            },
            "optional": {
                "language": (["auto"] + [s.capitalize() for s in sorted(list(whisper.tokenizer.LANGUAGES.values()))], {"default":"auto"}),
                "prompt": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "transcribe_batch"
    CATEGORY = "Audio/Transcription"

    def transcribe_batch(self, audio1, speaker1, audio2, speaker2, audio3, speaker3, model_name, language, prompt):
        audio_list = [audio1, audio2, audio3]
        speaker_list = [speaker1, speaker2, speaker3]

        # Load or reuse cached Whisper patcher
        if model_name not in WHISPER_PATCHER_CACHE:
            load_device = mm.get_torch_device()
            download_root = os.path.join(folder_paths.models_dir, WHISPER_MODEL_SUBDIR)
            wrapper = WhisperModelWrapper(model_name, download_root)
            patcher = WhisperPatcher(wrapper, load_device, mm.unet_offload_device(), size=0)
            WHISPER_PATCHER_CACHE[model_name] = patcher

        patcher = WHISPER_PATCHER_CACHE[model_name]
        mm.load_model_gpu(patcher)
        whisper_model = patcher.model.whisper_model
        if whisper_model is None:
            return ("Error: Whisper model failed to load.",)

        # Build language code
        if language != "auto":
            if BatchWhisperNode.languages_by_name is None:
                BatchWhisperNode.languages_by_name = {v.lower(): k for k,v in whisper.tokenizer.LANGUAGES.items()}
            language_code = BatchWhisperNode.languages_by_name.get(language.lower(), None)
        else:
            language_code = "auto"

        full_segments = []
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        # Transcribe each audio
        for speaker, audio in zip(speaker_list, audio_list):
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
            torchaudio.save(temp_path, waveform.squeeze(0), sample_rate)

            try:
                result = whisper_model.transcribe(
                    temp_path,
                    initial_prompt=prompt,
                    language=language_code,
                    word_timestamps=True
                )
                for segment in result.get("segments", []):
                    full_segments.append({
                        "speaker": speaker,
                        "text": segment["text"].strip(),
                        "start": segment["start"],
                        "end": segment["end"]
                    })
            except Exception as e:
                full_segments.append({
                    "speaker": speaker,
                    "text": f"[Error: {str(e)}]",
                    "start": 0,
                    "end": 0
                })

        # Sort all segments by start time
        full_segments.sort(key=lambda x: x["start"])

        # Format transcript with timestamps
        transcript_lines = [
            f"{seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}"
            for seg in full_segments
        ]
        full_transcript = "\n".join(transcript_lines)

        return (full_transcript,)

# ComfyUI node mappings
NODE_CLASS_MAPPINGS = {
    "BatchWhisperNode": BatchWhisperNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchWhisperNode": "Batch Whisper (3 Speakers, Timestamped)"
}
