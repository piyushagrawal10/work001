#!/usr/bin/env python3
"""LTX2B independent-GPU experiment queue. Uses your existing Python environment.
No pip, uv install, or package replacement is performed. GPU inference is untested
by the build host. --dry-run and --write-plan need only the standard library.
"""
import argparse, json, os, sys, hashlib
from pathlib import Path
SHARED_PLAN={'schema_version': 1, 'description': 'Shared image-to-video diagnostic experiments. Each runner uses the same supplied input images and prompts; model-specific settings belong in the runner configuration.', 'input_keys': {'pig': 'User-supplied approved single-pig image, pig.png.', 'pair': 'User-supplied approved two-character image, with both characters visible and clearly separated.', 'scene': 'User-supplied approved two-character image already composed into the intended background.', 'pig_object': 'Optional distinct pig_object.png: approved pig with a visibly present reachable interaction object. Do not replace the baseline pig.png.'}, 'execution_notes': ['Do not substitute a missing input or invent an unseen object. Skip the affected experiment with an explicit reason.', 'Test 05 uses the distinct optional pig_object.png input. It is disabled by default and must be skipped if that image is missing or does not visibly contain a reachable interaction object. Do not alter baseline pig.png to enable this test.', 'Test 10 is disabled by default and requires explicit opt-in plus a model-supported longer duration or frame count. Do not silently loop, duplicate, or interpolate frames to claim longer generation.', "Use the model's supported default short duration for tests 01 through 09; record actual frame count, FPS, resolution, seed, checkpoint, and settings alongside each result.", 'A shared numeric seed does not imply identical noise across different models. Compare tests 01 and 02 within a model for seed variability.', 'If a runner does not support negative prompts, omit that argument and record this limitation rather than failing.'], 'experiments': [{'id': '01_pig_silent_speech', 'input_key': 'pig', 'prompt': 'A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. Keep the same face, spectacles if present, clothing, proportions, pencil shading, colors, and background. The feet stay grounded. Locked camera, no cut, no new objects, no subtitles.', 'seed': 1701, 'purpose': 'Baseline identity preservation, mouth animation, and coordinated hand and upper-body acting.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity change, distorted face, warped spectacles, duplicate limbs, extra fingers, fused hands, floating feet, changing clothes, flickering lines, camera motion, scene cut, text, watermark'}, {'id': '02_pig_silent_speech_second_seed', 'input_key': 'pig', 'prompt': 'A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. Keep the same face, spectacles if present, clothing, proportions, pencil shading, colors, and background. The feet stay grounded. Locked camera, no cut, no new objects, no subtitles.', 'seed': 1702, 'purpose': 'Repeat the exact baseline prompt with a second seed to distinguish reproducible capability from a lucky sample.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity change, distorted face, warped spectacles, duplicate limbs, extra fingers, fused hands, floating feet, changing clothes, flickering lines, camera motion, scene cut, text, watermark'}, {'id': '03_weight_shift_walk_stop', 'input_key': 'pig', 'prompt': 'The pig in the supplied image shifts its weight naturally, takes two small deliberate steps sideways within the existing frame, then comes to a balanced stop. Show coordinated hips, knees, feet, shoulders, and gentle arm counter-motion. Each foot plants on the ground before the other lifts. Preserve the exact character design, face, clothing, short leg proportions, and hand-drawn painted texture. Keep the whole body visible, the background unchanged, and the camera locked. One continuous shot.', 'seed': 1701, 'purpose': 'Test grounded full-body coordination, short strides, foot sliding, and settling after locomotion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'sliding feet, gliding, skating, elongated legs, stretching body, duplicate feet, extra limbs, identity change, floating, camera tracking, scene cut, photorealism'}, {'id': '04_turn_return_glasses', 'input_key': 'pig', 'prompt': 'The pig in the supplied image slowly turns its head and upper torso a modest amount toward the other side, pauses briefly, then returns to its starting orientation. Preserve the same facial structure, eyes, cheeks, snout, ears, and spectacles if present. Facial features remain correctly attached through the turn, with natural occlusion of the far eye and spectacle arm. Add a subtle blink and breathing. Preserve the clothing and hand-drawn artwork. Feet remain planted, camera locked, background unchanged, no cut.', 'seed': 1701, 'purpose': 'Stress-test left/right view changes, facial identity, glasses geometry, and correct occlusion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'melting face, eye swapping, duplicated eyes, disappearing glasses, detached spectacles, changing cheeks, shifting snout, rubber neck, identity change, camera movement, scene cut'}, {'id': '05_visible_object_interaction', 'input_key': 'pig_object', 'prompt': "The pig interacts only with the object already visibly present within reach in the supplied image. The pig looks at it, reaches toward it with coordinated shoulder, elbow, wrist, and torso motion, briefly makes clear hand contact, then relaxes the arm. Keep the object's identity, shape, position, and material stable. Do not add or remove any object. Preserve the pig's face, clothing, proportions, and painted drawing style. Keep the feet planted and camera locked in one continuous shot.", 'seed': 1701, 'purpose': 'Test contact, hand-object geometry, eye direction, and coordinated reaching without inventing a prop.', 'experimental': False, 'enabled_by_default': False, 'requires': ['pig_object_input_available', 'pig_object_contains_visible_reachable_object'], 'skip_unless_requirements_confirmed': True, 'negative_prompt': 'new object, disappearing object, changing prop, object passing through hand, fused hand and object, detached arm, duplicate fingers, floating object, identity change, scene cut'}, {'id': '06_pair_alternating_speech', 'input_key': 'pair', 'prompt': 'The two characters in the supplied image have a brief silent conversation. First the character on the left speaks with clear mouth changes and one restrained hand gesture while the character on the right listens. Then the left character settles and the right character replies with mouth movement and a small expressive gesture. Preserve both distinct identities, faces, proportions, clothing, and hand-drawn textures. Keep their positions and feet grounded, with natural blinks and subtle body motion. Locked camera, unchanged background, no cut, no subtitles.', 'seed': 1701, 'purpose': 'Test speaker alternation, separate character identities, and independent acting without simultaneous uncontrolled motion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity swap, merged characters, synchronized identical gestures, duplicate limbs, both mouths constantly talking, face distortion, changing clothes, camera movement, scene cut, text'}, {'id': '07_pair_reacting', 'input_key': 'pair', 'prompt': 'The left character in the supplied image makes a small confident gesture toward the right character. The right character notices it, reacts with a surprised expression and a slight backward upper-body shift, then settles into a curious listening pose. The left character watches the reaction. Preserve the two distinct character designs and all clothing, facial features, and painted textures. Keep feet grounded and clear separation between bodies. The camera and background remain still in one continuous shot.', 'seed': 1701, 'purpose': 'Test cause-and-effect acting, reaction timing, gaze, and independent coordinated body motion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity swap, merging bodies, duplicated character, random motion, sliding feet, floating, distorted hands, facial drift, camera movement, scene cut, text'}, {'id': '08_pair_background_locked', 'input_key': 'scene', 'prompt': 'Animate only the two characters already present in the supplied scene. They exchange a small nod and a restrained conversational hand gesture with subtle breathing and natural blinks. Keep their identities, proportions, clothing, and hand-drawn style consistent. Keep every background object, architectural line, floor contact, lighting direction, and camera position fixed. No new objects, no moving scenery, no camera movement, no zoom, no cuts.', 'seed': 1701, 'purpose': 'Test character animation inside a composed scene while preserving background geometry and camera lock.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'background warping, moving scenery, shifting architecture, flickering lighting, camera pan, zoom, parallax, new objects, character identity change, floating feet, scene cut'}, {'id': '09_baseline_style_preservation', 'input_key': 'pig', 'prompt': "A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. The feet stay grounded. Preserve the source illustration's pencil contours, soft painted shading, paper-like texture, muted old-school palette, and gently imperfect drawn edges throughout every frame. Maintain the exact face, spectacles if present, clothing, proportions, and background. Do not reinterpret the art as glossy 3D or photorealism. Locked camera, no cut, no new objects, no subtitles.", 'seed': 1701, 'purpose': 'Compare against test 01 with the same seed and action but stronger source-style instructions; inspect texture flicker, line drift, and unwanted 3D conversion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'glossy 3D, plastic skin, photorealism, smooth vector art, excessive contrast, oversharpening, texture flicker, style drift, identity change, distorted face, warped spectacles, duplicate limbs, text, watermark'}, {'id': '10_longer_continuity_experimental', 'input_key': 'pig', 'prompt': 'In one continuous longer shot, the pig in the supplied image speaks silently, makes one restrained hand gesture with coordinated shoulder and torso motion, settles, looks slightly to one side, and returns to the starting orientation. Maintain the same identity, facial features, spectacles if present, clothing, proportions, pencil contours, and painted texture for the entire shot. Keep feet grounded, camera locked, and background unchanged. Natural pacing, no repeated loop, no cuts, no new objects, no subtitles.', 'seed': 1701, 'purpose': 'Optional longer-duration temporal consistency and memory-cost stress test, only with explicit opt-in and supported model settings.', 'experimental': True, 'enabled_by_default': False, 'requires': ['explicit_longer_test_opt_in', 'model_supported_longer_duration'], 'skip_unless_requirements_confirmed': True, 'negative_prompt': 'identity drift, changing costume, distorted spectacles, extra limbs, looping action, repeated frames, frozen character, camera motion, scene cut, background warping, text'}]}
WORKER_SOURCE='"""LTX 2B 0.9.8 distilled single-pass image-to-video. No remote Python execution."""\nimport argparse, json, os, time\nfrom pathlib import Path\n\ndef main():\n    p=argparse.ArgumentParser()\n    p.add_argument(\'--image\', required=True); p.add_argument(\'--output\', required=True)\n    p.add_argument(\'--prompt\', default=\'A hand-drawn cartoon pig in a brown patched suit turns his head, smiles, and makes one small friendly hand gesture. Full body visible. Fixed camera. Preserve his face, glasses, proportions, clothes and pencil texture.\')\n    p.add_argument(\'--width\', type=int, default=864); p.add_argument(\'--height\', type=int, default=480)\n    p.add_argument(\'--frames\', type=int, default=121); p.add_argument(\'--fps\', type=int, default=24)\n    p.add_argument(\'--seed\', type=int, default=42)\n    p.add_argument(\'--offload\', choices=[\'sequential\',\'model\',\'none\'], default=\'model\')\n    p.add_argument(\'--offline\', action=\'store_true\')\n    a=p.parse_args()\n    if a.offline:\n        os.environ[\'HF_HUB_OFFLINE\']=\'1\'\n        os.environ[\'TRANSFORMERS_OFFLINE\']=\'1\'\n    assert a.width%32==0 and a.height%32==0, \'Dimensions must be multiples of 32.\'\n    assert a.frames>=9 and (a.frames-1)%8==0, \'Frames must be 8*n+1.\'\n    import torch\n    from diffusers import LTXConditionPipeline, LTXVideoTransformer3DModel, AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler\n    from diffusers.utils import export_to_video\n    from transformers import T5EncoderModel, T5Tokenizer\n    from huggingface_hub import hf_hub_download, snapshot_download\n    from PIL import Image, ImageOps\n    if not torch.cuda.is_available(): raise RuntimeError(\'Select a GPU runtime before inference.\')\n    gpu=torch.cuda.get_device_properties(0)\n    if gpu.total_memory < 7*1024**3:\n        raise RuntimeError(\'This first-test profile is not validated below 8 GB VRAM. Use a Colab GPU with >=8 GB; 16 GB preferred.\')\n    native_bf16 = torch.cuda.get_device_capability(0)[0] >= 8\n    dtype = torch.bfloat16 if native_bf16 else torch.float16\n    root=Path(__file__).resolve().parents[1]\n    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)\n    print(json.dumps({\'gpu\':gpu.name,\'vram_gib\':gpu.total_memory/1024**3,\'dtype\':str(dtype),\'settings\':vars(a)},indent=2),flush=True)\n    ckpt=hf_hub_download(\'Lightricks/LTX-Video\',\'ltxv-2b-0.9.8-distilled.safetensors\',revision=\'8984fa25007f376c1a299016d0957a37a2f797bb\')\n    text_dir=snapshot_download(\'PixArt-alpha/PixArt-XL-2-1024-MS\',revision=\'b89adadeccd9ead2adcb9fa2825d3fabec48d404\',allow_patterns=[\'text_encoder/*.json\',\'text_encoder/*.safetensors\',\'tokenizer/*\'])\n    # Configs are vendored official 0.9.5 schemas, whose shapes match the actual\n    # 2B 0.9.8 checkpoint (validated from its safetensors header). We load the\n    # 0.9.8 WEIGHTS, never silently substitute the older 0.9.5 weights.\n    transformer=LTXVideoTransformer3DModel.from_single_file(ckpt,config=str(root/\'configs\'),subfolder=\'transformer\',torch_dtype=dtype,local_files_only=True)\n    vae=AutoencoderKLLTXVideo.from_single_file(ckpt,config=str(root/\'configs\'),subfolder=\'vae\',torch_dtype=dtype,local_files_only=True)\n    text_encoder=T5EncoderModel.from_pretrained(text_dir,subfolder=\'text_encoder\',torch_dtype=dtype,local_files_only=True)\n    tokenizer=T5Tokenizer.from_pretrained(text_dir,subfolder=\'tokenizer\',local_files_only=True)\n    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(root/\'configs\'),subfolder=\'scheduler\')\n    pipe=LTXConditionPipeline(transformer=transformer,vae=vae,text_encoder=text_encoder,tokenizer=tokenizer,scheduler=scheduler)\n    pipe.vae.enable_tiling()\n    if a.offload==\'sequential\': pipe.enable_sequential_cpu_offload()\n    elif a.offload==\'model\': pipe.enable_model_cpu_offload()\n    else: pipe.to(\'cuda:0\')\n    image=ImageOps.exif_transpose(Image.open(a.image)).convert(\'RGBA\')\n    bg=Image.new(\'RGBA\',image.size,(225,215,194,255));bg.alpha_composite(image)\n    image=ImageOps.pad(bg.convert(\'RGB\'),(a.width,a.height),method=Image.Resampling.LANCZOS,color=(225,215,194),centering=(.5,.5))\n    image.save(out.with_suffix(\'.input.png\'))\n    torch.cuda.reset_peak_memory_stats()\n    started=time.time()\n    def verify_latents(_pipe, step, timestep, values):\n        if not torch.isfinite(values[\'latents\']).all().item():\n            raise RuntimeError(f\'Non-finite LTX latents after step {step}; FP16 numerical stability is not guaranteed on T4\')\n        return values\n    # All eight allowed distilled timesteps; single-pass baseline, no latent upscaler.\n    video=pipe(image=image,prompt=a.prompt,width=a.width,height=a.height,num_frames=a.frames,frame_rate=a.fps,\n               timesteps=[1000,993.7,987.5,981.2,975,909.4,725,421.9],guidance_scale=1.0,\n               decode_timestep=0.05,decode_noise_scale=0.025,image_cond_noise_scale=0.0,\n               generator=torch.Generator(\'cpu\').manual_seed(a.seed),\n               callback_on_step_end=verify_latents,output_type=\'pil\').frames[0]\n    if len(video) != a.frames:\n        raise RuntimeError(f\'Expected {a.frames} frames, got {len(video)}\')\n    # Explicit encoder/pixel-format contract: all exports are H.264 yuv420p.\n    import imageio.v2 as imageio\n    import numpy as np\n    with imageio.get_writer(str(out), fps=a.fps, codec=\'libx264\', pixelformat=\'yuv420p\',\n                            macro_block_size=None, ffmpeg_log_level=\'warning\') as writer:\n        for frame in video:\n            pixels=np.asarray(frame.convert(\'RGB\'))\n            writer.append_data(pixels)\n    reader=imageio.get_reader(str(out)); encoded_count=reader.count_frames(); reader.close()\n    if encoded_count != a.frames:\n        raise RuntimeError(f\'Encoded video has {encoded_count} frames, expected {a.frames}\')\n    data={\'model\':\'ltxv-2b-0.9.8-distilled\',\'elapsed_seconds\':time.time()-started,\'gpu\':gpu.name,\'dtype\':str(dtype),\'settings\':vars(a),\'frames\':len(video),\'peak_vram_allocated_bytes\':torch.cuda.max_memory_allocated(),\'peak_vram_reserved_bytes\':torch.cuda.max_memory_reserved(),\'checkpoint_revision\':\'8984fa25007f376c1a299016d0957a37a2f797bb\',\'distilled_timesteps\':[1000,993.7,987.5,981.2,975,909.4,725,421.9],\'guidance_scale\':1.0,\'cuda_visible_devices\':os.environ.get(\'CUDA_VISIBLE_DEVICES\')}\n    out.with_suffix(\'.json\').write_text(json.dumps(data,indent=2))\n    print(\'Saved\',out,flush=True)\nif __name__==\'__main__\': main()\n'
CONFIG_FILES={'configs/scheduler/scheduler_config.json': '{\n  "_class_name": "FlowMatchEulerDiscreteScheduler",\n  "_diffusers_version": "0.33.0.dev0",\n  "base_image_seq_len": 256,\n  "base_shift": 0.5,\n  "invert_sigmas": false,\n  "max_image_seq_len": 4096,\n  "max_shift": 1.15,\n  "num_train_timesteps": 1000,\n  "shift": 1.0,\n  "shift_terminal": null,\n  "time_shift_type": "exponential",\n  "use_beta_sigmas": false,\n  "use_dynamic_shifting": false,\n  "use_exponential_sigmas": false,\n  "use_karras_sigmas": false\n}\n', 'configs/vae/config.json': '{\n  "_class_name": "AutoencoderKLLTXVideo",\n  "_diffusers_version": "0.33.0.dev0",\n  "block_out_channels": [\n    128,\n    256,\n    512,\n    1024,\n    2048\n  ],\n  "decoder_block_out_channels": [\n    256,\n    512,\n    1024\n  ],\n  "decoder_causal": false,\n  "decoder_inject_noise": [\n    false,\n    false,\n    false,\n    false\n  ],\n  "decoder_layers_per_block": [\n    5,\n    5,\n    5,\n    5\n  ],\n  "decoder_spatio_temporal_scaling": [\n    true,\n    true,\n    true\n  ],\n  "down_block_types": [\n    "LTXVideo095DownBlock3D",\n    "LTXVideo095DownBlock3D",\n    "LTXVideo095DownBlock3D",\n    "LTXVideo095DownBlock3D"\n  ],\n  "downsample_type": [\n    "spatial",\n    "temporal",\n    "spatiotemporal",\n    "spatiotemporal"\n  ],\n  "encoder_causal": true,\n  "in_channels": 3,\n  "latent_channels": 128,\n  "layers_per_block": [\n    4,\n    6,\n    6,\n    2,\n    2\n  ],\n  "out_channels": 3,\n  "patch_size": 4,\n  "patch_size_t": 1,\n  "resnet_norm_eps": 1e-06,\n  "scaling_factor": 1.0,\n  "spatial_compression_ratio": 32,\n  "spatio_temporal_scaling": [\n    true,\n    true,\n    true,\n    true\n  ],\n  "temporal_compression_ratio": 8,\n  "timestep_conditioning": true,\n  "upsample_factor": [\n    2,\n    2,\n    2\n  ],\n  "upsample_residual": [\n    true,\n    true,\n    true\n  ]\n}\n', 'configs/transformer/config.json': '{\n  "_class_name": "LTXVideoTransformer3DModel",\n  "_diffusers_version": "0.33.0.dev0",\n  "activation_fn": "gelu-approximate",\n  "attention_bias": true,\n  "attention_head_dim": 64,\n  "attention_out_bias": true,\n  "caption_channels": 4096,\n  "cross_attention_dim": 2048,\n  "in_channels": 128,\n  "norm_elementwise_affine": false,\n  "norm_eps": 1e-06,\n  "num_attention_heads": 32,\n  "num_layers": 28,\n  "out_channels": 128,\n  "patch_size": 1,\n  "patch_size_t": 1,\n  "qk_norm": "rms_norm_across_heads"\n}\n'}

"""Standard-library multi-GPU subprocess queue. No torch imports or installation.

A command must use the exact Python executable from a working model environment.
GPU tokens identify physical IDs/UUIDs; inherited CUDA_VISIBLE_DEVICES is respected.
One GPU per subprocess, one worker per requested GPU. Devices are never pooled.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()) + '-' + str(threading.get_ident()))
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize_gpu_ids(gpu_ids):
    """Return requested physical tokens, respecting externally allocated visibility.

    None/'auto': use inherited CUDA_VISIBLE_DEVICES if set, otherwise nvidia-smi.
    Explicit numeric IDs are physical tokens, NOT remapped visible ordinals.
    UUID allocation plus numeric requests is rejected rather than guessed.
    """
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    allocation = None if inherited is None else [x.strip() for x in inherited.split(',') if x.strip()]
    if gpu_ids is None or gpu_ids == 'auto':
        if allocation is not None:
            tokens = allocation
        else:
            try:
                found = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as e:
                raise ValueError('Cannot discover GPUs. Supply --gpus with allocated GPU IDs/UUIDs; no GPU run has started.') from e
            tokens = [x.strip() for x in found.splitlines() if x.strip()]
        tokens = tokens[:4]
    else:
        tokens = [str(x).strip() for x in (gpu_ids.split(',') if isinstance(gpu_ids, str) else gpu_ids)]
    if not tokens or len(tokens) > 4 or len(set(tokens)) != len(tokens):
        raise ValueError('Choose 1–4 distinct GPU IDs or UUIDs.')
    for token in tokens:
        if not re.fullmatch(r'(?:\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9_/-]+)', token):
            raise ValueError('Invalid GPU token: ' + repr(token))
    if allocation is not None and not set(tokens).issubset(allocation):
        raise ValueError('Requested GPUs are outside inherited CUDA_VISIBLE_DEVICES. Use the allocated tokens exactly, or --gpus auto.')
    return tokens



_ROOT_LOCKS = {}
_ROOT_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def output_root_lock(output_root):
    """Reserve an output root before writing configs, outputs, or queue state.

    Reentrant in the same process/thread so adapters may acquire before preparing
    job files, then call run_queue (which always takes the same lock). Fail-fast
    for other threads/processes. Lock file is outside output root; no output files
    are mutated before acquisition.
    """
    if os.name != 'posix':
        raise RuntimeError('This GPU queue targets Linux workstation environments.')
    import fcntl
    canonical = str(Path(output_root).expanduser().resolve())
    registry_key = (os.getpid(), canonical)
    owner = threading.get_ident()
    with _ROOT_LOCKS_GUARD:
        existing = _ROOT_LOCKS.get(registry_key)
        if existing is not None:
            if existing['owner'] != owner:
                raise RuntimeError('Output root is already active in another thread: ' + canonical)
            existing['depth'] += 1
        else:
            locks = Path(tempfile.gettempdir()) / ('pig-video-output-locks-' + str(os.getuid()))
            locks.mkdir(mode=0o700, exist_ok=True)
            key = hashlib.sha256(canonical.encode()).hexdigest()
            handle = open(locks / (key + '.lock'), 'a+', encoding='utf-8')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise RuntimeError('Output root is in use by another launcher; choose a separate --output-root: ' + canonical)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({'pid': os.getpid(), 'output_root': canonical}))
            handle.flush()
            _ROOT_LOCKS[registry_key] = {'owner': owner, 'depth': 1, 'handle': handle}
    try:
        yield
    finally:
        with _ROOT_LOCKS_GUARD:
            held = _ROOT_LOCKS[registry_key]
            held['depth'] -= 1
            if held['depth'] == 0:
                held['handle'].close()
                del _ROOT_LOCKS[registry_key]


def _gpu_lock_key(token):
    """Canonicalize integer/UUID aliases where nvidia-smi is available."""
    try:
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
        mapping = dict(tuple(x.strip() for x in line.split(',', 1)) for line in lines)
        return mapping.get(token, token)
    except (OSError, subprocess.SubprocessError, ValueError):
        return token


@contextlib.contextmanager
def _device_locks(tokens):
    """Advisory Unix locks protect against another copy of these launchers.

    Does not reserve GPUs against unrelated jobs or bypass a cluster scheduler.
    Locks are released by the OS after crash. Only affects this user's launchers.
    """
    if os.name != 'posix':
        raise RuntimeError('This GPU queue targets Linux workstation environments.')
    import fcntl
    root = Path(tempfile.gettempdir()) / ('pig-video-gpu-locks-' + str(os.getuid()))
    root.mkdir(mode=0o700, exist_ok=True)
    handles = []
    try:
        for token in sorted(tokens):
            key = hashlib.sha256(_gpu_lock_key(token).encode()).hexdigest()[:24]
            handle = open(root / (key + '.lock'), 'a+', encoding='utf-8')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                raise RuntimeError('GPU ' + token + ' is already reserved by another experiment launcher. Use disjoint GPUs or wait for it to finish.')
            handles.append(handle)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({'pid': os.getpid(), 'gpu': token, 'started': time.time()}))
            handle.flush()
        yield
    finally:
        for handle in handles:
            handle.close()


def _stop_process(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=8)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _prepare_jobs(jobs, output_root):
    seen = set()
    prepared = []
    for raw in jobs:
        job = dict(raw)
        jid = str(job['id'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', jid) or jid in seen:
            raise ValueError('Job IDs must be distinct safe directory names: ' + jid)
        seen.add(jid)
        command = job.get('command')
        if not isinstance(command, (list, tuple)) or not command or any(not isinstance(x, str) for x in command):
            raise ValueError('Job command must be a nonempty argv list of strings: ' + jid)
        env = job.get('env') or {}
        if any(key in env for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE')):
            raise ValueError('Per-job environment cannot override device/distributed isolation.')
        outputs = [str(Path(p).expanduser().resolve()) for p in job.get('expected_outputs', [])]
        if not outputs:
            raise ValueError('Each job needs at least one expected output file: ' + jid)
        job.update(id=jid, command=list(command), expected_outputs=outputs, env={str(k):str(v) for k,v in env.items()})
        cwd = job.get('cwd')
        job['cwd'] = str(Path(cwd).expanduser().resolve()) if cwd else None
        # Record worker source and explicitly supplied small inputs, not model weights.
        inputs = set(str(Path(p).expanduser().resolve()) for p in job.get('input_files', []))
        base = Path(job['cwd']) if job['cwd'] else Path.cwd()
        for argument in job['command'][1:]:
            candidate = Path(argument)
            if not candidate.is_absolute():
                candidate = base / candidate
            if candidate.suffix == '.py' and candidate.is_file():
                inputs.add(str(candidate.resolve()))
        missing_inputs = [p for p in inputs if not Path(p).is_file()]
        if missing_inputs:
            raise ValueError('Missing declared input files for ' + jid + ': ' + ', '.join(missing_inputs))
        job['input_hashes'] = {p: _file_digest(p) for p in sorted(inputs)}
        fingerprint_data = {k: job.get(k) for k in ('command', 'expected_outputs', 'metadata', 'cwd', 'env', 'input_hashes')}
        job['fingerprint'] = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True, default=str).encode()).hexdigest()
        job['record_dir'] = str(Path(output_root) / jid)
        prepared.append(job)
    all_outputs = [p for j in prepared for p in j['expected_outputs']]
    if len(set(all_outputs)) != len(all_outputs):
        raise ValueError('Jobs cannot share expected output paths.')
    return prepared


def _can_resume(job):
    try:
        state = json.loads((Path(job['record_dir']) / 'status.json').read_text())
        if state.get('status') != 'succeeded' or state.get('fingerprint') != job['fingerprint']:
            return False
        recorded = state['outputs']
        return all(Path(p).is_file() and Path(p).stat().st_size > 0 and recorded[p]['sha256'] == _file_digest(p) for p in job['expected_outputs'])
    except (OSError, ValueError, KeyError):
        return False


def _run_queue_unlocked(jobs, gpu_ids, output_root, *, dry_run=False, resume=True, timeout_seconds=None):
    """Run independent subprocess experiments and return a machine-readable summary.

    Each job: id, command list, expected_outputs list, optional metadata/cwd/env.
    Include input_files=[image, job_json, ...] to invalidate resume if inputs change.
    Python source paths in command arguments are hashed automatically.
    timeout_seconds applies individually per job. A failed job does not stop others.
    Successful resume requires matching configuration AND output SHA256 hashes.
    No shell invocation, downloads, installations, or imports of ML libraries.
    """
    tokens = normalize_gpu_ids(gpu_ids)
    root = Path(output_root).expanduser().resolve()
    prepared = _prepare_jobs(jobs, root)
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError('timeout_seconds must be positive.')
    plan = {'gpu_ids': tokens, 'workers': len(tokens), 'jobs': [{k:v for k,v in j.items() if k != 'env'} for j in prepared]}
    if dry_run:
        return {'status': 'dry_run', 'plan': plan, 'results': []}
    root.mkdir(parents=True, exist_ok=True)
    # Do not serialize environment overrides, which may contain secrets.
    _atomic_json(root / 'plan.json', plan)
    pending = queue.Queue()
    results = []
    guard = threading.Lock()
    stop = threading.Event()
    children = {}
    started = time.time()
    for job in prepared:
        if resume and _can_resume(job):
            results.append({'id': job['id'], 'status': 'skipped', 'reason': 'matching successful output hashes', 'fingerprint': job['fingerprint']})
        else:
            pending.put(job)

    def worker(gpu):
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            path = Path(job['record_dir'])
            state = {'id': job['id'], 'gpu': gpu, 'fingerprint': job['fingerprint'], 'status': 'running', 'started': time.time(), 'metadata': job.get('metadata', {})}
            proc = None
            try:
                path.mkdir(parents=True, exist_ok=True)
                for output in job['expected_outputs']:
                    Path(output).parent.mkdir(parents=True, exist_ok=True)
                _atomic_json(path / 'status.json', state)
                _atomic_json(path / 'run.json', {k:v for k,v in job.items() if k != 'env'})
                before = {p: (Path(p).stat().st_mtime_ns, Path(p).stat().st_size) if Path(p).is_file() else None for p in job['expected_outputs']}
                env = os.environ.copy()
                env.update(job['env'])
                env['CUDA_VISIBLE_DEVICES'] = gpu
                env['PYTHONUNBUFFERED'] = '1'
                # Prevent accidental inherited torchrun/Slurm distributed launch.
                for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                    env.pop(key, None)
                with open(path / 'stdout.log', 'a', encoding='utf-8') as log:
                    log.write('\n=== new attempt ' + time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ===\n')
                    log.flush()
                    proc = subprocess.Popen(job['command'], cwd=job['cwd'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    with guard:
                        children[gpu] = proc
                    while proc.poll() is None:
                        if stop.is_set():
                            _stop_process(proc)
                            state['status'] = 'cancelled'
                            break
                        if timeout_seconds and time.time() - state['started'] > timeout_seconds:
                            _stop_process(proc)
                            state.update(status='failed', error='timeout')
                            break
                        time.sleep(0.1)
                    state['returncode'] = proc.wait()
                    if state['status'] == 'running':
                        if state['returncode']:
                            state.update(status='failed', error='worker exit ' + str(state['returncode']))
                        else:
                            missing = [p for p in job['expected_outputs'] if not Path(p).is_file() or Path(p).stat().st_size == 0]
                            stale = [p for p in job['expected_outputs'] if Path(p).is_file() and before[p] is not None and before[p] == (Path(p).stat().st_mtime_ns, Path(p).stat().st_size)]
                            if missing or stale:
                                state.update(status='failed', error='expected output missing/empty or unchanged from prior attempt', missing=missing, stale=stale)
                            else:
                                state.update(status='succeeded', outputs={p: {'bytes':Path(p).stat().st_size, 'sha256':_file_digest(p)} for p in job['expected_outputs']})
            except Exception as e:
                state.update(status='failed', error=type(e).__name__ + ': ' + str(e))
            finally:
                if proc is not None:
                    _stop_process(proc)
                with guard:
                    children.pop(gpu, None)
                state['finished'] = time.time()
                state['runtime_seconds'] = state['finished'] - state['started']
                try:
                    _atomic_json(path / 'status.json', state)
                except OSError as e:
                    state['status_write_error'] = str(e)
                with guard:
                    results.append(state)
                    _atomic_json(root / 'summary.json', {'status':'running', 'results':results, 'gpu_ids':tokens})
                print(f"[{gpu}] {job['id']}: {state['status']}", flush=True)
                pending.task_done()

    interrupted = False
    with _device_locks(tokens):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(tokens))
        futures = [executor.submit(worker, gpu) for gpu in tokens]
        try:
            for future in futures:
                future.result()
        except KeyboardInterrupt:
            interrupted = True
            stop.set()
        finally:
            executor.shutdown(wait=True)
    while not pending.empty():
        job = pending.get_nowait()
        results.append({'id':job['id'], 'status':'not_started'})
    ordered = sorted(results, key=lambda r: [j['id'] for j in prepared].index(r['id']))
    failed = any(r['status'] not in ('succeeded', 'skipped') for r in ordered)
    summary = {'status':'interrupted' if interrupted else ('completed_with_failures' if failed else 'completed'), 'started':started, 'finished':time.time(), 'gpu_ids':tokens, 'results':ordered}
    _atomic_json(root / 'summary.json', summary)
    return summary


def run_queue(jobs, gpu_ids, output_root, *, dry_run=False, resume=True, timeout_seconds=None):
    """Run queue under an exclusive output-root lock; see _run_queue_unlocked.

    Adapters that materialize job JSON/source files must hold output_root_lock
    around that preparation AND this call. The same-thread lock is reentrant.
    """
    if dry_run:
        return _run_queue_unlocked(jobs, gpu_ids, output_root, dry_run=True, resume=resume, timeout_seconds=timeout_seconds)
    with output_root_lock(output_root):
        return _run_queue_unlocked(jobs, gpu_ids, output_root, dry_run=False, resume=resume, timeout_seconds=timeout_seconds)




def default_plan(a):
    return SHARED_PLAN

def expand_plan(raw,a):
    if isinstance(raw,dict): raw=raw.get('experiments')
    if not isinstance(raw,list) or not raw: raise ValueError('Manifest must contain nonempty experiments array')
    inputs={'pig':a.pig,'pair':a.pair,'scene':a.scene,'pig_object':a.pig_object}
    rows=[]
    for item in raw:
        if not isinstance(item,dict): raise ValueError('Every experiment must be an object')
        cfg=dict(item.get('model_settings',{}))
        rows.append(dict(item,image=item.get('image',inputs.get(item.get('input_key'),'')),width=cfg.get('width',a.width),height=cfg.get('height',a.height),frames=cfg.get('frames',a.long_frames if item.get('experimental') else a.frames),fps=cfg.get('fps',a.fps),offload=cfg.get('offload',a.offload)))
    return rows
def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',default=sys.executable,help='Existing working LTX interpreter; packages are never installed')
    p.add_argument('--gpus',default='auto',help='Comma-separated visible GPU IDs or UUIDs; defaults respect CUDA_VISIBLE_DEVICES, otherwise0,1,2,3')
    p.add_argument('--pig-object',default='inputs/pig_object.png'); p.add_argument('--pig',default='inputs/pig.png'); p.add_argument('--pair',default='inputs/pig_wolf.png'); p.add_argument('--scene',default='inputs/pig_wolf_background.png')
    p.add_argument('--output',default='outputs/ltx'); p.add_argument('--manifest',help='JSON list of experiment dictionaries')
    p.add_argument('--write-plan',metavar='PATH',help='Write editable JSON plan and exit')
    p.add_argument('--only',help='Comma-separated exact experiment IDs')
    p.add_argument('--width',type=int,default=864); p.add_argument('--height',type=int,default=480)
    p.add_argument('--frames',type=int,default=121); p.add_argument('--long-frames',type=int,default=241); p.add_argument('--fps',type=int,default=24)
    p.add_argument('--offload',choices=['model','sequential','none'],default='model')
    p.add_argument('--offline',action='store_true',help='Require checkpoints and text encoder already in local Hugging Face cache')
    p.add_argument('--confirm-object',action='store_true',help='Confirm pig image visibly contains a reachable interaction object for test05')
    p.add_argument('--include-experimental',action='store_true',help='Include experiment10 native241-frame duration stress test; quality not validated')
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--no-resume',action='store_true')
    p.add_argument('--timeout-seconds',type=float,default=None)
    a=p.parse_args()
    plan=json.loads(Path(a.manifest).read_text()) if a.manifest else default_plan(a)
    if a.write_plan:
        target=Path(a.write_plan); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(plan,indent=2)); print(target); return
    try: plan=expand_plan(plan,a)
    except ValueError as exc: p.error(str(exc))
    if not a.include_experimental:
        plan=[x for x in plan if not x.get('experimental',False)]
    if not a.confirm_object:
        plan=[x for x in plan if x.get('input_key')!='pig_object']
    if a.only:
        wanted=set(a.only.split(',')); plan=[x for x in plan if x.get('id') in wanted]
        if wanted-{x.get('id') for x in plan}: p.error('Unknown experiment IDs in --only')
    dest=Path(a.output).expanduser().resolve()
    with output_root_lock(dest):
        package=dest/'_ltx_worker'; (package/'scripts').mkdir(parents=True,exist_ok=True)
        (package/'scripts/run.py').write_text(WORKER_SOURCE)
        for rel,contents in CONFIG_FILES.items():
            path=package/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(contents)
        jobs=[]; skipped=[]
        allowed={'id','image','input_key','prompt','negative_prompt','purpose','seed','width','height','frames','fps','offload','experimental','comparison_note','enabled_by_default','requires','skip_unless_requirements_confirmed','model_settings'}
        for item in plan:
            if not isinstance(item,dict) or set(item)-allowed: p.error('Invalid or unknown manifest fields')
            name=item.get('id','')
            if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name): p.error('Invalid experiment ID')
            for key in ('width','height'):
                if not isinstance(item.get(key),int) or item[key]<=0 or item[key]%32: p.error('LTX dimensions must be positive multiples of32')
            n=item.get('frames'); fps=item.get('fps')
            if not isinstance(n,int) or n<9 or (n-1)%8: p.error('LTX frames must be8n+1, at least9')
            if not isinstance(fps,int) or fps<=0: p.error('FPS must be positive')
            if item.get('offload') not in ['model','sequential','none']: p.error('Unsupported offload mode')
            image=Path(item['image']).expanduser().resolve()
            if not a.dry_run and not image.is_file():
                skipped.append({'id':name,'reason':f'Missing input image: {image}'}); continue
            out=dest/name/'video.mp4'
            cmd=[a.python,str(package/'scripts/run.py'),'--image',str(image),'--output',str(out),'--prompt',item['prompt']]
            for key in ('width','height','frames','fps','seed','offload'): cmd += ['--'+key,str(item[key])]
            if a.offline: cmd.append('--offline')
            item['negative_prompt_handling']='Not used: distilled CFG1 recipe does not perform negative-prompt guidance'
            jobs.append({'id':name,'command':cmd,'input_files':[str(image)] if image.is_file() else [],'expected_outputs':[str(out),str(out.with_suffix('.json'))],'metadata':dict(item,model='LTX2B0.9.8-distilled',source_revision='8984fa25007f376c1a299016d0957a37a2f797bb',inference_tested_by_build_host=False,worker_source_sha256=hashlib.sha256(WORKER_SOURCE.encode()).hexdigest(),configs_sha256=hashlib.sha256(json.dumps(CONFIG_FILES,sort_keys=True).encode()).hexdigest())})
        (dest/'skipped_inputs.json').write_text(json.dumps(skipped,indent=2))
        if not jobs: p.error('No runnable jobs; check input paths and selection. See skipped_inputs.json')
        gpus=a.gpus
        result=run_queue(jobs,gpus,dest,dry_run=a.dry_run,resume=not a.no_resume,timeout_seconds=a.timeout_seconds)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['status'] in ('dry_run','completed') else 1)
if __name__=='__main__': cli()
