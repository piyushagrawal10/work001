#!/usr/bin/env python3
"""SkyReels V2 I2V 1.3B experiment queue. Uses an EXISTING working environment.
No installation, remote upload, or model download is performed by this file.
"""
from __future__ import annotations
import argparse, json, os, sys, time, subprocess, shutil, hashlib
from pathlib import Path

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




MODEL_ID = 'Skywork/SkyReels-V2-I2V-1.3B-540P'
BASE_STYLE = ('Preserve the supplied hand-drawn old-school pig character, spectacles, cheeks, '
              'brown suit, proportions, pencil shading and muted painted texture. '
              'Full body remains visible. Natural coordinated body acting, grounded feet. '
              'One continuous shot, locked camera, no cuts, no captions. Silent animation. ')
NEGATIVE = ('extra limbs, missing limbs, fused fingers, disconnected hands, distorted face, '
            'changing spectacles, changed clothing, sliding feet, cropped feet, text, watermark, '
            'camera cuts, duplicate characters, flickering outlines')

SHARED_MANIFEST = {'schema_version': 1, 'description': 'Shared image-to-video diagnostic experiments. Each runner uses the same supplied input images and prompts; model-specific settings belong in the runner configuration.', 'input_keys': {'pig': 'User-supplied approved single-pig image, pig.png.', 'pair': 'User-supplied approved two-character image, with both characters visible and clearly separated.', 'scene': 'User-supplied approved two-character image already composed into the intended background.', 'pig_object': 'Optional distinct pig_object.png: approved pig with a visibly present reachable interaction object. Do not replace the baseline pig.png.'}, 'execution_notes': ['Do not substitute a missing input or invent an unseen object. Skip the affected experiment with an explicit reason.', 'Test 05 uses the distinct optional pig_object.png input. It is disabled by default and must be skipped if that image is missing or does not visibly contain a reachable interaction object. Do not alter baseline pig.png to enable this test.', 'Test 10 is disabled by default and requires explicit opt-in plus a model-supported longer duration or frame count. Do not silently loop, duplicate, or interpolate frames to claim longer generation.', "Use the model's supported default short duration for tests 01 through 09; record actual frame count, FPS, resolution, seed, checkpoint, and settings alongside each result.", 'A shared numeric seed does not imply identical noise across different models. Compare tests 01 and 02 within a model for seed variability.', 'If a runner does not support negative prompts, omit that argument and record this limitation rather than failing.'], 'experiments': [{'id': '01_pig_silent_speech', 'input_key': 'pig', 'prompt': 'A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. Keep the same face, spectacles if present, clothing, proportions, pencil shading, colors, and background. The feet stay grounded. Locked camera, no cut, no new objects, no subtitles.', 'seed': 1701, 'purpose': 'Baseline identity preservation, mouth animation, and coordinated hand and upper-body acting.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity change, distorted face, warped spectacles, duplicate limbs, extra fingers, fused hands, floating feet, changing clothes, flickering lines, camera motion, scene cut, text, watermark'}, {'id': '02_pig_silent_speech_second_seed', 'input_key': 'pig', 'prompt': 'A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. Keep the same face, spectacles if present, clothing, proportions, pencil shading, colors, and background. The feet stay grounded. Locked camera, no cut, no new objects, no subtitles.', 'seed': 1702, 'purpose': 'Repeat the exact baseline prompt with a second seed to distinguish reproducible capability from a lucky sample.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity change, distorted face, warped spectacles, duplicate limbs, extra fingers, fused hands, floating feet, changing clothes, flickering lines, camera motion, scene cut, text, watermark'}, {'id': '03_weight_shift_walk_stop', 'input_key': 'pig', 'prompt': 'The pig in the supplied image shifts its weight naturally, takes two small deliberate steps sideways within the existing frame, then comes to a balanced stop. Show coordinated hips, knees, feet, shoulders, and gentle arm counter-motion. Each foot plants on the ground before the other lifts. Preserve the exact character design, face, clothing, short leg proportions, and hand-drawn painted texture. Keep the whole body visible, the background unchanged, and the camera locked. One continuous shot.', 'seed': 1701, 'purpose': 'Test grounded full-body coordination, short strides, foot sliding, and settling after locomotion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'sliding feet, gliding, skating, elongated legs, stretching body, duplicate feet, extra limbs, identity change, floating, camera tracking, scene cut, photorealism'}, {'id': '04_turn_return_glasses', 'input_key': 'pig', 'prompt': 'The pig in the supplied image slowly turns its head and upper torso a modest amount toward the other side, pauses briefly, then returns to its starting orientation. Preserve the same facial structure, eyes, cheeks, snout, ears, and spectacles if present. Facial features remain correctly attached through the turn, with natural occlusion of the far eye and spectacle arm. Add a subtle blink and breathing. Preserve the clothing and hand-drawn artwork. Feet remain planted, camera locked, background unchanged, no cut.', 'seed': 1701, 'purpose': 'Stress-test left/right view changes, facial identity, glasses geometry, and correct occlusion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'melting face, eye swapping, duplicated eyes, disappearing glasses, detached spectacles, changing cheeks, shifting snout, rubber neck, identity change, camera movement, scene cut'}, {'id': '05_visible_object_interaction', 'input_key': 'pig_object', 'prompt': "The pig interacts only with the object already visibly present within reach in the supplied image. The pig looks at it, reaches toward it with coordinated shoulder, elbow, wrist, and torso motion, briefly makes clear hand contact, then relaxes the arm. Keep the object's identity, shape, position, and material stable. Do not add or remove any object. Preserve the pig's face, clothing, proportions, and painted drawing style. Keep the feet planted and camera locked in one continuous shot.", 'seed': 1701, 'purpose': 'Test contact, hand-object geometry, eye direction, and coordinated reaching without inventing a prop.', 'experimental': False, 'enabled_by_default': False, 'requires': ['pig_object_input_available', 'pig_object_contains_visible_reachable_object'], 'skip_unless_requirements_confirmed': True, 'negative_prompt': 'new object, disappearing object, changing prop, object passing through hand, fused hand and object, detached arm, duplicate fingers, floating object, identity change, scene cut'}, {'id': '06_pair_alternating_speech', 'input_key': 'pair', 'prompt': 'The two characters in the supplied image have a brief silent conversation. First the character on the left speaks with clear mouth changes and one restrained hand gesture while the character on the right listens. Then the left character settles and the right character replies with mouth movement and a small expressive gesture. Preserve both distinct identities, faces, proportions, clothing, and hand-drawn textures. Keep their positions and feet grounded, with natural blinks and subtle body motion. Locked camera, unchanged background, no cut, no subtitles.', 'seed': 1701, 'purpose': 'Test speaker alternation, separate character identities, and independent acting without simultaneous uncontrolled motion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity swap, merged characters, synchronized identical gestures, duplicate limbs, both mouths constantly talking, face distortion, changing clothes, camera movement, scene cut, text'}, {'id': '07_pair_reacting', 'input_key': 'pair', 'prompt': 'The left character in the supplied image makes a small confident gesture toward the right character. The right character notices it, reacts with a surprised expression and a slight backward upper-body shift, then settles into a curious listening pose. The left character watches the reaction. Preserve the two distinct character designs and all clothing, facial features, and painted textures. Keep feet grounded and clear separation between bodies. The camera and background remain still in one continuous shot.', 'seed': 1701, 'purpose': 'Test cause-and-effect acting, reaction timing, gaze, and independent coordinated body motion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity swap, merging bodies, duplicated character, random motion, sliding feet, floating, distorted hands, facial drift, camera movement, scene cut, text'}, {'id': '08_pair_background_locked', 'input_key': 'scene', 'prompt': 'Animate only the two characters already present in the supplied scene. They exchange a small nod and a restrained conversational hand gesture with subtle breathing and natural blinks. Keep their identities, proportions, clothing, and hand-drawn style consistent. Keep every background object, architectural line, floor contact, lighting direction, and camera position fixed. No new objects, no moving scenery, no camera movement, no zoom, no cuts.', 'seed': 1701, 'purpose': 'Test character animation inside a composed scene while preserving background geometry and camera lock.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'background warping, moving scenery, shifting architecture, flickering lighting, camera pan, zoom, parallax, new objects, character identity change, floating feet, scene cut'}, {'id': '09_baseline_style_preservation', 'input_key': 'pig', 'prompt': "A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. The feet stay grounded. Preserve the source illustration's pencil contours, soft painted shading, paper-like texture, muted old-school palette, and gently imperfect drawn edges throughout every frame. Maintain the exact face, spectacles if present, clothing, proportions, and background. Do not reinterpret the art as glossy 3D or photorealism. Locked camera, no cut, no new objects, no subtitles.", 'seed': 1701, 'purpose': 'Compare against test 01 with the same seed and action but stronger source-style instructions; inspect texture flicker, line drift, and unwanted 3D conversion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'glossy 3D, plastic skin, photorealism, smooth vector art, excessive contrast, oversharpening, texture flicker, style drift, identity change, distorted face, warped spectacles, duplicate limbs, text, watermark'}, {'id': '10_longer_continuity_experimental', 'input_key': 'pig', 'prompt': 'In one continuous longer shot, the pig in the supplied image speaks silently, makes one restrained hand gesture with coordinated shoulder and torso motion, settles, looks slightly to one side, and returns to the starting orientation. Maintain the same identity, facial features, spectacles if present, clothing, proportions, pencil contours, and painted texture for the entire shot. Keep feet grounded, camera locked, and background unchanged. Natural pacing, no repeated loop, no cuts, no new objects, no subtitles.', 'seed': 1701, 'purpose': 'Optional longer-duration temporal consistency and memory-cost stress test, only with explicit opt-in and supported model settings.', 'experimental': True, 'enabled_by_default': False, 'requires': ['explicit_longer_test_opt_in', 'model_supported_longer_duration'], 'skip_unless_requirements_confirmed': True, 'negative_prompt': 'identity drift, changing costume, distorted spectacles, extra limbs, looping action, repeated frames, frozen character, camera motion, scene cut, background warping, text'}]}

def experiments(frames=97,long_frames=121):
    rows=json.loads(json.dumps(SHARED_MANIFEST['experiments']))
    for r in rows:
        r['frames']=long_frames if r.get('experimental') else frames
        r['steps']=30
    return rows

def python_executable(value):
    found=shutil.which(os.path.expanduser(value))
    if found is None:raise FileNotFoundError('Working Python executable not found or not executable: '+value)
    return os.path.abspath(found)  # Keep venv symlink identity; do not Path.resolve().

def select_records(records,selection):
    if selection=='all':return records
    chosen={token.strip() for token in selection.split(',')}
    return [r for r in records if r['id'] in chosen or str(int(r['id'].split('_',1)[0])) in chosen]

def validate_config(c):
    if c['frames'] < 5 or (c['frames']-1)%4: raise ValueError('SkyReels frames must be 4*n+1, at least 5')
    for k in ('width','height'):
        if c[k]<64 or c[k]%16: raise ValueError(k+' must be >=64 and divisible by 16')
    if not 1<=c['steps']<=100 or not 1<=c['fps']<=60: raise ValueError('Invalid steps or fps')
    if not 0<=c['seed']<2**63: raise ValueError('Seed out of range')
    if not 0<c['guidance']<=30 or not 0<c['shift']<=30: raise ValueError('Invalid guidance or shift')

def worker(job_path):
    c=json.loads(Path(job_path).read_text()); validate_config(c)
    output=Path(c['output']); output.parent.mkdir(parents=True,exist_ok=True)
    source=Path(c['source']); weights=Path(c['weights'])
    if not (source/'skyreels_v2_infer').is_dir(): raise FileNotFoundError('Expected existing SkyReels source: '+str(source))
    if not weights.is_dir(): raise FileNotFoundError('Expected existing model weights: '+str(weights))
    if not Path(c['image']).is_file(): raise FileNotFoundError(c['image'])
    sys.path.insert(0,str(source))
    import torch, numpy as np, imageio.v2 as imageio
    from PIL import Image, ImageOps
    from skyreels_v2_infer.pipelines import Image2VideoPipeline
    if not torch.cuda.is_available(): raise RuntimeError('A CUDA GPU is required; worker must use the model working Python environment')
    if torch.cuda.device_count()!=1: raise RuntimeError('Each worker must see exactly one GPU through CUDA_VISIBLE_DEVICES')
    torch.cuda.reset_peak_memory_stats()
    image=ImageOps.exif_transpose(Image.open(c['image'])).convert('RGBA')
    bg=Image.new('RGBA',image.size,(220,210,190,255)); bg.alpha_composite(image)
    image=ImageOps.pad(bg.convert('RGB'),(c['width'],c['height']),color=(220,210,190))
    image.save(output.with_suffix('.input.png'))
    started=time.time()
    pipe=Image2VideoPipeline(str(weights),str(weights),weight_dtype=torch.bfloat16,
                            offload=c['offload'],use_usp=False)
    with torch.inference_mode():
        frames=pipe(image=image,prompt=c['prompt'],negative_prompt=c['negative_prompt'],
                    height=c['height'],width=c['width'],num_frames=c['frames'],
                    num_inference_steps=c['steps'],guidance_scale=c['guidance'],shift=c['shift'],
                    generator=torch.Generator(device='cuda').manual_seed(c['seed']))[0]
    frames=np.asarray(frames)
    if frames.shape!=(c['frames'],c['height'],c['width'],3) or not np.isfinite(frames).all():
        raise RuntimeError('Invalid returned frames: '+str(frames.shape))
    temp=output.with_name(output.stem+'.partial.mp4')
    imageio.mimwrite(temp,frames,fps=c['fps'],quality=8,codec='libx264',pixelformat='yuv420p',macro_block_size=1)
    reader=imageio.get_reader(temp)
    try:
        count=sum(1 for _ in reader); meta=reader.get_meta_data()
    finally: reader.close()
    if count!=c['frames'] or abs(float(meta['fps'])-c['fps'])>0.01: raise RuntimeError('MP4 frame count/fps verification failed')
    temp.replace(output)
    version={}
    for module in ('torch','numpy','PIL','imageio'):
        m=sys.modules.get(module);version[module]=getattr(m,'__version__','unknown')
    try: revision=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired): revision='unknown'
    info=dict(model=MODEL_ID,config=c,elapsed_seconds=time.time()-started,decoded_frames=count,
              fps=c['fps'],gpu_name=torch.cuda.get_device_name(0),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),versions=version,source_revision=revision,
              conditioning_fps='not an exposed native pipeline argument; fps parameter controls encoding',
              input_sha256=hashlib.sha256(Path(c['image']).read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(info,indent=2))
    print('VERIFIED OUTPUT:',output,flush=True)

def choose_gpus(value,dry_run):
    if value!='auto': result=[s.strip() for s in value.split(',') if s.strip()]
    elif os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        result=[s.strip() for s in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if s.strip() and s.strip()!='-1']
    elif dry_run: result=['0','1','2','3']
    else:
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True)
            result=p.stdout.strip().splitlines()[:4]
        except (OSError,subprocess.CalledProcessError) as e: raise RuntimeError('Cannot detect GPUs. Supply --gpus with allocated GPU IDs/UUIDs.') from e
    if not result or len(result)>4 or len(set(result))!=len(result): raise ValueError('Select one to four distinct allocated GPUs')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--python',default=sys.executable,help='Python executable from the existing working SkyReels environment')
    p.add_argument('--root',type=Path,default=Path(os.environ.get('SKYREELS_ROOT','./skyreels')))
    p.add_argument('--source',type=Path);p.add_argument('--weights',type=Path)
    p.add_argument('--pig',type=Path,default=Path('inputs/pig.png'));p.add_argument('--pair',type=Path,default=Path('inputs/pig_wolf.png'));p.add_argument('--scene',type=Path,default=Path('inputs/pig_wolf_background.png'))
    p.add_argument('--pig-object',type=Path,default=Path('inputs/pig_object.png'),help='Separate pig image with visible reachable prop for optional test05')
    p.add_argument('--output-root',type=Path,default=Path('skyreels_experiments'))
    p.add_argument('--gpus',default='auto',help='One to four allocated physical IDs or GPU UUIDs, comma separated; auto respects CUDA_VISIBLE_DEVICES')
    p.add_argument('--select',default='all',help='Comma-separated experiment numbers (1-10) or IDs')
    p.add_argument('--manifest',type=Path,help='Optional JSON list overriding the default experiment list')
    p.add_argument('--write-manifest',type=Path,help='Write default experiment definitions and exit')
    p.add_argument('--frames',type=int,default=97);p.add_argument('--long-frames',type=int,default=121)
    p.add_argument('--width',type=int,default=960);p.add_argument('--height',type=int,default=544)
    p.add_argument('--fps',type=int,default=24);p.add_argument('--guidance',type=float,default=5.0)
    p.add_argument('--shift',type=float,default=3.0,help='Native official I2V baseline. Set 5 to reproduce old wrapper shift.')
    p.add_argument('--offload',action='store_true',help='CPU offload; may greatly increase host RAM demand with four workers')
    p.add_argument('--confirm-object-present',action='store_true',help='Enable test05 only after confirming a visible reachable object in pig input')
    p.add_argument('--include-experimental',action='store_true',help='Include duration stress job 10; not validated native long-video quality')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--no-resume',action='store_true')
    p.add_argument('--timeout-seconds',type=float,default=None)
    a=p.parse_args()
    if a.worker:return worker(a.worker)
    records=experiments(a.frames,a.long_frames)
    if a.write_manifest:
        a.write_manifest.parent.mkdir(parents=True,exist_ok=True);a.write_manifest.write_text(json.dumps(records,indent=2));return
    if a.manifest:
        loaded=json.loads(a.manifest.read_text());records=loaded.get('experiments') if isinstance(loaded,dict) else loaded
    skipped=[]
    filtered=[]
    for r in records:
        if r.get('experimental') and not a.include_experimental:
            skipped.append({'id':r['id'],'reason':'experimental longer duration not enabled'});continue
        if r['id'].startswith('05_') and not a.confirm_object_present:
            skipped.append({'id':r['id'],'reason':'visible object prerequisite not confirmed'});continue
        filtered.append(r)
    records=filtered
    if not isinstance(records,list) or not records:raise ValueError('Manifest must be a nonempty JSON list')
    if a.select!='all':
        records=select_records(records,a.select)
        if not records:raise ValueError('Selection matched no experiments')
    out=a.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    with output_root_lock(out):
        images={k:getattr(a,k) for k in ('pig','pair','scene','pig_object')}
        source=(a.source or a.root/'source').resolve();weights=(a.weights or a.root/'weights').resolve()
        py=python_executable(a.python)
        jobs=[]
        for r in records:
            ident=r['id']
            if not ident or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in ident):raise ValueError('Unsafe experiment id')
            raw_image=r.get('image') or images.get(r.get('input_key',r.get('image_key','pig')))
            if raw_image is None:
                skipped.append({'id':ident,'reason':'missing input path'});continue
            image=Path(raw_image).resolve()
            if not image.is_file():
                skipped.append({'id':ident,'reason':'missing input file: '+str(image)});continue
            jobdir=out/ident;jobdir.mkdir(exist_ok=True)
            c=dict(width=a.width,height=a.height,frames=a.long_frames if r.get('experimental') else a.frames,steps=30,guidance=a.guidance,shift=a.shift,
                   fps=a.fps,seed=12345,prompt=BASE_STYLE,negative_prompt=NEGATIVE,offload=a.offload)
            c.update({k:v for k,v in r.items() if k not in ('image','image_key')})
            c.update(source=str(source),weights=str(weights),image=str(image),output=str(jobdir/'video.mp4'))
            validate_config(c)
            path=jobdir/'config.json';path.write_text(json.dumps(c,indent=2))
            jobs.append(dict(id=ident,command=[py,str(Path(__file__).resolve()),'--worker',str(path)],
                             expected_outputs=[c['output'],str(Path(c['output']).with_suffix('.json'))],
                             metadata=c,input_files=[str(image),str(path)],cwd=str(out),env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}))
        (out/'skipped.json').write_text(json.dumps(skipped,indent=2))
        if not jobs:
            print(json.dumps({'status':'no_jobs','skipped':skipped},indent=2));return 2
        result=run_queue(jobs,choose_gpus(a.gpus,a.dry_run),str(out),dry_run=a.dry_run,
                         resume=not a.no_resume,timeout_seconds=a.timeout_seconds)
        print(json.dumps(result,indent=2))
        return 0 if result['status'] in ('dry_run','completed') else 1

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
