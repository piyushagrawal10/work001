#!/usr/bin/env python3
"""Standard-library multi-GPU subprocess queue. No torch imports or installation.

A command must use the exact Python executable from a working model environment.
GPU tokens identify physical IDs/UUIDs; inherited CUDA_VISIBLE_DEVICES is respected.
One GPU per subprocess, one worker per requested GPU. Devices are never pooled.
"""
from __future__ import annotations
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


DEFAULT_MANIFEST = {'schema_version': 1, 'description': 'Shared image-to-video diagnostic experiments. Each runner uses the same supplied input images and prompts; model-specific settings belong in the runner configuration.', 'input_keys': {'pig': 'User-supplied approved single-pig image, pig.png.', 'pair': 'User-supplied approved two-character image, with both characters visible and clearly separated.', 'scene': 'User-supplied approved two-character image already composed into the intended background.', 'pig_object': 'Optional distinct pig_object.png: approved pig with a visibly present reachable interaction object. Do not replace the baseline pig.png.'}, 'execution_notes': ['Do not substitute a missing input or invent an unseen object. Skip the affected experiment with an explicit reason.', 'Test 05 uses the distinct optional pig_object.png input. It is disabled by default and must be skipped if that image is missing or does not visibly contain a reachable interaction object. Do not alter baseline pig.png to enable this test.', 'Test 10 is disabled by default and requires explicit opt-in plus a model-supported longer duration or frame count. Do not silently loop, duplicate, or interpolate frames to claim longer generation.', "Use the model's supported default short duration for tests 01 through 09; record actual frame count, FPS, resolution, seed, checkpoint, and settings alongside each result.", 'A shared numeric seed does not imply identical noise across different models. Compare tests 01 and 02 within a model for seed variability.', 'If a runner does not support negative prompts, omit that argument and record this limitation rather than failing.'], 'experiments': [{'id': '01_pig_silent_speech', 'input_key': 'pig', 'prompt': 'A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. Keep the same face, spectacles if present, clothing, proportions, pencil shading, colors, and background. The feet stay grounded. Locked camera, no cut, no new objects, no subtitles.', 'seed': 1701, 'purpose': 'Baseline identity preservation, mouth animation, and coordinated hand and upper-body acting.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity change, distorted face, warped spectacles, duplicate limbs, extra fingers, fused hands, floating feet, changing clothes, flickering lines, camera motion, scene cut, text, watermark'}, {'id': '02_pig_silent_speech_second_seed', 'input_key': 'pig', 'prompt': 'A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. Keep the same face, spectacles if present, clothing, proportions, pencil shading, colors, and background. The feet stay grounded. Locked camera, no cut, no new objects, no subtitles.', 'seed': 1702, 'purpose': 'Repeat the exact baseline prompt with a second seed to distinguish reproducible capability from a lucky sample.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity change, distorted face, warped spectacles, duplicate limbs, extra fingers, fused hands, floating feet, changing clothes, flickering lines, camera motion, scene cut, text, watermark'}, {'id': '03_weight_shift_walk_stop', 'input_key': 'pig', 'prompt': 'The pig in the supplied image shifts its weight naturally, takes two small deliberate steps sideways within the existing frame, then comes to a balanced stop. Show coordinated hips, knees, feet, shoulders, and gentle arm counter-motion. Each foot plants on the ground before the other lifts. Preserve the exact character design, face, clothing, short leg proportions, and hand-drawn painted texture. Keep the whole body visible, the background unchanged, and the camera locked. One continuous shot.', 'seed': 1701, 'purpose': 'Test grounded full-body coordination, short strides, foot sliding, and settling after locomotion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'sliding feet, gliding, skating, elongated legs, stretching body, duplicate feet, extra limbs, identity change, floating, camera tracking, scene cut, photorealism'}, {'id': '04_turn_return_glasses', 'input_key': 'pig', 'prompt': 'The pig in the supplied image slowly turns its head and upper torso a modest amount toward the other side, pauses briefly, then returns to its starting orientation. Preserve the same facial structure, eyes, cheeks, snout, ears, and spectacles if present. Facial features remain correctly attached through the turn, with natural occlusion of the far eye and spectacle arm. Add a subtle blink and breathing. Preserve the clothing and hand-drawn artwork. Feet remain planted, camera locked, background unchanged, no cut.', 'seed': 1701, 'purpose': 'Stress-test left/right view changes, facial identity, glasses geometry, and correct occlusion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'melting face, eye swapping, duplicated eyes, disappearing glasses, detached spectacles, changing cheeks, shifting snout, rubber neck, identity change, camera movement, scene cut'}, {'id': '05_visible_object_interaction', 'input_key': 'pig_object', 'prompt': "The pig interacts only with the object already visibly present within reach in the supplied image. The pig looks at it, reaches toward it with coordinated shoulder, elbow, wrist, and torso motion, briefly makes clear hand contact, then relaxes the arm. Keep the object's identity, shape, position, and material stable. Do not add or remove any object. Preserve the pig's face, clothing, proportions, and painted drawing style. Keep the feet planted and camera locked in one continuous shot.", 'seed': 1701, 'purpose': 'Test contact, hand-object geometry, eye direction, and coordinated reaching without inventing a prop.', 'experimental': False, 'enabled_by_default': False, 'requires': ['pig_object_input_available', 'pig_object_contains_visible_reachable_object'], 'skip_unless_requirements_confirmed': True, 'negative_prompt': 'new object, disappearing object, changing prop, object passing through hand, fused hand and object, detached arm, duplicate fingers, floating object, identity change, scene cut'}, {'id': '06_pair_alternating_speech', 'input_key': 'pair', 'prompt': 'The two characters in the supplied image have a brief silent conversation. First the character on the left speaks with clear mouth changes and one restrained hand gesture while the character on the right listens. Then the left character settles and the right character replies with mouth movement and a small expressive gesture. Preserve both distinct identities, faces, proportions, clothing, and hand-drawn textures. Keep their positions and feet grounded, with natural blinks and subtle body motion. Locked camera, unchanged background, no cut, no subtitles.', 'seed': 1701, 'purpose': 'Test speaker alternation, separate character identities, and independent acting without simultaneous uncontrolled motion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity swap, merged characters, synchronized identical gestures, duplicate limbs, both mouths constantly talking, face distortion, changing clothes, camera movement, scene cut, text'}, {'id': '07_pair_reacting', 'input_key': 'pair', 'prompt': 'The left character in the supplied image makes a small confident gesture toward the right character. The right character notices it, reacts with a surprised expression and a slight backward upper-body shift, then settles into a curious listening pose. The left character watches the reaction. Preserve the two distinct character designs and all clothing, facial features, and painted textures. Keep feet grounded and clear separation between bodies. The camera and background remain still in one continuous shot.', 'seed': 1701, 'purpose': 'Test cause-and-effect acting, reaction timing, gaze, and independent coordinated body motion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'identity swap, merging bodies, duplicated character, random motion, sliding feet, floating, distorted hands, facial drift, camera movement, scene cut, text'}, {'id': '08_pair_background_locked', 'input_key': 'scene', 'prompt': 'Animate only the two characters already present in the supplied scene. They exchange a small nod and a restrained conversational hand gesture with subtle breathing and natural blinks. Keep their identities, proportions, clothing, and hand-drawn style consistent. Keep every background object, architectural line, floor contact, lighting direction, and camera position fixed. No new objects, no moving scenery, no camera movement, no zoom, no cuts.', 'seed': 1701, 'purpose': 'Test character animation inside a composed scene while preserving background geometry and camera lock.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'background warping, moving scenery, shifting architecture, flickering lighting, camera pan, zoom, parallax, new objects, character identity change, floating feet, scene cut'}, {'id': '09_baseline_style_preservation', 'input_key': 'pig', 'prompt': "A short, continuous hand-drawn animation of the pig in the supplied image speaking silently. The pig makes a small deliberate hand gesture, coordinated with a slight shoulder and torso movement, natural blinks, and clearly changing mouth shapes. The feet stay grounded. Preserve the source illustration's pencil contours, soft painted shading, paper-like texture, muted old-school palette, and gently imperfect drawn edges throughout every frame. Maintain the exact face, spectacles if present, clothing, proportions, and background. Do not reinterpret the art as glossy 3D or photorealism. Locked camera, no cut, no new objects, no subtitles.", 'seed': 1701, 'purpose': 'Compare against test 01 with the same seed and action but stronger source-style instructions; inspect texture flicker, line drift, and unwanted 3D conversion.', 'experimental': False, 'enabled_by_default': True, 'negative_prompt': 'glossy 3D, plastic skin, photorealism, smooth vector art, excessive contrast, oversharpening, texture flicker, style drift, identity change, distorted face, warped spectacles, duplicate limbs, text, watermark'}, {'id': '10_longer_continuity_experimental', 'input_key': 'pig', 'prompt': 'In one continuous longer shot, the pig in the supplied image speaks silently, makes one restrained hand gesture with coordinated shoulder and torso motion, settles, looks slightly to one side, and returns to the starting orientation. Maintain the same identity, facial features, spectacles if present, clothing, proportions, pencil contours, and painted texture for the entire shot. Keep feet grounded, camera locked, and background unchanged. Natural pacing, no repeated loop, no cuts, no new objects, no subtitles.', 'seed': 1701, 'purpose': 'Optional longer-duration temporal consistency and memory-cost stress test, only with explicit opt-in and supported model settings.', 'experimental': True, 'enabled_by_default': False, 'requires': ['explicit_longer_test_opt_in', 'model_supported_longer_duration'], 'skip_unless_requirements_confirmed': True, 'negative_prompt': 'identity drift, changing costume, distorted spectacles, extra limbs, looping action, repeated frames, frozen character, camera motion, scene cut, background warping, text'}]}
"""Wan 2.2 TI2V-5B experiment adapter. Embed after common_queue.py.

Uses the user's existing Python/ComfyUI installation; no install or downloads.
GPU inference is not validated on this CPU build host.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time


WAN_DEFAULTS = dict(width=960, height=544, frames=121, fps=24, steps=20,
                    guidance=5.0, shift=8.0, sampler="uni_pc", scheduler="simple")
WAN_NEGATIVE = "distorted anatomy, extra limbs, duplicated character, changing clothing, flicker, blurry, text, watermark"


def wan_validate(s):
    for k in ("width", "height", "frames", "fps", "steps"):
        if not isinstance(s[k], int) or s[k] <= 0:
            raise ValueError(k + " must be a positive integer")
    if s["width"] % 32 or s["height"] % 32 or s["frames"] % 4 != 1:
        raise ValueError("Wan needs width/height divisible by 32 and frames=4*n+1")
    if not 0 <= float(s["guidance"]) <= 30 or not 0 < float(s["shift"]) <= 30:
        raise ValueError("Invalid guidance or shift")


def wan_graph(s, prompt, negative, seed):
    def n(t, **inputs): return {"class_type": t, "inputs": inputs}
    return {
        "1": n("UNETLoader", unet_name="wan2.2_ti2v_5B_fp16.safetensors", weight_dtype="default"),
        "2": n("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="cpu"),
        "3": n("VAELoader", vae_name="wan2.2_vae.safetensors"),
        "4": n("CLIPTextEncode", clip=["2", 0], text=prompt),
        "5": n("CLIPTextEncode", clip=["2", 0], text=negative),
        "6": n("LoadImage", image="source.png"),
        "7": n("Wan22ImageToVideoLatent", vae=["3", 0], width=s["width"], height=s["height"], length=s["frames"], batch_size=1, start_image=["6", 0]),
        "8": n("ModelSamplingSD3", model=["1", 0], shift=s["shift"]),
        "9": n("KSampler", model=["8", 0], positive=["4", 0], negative=["5", 0], latent_image=["7", 0], seed=seed, steps=s["steps"], cfg=s["guidance"], sampler_name=s["sampler"], scheduler=s["scheduler"], denoise=1.0),
        "10": n("VAEDecodeTiled", samples=["9", 0], vae=["3", 0], tile_size=256, overlap=64, temporal_size=16, temporal_overlap=4),
        "11": n("SaveImage", images=["10", 0], filename_prefix="frames/wan")}


def wan_encode(paths, target, s):
    import av
    import numpy as np
    from PIL import Image
    temporary = target.with_name(target.stem + ".partial.mp4")
    with av.open(str(temporary), "w") as container:
        stream = container.add_stream("libx264", rate=s["fps"])
        stream.width, stream.height = s["width"], s["height"]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        for path in paths:
            with Image.open(path) as im:
                if im.size != (s["width"], s["height"]):
                    raise ValueError("Unexpected decoded dimensions")
                frame = av.VideoFrame.from_ndarray(np.asarray(im.convert("RGB")), format="rgb24")
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    with av.open(str(temporary)) as container:
        fps = float(container.streams.video[0].average_rate)
        count = sum(1 for _ in container.decode(video=0))
    if count != s["frames"] or abs(fps - s["fps"]) > 0.001:
        raise RuntimeError("Encoded MP4 failed frame/FPS verification")
    temporary.replace(target)
    return {"frames": count, "fps": fps, "duration_seconds": count / fps}


def wan_worker(config_path):
    from PIL import Image, ImageOps
    import requests
    session = requests.Session()
    session.trust_env = False  # Local-only ComfyUI traffic must not use a proxy.
    cfg = json.loads(Path(config_path).read_text())
    s = cfg["settings"]; wan_validate(s)
    run = Path(cfg["run_dir"]); run.mkdir(parents=True, exist_ok=True)
    repo = Path(cfg["repo"])
    for name in ("input", "backend_output", "temp", "frontend", "user"):
        (run / name).mkdir(exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(cfg["image"])).convert("RGBA")
    bg = Image.new("RGBA", im.size, (217, 207, 187, 255)); bg.alpha_composite(im)
    im = ImageOps.pad(bg.convert("RGB"), (s["width"], s["height"]), method=Image.Resampling.LANCZOS, color=(217, 207, 187))
    im.save(run / "input/source.png")
    graph = wan_graph(s, cfg["prompt"], cfg.get("negative", WAN_NEGATIVE), cfg["seed"])
    (run / "graph.json").write_text(json.dumps(graph, indent=2))
    proc = None; log = None
    def shutdown(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, shutdown)
    def stop():
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    started = time.monotonic()
    try:
        # The backend has no inherited-listener interface. Re-allocate/retry on
        # bind conflict; never connect to a foreign service on the chosen port.
        for attempt in range(5):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0)); port = probe.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            logpath = run / f"backend_{attempt}.log"
            log = logpath.open("w")
            command = [sys.executable, str(repo / "main.py"), "--listen", "127.0.0.1", "--port", str(port), "--disable-auto-launch", "--disable-all-custom-nodes", "--disable-api-nodes", "--lowvram", "--fp16-unet", "--fp32-vae", "--use-pytorch-cross-attention", "--preview-method", "none", "--input-directory", str(run / "input"), "--output-directory", str(run / "backend_output"), "--temp-directory", str(run / "temp"), "--user-directory", str(run / "user"), "--front-end-root", str(run / "frontend")]
            command += ["--database-url", "sqlite:///" + str(run / "user/comfyui.db")]
            env = os.environ.copy(); env.pop("PYTHONPATH", None); env.pop("PYTHONHOME", None)
            env["PYTHONNOUSERSITE"] = "1"
            proc = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, env=env)
            deadline = time.monotonic() + cfg.get("startup_timeout", 300)
            info = None
            while time.monotonic() < deadline:
                text = logpath.read_text(errors="replace")
                if proc.poll() is not None: break
                # Wait for *this process* to announce successful HTTP startup.
                if "To see the GUI go to:" in text:
                    try:
                        response = session.get(base + "/object_info", timeout=5)
                        if response.ok: info = response.json(); break
                    except requests.RequestException: pass
                time.sleep(1)
            if info is not None: break
            stop(); log.close(); log = None
            text = logpath.read_text(errors="replace")
            if "address already in use" not in text.lower():
                raise RuntimeError("ComfyUI startup failed or timed out; see " + str(logpath) + "\n" + text[-5000:])
        else: raise RuntimeError("Could not allocate an available ComfyUI port after five attempts")
        for node in graph.values():
            if node["class_type"] not in info:
                raise RuntimeError("Required ComfyUI node missing: " + node["class_type"])
        response = session.post(base + "/prompt", json={"prompt": graph}, timeout=60)
        response.raise_for_status(); submission = response.json()
        (run / "submission.json").write_text(json.dumps(submission, indent=2))
        pid = submission["prompt_id"]
        deadline = time.monotonic() + cfg.get("inference_timeout", 21600)
        while time.monotonic() < deadline:
            if proc.poll() is not None: raise RuntimeError("ComfyUI exited during inference; inspect backend log")
            response = session.get(base + "/history/" + pid, timeout=30); response.raise_for_status()
            hist = response.json()
            if pid in hist:
                h = hist[pid]; (run / "history.json").write_text(json.dumps(h, indent=2))
                if h.get("status", {}).get("status_str") == "error":
                    raise RuntimeError("ComfyUI execution error: " + json.dumps(h.get("status", {})))
                files = h.get("outputs", {}).get("11", {}).get("images", [])
                if len(files) != s["frames"]: raise RuntimeError("Wrong output frame count")
                break
            time.sleep(3)
        else: raise TimeoutError("Inference timed out")
        root = (run / "backend_output").resolve()
        paths = [(root / f["subfolder"] / f["filename"]).resolve() for f in files]
        if any(root not in path.parents or not path.is_file() for path in paths):
            raise RuntimeError("Invalid/missing ComfyUI output frame")
        result = wan_encode(paths, run / "video.mp4", s)
        result.update(settings=s, seed=cfg["seed"], runtime_seconds=time.monotonic() - started,
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      peak_vram_bytes=None, peak_vram_note="Not instrumented inside ComfyUI; do not infer from parent process.")
        (run / "result.json").write_text(json.dumps(result, indent=2))
    finally:
        stop()
        if log: log.close()
        session.close()


def wan_main():
    p = argparse.ArgumentParser(description="Wan 2.2 TI2V-5B: isolated independent-GPU experiment queue. No installation or downloads.")
    p.add_argument("--worker-config", help=argparse.SUPPRESS)
    p.add_argument("--python", default=sys.executable, help="Existing working Wan environment Python executable")
    p.add_argument("--comfy-root", type=Path, help="Existing ComfyUI directory containing main.py and installed model files")
    p.add_argument("--manifest", type=Path, required=False, help="Optional custom JSON; default ten experiments are embedded")
    p.add_argument("--input-dir", type=Path, default=Path("inputs"))
    p.add_argument("--pig", type=Path); p.add_argument("--pair", type=Path); p.add_argument("--scene", type=Path)
    p.add_argument("--pig-object", type=Path, help="Separate pig-plus-visible-object image; does not replace baseline pig")
    p.add_argument("--confirm-object", action="store_true", help="Confirm pig input includes a reachable visible object for test05")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wan"))
    p.add_argument("--gpus", default="auto", help="auto uses the allocation, or supply comma-separated physical GPU IDs/UUIDs")
    p.add_argument("--experiments", help="Comma-separated experiment IDs to run")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--timeout", type=float, default=25200)
    p.add_argument("--allow-experimental-long", action="store_true")
    for key in ("width", "height", "frames", "fps", "steps"):
        p.add_argument("--" + key, type=int)
    p.add_argument("--guidance", type=float); p.add_argument("--shift", type=float)
    a = p.parse_args()
    if a.worker_config: wan_worker(a.worker_config); return
    if not a.comfy_root: p.error("--comfy-root is required")
    repo = a.comfy_root.resolve()
    if not a.dry_run and not (repo / "main.py").is_file(): p.error("ComfyUI main.py not found")
    data = json.loads(a.manifest.read_text()) if a.manifest else DEFAULT_MANIFEST
    experiments = data.get("experiments", []) if isinstance(data, dict) else data
    selected = set(a.experiments.split(",")) if a.experiments else None
    python_candidate = os.path.expanduser(a.python)
    python_found = shutil.which(python_candidate)
    if not python_found: p.error("Python executable not found: " + a.python)
    python_executable = os.path.abspath(python_found)  # Preserve venv symlink path.
    out = a.output_dir.resolve()
    with output_root_lock(out):
        out.mkdir(parents=True, exist_ok=True)
        jobs = []; skipped = []
        images = {"pig": a.pig or a.input_dir / "pig.png", "pig_object": a.pig_object or a.input_dir / "pig_object.png", "pair": a.pair or a.input_dir / "pig_wolf.png", "scene": a.scene or a.input_dir / "pig_wolf_background.png"}
        for e in experiments:
            eid = str(e["id"])
            if selected and eid not in selected: continue
            if not eid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in eid):
                raise ValueError("Unsafe experiment ID: " + eid)
            requirements = e.get("requires", [])
            reason = None
            if any("visible_reachable_object" in r for r in requirements) and not a.confirm_object:
                reason = "test requires --confirm-object and a visible reachable object in the pig input"
            if e.get("experimental") and not a.allow_experimental_long:
                reason = "experimental long generation requires --allow-experimental-long"
            if reason:
                skipped.append({"id": eid, "reason": reason}); print("Skipping", eid, ":", reason); continue
            settings = dict(WAN_DEFAULTS)
            if e.get("experimental"): settings["frames"] = 241
            settings.update(e.get("settings", {}))
            settings.update(e.get("model_settings", {}).get("wan", {}))
            for key in WAN_DEFAULTS:
                value = getattr(a, key, None)
                if value is not None: settings[key] = value
            wan_validate(settings)
            if settings["frames"] > 121 and not a.allow_experimental_long:
                print("Skipping experimental long run", eid, "(use --allow-experimental-long)"); continue
            if "image" in e:
                image = Path(e["image"])
                if not image.is_absolute(): image = a.input_dir / image
            else:
                image = images[e["input_key"]]
            image = image.resolve()
            if not a.dry_run and not image.is_file():
                reason = "missing input image: " + str(image)
                skipped.append({"id": eid, "reason": reason}); print("Skipping", eid, ":", reason); continue
            run = out / eid; run.mkdir(exist_ok=True)
            cfg = dict(run_dir=str(run), repo=str(repo), image=str(image), settings=settings,
                       prompt=e["prompt"], seed=int(e.get("seed", 42)), negative=e.get("negative_prompt", e.get("negative", WAN_NEGATIVE)))
            config_path = run / "worker_config.json"
            config_path.write_text(json.dumps(cfg, indent=2))
            metadata = dict(cfg)
            metadata["input_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else None
            jobs.append({"id": eid, "command": [python_executable, str(Path(__file__).resolve()), "--worker-config", str(config_path)],
                         "expected_outputs": [str(run / "video.mp4"), str(run / "result.json")], "metadata": metadata,
                         "input_files": [str(config_path)] + ([str(image)] if image.is_file() else [])})
        (out / "skipped_experiments.json").write_text(json.dumps(skipped, indent=2))
        if not jobs: p.error("No runnable experiments selected; inspect skipped_experiments.json")
        return run_queue(jobs, a.gpus, str(out),
                         dry_run=a.dry_run, resume=not a.no_resume, timeout_seconds=a.timeout)


if __name__ == "__main__":
    result = wan_main()
    if isinstance(result, int): sys.exit(result)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("status") in ("dry_run", "completed") else 1)
