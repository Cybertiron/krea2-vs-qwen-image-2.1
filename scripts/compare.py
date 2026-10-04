"""Krea 2 Turbo vs Qwen-Image-2.1 vs Qwen-Image-2.1 + Fix LoRA — same prompts/seeds via ComfyUI API.
Usage: python compare.py [--port 8189]   (ComfyUI must be running)
Prompt enhancers are OFF for all variants so every model sees the identical prompt."""
import json, sys, time, urllib.request, uuid, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8189)
ap.add_argument("--seeds", default="1001,2002")
ap.add_argument("--size", default="1024x1024")
ap.add_argument("--only", default="")  # comma list of variant names
args = ap.parse_args()
URL = f"http://127.0.0.1:{args.port}"
W, H = map(int, args.size.split("x"))
SEEDS = [int(s) for s in args.seeds.split(",")]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

PROMPTS = {
    "portrait": "Close-up photograph of an elderly fisherman with a weathered, deeply wrinkled face and a grey stubble beard, wearing a yellow oilskin hat, droplets of sea spray on his skin, overcast soft daylight, harbor softly blurred in the background, 85mm lens, shallow depth of field, natural skin texture.",
    "landscape": "A misty Nordic fjord at sunrise, a small village of red wooden houses on the shore, steep green cliffs disappearing into low clouds, a calm mirror-like water surface reflecting golden light, a lone rowing boat in the foreground, cinematic wide shot, high detail.",
    "text": "A cozy bakery storefront at dusk with a large chalkboard sign by the door that clearly reads \"FRESH BREAD - 3 COINS\" in neat white hand-lettering, and a painted wooden sign above the window reading \"MOONFLOUR BAKERY\". Warm light glows from inside, loaves visible on shelves.",
    "anime_fantasy": "Anime illustration of an original character: a young witch with short silver hair and amber eyes, wearing a dark blue cloak with brass buttons, standing in a cobblestone medieval town square next to a primitive iron steam engine puffing white smoke, curious townsfolk watching from a distance, late afternoon light, clean lineart, vibrant cel shading, detailed background.",
    "composition": "A red cube balanced on top of a blue sphere on a wooden table, to the left a transparent glass half-filled with water, to the right a green apple, and under the table a sleeping orange cat. Studio lighting, white background, photorealistic.",
}
NEG_QWEN = "low quality, blurry, distorted"
NEG_FIX = "artifacts, gpt-image, washed-out colors, low quality, low resolution, AI slop, deviantart, sloppy lines, rough sketch, blurry, indistinct, missing fingers, badly drawn hands, wrong number of fingers"


def krea(prompt, seed, unet="krea2_turbo_fp8_scaled.safetensors"):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
                                                   "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "cmp/krea2"}},
    }


def qwen(prompt, seed, fix=False):
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0], "prompt": prompt, "negative_prompt": NEG_FIX if fix else NEG_QWEN, "resolution": 1024}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "10": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "cmp/qwen21fix" if fix else "cmp/qwen21"}},
    }
    if fix:  # author's recipe: LoRA 1.0 -> APG(1,10,0.3) -> FreSca(1,2,8), 20 steps, cfg 3, seeds_2 / sgm_uniform
        g["11"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["10", 0], "lora_name": "qwen-image-2.1-fix-1.0-comfy.safetensors", "strength_model": 1.0}}
        g["12"] = {"class_type": "APG", "inputs": {"model": ["11", 0], "eta": 1.0, "norm_threshold": 10.0, "momentum": 0.3}}
        g["13"] = {"class_type": "FreSca", "inputs": {"model": ["12", 0], "scale_low": 1.0, "scale_high": 2.0, "freq_cutoff": 8}}
        model, steps, cfg, sampler, sched = ["13", 0], 20, 3.0, "seeds_2", "sgm_uniform"
    else:  # official template defaults
        model, steps, cfg, sampler, sched = ["10", 0], 25, 1.0, "euler", "simple"
    g["7"] = {"class_type": "KSampler", "inputs": {"model": model, "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": sched,
                                                  "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["6", 0], "denoise": 1.0}}
    return g


VARIANTS = {"krea2": krea, "krea2int8": lambda p, s: krea(p, s, "krea2_turbo_int8_convrot.safetensors"), "qwen21": lambda p, s: qwen(p, s), "qwen21fix": lambda p, s: qwen(p, s, fix=True)}
if args.only:
    VARIANTS = {k: v for k, v in VARIANTS.items() if k in args.only.split(",")}


def post(path, data):
    req = urllib.request.Request(URL + path, json.dumps(data).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def run(graph):
    cid = str(uuid.uuid4())
    t0 = time.time()
    try:
        pid = post("/prompt", {"prompt": graph, "client_id": cid})["prompt_id"]
    except urllib.error.HTTPError as e:
        sys.exit("prompt rejected: " + e.read().decode()[:2000])
    while True:
        time.sleep(0.5)
        h = json.load(urllib.request.urlopen(f"{URL}/history/{pid}"))
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                sys.exit("failed: " + json.dumps(st.get("messages"))[:3000])
            imgs = [i for o in h[pid]["outputs"].values() for i in o.get("images", [])]
            return time.time() - t0, imgs[0]


log_path = os.path.join(OUT, "timings.json")
log = json.load(open(log_path)) if os.path.exists(log_path) else {}
for vname, fn in VARIANTS.items():
    # warmup = model load, not counted
    dt, _ = run(fn("warmup", 1))
    print(f"[{vname}] warmup/load {dt:.1f}s", flush=True)
    for pname, ptext in PROMPTS.items():
        for seed in SEEDS:
            dt, img = run(fn(ptext, seed))
            src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ComfyUI", "output", img["subfolder"], img["filename"])
            dst = os.path.join(OUT, f"{pname}_{seed}_{vname}.png")
            os.replace(src, dst)
            log[f"{pname}_{seed}_{vname}"] = round(dt, 2)
            print(f"[{vname}] {pname} seed {seed}: {dt:.1f}s", flush=True)
            json.dump(log, open(log_path, "w"), indent=1)
print("done")
