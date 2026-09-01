#!/usr/bin/env python3
"""Extract θ decomposition from LLM hidden states - three views:
1. δ vs model size (final layer, main checkpoint)
2. δ vs training step (final layer, across checkpoints)  
3. δ vs layer depth (all layers, one checkpoint)
"""
import torch, json, numpy as np, os, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is", "Machine learning models can",
    "In physics, entropy measures", "The largest ocean on Earth is",
    "Neural networks learn by", "Democracy is a form of",
    "The speed of light is", "Photosynthesis converts sunlight into",
    "Shakespeare wrote many", "The human genome contains",
    "Quantum mechanics describes", "The Industrial Revolution began",
    "Artificial intelligence aims to", "The Pythagorean theorem states",
    "Climate change affects", "The periodic table organizes",
]

def compute_pr(H):
    """Participation ratio from a (n_samples, d) matrix."""
    if len(H) < 3: return float('nan')
    cov = np.cov(H.T)
    eigs = np.linalg.eigvalsh(cov)
    eigs = eigs[eigs > 1e-10]
    if len(eigs) == 0: return float('nan')
    return float((np.sum(eigs)**2) / np.sum(eigs**2))

def theta_decompose(zoom_sizes, prs):
    """θ = θ_floor + δ from zoom-level PR values."""
    log_z = np.log(zoom_sizes)
    log_pr = np.log(np.clip(prs, 1e-10, None))
    theta_obs = float(np.polyfit(log_z, log_pr, 1)[0])
    d_eff = prs[-1]
    pr_null = [(k-1)*d_eff / ((k-1) + d_eff) for k in zoom_sizes]
    log_null = np.log(np.clip(pr_null, 1e-10, None))
    theta_floor = float(np.polyfit(log_z, log_null, 1)[0])
    delta = theta_obs - theta_floor
    return theta_obs, theta_floor, delta

def extract_all_layers(model, tokenizer, prompts, device='cpu'):
    """Extract hidden states at EVERY layer, compute PR ladder per layer."""
    model.eval()
    zoom_sizes = [4, 6, 8, 10, 12, 14, 16]
    results_per_layer = {}
    
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            # Take last token per prompt
            H = hidden[:, -1, :].cpu().numpy()  # (n_prompts, d_model)
            
            prs = []
            for k in zoom_sizes:
                pr_samples = []
                for _ in range(20):
                    idx = np.random.choice(len(H), min(k, len(H)), replace=False)
                    pr_samples.append(compute_pr(H[idx]))
                prs.append(float(np.nanmean(pr_samples)))
            
            theta_obs, theta_floor, delta = theta_decompose(zoom_sizes, prs)
            results_per_layer[layer_idx] = {
                'theta_obs': theta_obs, 'theta_floor': theta_floor,
                'delta': delta, 'pr_final': prs[-1]
            }
    
    return results_per_layer

def extract_final_layer(model, tokenizer, prompts, device='cpu'):
    """Quick extraction: final layer only, for size/checkpoint sweeps."""
    model.eval()
    zoom_sizes = [4, 6, 8, 10, 12, 14, 16]
    
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = model(**inputs, output_hidden_states=True)
        H = outputs.hidden_states[-1][:, -1, :].cpu().numpy()
    
    prs = []
    for k in zoom_sizes:
        pr_samples = []
        for _ in range(20):
            idx = np.random.choice(len(H), min(k, len(H)), replace=False)
            pr_samples.append(compute_pr(H[idx]))
        prs.append(float(np.nanmean(pr_samples)))
    
    return theta_decompose(zoom_sizes, prs)

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'EleutherAI/pythia-410m-deduped'
    mode = sys.argv[2] if len(sys.argv) > 2 else 'final'  # 'final' or 'all_layers'
    
    out_dir = 'checkpoint_time/theta_embeddings'
    os.makedirs(out_dir, exist_ok=True)
    
    tag = model_path.split('/')[-1]
    print(f"θ extraction: {model_path} mode={mode}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    
    if mode == 'all_layers':
        results = extract_all_layers(model, tokenizer, PROMPTS)
        out = {'model': model_path, 'mode': 'per_layer', 'layers': {}}
        print(f"{'Layer':>5} {'θ_obs':>7} {'θ_floor':>8} {'δ':>7}")
        for l in sorted(results.keys()):
            r = results[l]
            print(f"{l:>5} {r['theta_obs']:7.4f} {r['theta_floor']:8.4f} {r['delta']:7.4f}")
            out['layers'][str(l)] = r
        json.dump(out, open(f'{out_dir}/{tag}_layers.json', 'w'), indent=2)
    else:
        theta_obs, theta_floor, delta = extract_final_layer(model, tokenizer, PROMPTS)
        out = {'model': model_path, 'mode': 'final_layer',
               'theta_obs': theta_obs, 'theta_floor': theta_floor, 'delta': delta}
        print(f"θ_obs={theta_obs:.4f} = θ_floor={theta_floor:.4f} + δ={delta:.4f}")
        json.dump(out, open(f'{out_dir}/{tag}.json', 'w'), indent=2)
    
