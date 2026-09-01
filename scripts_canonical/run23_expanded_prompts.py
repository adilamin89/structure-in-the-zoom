"""Run 23 - Expanded prompts (32/class) on the two flagship axes + OLMo-1B
cross-family check. Fixes T4's resolution floor by doubling prompt count.

Design: world_knowledge + language_type each get 32 prompts per class
(16 existing + 16 new). Run on Pythia-160m (comparison) and OLMo-1B-hf
(cross-family). Report: per-layer δ profiles, embedding excess, and the
split-half cross-axis correlation (T4 rerun with 16/half instead of 8/half).

REGISTERED:
R1: Pythia-160m profiles replicate run17 (same pattern, tighter).
R2: OLMo-1B shows the same content-vs-structure contrast (cross-family).
R3: T4 rerun with 16/half gives positive ρ (was −0.94 at 8/half).

Out: feedback_runs/run23_expanded_prompts.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)

# 16 NEW prompts per class for the two flagship axes
EXTRA_WORLD = {
    "mathematics": [
        "Abstract algebra studies groups rings and fields",
        "Combinatorics counts arrangements and combinations of objects",
        "Numerical analysis approximates solutions to continuous problems",
        "Fractal geometry describes self-similar patterns at every scale",
        "Cryptography uses number theory to secure digital communications",
        "Measure theory extends the concept of length and area",
        "Optimization finds the best solution among many alternatives",
        "Chaos theory shows how small changes lead to unpredictable outcomes",
        "Mathematical induction proves statements about natural numbers",
        "Fourier analysis decomposes signals into frequency components",
        "Algebraic topology classifies spaces by their holes and connectivity",
        "Stochastic processes model random phenomena evolving over time",
        "Complex analysis studies functions of complex variables",
        "Information theory quantifies the limits of data compression",
        "Knot theory classifies mathematical knots and their properties",
        "Category theory provides a unifying framework for mathematics"],
    "geography": [
        "The Mariana Trench reaches nearly eleven thousand meters deep",
        "River deltas form where flowing water deposits sediment at coast",
        "The Ring of Fire circles the Pacific with volcanic activity",
        "Fjords are deep narrow inlets carved by glacial erosion",
        "Latitude and longitude provide coordinates for any location",
        "Coral atolls form rings around submerged volcanic peaks",
        "The jet stream influences weather patterns across continents",
        "Karst landscapes develop from dissolving limestone bedrock",
        "Continental shelves extend underwater from coastlines to deep ocean",
        "The Coriolis effect deflects moving objects on rotating Earth",
        "Estuaries are where rivers meet the sea in brackish mixing zones",
        "The Gobi Desert stretches across northern China and Mongolia",
        "Permafrost is ground that remains frozen for two or more years",
        "The Great Rift Valley extends from Lebanon to Mozambique",
        "Barrier islands protect coastlines from ocean waves and storms",
        "Pangaea was the supercontinent that began breaking apart millions"],
    "history": [
        "The Magna Carta limited the power of English monarchs in 1215",
        "The Scientific Revolution transformed understanding of nature",
        "Colonial empires controlled vast territories across the globe",
        "The Reformation split Western Christianity into Catholic and Protestant",
        "The Meiji Restoration modernized Japan in the late nineteenth century",
        "The Crusades were military campaigns to reclaim the Holy Land",
        "The Enlightenment emphasized reason and individual rights",
        "The Partition of India created two independent nations in 1947",
        "The Hundred Years War was fought between England and France",
        "The Columbian Exchange transferred plants animals and diseases",
        "Feudalism organized medieval society around land and loyalty",
        "The Treaty of Versailles ended World War One with harsh terms",
        "The Neolithic Revolution introduced agriculture and settlements",
        "The Scramble for Africa divided the continent among European powers",
        "The Ming Dynasty ruled China for nearly three hundred years",
        "The Emancipation Proclamation declared enslaved people free"],
    "technology": [
        "Virtual reality creates immersive digital environments for users",
        "Edge computing processes data near the source rather than centrally",
        "Natural language processing enables computers to understand text",
        "Digital twins are virtual replicas of physical systems",
        "Augmented reality overlays digital information on the real world",
        "Container orchestration manages deployment of microservices at scale",
        "Lidar uses laser pulses to create detailed three dimensional maps",
        "Brain computer interfaces translate neural signals into commands",
        "Serverless computing runs code without managing infrastructure",
        "Generative models create new content from learned patterns",
        "The Internet of Things connects everyday objects to networks",
        "Additive manufacturing builds objects layer by layer from materials",
        "Computer vision enables machines to interpret visual information",
        "Federated learning trains models across distributed devices",
        "Homomorphic encryption allows computation on encrypted data",
        "Swarm robotics coordinates many simple robots for complex tasks"],
    "law": [
        "Statutory law consists of rules enacted by legislative bodies",
        "Precedent requires courts to follow earlier decisions on similar cases",
        "Regulatory agencies create and enforce specific industry rules",
        "Constitutional amendments modify the fundamental governing document",
        "Mediation resolves disputes through a neutral third party",
        "Sovereign immunity protects governments from certain lawsuits",
        "Bankruptcy law provides a process for debt relief and reorganization",
        "Class action lawsuits allow groups to sue collectively",
        "Extradition treaties govern the transfer of accused persons",
        "Whistleblower protections shield those who report wrongdoing",
        "Eminent domain allows governments to acquire private property",
        "Probate courts oversee the distribution of deceased persons estates",
        "Administrative law governs the activities of government agencies",
        "Fiduciary duty requires acting in another persons best interest",
        "Judicial independence ensures courts are free from political pressure",
        "Customary international law derives from consistent state practice"],
    "medicine": [
        "Immunotherapy harnesses the immune system to fight cancer cells",
        "Genetic testing identifies inherited risk factors for disease",
        "Telemedicine delivers healthcare remotely through digital technology",
        "Stem cell therapy aims to regenerate damaged tissues and organs",
        "Biomarkers indicate the presence or progression of disease",
        "Precision medicine tailors treatment to individual genetic profiles",
        "Palliative care focuses on comfort for patients with serious illness",
        "Pharmacogenomics studies how genes affect drug response",
        "Epidemiological surveillance tracks disease outbreaks in populations",
        "Radiology uses imaging techniques to diagnose internal conditions",
        "Microbiome research reveals how gut bacteria influence health",
        "Prosthetics replace missing limbs with increasingly sophisticated devices",
        "Pathology examines tissues and fluids to identify disease",
        "Emergency medicine provides immediate care for acute conditions",
        "Regenerative medicine seeks to restore damaged organ function",
        "Public health interventions prevent disease at the population level"],
    "philosophy": [
        "Virtue ethics focuses on developing good character traits",
        "The mind body problem asks how consciousness relates to matter",
        "Political philosophy examines justice power and the state",
        "Relativism holds that truth depends on perspective and context",
        "The problem of evil questions how suffering exists alongside divinity",
        "Social contract theory explains political obligation through agreement",
        "Analytic philosophy emphasizes clarity and logical argument",
        "Continental philosophy explores experience meaning and interpretation",
        "Ethical egoism argues that self interest should guide moral decisions",
        "Free will debates whether our choices are truly our own",
        "Philosophy of science examines how scientific knowledge is produced",
        "Bioethics addresses moral questions raised by medical advances",
        "Feminist philosophy critiques traditional assumptions about gender",
        "Environmental ethics considers moral obligations to nature",
        "Philosophy of language studies meaning reference and communication",
        "Skepticism doubts the possibility of certain knowledge"],
    "sports": [
        "The World Cup is the most watched sporting event globally",
        "Archery requires precision focus and controlled breathing",
        "Professional cycling races cover thousands of kilometers in stages",
        "Fencing combines speed strategy and precise blade work",
        "Table tennis requires rapid reflexes and spin control",
        "Rock climbing tests strength endurance and problem solving",
        "Synchronized swimming combines athletic skill with artistic expression",
        "Rowing requires coordination between multiple team members",
        "Skiing descends snowy slopes at speeds exceeding one hundred",
        "Surfing rides ocean waves using balance and board control",
        "Weightlifting tests maximum strength in two competition lifts",
        "Triathlon combines swimming cycling and running in one event",
        "Badminton shuttlecocks can travel over three hundred kilometers",
        "Water polo combines swimming with ball handling and teamwork",
        "Judo uses an opponents momentum to execute throws",
        "Esports competitions draw millions of online viewers worldwide"],
}

EXTRA_LANG = {
    "question": [
        "Can renewable energy fully replace fossil fuels by 2050",
        "Does regular exercise reduce the risk of dementia",
        "Which programming language is best for beginners to learn",
        "Are there undiscovered species in the deep ocean",
        "Could asteroid mining become economically feasible",
        "Will artificial intelligence surpass human creativity",
        "Is there a limit to how fast computers can process data",
        "Do animals experience emotions similar to humans",
        "What role does sleep play in memory consolidation",
        "How do coral reefs adapt to changing ocean temperatures",
        "Are there practical applications for quantum entanglement",
        "Why do some languages have more vowels than others",
        "Can cities be designed to be completely carbon neutral",
        "Does music training improve mathematical ability",
        "How do migratory birds navigate across vast distances",
        "What makes some materials superconducting at low temperatures"],
    "definition": [
        "A paradigm is a framework of assumptions and methods",
        "Symbiosis is a close relationship between different species",
        "A monopoly is a market dominated by a single seller",
        "Photovoltaic cells convert sunlight directly into electricity",
        "A thesis is the central argument of an academic work",
        "Osmosis is the movement of water through a semipermeable membrane",
        "A derivative measures the rate of change of a function",
        "An archipelago is a chain or cluster of islands",
        "A hypothesis is a testable prediction about natural phenomena",
        "Mitosis is the process of cell division for growth",
        "A constitution is the supreme legal framework of a nation",
        "An alloy is a mixture of two or more metallic elements",
        "A biome is a large community of plants and animals",
        "Thermodynamics is the study of heat energy and work",
        "A renaissance is a period of cultural and intellectual rebirth",
        "An axiom is a statement accepted as true without proof"],
    "comparison": [
        "While rivers flow continuously lakes are standing bodies of water",
        "Unlike hardwood softwood comes from coniferous trees that grow faster",
        "Compared to oil natural gas produces fewer carbon emissions",
        "In contrast to monarchies republics elect their heads of state",
        "Similar to radar sonar uses waves to detect objects underwater",
        "Whereas deserts receive minimal rainfall tropical forests get abundant",
        "Just as roots anchor plants foundations support buildings",
        "The difference between speed and velocity is that velocity includes",
        "Unlike solids liquids take the shape of their container",
        "While telescopes observe distant objects microscopes examine tiny ones",
        "Compared to analog signals digital signals resist noise better",
        "In contrast to herbivores carnivores obtain energy from consuming",
        "Similar to how rivers erode rock wind shapes desert landforms",
        "Whereas conduction transfers heat through contact convection uses",
        "Just as evolution shapes species market forces shape economies",
        "The distinction between empathy and sympathy is that empathy involves"],
    "narrative": [
        "The caravan crossed the mountain pass just before winter sealed it",
        "For generations the village had depended on the annual salmon run",
        "The mathematician spent years searching for an elegant proof",
        "When the dam broke the downstream communities had only hours",
        "The translator worked through the night to finish the manuscript",
        "Decades of careful breeding produced a drought resistant crop",
        "The astronaut looked back at Earth from the station window",
        "Through trial and error the engineer perfected the design",
        "The pandemic forced the entire world to reconsider daily routines",
        "A chance discovery in the attic revealed letters from the war",
        "The orchestra rehearsed for months before the premiere performance",
        "After the eruption the landscape was unrecognizable to survivors",
        "The journalist traveled to remote regions to document disappearing",
        "By the time the rescue team arrived the flood waters had receded",
        "The apprentice gradually mastered the techniques passed down through",
        "When electricity first reached the rural town everything changed"],
    "cause_effect": [
        "Melting ice caps contribute to rising sea levels worldwide",
        "Increased screen brightness at night disrupts circadian rhythms",
        "Volcanic eruptions inject particles that temporarily cool the climate",
        "Antibiotic overuse accelerates the development of resistant bacteria",
        "Removing apex predators causes prey populations to increase rapidly",
        "Soil compaction from heavy machinery reduces water absorption",
        "Light pollution prevents many people from seeing stars at night",
        "Ocean acidification weakens the shells of marine organisms",
        "Sleep deprivation impairs cognitive function and decision making",
        "Invasive plants outcompete native species for available resources",
        "Prolonged drought increases the likelihood of devastating wildfires",
        "Air pollution from vehicles contributes to respiratory diseases",
        "Glacial retreat exposes new land that slowly becomes vegetated",
        "Noise pollution in oceans interferes with whale communication",
        "Habitat fragmentation isolates animal populations and reduces diversity",
        "Microplastic contamination affects organisms throughout the food chain"],
    "instruction": [
        "Measure the ingredients precisely before combining them together",
        "Allow the engine to warm up before driving in cold weather",
        "Position the antenna facing south for optimal satellite reception",
        "Sterilize all equipment before beginning the laboratory experiment",
        "Back up your files regularly to prevent data loss",
        "Calibrate the instrument using the reference standard provided",
        "Rotate the crops each season to maintain soil fertility",
        "Tighten the bolts in a star pattern to distribute pressure evenly",
        "Filter the solution through cheesecloth to remove solid particles",
        "Label each sample clearly with the date and contents",
        "Dilute the concentrate according to the ratio specified",
        "Sand the surface smooth before applying the final finish",
        "Verify the voltage before connecting any electrical components",
        "Align the mirrors carefully to direct the beam accurately",
        "Record all observations immediately to avoid memory errors",
        "Let the concrete cure for at least seven days before loading"],
    "opinion": [
        "Several studies suggest that reading fiction improves empathy",
        "Public health officials recommend limiting processed food intake",
        "Engineers argue that nuclear power deserves reconsideration",
        "Teachers report that collaborative learning increases engagement",
        "Economists predict that automation will transform labor markets",
        "Conservationists believe that rewilding can restore ecosystems",
        "Legal scholars debate whether privacy laws keep pace with technology",
        "Climate scientists urge immediate reduction of greenhouse emissions",
        "Architects advocate for sustainable building materials in construction",
        "Linguists observe that languages evolve faster in urban environments",
        "Sociologists note that social media reshapes community formation",
        "Biologists propose that microbiome diversity indicates overall health",
        "Technologists envision decentralized systems replacing central platforms",
        "Historians argue that trade networks shaped civilizations more than wars",
        "Astronomers speculate about the possibility of life beyond Earth",
        "Ethicists question whether genetic enhancement is morally permissible"],
    "negation": [
        "The moon does not produce its own light but reflects sunlight",
        "Cold weather alone does not cause colds which are viral infections",
        "Ostriches do not actually bury their heads in sand when frightened",
        "Hair and fingernails do not continue growing after death",
        "Antibiotics are not effective against viral infections like the flu",
        "The tongue does not have separate zones for different tastes",
        "Lemmings do not deliberately jump off cliffs in mass suicide",
        "Humans do not have just five senses but many more including balance",
        "Napoleon was not especially short for his time period",
        "Glass is not a slowly flowing liquid despite the common myth",
        "Dropping a penny from a tall building will not cause serious injury",
        "Searing meat does not actually seal in its juices",
        "We do not swallow spiders in our sleep as the myth claims",
        "Chameleons do not change color primarily for camouflage",
        "Thomas Edison did not invent the light bulb from scratch",
        "Adding salt to water does not significantly change its boiling point"],
}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr

    # Build expanded axes (16 original + 16 new = 32 per class)
    axes = r17.build_axes()
    for cls, extra in EXTRA_WORLD.items():
        axes["world_knowledge"][cls] = axes["world_knowledge"][cls] + extra
    for cls, extra in EXTRA_LANG.items():
        axes["language_type"][cls] = axes["language_type"][cls] + extra

    target_axes = ["world_knowledge", "language_type"]
    models_to_run = [
        ("EleutherAI/pythia-160m", "pythia160m"),
        ("allenai/OLMo-1B-hf", "olmo1b"),
    ]

    out = {}
    for model_name, tag in models_to_run:
        print(f"\n{'='*60}\n[{tag}] loading {model_name}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if "olmo" in tag else None)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        n_layers = model.config.num_hidden_layers
        print(f"  {n_layers} layers, d={model.config.hidden_size}", flush=True)

        model_out = {"model": model_name, "n_layers": n_layers, "axes": {}}

        for axis_name in target_axes:
            classes = axes[axis_name]
            class_names = list(classes.keys())
            n_classes = len(class_names)
            all_prompts, labels = [], []
            for ci, cn in enumerate(class_names):
                for p in classes[cn]:
                    all_prompts.append(p)
                    labels.append(ci)
            labels = np.array(labels)
            print(f"\n  [{axis_name}] {n_classes} classes, "
                  f"{len(all_prompts)} prompts ({len(all_prompts)//n_classes}/class)",
                  flush=True)

            per_layer = r17.get_hidden_states(model, tokenizer, all_prompts)

            axis_results = {"n_classes": n_classes,
                            "n_prompts": len(all_prompts),
                            "prompts_per_class": len(all_prompts) // n_classes,
                            "layers": []}
            for l in range(len(per_layer)):
                X = per_layer[l].astype(np.float32)
                X /= (X.std() + 1e-9)
                d, sh = r17.ladder_delta(X, labels, n_classes,
                                         np.random.default_rng(l * 100 + 1))
                if d is not None:
                    axis_results["layers"].append({
                        "layer": l, "delta": d, "shuffle_mean": sh})

            if axis_results["layers"]:
                emb_delta = axis_results["layers"][0]["delta"]
                for lr in axis_results["layers"]:
                    lr["delta_excess"] = lr["delta"] - emb_delta

            model_out["axes"][axis_name] = axis_results
            if axis_results["layers"]:
                ds = [lr["delta"] for lr in axis_results["layers"]]
                print(f"    delta: [{min(ds):+.3f}, {max(ds):+.3f}]", flush=True)

        # T4 rerun: split-half with 16/half
        print(f"\n  [T4 rerun] split-half cross-axis (16/half)...", flush=True)
        delta_A, delta_B = {}, {}
        for axis_name in target_axes:
            classes = axes[axis_name]
            class_names = list(classes.keys())
            n_classes = len(class_names)
            for half, start, end, store in [
                ("A", 0, 16, delta_A), ("B", 16, 32, delta_B)
            ]:
                prompts, labs = [], []
                for ci, cn in enumerate(class_names):
                    ps = classes[cn][start:end]
                    prompts.extend(ps)
                    labs.extend([ci] * len(ps))
                labs = np.array(labs)
                per_layer = r17.get_hidden_states(model, tokenizer, prompts)
                ds = []
                for l in range(len(per_layer)):
                    X = per_layer[l].astype(np.float32)
                    X /= (X.std() + 1e-9)
                    d, _ = r17.ladder_delta(X, labs, n_classes,
                                            np.random.default_rng(l * 100 + 1))
                    ds.append(d if d is not None else 0)
                store[axis_name] = float(np.mean(ds))

        axes_common = [a for a in target_axes if a in delta_A and a in delta_B]
        if len(axes_common) >= 2:
            x = [delta_A[a] for a in axes_common]
            y = [delta_B[a] for a in axes_common]
            # With only 2 axes, Spearman = sign agreement
            agree = (x[0] > x[1]) == (y[0] > y[1])
            model_out["t4_rerun"] = {
                "half_A": {a: delta_A[a] for a in axes_common},
                "half_B": {a: delta_B[a] for a in axes_common},
                "ranking_agrees": bool(agree)}
            print(f"    A: {delta_A} | B: {delta_B} | agree: {agree}",
                  flush=True)

        out[tag] = model_out
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    json.dump(out, open(HERE / "run23_expanded_prompts.json", "w"), indent=1)
    print("\nDONE run23", flush=True)


if __name__ == "__main__":
    main()
