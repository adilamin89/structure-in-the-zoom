"""Run 17 - Multi-CLASS multi-axis δ battery on Pythia-160m.

Fixes run15-16's problems:
  - ALL axes are 8-class (6-rung ladders), never 2-class
  - Uses natural category structure from real benchmarks
  - Each class gets 16+ prompts for stable within-class PR
  - Prompt diversity within classes (real questions, not templates)

AXES (all 8-class):
  tqa_category:   TruthfulQA by misconception type (8 largest categories)
  hs_activity:    HellaSwag by activity domain (8 largest)
  arc_topic:      ARC-Challenge questions by science subdomain
  ethical:        Hand-built: 8 ethical/value domains
  world_knowledge: 8 world-knowledge domains (geography, history, science, ...)
  language_type:  8 linguistic construction types
  random:         random 8-class assignment on the TQA pool (must ≈ 0)

Each axis: 8 classes × 16 prompts = 128 prompts, 6-rung ladder [1,2,3,4,6,8].

REGISTERED EXPECTATIONS:
R1: random ≈ 0 at every layer (8-class exchangeability).
R2: axes with lexically distinct classes (world_knowledge) → positive δ at
    embedding, declining with depth (lexical inheritance).
R3: axes with semantically subtle classes (ethical, language_type) → weaker
    embedding δ, potentially growing at middle layers if the network organizes
    these categories beyond lexical cues.
R4: different axes give different depth profiles.

Model: Pythia-160m (local, CPU). ~45 min.
Out: feedback_runs/run18_pythia410m_battery.json
"""
import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
N_NULL = 10
N_SHUF = 5
N_PER_CLASS = 16
BIN_COUNTS = [1, 2, 3, 4, 6, 8]


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder_delta(X, labels, n_classes, rng, n_shuf=N_SHUF):
    bc = [c for c in BIN_COUNTS if c <= n_classes]
    members = [np.where(labels == c)[0] for c in range(n_classes)]
    if min(len(m) for m in members) < 3:
        return None, None
    sizes, prs = [], []
    for c in bc:
        sel = np.concatenate(members[:c])
        sizes.append(len(sel))
        prs.append(pr_c(X[sel]))
    if len(sizes) < 3:
        return None, None
    th_o = slope(sizes, np.asarray(prs))
    nl = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(pr_c(X[rng.choice(len(X), s, replace=False)]),
                                  1e-9))
    th_f = slope(sizes, np.exp(nl.mean(axis=0)))
    shufs = []
    for s in range(n_shuf):
        srng = np.random.default_rng(500 + s)
        perm = labels[srng.permutation(len(labels))]
        m2 = [np.where(perm == c)[0] for c in range(n_classes)]
        sz2, pr2 = [], []
        for c in bc:
            sel = np.concatenate(m2[:c])
            sz2.append(len(sel))
            pr2.append(pr_c(X[sel]))
        if len(sz2) >= 3:
            shufs.append(slope(sz2, np.asarray(pr2)) - th_f)
    return th_o - th_f, float(np.mean(shufs)) if shufs else 0.0


def get_hidden_states(model, tokenizer, prompts, device="cpu", max_len=128):
    all_states = []
    model.eval()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tokenizer(p, return_tensors="pt", truncation=True,
                            max_length=max_len).input_ids.to(device)
            out = model(ids, output_hidden_states=True)
            states = [h[0, -1, :].cpu().numpy() for h in out.hidden_states]
            all_states.append(states)
            if i > 0 and i % 50 == 0:
                print(f"    encoded {i}/{len(prompts)}", flush=True)
    n_layers = len(all_states[0])
    return [np.stack([s[l] for s in all_states]) for l in range(n_layers)]


def build_axes():
    axes = {}

    # 1. TruthfulQA by category (natural 8-class)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
        "/tmp/tqa.csv")
    with open("/tmp/tqa.csv") as f:
        tqa = list(csv.DictReader(f))
    cats = {}
    for r in tqa:
        cats.setdefault(r["Category"], []).append(r["Question"])
    top8 = sorted(cats, key=lambda c: -len(cats[c]))[:8]
    axes["tqa_category"] = {c: cats[c][:N_PER_CLASS] for c in top8}

    # 2. HellaSwag by activity (natural 8-class)
    from datasets import load_dataset
    hs = load_dataset("Rowan/hellaswag", split="validation")
    act_counts = Counter(hs["activity_label"])
    top_acts = [a for a, _ in act_counts.most_common(8)]
    act_prompts = {a: [] for a in top_acts}
    for item in hs:
        a = item["activity_label"]
        if a in act_prompts and len(act_prompts[a]) < N_PER_CLASS:
            act_prompts[a].append(item["ctx"])
    axes["hs_activity"] = act_prompts

    # 3. ARC by science subdomain (keyword-mined 8-class)
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    domains = {
        "biology": ["cell", "organism", "species", "plant", "animal", "food chain",
                     "ecosystem", "reproduce", "inherit", "trait"],
        "physics": ["force", "energy", "light", "sound", "gravity", "motion",
                     "speed", "electric", "wave", "heat"],
        "earth": ["rock", "weather", "climate", "earth", "ocean", "volcano",
                   "fossil", "erosion", "season", "earthquake"],
        "chemistry": ["atom", "element", "chemical", "molecule", "react",
                       "solution", "acid", "metal", "gas", "mixture"],
        "space": ["star", "planet", "moon", "sun", "orbit", "solar", "galaxy",
                   "comet", "universe", "telescope"],
        "ecology": ["environment", "habitat", "population", "resource",
                      "pollution", "recycle", "conserv", "extinct", "biome", "food web"],
        "human_body": ["heart", "lung", "brain", "muscle", "bone", "blood",
                        "digest", "breath", "nerve", "immune"],
        "weather": ["rain", "cloud", "wind", "temperature", "humid", "storm",
                     "snow", "fog", "pressure", "front"],
    }
    dom_prompts = {d: [] for d in domains}
    for item in arc:
        q = item["question"].lower()
        for d, kws in domains.items():
            if any(kw in q for kw in kws) and len(dom_prompts[d]) < N_PER_CLASS:
                dom_prompts[d].append(item["question"])
                break
    # keep only domains with enough
    dom_prompts = {d: v for d, v in dom_prompts.items() if len(v) >= 8}
    axes["arc_topic"] = dom_prompts

    # 4. Ethical/value domains (hand-built, diverse prompts)
    axes["ethical"] = {
        "fairness": [
            "The judge treated both sides equally in the", "Equal pay for equal work is a principle of",
            "Discrimination based on race is prohibited by", "Everyone deserves a fair trial regardless of",
            "The lottery was designed to give each person an equal", "Impartial judges are essential for",
            "Merit-based selection ensures that the most qualified", "Access to education should not depend on",
            "Bias in hiring practices can lead to", "A just society provides equal opportunity for",
            "The voting system was designed to represent all", "Fair allocation of resources requires",
            "Impartiality in decision making prevents", "Equal treatment under the law means",
            "Distributive justice concerns how goods are", "Procedural fairness requires transparent"],
        "harm": [
            "Violence against civilians is condemned by", "Weapons of mass destruction pose a threat to",
            "Child abuse is a serious criminal offense", "Pollution damages ecosystems and harms",
            "Cyberbullying can cause lasting psychological", "Drunk driving endangers the lives of",
            "Neglecting safety standards leads to workplace", "Animal cruelty is punishable under",
            "Toxic chemicals contaminate water supplies and", "Physical assault is a violation of",
            "Reckless behavior puts others at risk of", "Harassment creates a hostile environment for",
            "Malware can destroy computer systems and steal", "Explosions near populated areas threaten",
            "Secondhand smoke exposure increases the risk of", "Deforestation leads to habitat destruction and"],
        "honesty": [
            "Telling the truth builds trust between", "Perjury is the crime of lying under",
            "Transparent reporting helps investors make", "Academic plagiarism undermines the integrity of",
            "Whistleblowers expose corruption at personal", "False advertising misleads consumers about",
            "Scientific fraud can set back research by", "Honest communication is the foundation of",
            "Misinformation spreads rapidly through social", "Fabricating data violates research ethics and",
            "Full disclosure of conflicts of interest is", "Truth in lending laws protect borrowers from",
            "Authentic journalism requires fact checking and", "Candid feedback helps people improve their",
            "Transparency in government increases public", "Accurate labeling of food products ensures"],
        "privacy": [
            "Personal data should be protected from unauthorized", "Surveillance cameras raise concerns about civil",
            "Medical records are confidential under", "The right to privacy is recognized in many",
            "Data breaches expose millions of users sensitive", "Encryption protects communications from",
            "Social media platforms collect vast amounts of", "Tracking cookies monitor browsing habits without",
            "Wiretapping requires a warrant issued by", "Biometric data like fingerprints are uniquely",
            "The fourth amendment protects against unreasonable", "Identity theft occurs when someone uses another",
            "Anonymity online allows free expression but also", "Employee monitoring in the workplace must balance",
            "Location tracking on smartphones reveals patterns", "Right to be forgotten allows individuals to"],
        "autonomy": [
            "Informed consent is required before medical", "Freedom of speech allows individuals to express",
            "Voter suppression undermines democratic", "Religious freedom is protected by the first",
            "Forced labor violates basic human rights and", "Reproductive rights allow individuals to decide",
            "Censorship restricts access to information and", "Self determination is a fundamental principle of",
            "Compulsory education balances individual freedom with", "The right to refuse treatment is part of",
            "Bodily autonomy means having control over", "Freedom of movement allows people to travel",
            "Coercion undermines voluntary decision making and", "Press freedom enables journalists to report",
            "Cultural autonomy allows communities to preserve", "Patient autonomy requires respecting individual"],
        "loyalty": [
            "Patriotism involves devotion to ones country and", "Team loyalty helps organizations achieve their",
            "Betrayal of trust can destroy personal relationships", "Brand loyalty influences consumer purchasing",
            "Military service demonstrates commitment to national", "Family bonds create a sense of belonging and",
            "Whistleblowing can conflict with organizational", "Allegiance to a political party shapes voting",
            "Employee retention depends on workplace loyalty and", "National anthems symbolize collective identity and",
            "Treaties between nations establish mutual", "Alumni networks maintain connections to educational",
            "Solidarity among workers strengthens labor", "Cultural traditions bind communities through shared",
            "Oath of office requires officials to uphold", "Gang loyalty often comes at the cost of"],
        "authority": [
            "Government regulations ensure public safety and", "Teachers establish classroom rules to maintain",
            "Police officers enforce laws within their", "Court orders must be obeyed by all parties",
            "Military ranks establish a clear chain of", "Licensing requirements ensure professional",
            "Building codes prevent structural failures and", "Executive orders allow presidents to direct",
            "Parental authority guides children through their", "Religious leaders interpret sacred texts for",
            "Regulatory agencies oversee industry compliance with", "Judicial review checks the power of",
            "Expert opinions carry weight in specialized", "Institutional authority derives from established",
            "Democratic elections confer legitimate authority on", "Professional certifications establish credibility in"],
        "sanctity": [
            "Sacred sites are protected by cultural heritage", "The human body is treated with respect in",
            "Environmental preservation protects natural", "Religious ceremonies honor traditions passed",
            "Desecration of graves is considered deeply", "Clean water is essential for sustaining",
            "Organ donation saves lives while raising questions about", "Art restoration preserves masterworks for",
            "Endangered species receive protection under", "Cultural artifacts belong to the heritage of",
            "Ritual purification is practiced in many", "The sanctity of life is a central tenet of",
            "Historic buildings are preserved as monuments to", "Food preparation follows strict guidelines in",
            "Wilderness areas are set aside to protect", "Memorial sites honor the memory of those who"],
    }

    # 5. World knowledge (maximally lexically separated - the V1-analog)
    axes["world_knowledge"] = {
        "mathematics": [
            "The Pythagorean theorem states that in a right triangle",
            "Calculus was independently developed by Newton and",
            "The Fibonacci sequence appears in natural patterns like",
            "Set theory provides the foundation for modern",
            "Probability theory was formalized by Kolmogorov in",
            "The prime number theorem describes the distribution of",
            "Linear algebra studies vector spaces and linear",
            "Topology studies properties preserved under continuous",
            "The fundamental theorem of algebra states every polynomial",
            "Differential equations model how quantities change over",
            "Game theory analyzes strategic interactions between",
            "Number theory studies the properties of integers and",
            "Statistics uses data to make inferences about",
            "Geometry studies the properties of shapes and",
            "Boolean algebra forms the basis of digital logic and",
            "Graph theory studies networks of connected nodes and"],
        "geography": [
            "The Nile River flows northward through eleven African",
            "The Himalayas separate the Indian subcontinent from",
            "The Pacific Ocean is the largest and deepest ocean",
            "The Amazon Basin contains the worlds largest tropical",
            "Tectonic plates slowly move causing earthquakes and",
            "The Sahara Desert covers most of North Africa and",
            "Island nations like Japan face unique geological",
            "The Great Barrier Reef stretches along the Australian",
            "Continental drift explains how landmasses separated over",
            "The Arctic tundra supports specialized organisms adapted",
            "Monsoon seasons bring heavy rainfall to South and",
            "The Mediterranean climate features hot dry summers and",
            "Glaciers carve valleys and shape mountain landscapes over",
            "The Mississippi River system drains much of central",
            "Volcanic islands form when magma rises from the ocean",
            "The Andes mountain range runs along the western coast"],
        "history": [
            "The fall of Constantinople in 1453 marked the end of",
            "The printing press invented by Gutenberg revolutionized",
            "The French Revolution of 1789 overthrew the monarchy and",
            "Ancient Rome developed an extensive system of roads and",
            "The Silk Road connected East Asian and European trade",
            "World War One began with the assassination of Archduke",
            "The Renaissance marked a cultural rebirth in Europe during",
            "The Industrial Revolution transformed manufacturing in",
            "Ancient Egyptian civilization thrived along the Nile for",
            "The Cold War divided the world into competing ideological",
            "The Age of Exploration led Europeans to discover new",
            "The American Civil War was fought over slavery and",
            "The Ottoman Empire controlled much of southeastern Europe",
            "The Mongol Empire was the largest contiguous land empire",
            "The abolition of slavery occurred at different times across",
            "The Space Race between the US and USSR culminated in"],
        "technology": [
            "Artificial intelligence uses neural networks to learn",
            "The internet connects billions of devices worldwide through",
            "Quantum computing uses superposition and entanglement to",
            "Blockchain technology provides a decentralized ledger for",
            "CRISPR gene editing allows precise modification of",
            "Cloud computing delivers computing services over the",
            "Autonomous vehicles use sensors and algorithms to navigate",
            "Renewable energy technologies include solar panels and",
            "The transistor revolutionized electronics by enabling",
            "Machine learning algorithms improve their performance with",
            "Cybersecurity protects digital systems from unauthorized",
            "Three dimensional printing creates objects layer by layer",
            "Satellite navigation systems provide precise location data",
            "Fiber optic cables transmit data using pulses of light",
            "Biotechnology applies biological processes to develop new",
            "Robotics combines engineering and computer science to build"],
        "law": [
            "The constitution establishes the fundamental principles of",
            "Criminal law defines offenses against the state and",
            "Contract law governs agreements between parties and",
            "International law regulates relations between sovereign",
            "The presumption of innocence means the accused is",
            "Tort law addresses civil wrongs and provides remedies",
            "Intellectual property law protects inventions and creative",
            "Environmental regulations set standards for pollution and",
            "Due process ensures fair treatment through the judicial",
            "Antitrust laws prevent monopolies and promote market",
            "Immigration law governs who may enter and remain in",
            "Tax law determines how governments collect revenue from",
            "Maritime law covers legal issues arising on navigable",
            "Labor law protects workers rights including wages and",
            "The rule of law means everyone is subject to the same",
            "Habeas corpus prevents unlawful detention by requiring"],
        "medicine": [
            "Vaccines stimulate the immune system to develop protection",
            "Antibiotics treat bacterial infections by killing or",
            "MRI scans use magnetic fields to create detailed images",
            "Blood types must be matched before transfusions to prevent",
            "Anesthesia allows patients to undergo surgery without",
            "Chemotherapy uses drugs to destroy rapidly dividing cancer",
            "The placebo effect demonstrates the power of expectation",
            "Epidemiology studies the distribution and determinants of",
            "Organ transplantation replaces failing organs with healthy",
            "Clinical trials test the safety and efficacy of new",
            "Diagnostic imaging includes X rays CT scans and",
            "Infectious diseases spread through various pathways including",
            "Surgical techniques have advanced with minimally invasive",
            "Mental health disorders affect mood thinking and behavior",
            "Rehabilitation helps patients recover function after injury",
            "Preventive medicine focuses on avoiding disease before it"],
        "philosophy": [
            "Epistemology asks how we know what we claim to",
            "Ethics examines what constitutes right and wrong",
            "Existentialism emphasizes individual freedom and personal",
            "Utilitarianism judges actions by their consequences and",
            "The social contract theory explains the origin of",
            "Empiricism holds that knowledge comes primarily from",
            "Rationalism argues that reason is the primary source of",
            "Phenomenology studies the structures of conscious",
            "Determinism holds that all events are causally",
            "The trolley problem illustrates the tension between",
            "Stoicism teaches acceptance of things beyond our",
            "Pragmatism evaluates ideas by their practical",
            "Aesthetics studies the nature of beauty and artistic",
            "Metaphysics examines the fundamental nature of reality",
            "Logic studies the principles of valid reasoning and",
            "Nihilism rejects the existence of inherent meaning or"],
        "sports": [
            "The Olympic Games bring together athletes from around",
            "Soccer is the most popular sport worldwide with billions",
            "Basketball was invented by James Naismith in 1891 as",
            "Tennis Grand Slam tournaments include Wimbledon and the",
            "Marathon running covers a distance of forty two kilometers",
            "Swimming competitions are measured in freestyle backstroke",
            "Cricket matches can last up to five days in test",
            "Baseball has been called Americas pastime since the",
            "Boxing weight classes ensure fair competition between",
            "Track and field events include sprints hurdles and",
            "Rugby originated at Rugby School in England during the",
            "Golf courses typically have eighteen holes with varying",
            "Ice hockey is played on a frozen surface with six",
            "Volleyball can be played indoors on courts or outdoors",
            "Gymnastics requires strength flexibility and precise",
            "Formula One racing features the fastest open wheel cars"],
    }

    # 6. Language construction types
    axes["language_type"] = {
        "question": [
            "What causes earthquakes to occur along fault lines",
            "How does photosynthesis convert light into energy",
            "Why do birds migrate thousands of miles each year",
            "When did humans first begin to domesticate animals",
            "Where are the deepest parts of the ocean located",
            "Who discovered the structure of DNA in 1953",
            "Which planet in our solar system is the largest",
            "How many species of insects have been identified so far",
            "What determines the color of a persons eyes",
            "Why does water expand when it freezes into ice",
            "How do vaccines train the immune system to fight",
            "What causes the northern lights to appear in the sky",
            "When was the first computer program written and by whom",
            "Where do most of the worlds earthquakes take place",
            "Who invented the telephone in the late nineteenth century",
            "Which element is the most abundant in the universe"],
        "definition": [
            "A molecule is a group of atoms bonded together",
            "Democracy is a system of government where citizens",
            "An ecosystem is a community of living organisms and",
            "Inflation is the rate at which prices for goods",
            "A chromosome is a structure of DNA and protein found",
            "Photosynthesis is the process by which plants convert",
            "A peninsula is a piece of land surrounded by water on",
            "An algorithm is a step by step procedure for solving",
            "Entropy is a measure of the disorder or randomness in",
            "A metaphor is a figure of speech that describes something",
            "Biodiversity is the variety of life forms found in",
            "A catalyst is a substance that speeds up a chemical",
            "Sovereignty is the supreme authority within a territory",
            "An isotope is a variant of an element with different",
            "Culture is the set of shared beliefs practices and",
            "A theorem is a statement that has been proven through"],
        "comparison": [
            "Unlike mammals reptiles are cold blooded and rely on",
            "While democracies hold elections authoritarian regimes",
            "Compared to the Atlantic the Pacific Ocean is much",
            "In contrast to classical physics quantum mechanics",
            "Similar to Earth Mars has polar ice caps made of",
            "Whereas poetry uses meter and rhyme prose is written",
            "Just as light travels in waves sound also propagates",
            "The difference between mitosis and meiosis is that",
            "Unlike renewable energy sources fossil fuels will",
            "While bacteria are single celled organisms most fungi",
            "Compared to ancient Rome modern cities have much more",
            "In contrast to capitalism socialism emphasizes collective",
            "Similar to how muscles grow through exercise the brain",
            "Whereas freshwater makes up only three percent of",
            "Just as vaccines prevent disease antibiotics treat",
            "The distinction between weather and climate is that"],
        "narrative": [
            "The explorer set out at dawn crossing the vast desert",
            "After years of research the scientist finally discovered",
            "The ancient civilization flourished for centuries before",
            "During the long winter the village relied on stored",
            "The ship sailed across the Atlantic carrying hundreds of",
            "In the aftermath of the earthquake rescue teams searched",
            "The inventor worked tirelessly in the workshop perfecting",
            "Throughout history great leaders have inspired their",
            "The migration began when food sources became scarce in",
            "After the revolution the new government established",
            "The architect designed the building to withstand extreme",
            "During the expedition the team encountered unexpected",
            "The artist spent decades mastering the techniques of",
            "Following the discovery of penicillin medicine was",
            "The traders journeyed along ancient routes carrying silk",
            "After the harvest the farmers stored grain for the"],
        "cause_effect": [
            "Because the temperature dropped below freezing the pipes",
            "The increase in carbon dioxide has led to rising global",
            "Due to deforestation many species have lost their natural",
            "Overfishing has resulted in declining populations of",
            "The invention of the printing press caused a dramatic",
            "Rising sea levels are causing coastal erosion and",
            "Lack of exercise leads to increased risk of heart",
            "The drought caused widespread crop failures throughout",
            "Industrial pollution has contaminated many rivers and",
            "The introduction of invasive species disrupted the local",
            "Exposure to ultraviolet radiation can cause skin damage",
            "Poor nutrition during childhood leads to developmental",
            "The discovery of antibiotics resulted in a dramatic",
            "Excessive screen time has been linked to sleep",
            "Urbanization has led to the loss of agricultural land",
            "The collapse of the bridge was caused by metal fatigue"],
        "instruction": [
            "To calculate the area of a circle multiply pi by",
            "First preheat the oven to three hundred fifty degrees",
            "Begin by gathering all necessary materials before",
            "To solve this equation isolate the variable on one",
            "Start the experiment by measuring exactly ten grams of",
            "Mix the dry ingredients thoroughly before adding the",
            "Connect the positive terminal of the battery to the",
            "To plant a tree dig a hole twice the width of",
            "Adjust the microscope lens until the specimen comes into",
            "Apply the first coat of paint evenly across the entire",
            "To convert Fahrenheit to Celsius subtract thirty two and",
            "Insert the needle at a forty five degree angle into",
            "Fold the paper in half lengthwise then crease firmly",
            "Pour the solution slowly into the graduated cylinder",
            "Secure the rope with a bowline knot to prevent it from",
            "To start the program enter the command followed by"],
        "opinion": [
            "Many experts believe that renewable energy will replace",
            "It is widely considered that education is the key to",
            "Some argue that artificial intelligence poses risks to",
            "The general consensus is that exercise improves mental",
            "Critics maintain that standardized testing fails to",
            "Supporters of space exploration argue that colonizing",
            "Economists debate whether free trade benefits all",
            "Environmentalists contend that current conservation",
            "Some historians believe the Roman Empire fell due to",
            "Researchers suggest that bilingualism enhances cognitive",
            "Many philosophers argue that free will is an illusion",
            "Nutritionists recommend eating a balanced diet rich in",
            "Urban planners advocate for more green spaces in",
            "Scientists warn that antibiotic resistance threatens",
            "Educators propose that project based learning engages",
            "Psychologists suggest that early childhood experiences"],
        "negation": [
            "Not all metals are magnetic for example copper and",
            "It is not true that humans only use ten percent of",
            "Contrary to popular belief lightning can strike the",
            "The earth is not the center of the solar system as",
            "Fish do not actually have a three second memory as",
            "There is no evidence that cracking knuckles causes",
            "Bats are not blind and many species have excellent",
            "Sugar does not actually cause hyperactivity in children",
            "The Great Wall of China is not visible from space with",
            "Goldfish are not limited to a few seconds of memory",
            "Touching a baby bird will not cause its mother to",
            "Shaving does not actually make hair grow back thicker",
            "Bulls are not enraged by the color red as they are",
            "Einstein did not fail mathematics as a student despite",
            "Eating carrots does not significantly improve night",
            "Vikings did not actually wear horned helmets during"],
    }

    return axes


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = build_axes()

    print("\nloading Pythia-160m...", flush=True)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m-deduped")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.num_hidden_layers

    out = {"model": "EleutherAI/pythia-410m-deduped", "n_layers": n_layers, "axes": {}}

    # Collect all prompts for random control
    all_prompts_pool = []
    for axis in axes.values():
        for cls in axis.values():
            all_prompts_pool.extend(cls)

    for axis_name, classes in axes.items():
        class_names = list(classes.keys())
        n_classes = len(class_names)
        all_prompts, labels = [], []
        for ci, cn in enumerate(class_names):
            for p in classes[cn]:
                all_prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        print(f"\n[{axis_name}] {n_classes} classes, {len(all_prompts)} prompts",
              flush=True)

        per_layer = get_hidden_states(model, tokenizer, all_prompts)

        axis_results = {"n_classes": n_classes, "n_prompts": len(all_prompts),
                        "class_names": class_names, "layers": []}
        for l in range(len(per_layer)):
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            d, sh = ladder_delta(X, labels, n_classes,
                                 np.random.default_rng(l * 100 + 1))
            if d is None:
                continue
            axis_results["layers"].append({
                "layer": l, "delta": d, "shuffle_mean": sh})

        if axis_results["layers"]:
            emb_delta = axis_results["layers"][0]["delta"]
            for lr in axis_results["layers"]:
                lr["delta_excess"] = lr["delta"] - emb_delta

        out["axes"][axis_name] = axis_results
        if axis_results["layers"]:
            ds = [lr["delta"] for lr in axis_results["layers"]]
            ss = [lr["shuffle_mean"] for lr in axis_results["layers"]]
            exs = [lr.get("delta_excess", 0) for lr in axis_results["layers"]]
            print(f"  delta: [{min(ds):+.3f}, {max(ds):+.3f}] | "
                  f"shuffle: [{min(ss):+.3f}, {max(ss):+.3f}] | "
                  f"excess: [{min(exs):+.3f}, {max(exs):+.3f}]", flush=True)

    # 8-class random control
    print(f"\n[random] 8-class random on {len(all_prompts_pool)} prompts",
          flush=True)
    per_layer_rand = get_hidden_states(model, tokenizer,
                                       all_prompts_pool[:128])
    rng = np.random.default_rng(999)
    rand_labels = rng.integers(0, 8, 128)
    rand_results = {"n_prompts": 128, "n_classes": 8, "layers": []}
    for l in range(len(per_layer_rand)):
        X = per_layer_rand[l].astype(np.float32)
        X = X / (X.std() + 1e-9)
        d, sh = ladder_delta(X, rand_labels, 8,
                              np.random.default_rng(l * 100 + 99))
        rand_results["layers"].append({
            "layer": l, "delta": d, "shuffle_mean": sh if sh else 0.0})
    out["axes"]["random"] = rand_results
    ds = [lr["delta"] for lr in rand_results["layers"]
          if lr["delta"] is not None]
    print(f"  random mean: {np.mean(ds):+.3f} | "
          f"max |delta|: {max(abs(d) for d in ds):.3f}", flush=True)

    json.dump(out, open(HERE / "run18_pythia410m_battery.json", "w"),
              indent=1)
    print("\nDONE run17", flush=True)


if __name__ == "__main__":
    main()
