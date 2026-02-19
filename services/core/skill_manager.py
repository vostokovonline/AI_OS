import os, json, subprocess, httpx
SKILLS_DIR = "/app/skills"
REGISTRY_FILE = os.path.join(SKILLS_DIR, "registry.json")
MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8001")

if not os.path.exists(REGISTRY_FILE):
    with open(REGISTRY_FILE, "w") as f: json.dump({}, f)
if not os.path.exists(os.path.join(SKILLS_DIR, ".git")):
    try:
        subprocess.run(["git", "init"], cwd=SKILLS_DIR)
        subprocess.run(["git", "config", "user.email", "ai@os"], cwd=SKILLS_DIR)
        subprocess.run(["git", "config", "user.name", "AI"], cwd=SKILLS_DIR)
    except: pass

def load_registry():
    try:
        with open(REGISTRY_FILE, "r") as f: return json.load(f)
    except: return {}

def save_registry(data):
    with open(REGISTRY_FILE, "w") as f: json.dump(data, f, indent=2)

async def create_skill(name, code, description, example):
    safe_name = "".join(x for x in name if x.isalnum() or x == "_")
    filename = f"{safe_name}.py"
    filepath = os.path.join(SKILLS_DIR, filename)
    with open(filepath, "w") as f: f.write(code)
    registry = load_registry()
    registry[safe_name] = {"description": description, "example": example}
    save_registry(registry)
    try:
        subprocess.run(["git", "add", "."], cwd=SKILLS_DIR)
        subprocess.run(["git", "commit", "-m", f"Add {safe_name}"], cwd=SKILLS_DIR)
    except: pass
    async with httpx.AsyncClient() as client:
        try:
            text = f"Skill: {safe_name}\nDescription: {description}\nUsage: {example}"
            await client.post(f"{MEMORY_URL}/remember", json={"text": text, "metadata": {"source": "skill", "name": safe_name}})
        except: pass
    return f"Skill {safe_name} created."
