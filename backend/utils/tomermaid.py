import argparse
import ast
from pathlib import Path

def find_python_files(root: Path):
    return [p for p in root.rglob("*.py") if not any(part.startswith(".") for part in p.parts)]

def detect_internal_packages(root: Path):
    pkgs = set()
    for p in root.rglob("__init__.py"):
        try:
            rel = p.parent.relative_to(root)
            if rel.parts:
                pkgs.add(".".join(rel.parts))
        except Exception:
            pass
    # also treat top-level .py files as modules
    for p in root.glob("*.py"):
        pkgs.add(p.stem)
    # fallback: root folder name as package if it has any __init__.py under it
    if pkgs:
        pkgs.add(root.name)
    return pkgs

def module_name_from_path(pyfile: Path, root: Path):
    rel = pyfile.relative_to(root)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # drop __init__ ending
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else root.name

def parse_imports(py_path: Path):
    try:
        src = py_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = py_path.read_text(encoding="latin-1", errors="ignore")
    try:
        tree = ast.parse(src, filename=str(py_path))
    except SyntaxError:
        return set()

    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                targets.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                targets.add(top)
    return targets

def build_edges(root: Path):
    pyfiles = find_python_files(root)
    internal_pkgs = detect_internal_packages(root)
    nodes = set()
    edges = set()

    # map top-level name -> best guess “internal package root”
    # e.g., if we have package "kgchat" -> treat "kgchat" as internal
    internal_tops = {name.split(".")[0] for name in internal_pkgs}

    for f in pyfiles:
        src_mod = module_name_from_path(f, root)
        nodes.add(src_mod)
        imports = parse_imports(f)
        for top in imports:
            if top in internal_tops:
                # connect to the *top-level* internal package; we keep it simple & readable
                dst = top
                if src_mod != dst:
                    edges.add((src_mod, dst))

    return nodes, edges

def to_mermaid(nodes, edges, direction="LR", max_nodes=None):
    lines = [f"graph {direction}"]
    # Optionally cap nodes to avoid huge graphs
    node_list = list(nodes)
    if max_nodes and len(node_list) > max_nodes:
        node_list = node_list[:max_nodes]
        # filter edges accordingly
        allowed = set(node_list)
        edges = {(a, b) for (a, b) in edges if a in allowed and b in allowed}

    def safe_id(s: str):
        # Mermaid node ids: keep alnum & _ . Convert others to _
        import re
        sid = re.sub(r"[^0-9A-Za-z_.]", "_", s)
        if not sid:
            sid = "node"
        return sid

    # Emit nodes as labels (optional but helps readability)
    emitted_nodes = set()
    for n in node_list:
        nid = safe_id(n)
        lines.append(f'    {nid}["{n}"]')
        emitted_nodes.add(nid)

    # Emit edges
    for a, b in sorted(edges):
        aid = safe_id(a)
        bid = safe_id(b)
        # ensure nodes exist (in case of filtering)
        if aid not in emitted_nodes:
            lines.append(f'    {aid}["{a}"]'); emitted_nodes.add(aid)
        if bid not in emitted_nodes:
            lines.append(f'    {bid}["{b}"]'); emitted_nodes.add(bid)
        lines.append(f"    {aid} --> {bid}")

    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser(description="Generate Mermaid module dependency graph for a Python project.")
    ap.add_argument("project_root", help="Path to project root, e.g., /Documents/codes/KGChat-03")
    ap.add_argument("-o", "--output", default="dependencies.mmd", help="Output .mmd file")
    ap.add_argument("--direction", choices=["LR", "TB", "RL", "BT"], default="LR", help="Graph direction (Mermaid)")
    ap.add_argument("--max-nodes", type=int, default=None, help="Limit node count to keep graph readable")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    nodes, edges = build_edges(root)
    mermaid = to_mermaid(nodes, edges, direction=args.direction, max_nodes=args.max_nodes)

    out_path = Path(args.output).resolve()
    out_path.write_text(mermaid, encoding="utf-8")
    print(f"[OK] Mermaid graph written to: {out_path}")
    print("\nPreview:\n")
    print(mermaid)

if __name__ == "__main__":
    main()
