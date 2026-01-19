import json
import re
from pathlib import Path


def parse_verdict(text):

    if not isinstance(text, str) or not text.strip():
        return None

    text = text.lower().strip()

    m = re.search(
        r"verdict\s*[:=]\s*\[?\s*(true|false|maybe|uncertain|unknown)\s*\]?",
        text
    )
    if m:
        v = m.group(1)
        if v in {"uncertain", "unknown"}:
            return "maybe"
        return v

    first_line = text.split("\n", 1)[0].strip()

    m = re.match(
        r"^[\[\(\{]?\s*(true|false|maybe|uncertain|unknown)\s*[\]\)\}]?\b",
        first_line
    )
    if m:
        v = m.group(1)
        if v in {"uncertain", "unknown"}:
            return "maybe"
        return v

    return None



def parse_confidence(text: str):
    if not text:
        return None

    text = text.strip()

    m = re.search(
        r"confidence\s*score\s*:\s*\[?\s*(\d{1,3})\s*\]?",
        text,
        re.IGNORECASE
    )
    if m:
        score = int(m.group(1))
        return score if 0 <= score <= 100 else None
    m = re.search(
        r"\[?\b(true|false|maybe)\b\]?\s*[\s:\-\(\[]*\s*(\d{1,3})\s*\]?",
        text,
        re.IGNORECASE
    )
    if m:
        score = int(m.group(2))
        return score if 0 <= score <= 100 else None
    first_line = text.split("\n", 1)[0]

    m = re.findall(r"\[?\b(\d{1,3})\b\]?", first_line)
    if m:
        score = int(m[-1])  # take the LAST number
        return score if 0 <= score <= 100 else None

    return None



def parse_internal_knowledge(text: str):
    if not text:
        return None

    t = text.lower().strip()

    m = re.search(
        r"\binternal(?:ly)?[-\s]+knowledge\b\s*[:=]\s*\[?\s*(true|false|yes|no)\s*\]?",
        t
    )
    if m:
        val = m.group(1)
        if val == "yes": return "true"
        if val == "no": return "false"
        return val


    if re.search(r"\bno\s+internal(?:ly)?\s+knowledge\b", t):
        return "false"

    if re.search(r"does\s+not\s+indicate\s+any\s+internal", t):
        return "false"


    m2 = re.search(r"\binternally\s*[:=]\s*(true|false)\s*$", t)
    if m2:
        return m2.group(1)


    return None


def parse_bullets(text: str):
    if not text:
        return []

    header_pattern = (
        r"(?:"
        r"here\s+(?:is|are)\s+the\s+analysis"
        r"|here\s+are\s+the\s+key\s+factual\s+insights"
        r"|here's\s+the\s+analysis"
        r"|bullets"
        r")\s*:\s*([\s\S]*)"
    )
    m = re.search(header_pattern, text, re.IGNORECASE)
    bullet_block = m.group(1) if m else text

    bullet_block = re.split(
        r"(?=\b(?:internal(?:ly)?|internal\s+knowledg|internal\s*knowledge|verdict|confidence\s+score)\b)",
        bullet_block,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    bullets = []


    bullet_prefix = r"(?<!\w)(?:--|\-|\d+[.)])"

    dash_style = re.findall(
        rf"{bullet_prefix}\s+(.*?)(?={bullet_prefix}\s+|$)",
        bullet_block,
        flags=re.DOTALL
    )

    bullets.extend([b.strip().strip("[]") for b in dash_style if b.strip()])


    numbered_quote = re.findall(r'\b\d+\s*"([^"]+)"', bullet_block)
    bullets.extend([b.strip() for b in numbered_quote if b.strip()])


    bullets = [b.strip("[] ").strip() for b in bullets if b.strip()]

    return bullets





    

def parse_and_update_json(basedir, modes, models, file_dir, file_suffix):
 

    for mode in modes:
        for model in models:   

            _dir = basedir / file_dir  

            _file = f"{model}_{mode}" + file_suffix
            file_path = _dir / _file
        
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            items = data if isinstance(data, list) else data["data"]
    

    
            for i, item in enumerate(items):
                rb = item.get("response_basic", "")
                ri = item.get("response_internal", "")
                rc = item.get("response_claim", "")
            
                vb = parse_verdict(rb)
                vi = parse_verdict(ri)
                vc = parse_verdict(rc)
            

                cb = parse_confidence(rb)
                ci = parse_confidence(ri)
                cc = parse_confidence(rc)


                bb = parse_bullets(rb)
                bi = parse_bullets(ri)
                bc = parse_bullets(rc)
                

                kb = parse_internal_knowledge(rb)
                ki = parse_internal_knowledge(ri)
                kc = parse_internal_knowledge(rc)
                

                item["response_basic_bullets"] = bb
                item["response_internal_bullets"] = bi
                item["response_claim_bullets"] = bc
                
                item["response_basic_internal_knowledge"] = kb
                item["response_internal_internal_knowledge"] = ki
                item["response_claim_internal_knowledge"] = kc
                


                item["response_basic_verdict"] = vb
                item["response_internal_verdict"] = vi
                item["response_claim_verdict"] = vc
            
                item["response_basic_confidence"] = cb
                item["response_internal_confidence"] = ci
                item["response_claim_confidence"] = cc
            

                if vb is None:
                    print(f"[WARN] {file_path} | item {i} | response_basic unparsable verdict")
            
                if vi is None:
                    print(f"[WARN] {file_path} | item {i} | response_internal unparsable verdict")
            
                if vc is None:
                    print(f"[WARN] {file_path} | item {i} | response_claim unparsable verdict")

                    

                if cb is None:
                    print(f"[WARN] {file_path} | item {i} | response_basic confidence missing")
            
                if ci is None:
                    print(f"[WARN] {file_path} | item {i} | response_internal confidence missing")
            
                if cc is None:
                    print(f"[WARN] {file_path} | item {i} | response_claim confidence missing")



                if not bb:
                    print(f"[WARN] {file_path} | item {i} | response_basic bullets missing")


                if kb is None:
                    print(f"[WARN] {file_path} | item {i} | response_basic internal knowledge missing")



            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Updated & overwritten: {file_path}")

