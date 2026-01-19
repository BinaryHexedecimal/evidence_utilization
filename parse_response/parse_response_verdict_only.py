import json
import re


def parse_verdict(text):
    if not isinstance(text, str):
        return None

    text = text.lower()


    if re.search(r"\btrue\b", text):
        return "true"
    elif re.search(r"\bfalse\b", text):
        return "false"
    elif re.search(r"\bmaybe\b", text) or re.search(r"\bunknown\b", text) or re.search(r"\buncertain\b", text):
        return "maybe"
    else:
        return None




def parse_confidence(text: str):

    if not text:
        print(text)
        return None

    text = text.strip()


    m = re.search(r"confidence\s*score\s*:\s*(\d{1,3})", text, re.IGNORECASE)
    if m:
        score = int(m.group(1))
        return score if 0 <= score <= 100 else None


    m = re.search(r"\b(true|false|maybe)\b[\s:\-\(\[]*(\d{1,3})", text, re.IGNORECASE)
    if m:
        score = int(m.group(2))
        return score if 0 <= score <= 100 else None


    first_line = text.split("\n", 1)[0]
    m = re.search(r"\b(\d{1,3})\b", first_line)
    if m:
        score = int(m.group(1))
        return score if 0 <= score <= 100 else None
    print(text)
    return None





def parse_and_update_json(_dir, modes, models, suffix, conf = False ):
    for mode in modes:
        for model in models:   

            _file = f"{model}_{mode}" + suffix 

            file_path = _dir / _file
            
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            items = data if isinstance(data, list) else data["data"]
    
            error_count = 0
    
            for i, item in enumerate(items):
                rb = item.get("response_basic", "")
                ri = item.get("response_internal", "")
                rc = item.get("response_claim", "")
            

                vb = parse_verdict(rb)
                vi = parse_verdict(ri)
                vc = parse_verdict(rc)

                if conf:
                    cb = parse_confidence(rb)
                    ci = parse_confidence(ri)
                    cc = parse_confidence(rc)
            

                item["response_basic_verdict"] = vb
                item["response_internal_verdict"] = vi
                item["response_claim_verdict"] = vc

                if conf:
                    item["response_basic_confidence"] = cb
                    item["response_internal_confidence"] = ci
                    item["response_claim_confidence"] = cc
            

                if vb is None:
                    print(f"[WARN] {file_path} | item {i} | response_basic unparsable → {rb}")
            
                if vi is None:
                    print(f"[WARN] {file_path} | item {i} | response_internal unparsable → {ri}")
            
                if vc is None:
                    print(f"[WARN] {file_path} | item {i} | response_claim unparsable → {rc}")
            

                if conf:
                    if cb is None:
                        print(f"[WARN] {file_path} | item {i} | response_basic confidence missing")
                
                    if ci is None:
                        print(f"[WARN] {file_path} | item {i} | response_internal confidence missing")
                
                    if cc is None:
                        print(f"[WARN] {file_path} | item {i} | response_claim confidence missing")


            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Updated & overwritten: {file_path}")

