import os
import tokenize
import io

def remove_comments(source):
    io_obj = io.StringIO(source)
    out = ""
    last_lineno = -1
    last_col = 0
    
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
                
            if token_type == tokenize.COMMENT:
                pass
            else:
                out += token_string
                
            last_col = end_col
            last_lineno = end_line
            
        # Clean up empty lines
        lines = out.splitlines()
        cleaned_lines = []
        for line in lines:
            if line.strip() != "":
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines) + "\n"
    except Exception as e:
        print("Error parsing:", e)
        return source

directories_to_clean = [".", "src"]
for directory in directories_to_clean:
    for filename in os.listdir(directory):
        if filename.endswith(".py") and filename != "remove_comments.py":
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                print(f"Cleaning {filepath}...")
                cleaned_content = remove_comments(content)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned_content)
print("Done cleaning comments.")
