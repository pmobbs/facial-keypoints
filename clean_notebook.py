import json

notebook_path = '3. Facial Keypoint Detection, Complete Pipeline.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            # Skip lines with TODO markers and associated instruction text
            if 'TODO:' in line:
                continue
            if "You'll need to un-comment" in line:
                continue

            # Specific line cleanup
            if 'print out your net and prepare it for testing' in line:
                line = line.replace('## print out your net and prepare it for testing (uncomment the line below)', '## prepare it for testing')

            new_source.append(line)
        cell['source'] = new_source

    elif cell['cell_type'] == 'markdown':
        new_source = []
        for line in cell['source']:
            # Remove TODO: from headers
            if '### TODO:' in line:
                line = line.replace('### TODO: ', '### ')

            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
