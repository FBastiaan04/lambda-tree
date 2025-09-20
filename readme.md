# Installation
```bash
git clone https://github.com/FBastiaan04/lambda-tree
cd lambda-tree
pip install pygame
python main.py
```

# Usage
What most keys do depends on if the tree is complete (no nodes are missing)

|key|effect complete|effect incomplete|
|---|---|---|
|`backspace`|remove last added node|remove last added node|
|`l`|n/a|insert Lambda node|
|`space`|n/a|insert Apply node|
|any letter (other than `l`)|n/a|insert Variable node|
|shift + any letter|save as shorthand|insert shorthand as node|
|shift + any number|n/a|insert Church number 0-9|
|`enter`|reduce leftmost-outermost|insert term from terminal as node|
|`delete`|delete whole tree|delete whole tree|
|`escape`|exit|exit|

You can also click on the highlighted Apply node of any redex to reduce it

# Shorthands
The shorthands are stored in the `/shorthands` file. This can be manually edited, but saving a shorthand in the program also updates this file. The default shorthands are Identity, True, False and Omega.
Church numbers 0-9 are also shorthands, but those are generated dynamically.

# TODO
? means might not do
- Indicate where you're typing ?
- Output `x Lx.x` instead of `x (Lx.x)` ?
- Correctly structure trees like `x (Lx.x x) Ly.Lz.y z`
- Save user-made shorthands to file
- Zoom