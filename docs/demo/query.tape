# Records docs/demo/query.gif from a real run: `vhs docs/demo/query.tape` from the repo root.
Output docs/demo/query.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 520
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms

Hide
Type "export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear" Enter
Sleep 4s
Show
Type `python -m hybrid_rag query "how do plants make energy from sunlight" --pipeline hybrid` Enter
Sleep 2s
Type "python -m hybrid_rag ablate --k 3" Enter
Sleep 5s
