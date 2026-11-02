# Records docs/demo/pipelines.gif from a real run: `vhs docs/demo/pipelines.tape` from the repo root.
Output docs/demo/pipelines.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 460
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m hybrid_rag query "how do plants make energy from sunlight" --pipeline bm25` Enter
Sleep 2s
Type `python -m hybrid_rag query "how do plants make energy from sunlight" --pipeline dense` Enter
Sleep 2s
Type `python -m hybrid_rag query "how do plants make energy from sunlight" --pipeline reranked` Enter
Sleep 3s
